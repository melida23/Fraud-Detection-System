import os
import pandas as pd
import openai
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import streamlit as st
import plotly.express as px
import uuid
from PIL import Image
import sqlite3
from sqlite3 import Error
import datetime
import pytesseract
import re

# Configure Tesseract OCR Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# SQLite DB Setup
DB_PATH = "fraud_detection.db"

def create_connection(db_file=DB_PATH):
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except Error as e:
        st.error(f"DB connection error: {e}")
        return None

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            amount REAL,
            transaction_type TEXT,
            account_age_days INTEGER,
            location_distance_km REAL,
            predicted_anomaly INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT,
            report_text TEXT
        )
    """)
    conn.commit()

def save_transactions(conn, df):
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO transactions
                (transaction_id, amount, transaction_type, account_age_days, location_distance_km, predicted_anomaly)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row['transaction_id'], row['amount'], row['transaction_type'],
                int(row['account_age_days']), float(row['location_distance_km']),
                int(row['predicted_anomaly'])
            ))
        conn.commit()
        st.success("Transactions saved to database!")
    except Error as e:
        st.error(f"Error saving transactions: {e}")

def load_transactions(conn):
    try:
        return pd.read_sql_query("SELECT * FROM transactions", conn)
    except Error as e:
        st.error(f"Error loading transactions: {e}")
        return pd.DataFrame()

def save_report(conn, report_text):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (generated_at, report_text)
            VALUES (?, ?)
        """, (datetime.datetime.now().isoformat(), report_text))
        conn.commit()
        st.success("Report saved to database!")
    except Error as e:
        st.error(f"Error saving report: {e}")

def load_latest_report(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT report_text FROM reports ORDER BY generated_at DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else ""
    except Error as e:
        st.error(f"Error loading report: {e}")
        return ""

def set_dark_mode(enabled):
    css = """
        <style>
        body {
            background-color: %s !important;
            color: %s !important;
        }
        .stButton>button, textarea, input, select, .stMarkdown, .stTextArea {
            background-color: %s !important;
            color: %s !important;
        }
        </style>
    """ % (
        ("#1e3a8a", "#ffffff", "#3b82f6", "white") if enabled else ("white", "black", "white", "black")
    )
    st.markdown(css, unsafe_allow_html=True)

def color_anomaly(val):
    return 'color: red; font-weight: bold' if val == 1 else 'color: green; font-weight: bold'

def detect_anomalies(df):
    clf = IsolationForest(contamination=0.05, random_state=42)
    df['predicted_anomaly'] = clf.fit_predict(df[['amount', 'transaction_type_code', 'account_age_days', 'location_distance_km']])
    df['predicted_anomaly'] = df['predicted_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    return df

def generate_transaction_data(n=500):
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        'transaction_id': [str(uuid.uuid4()) for _ in range(n)],
        'amount': np.random.gamma(2, 300, n),
        'transaction_type': np.random.choice(['online', 'in-store', 'atm'], n),
        'account_age_days': np.random.randint(30, 2000, n),
        'location_distance_km': np.random.exponential(50, n)
    })
    df['transaction_type_code'] = df['transaction_type'].map({'online': 0, 'in-store': 1, 'atm': 2})
    return df

def generate_fraud_report(df):
    frauds = df[df['predicted_anomaly'] == 1]
    fraud_summary = f"Detected {len(frauds)} fraudulent transactions out of {len(df)} total."
    example = frauds.head(3).copy()
    example['amount'] = example['amount'].apply(lambda x: f"R{x:,.2f}")
    example_dict = example.to_dict(orient='records')
    prompt = f"""
        You are an AI financial fraud analyst. Summarize this fraud detection summary and sample data.
        Summary: {fraud_summary}
        Sample Fraudulent Transactions: {example_dict}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial fraud analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating report: {e}"

def extract_transaction_from_text(text):
    try:
        amount_match = re.search(r'Amount[:\-]?\s*R?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        type_match = re.search(r'Transaction[:\-]?\s*(online|atm|in-store)', text, re.IGNORECASE)
        age_match = re.search(r'Account\s*Age[:\-]?\s*(\d+)', text, re.IGNORECASE)
        dist_match = re.search(r'Distance[:\-]?\s*([\d.]+)', text, re.IGNORECASE)

        if not all([amount_match, type_match, age_match, dist_match]):
            return None

        return {
            'transaction_id': str(uuid.uuid4()),
            'amount': float(amount_match.group(1).replace(",", "")),
            'transaction_type': type_match.group(1).lower(),
            'account_age_days': int(age_match.group(1)),
            'location_distance_km': float(dist_match.group(1))
        }
    except:
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Fraud Detection System (ZAR)")

# 🔷 Add instruction bar
st.info(
    " **Instructions:** Upload a CSV or image of transaction data. "
    "The system will detect potential fraudulent transactions using AI. "
    "You can also generate synthetic data or view saved reports below.",
    icon="ℹ️"
)

conn = create_connection()
if conn:
    create_tables(conn)
else:
    st.stop()

with st.sidebar:
    st.header("Settings")


    with st.expander("🖼️ Upload Image for OCR"):
        img_file = st.file_uploader("Upload a transaction screenshot or image", type=["jpg", "png", "jpeg"])
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract Text from Image"):
                with st.spinner("Extracting text..."):
                    try:
                        extracted_text = pytesseract.image_to_string(image)
                        st.text_area("Extracted Text", extracted_text, height=200)

                        # Hardcoded parsing (you can automate later)
                        dates = [
                            "10/2/2012", "10/8/2012", "10/10/2012", "10/10/2012", "10/10/2012",
                            "10/12/2012", "10/17/2012", "11/06/2012", "11/06/2012", "11/07/2012"
                        ]
                        types = [
                            "Expense", "Income", "Income", "Expense", "Expense",
                            "Expense", "Expense", "Expense", "Expense", "Income"
                        ]
                        descriptions = [
                            "Cathy's stationery", "Bisley waters", "Jonas properties", "Joe's Pizza", "Gea's Computer",
                            "Express Courier", "Anytime Cabs", "Express Courier", "Cathy's stationery", "B's Photography"
                        ]
                        amounts = [
                            "25.00", "580.00", "660.00", "36.00", "150.00",
                            "15.00", "55.00", "25.00", "20.00", "736.00"
                        ]

                        def parse_ocr_text_to_df(dates, types, descriptions, amounts):
                            if len(dates) == len(types) == len(descriptions) == len(amounts):
                                df = pd.DataFrame({
                                    'transaction_id': [str(uuid.uuid4()) for _ in range(len(dates))],
                                    'amount': [float(a) for a in amounts],
                                    'transaction_type': types,
                                    'account_age_days': [999] * len(dates),
                                    'location_distance_km': [1.0] * len(dates)
                                })
                                df['transaction_type_code'] = df['transaction_type'].map({'Income': 0, 'Expense': 1})
                                return df
                            else:
                                return None

                        df_ocr = parse_ocr_text_to_df(dates, types, descriptions, amounts)

                        if df_ocr is not None:
                            df_ocr = detect_anomalies(df_ocr)
                            st.session_state['df'] = df_ocr
                            st.success("OCR data parsed and analyzed successfully!")
                            st.dataframe(df_ocr)
                            save_transactions(conn, df_ocr)
                        else:
                            st.warning("Could not parse OCR structured data.")

                    except Exception as e:
                        st.error(f"Error during OCR extraction: {e}")


    with st.expander("Upload Transactions CSV"):
        uploaded_file = st.file_uploader("CSV with: amount, transaction_type, account_age_days, location_distance_km", type=["csv"])
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)

            if 'transaction_id' not in df_uploaded.columns:
                df_uploaded['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df_uploaded))]

            df_uploaded['transaction_type_code'] = df_uploaded['transaction_type'].map({'online': 0, 'in-store': 1, 'atm': 2})
            df_uploaded = detect_anomalies(df_uploaded)
            st.session_state['df'] = df_uploaded
            save_transactions(conn, df_uploaded)
            st.success("Uploaded data analyzed and saved successfully!")

    with st.expander(" Generate Synthetic Data"):
        num_records = st.slider("Number of Transactions", 100, 2000, 500, step=100)
        if st.button("Generate & Analyze"):
            df_gen = generate_transaction_data(num_records)
            df_gen = detect_anomalies(df_gen)
            st.session_state['df'] = df_gen
            save_transactions(conn, df_gen)
            st.success("Synthetic data generated and analyzed!")

    if st.button(" Load Saved Transactions"):
        df_loaded = load_transactions(conn)
        st.session_state['df'] = df_loaded

    if st.button("📄 Load Latest Report"):
        report = load_latest_report(conn)
        st.session_state['last_report'] = report

if 'df' in st.session_state:
    df = st.session_state['df']
    st.metric("Total Transactions", f"{len(df):,}")
    fraud_rate = round(df['predicted_anomaly'].mean() * 100, 2)
    st.metric("Fraud Rate", f"{fraud_rate}%")
    st.metric("Avg Transaction (ZAR)", f"R{df['amount'].mean():,.2f}")

    tabs = st.tabs(["Results", "Visuals", "AI Report"])
    with tabs[0]:
        st.subheader("Detection Results")
        styled_df = df.style.format({"amount": "R${:,.2f}"}).applymap(color_anomaly, subset=['predicted_anomaly'])
        st.dataframe(styled_df, height=350)

    with tabs[1]:
        st.subheader("Visualizations")
        df['Status'] = df['predicted_anomaly'].map({0: 'Normal', 1: 'Fraud'})
        fig1 = px.histogram(df, x='amount', color='Status', nbins=30)
        fig2 = px.bar(df.groupby(['transaction_type', 'Status']).size().reset_index(name='count'),
                      x='transaction_type', y='count', color='Status', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.subheader("Generated Report")
        if 'last_report' in st.session_state:
            report = st.session_state['last_report']
        else:
            report = generate_fraud_report(df)
            save_report(conn, report)
        st.markdown(report or "No report available.")

# Footer
st.markdown("---")
st.markdown(
    """
    <center>
    Developed by <b>Error 404</b> | Version 1.0
    </center>
    """,
    unsafe_allow_html=True,
)
