import os
import pandas as pd
import openai
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import streamlit as st
import plotly.express as px
import uuid
import pytesseract
from PIL import Image
import sqlite3
from sqlite3 import Error
import datetime
 
# --- SQLite DB Setup ---
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
                row['transaction_id'],
                row['amount'],
                row['transaction_type'],
                int(row['account_age_days']),
                float(row['location_distance_km']),
                int(row['predicted_anomaly']),
            ))
        conn.commit()
        st.success("Transactions saved to database!")
    except Error as e:
        st.error(f"Error saving transactions: {e}")
 
def load_transactions(conn):
    try:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
        return df
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
        cursor.execute("""
            SELECT report_text FROM reports
            ORDER BY generated_at DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return row[0] if row else ""
    except Error as e:
        st.error(f"Error loading report: {e}")
        return ""
 
# --- Streamlit app setup ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")
 
# OCR Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
 
# Load env variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"
 
# Initialize DB connection and tables
conn = create_connection()
if conn:
    create_tables(conn)
else:
    st.error("Cannot initialize database connection.")
 
# --- Dark mode CSS function ---
def set_dark_mode(enabled):
    if enabled:
        st.markdown(
            """
            <style>
            body {
                background-color: #0e1117 !important;
                color: #e6e6e6 !important;
            }
            .stButton>button {
                background-color: #1e2a38 !important;
                color: white !important;
            }
            textarea, input, select {
                background-color: #22272e !important;
                color: #e6e6e6 !important;
            }
            .stMarkdown, .stTextArea {
                color: #e6e6e6 !important;
            }
            .css-1d391kg {
                background-color: #0e1117 !important;
            }
            .css-1v0mbdj {
                color: #e6e6e6 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: white !important;
                color: black !important;
            }
            .stButton>button {
                background-color: #f0f0f0 !important;
                color: black !important;
            }
            textarea, input, select {
                background-color: white !important;
                color: black !important;
            }
            .stMarkdown, .stTextArea {
                color: black !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
 
# --- Anomaly color for DataFrame ---
def color_anomaly(val):
    color = 'red' if val == 1 else 'green'
    return f'color: {color}; font-weight: bold'
 
# --- Core functions ---
def detect_anomalies(df):
    features = ['amount', 'transaction_type_code', 'account_age_days', 'location_distance_km']
    clf = IsolationForest(contamination=0.05, random_state=42)
    df['predicted_anomaly'] = clf.fit_predict(df[features])
    df['predicted_anomaly'] = df['predicted_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    return df
 
def generate_transaction_data(n=500):
    import numpy as np
    np.random.seed(42)
    data = {
        'transaction_id': [str(uuid.uuid4()) for _ in range(n)],
        'amount': np.random.gamma(2, 300, n),
        'transaction_type': np.random.choice(['online', 'in-store', 'atm'], n),
        'account_age_days': np.random.randint(30, 2000, n),
        'location_distance_km': np.random.exponential(50, n),
    }
    df = pd.DataFrame(data)
    df['transaction_type_code'] = df['transaction_type'].map({'online': 0, 'in-store': 1, 'atm': 2})
    return df
 
def generate_fraud_report(df):
    frauds = df[df['predicted_anomaly'] == 1]
    fraud_summary = f"Detected {len(frauds)} fraudulent transactions out of {len(df)} total."
    example = frauds.head(3).copy()
    example['amount'] = example['amount'].apply(lambda x: f"R{x:,.2f}")
    example_dict = example.to_dict(orient='records')
    prompt = (
        f"You are an AI financial fraud analyst. Summarize the following fraud detection summary and sample data.\n"
        f"Summary: {fraud_summary}\n"
        f"Sample Fraudulent Transactions (Amounts in ZAR): {example_dict}\n"
        f"Provide a concise, professional report."
    )
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
 
# Blue background theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f0ff;  /* Light blue background */
        color: #003366;
    }
    section[data-testid="stSidebar"] {
        background-color: #cce0ff;
    }
    .css-1d391kg, .css-ffhzg2 {
        background-color: #f0f8ff !important;
        color: #003366 !important;
    }
    .stTextInput > div > input {
        background-color: #ffffff !important;
        color: #003366 !important;
    }
    .css-10trblm, .css-hxt7ib {
        color: #002244 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
with st.sidebar:
    st.header("Settings")
 
    dark_mode = st.checkbox("🌙 Enable Dark Mode", value=st.session_state.get("dark_mode", False))
    st.session_state["dark_mode"] = dark_mode
    set_dark_mode(dark_mode)
 
    with st.expander("🖼️ Upload Image for OCR"):
        img_file = st.file_uploader("Upload a transaction screenshot or image", type=["jpg", "png", "jpeg"])
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Extract Text from Image"):
                with st.spinner("Extracting text..."):
                    try:
                        extracted_text = pytesseract.image_to_string(image)
                        st.success("Extracted Text:")
                        st.text_area("Text Output", extracted_text, height=200)
                    except Exception as e:
                        st.error(f"Error during OCR: {e}")
 
    with st.expander("Upload Your Transactions CSV"):
        uploaded_file = st.file_uploader(
            "CSV with columns: amount, transaction_type, account_age_days, location_distance_km",
            type=["csv"]
        )
        if uploaded_file:
            st.info("File uploaded — processing...")
 
    with st.expander("Or Generate Synthetic Data"):
        num_records = st.slider("Number of Transactions", 100, 2000, 500, step=100)
        gen_btn = st.button("Generate & Analyze Synthetic Data")
 
    # Add buttons to load saved data
    if st.button("Load Saved Transactions"):
        if conn:
            df_loaded = load_transactions(conn)
            if not df_loaded.empty:
                st.session_state['df'] = df_loaded
                st.success("Loaded saved transactions from DB")
            else:
                st.warning("No saved transactions found in database.")
 
    if st.button("Load Latest Report"):
        if conn:
            latest_report = load_latest_report(conn)
            if latest_report:
                st.session_state['last_report'] = latest_report
                st.success("Loaded latest report from DB")
            else:
                st.warning("No reports found in database.")
 
# ---- Main app UI ----
st.title("💳 Fraud Detection System (Amounts in ZAR)")
 
use_uploaded = False
if 'uploaded_file' in locals() and uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        required_cols = ['amount', 'transaction_type', 'account_age_days', 'location_distance_km']
        if all(col in df_uploaded.columns for col in required_cols):
            df = df_uploaded.copy()
            df['transaction_type_code'] = df['transaction_type'].map({'online': 0, 'in-store': 1, 'atm': 2})
            with st.spinner("Analyzing uploaded data..."):
                df = detect_anomalies(df)
                if conn:
                    save_transactions(conn, df)
            st.session_state['df'] = df
            st.success("Uploaded data analyzed successfully!")
            use_uploaded = True
        else:
            st.error(f"Uploaded file is missing required columns: {required_cols}")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
 
if not use_uploaded and 'gen_btn' in locals() and gen_btn:
    with st.spinner("Generating and analyzing synthetic data..."):
        df = generate_transaction_data(num_records)
        df = detect_anomalies(df)
        if conn:
            save_transactions(conn, df)
    st.session_state['df'] = df
    st.success("Synthetic data generated and analyzed!")
 
if 'df' in st.session_state:
    df = st.session_state['df']
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    fraud_rate = round(df['predicted_anomaly'].mean() * 100, 2)
    col2.metric(
        "Fraud Rate",
        f"{fraud_rate}%",
        delta=f"{fraud_rate - 5:.2f}%",
        delta_color="inverse" if fraud_rate > 5 else "normal",
    )
    col3.metric("Avg Transaction (ZAR)", f"R{df['amount'].mean():,.2f}")
 
    tabs = st.tabs(["Results Table", "Visualizations", "AI Report"])
 
    with tabs[0]:
        st.subheader("Fraud Detection Results")
        df_display = df[[
            'transaction_id', 'amount', 'transaction_type',
            'account_age_days', 'location_distance_km', 'predicted_anomaly'
        ]].reset_index(drop=True)
 
        styled_df = (
            df_display.style
            .format({"amount": "R${:,.2f}"})
            .applymap(color_anomaly, subset=['predicted_anomaly'])
        )
        st.dataframe(styled_df, height=350)
 
    with tabs[1]:
        st.subheader("📈 Visualizations")
        df['Status'] = df['predicted_anomaly'].map({0: 'Normal', 1: 'Fraud'})
 
        fig1 = px.histogram(
            df,
            x='amount',
            color='Status',
            nbins=30,
            title="Transaction Amount Distribution by Fraud Status",
            labels={"amount": "Amount (ZAR)"},
            color_discrete_map={'Normal': 'blue', 'Fraud': 'red'},
            opacity=0.75
        )
        st.plotly_chart(fig1, use_container_width=True)
 
        fig2 = px.bar(
            df.groupby(['transaction_type', 'Status']).size().reset_index(name='count'),
            x='transaction_type',
            y='count',
            color='Status',
            barmode='group',
            title="Transaction Type by Fraud Status",
            labels={"transaction_type": "Transaction Type", "count": "Count"},
            color_discrete_map={'Normal': 'blue', 'Fraud': 'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    with tabs[2]:
        st.subheader("📝 Generated Report")
        with st.spinner("Generating report..."):
            if 'last_report' in st.session_state:
                report = st.session_state['last_report']
            else:
                report = generate_fraud_report(df)
                if report and conn:
                    save_report(conn, report)
        st.markdown(report or "No report available.")
 
        if st.button("Copy Report Text"):
            st.code(report)
 
else:
    st.info("Upload a CSV file or generate synthetic transactions from the sidebar to get started.")
 
# Footer
st.markdown("---")
st.markdown(
    """
    <center>
    Developed by **Error 404** | Version 1.0 |
    </center>
    """,
    unsafe_allow_html=True,
)
 