Establishing rules and timing is like setting up a formation for the Qimen Dunjia, and creating a holographic AI linked to the Earth system to simulate the operation status and laws of celestial bodies
# =============================
# 🚀 JTYYLSPH — AI Governance Platform
# =============================
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI
# =============================
# CONSTANTS
# =============================
LOG_FILE = "/tmp/audit_log.jsonl"
# =============================
# AUDIT FUNCTIONS
# =============================
def log_run(model_name, drift, fairness, stability, jurisdiction):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "drift": round(float(drift), 4),
        "fairness": round(float(fairness), 4),
        "stability": round(float(stability), 4),
        "jurisdiction": jurisdiction
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except:
        pass
def load_logs():
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                continue
    return logs
# =============================
# STREAM SIMULATION
# =============================
def simulate_stream(X_test, steps=20, noise_level=0.05):
    current = X_test.copy().astype(float)
    for step in range(steps):
        try:
            noise = np.random.normal(0, noise_level, current.shape)
            current = current + noise
            current = current.fillna(0)
            yield step, current
            time.sleep(0.2)
        except:
            yield step, current
# =============================
# GOVERNANCE METRICS
# =============================
def compute_drift(X_train, X_test):
    try:
        col = X_train.select_dtypes(include=[np.number]).columns[0]
        x1 = X_train[col].fillna(0)
        x2 = X_test[col].fillna(0)
        return float(wasserstein_distance(x1, x2))
    except:
        return 0.0
def compute_fairness(preds, y_true):
    try:
        return float(abs(np.mean(preds) - np.mean(y_true)))
    except:
        return 0.0
def system_stability_score(drift, fairness):
    score = (1 - drift) * 0.5 + (1 - fairness) * 0.5
    return max(0.0, min(1.0, float(score)))
def status_label(value):
    if value < 0.3:
        return "🟢"
    elif value < 0.6:
        return "🟡"
    return "🔴"
# =============================
# PDF REPORT
# =============================
def generate_pdf_report(drift, fairness, stability, filename="risk_report.pdf"):
    file_path = os.path.join("/tmp", filename)
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []
    content.append(Paragraph("AI Governance Risk Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Executive Summary", styles["Heading2"]))
    content.append(Paragraph(
        "This report evaluates model performance across drift, fairness, and system stability.",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Key Metrics", styles["Heading2"]))
    content.append(Paragraph(f"Drift Score: {round(drift,3)}", styles["Normal"]))
    content.append(Paragraph(f"Fairness Gap: {round(fairness,3)}", styles["Normal"]))
    content.append(Paragraph(f"System Stability: {round(stability,3)}", styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Risk Assessment", styles["Heading2"]))
    content.append(Paragraph(
        "⚠ Data drift detected." if drift > 0.3 else "✔ Drift acceptable.",
        styles["Normal"]
    ))
    content.append(Paragraph(
        "⚠ Bias risk detected." if fairness > 0.1 else "✔ Fairness acceptable.",
        styles["Normal"]
    ))
    content.append(Paragraph(
        "❌ System unstable." if stability < 0.5 else "✔ System stable.",
        styles["Normal"]
    ))
    verdict = (
        "HIGH RISK — Deployment not recommended."
        if stability < 0.5 else
        "MEDIUM RISK — Monitoring required."
        if drift > 0.3 or fairness > 0.1 else
        "LOW RISK — System acceptable."
    )
    content.append(Spacer(1, 12))
    content.append(Paragraph("Final Verdict", styles["Heading2"]))
    content.append(Paragraph(verdict, styles["Normal"]))
    doc.build(content)
    return file_path
# =============================
# FILE INGESTION
# =============================
def ingest_file(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        elif file.name.endswith(".json"):
            return pd.read_json(file)
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to read {file.name}: {e}")
        return None
# =============================
# 🔒 YOUR BLOCK (UNCHANGED)
# =============================
# =============================
# DARK MODE TOGGLE
# =============================
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
# =============================
# PAGE TITLE
# =============================
st.title("🚀 JTYYLSPH — AI Governance Platform")
# =============================
# SESSION STATE
# =============================
if "model" not in st.session_state:
    st.session_state.model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# =============================
# SIDEBAR
# =============================
st.sidebar.header("Compliance Mode")
jurisdiction = st.sidebar.selectbox(
    "Select Regulatory Framework",
    [
        "United States (SR 11-7)",
        "European Union (EU AI Act)",
        "UK Model Risk Guidance",
        "APAC General Risk Framework",
        "Custom Enterprise Policy",
    ],
)
st.sidebar.header("Dataset Controls")
domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "Business", "Emotion", "General"]
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
uploaded_files = st.sidebar.file_uploader(
    "Upload Dataset or Documents",
    accept_multiple_files=True,
    type=[
        "csv", "xlsx", "json", "parquet",
        "pdf", "docx", "txt", "log",
        "xml", "sql",
        "png", "jpg", "jpeg",
    ],
)
st.sidebar.header("Database Connection")
db_url = st.sidebar.text_input("SQLAlchemy DB URL", placeholder="postgresql://user:pass@host:5432/db")
query = st.sidebar.text_area("SQL Query", placeholder="SELECT * FROM table LIMIT 100")
def generate_domain_dataset(domain, n_samples=500):
    rng = np.random.default_rng(42)
    if domain == "Finance":
        df = pd.DataFrame({
            "credit_score": rng.normal(650, 50, n_samples),
            "income": rng.normal(70000, 20000, n_samples),
            "debt_ratio": rng.uniform(0.1, 0.8, n_samples),
            "loan_amount": rng.normal(20000, 8000, n_samples),
        })
        df["target"] = (
            (df["credit_score"] < 620) |
            (df["debt_ratio"] > 0.5)
        ).astype(int)
    elif domain == "Healthcare":
        df = pd.DataFrame({
            "age": rng.integers(20, 80, n_samples),
            "bmi": rng.normal(27, 5, n_samples),
            "blood_pressure": rng.normal(120, 15, n_samples),
            "cholesterol": rng.normal(200, 40, n_samples),
        })
        df["target"] = (
            (df["bmi"] > 30) |
            (df["blood_pressure"] > 140)
        ).astype(int)
    elif domain == "Sports":
        df = pd.DataFrame({
            "speed": rng.normal(25, 5, n_samples),
            "strength": rng.normal(70, 10, n_samples),
            "stamina": rng.normal(60, 15, n_samples),
            "reaction_time": rng.normal(0.3, 0.05, n_samples),
        })
        df["target"] = (
            (df["speed"] > 28) &
            (df["reaction_time"] < 0.28)
        ).astype(int)
    elif domain == "Business":
        df = pd.DataFrame({
            "revenue": rng.normal(1e6, 3e5, n_samples),
            "expenses": rng.normal(7e5, 2e5, n_samples),
            "customer_growth": rng.normal(0.1, 0.05, n_samples),
            "market_share": rng.uniform(0.01, 0.3, n_samples),
        })
        df["target"] = (
            (df["revenue"] - df["expenses"] > 2e5) &
            (df["customer_growth"] > 0.1)
        ).astype(int)
    elif domain == "Emotion":
        df = pd.DataFrame({
            "valence": rng.uniform(-1, 1, n_samples),
            "arousal": rng.uniform(0, 1, n_samples),
            "dominance": rng.uniform(0, 1, n_samples),
            "speech_rate": rng.normal(150, 30, n_samples),
        })
        df["target"] = (
            (df["valence"] > 0.2) &
            (df["arousal"] > 0.5)
        ).astype(int)
    else:  # General
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=6, random_state=42)
        df = pd.DataFrame(X)
        df["target"] = y
    return df
# =============================
# DATA LOADING
# =============================
X, y, df = None, None, None
if query and db_url:
    from sqlalchemy import create_engine
    try:
        engine = create_engine(db_url)
        df = pd.read_sql(query, engine)
        st.write("Database Data")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()
elif uploaded_files:
    dfs = []
    for f in uploaded_files:
        df_part = ingest_file(f)
        if df_part is not None:
            dfs.append(df_part)
    if dfs:
        df = pd.concat(dfs, ignore_index=True, sort=False)
        st.write("Combined Dataset")
        st.dataframe(df.head())
elif uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded Dataset")
    st.dataframe(df.head())
else:
    df = generate_domain_dataset(domain)
st.info(f"Using synthetic dataset: {domain}")
    df = pd.DataFrame(X_data)
    df["target"] = y_data
    st.info(f"Using synthetic dataset: {domain}")
# =============================
# TARGET SELECTION
# =============================
if df is not None:
    if len(df.columns) > 1:
        target_col = st.sidebar.selectbox("Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = pd.Series(np.random.randint(0, 2, len(df)))
if X is None or len(X) == 0:
    st.error("No valid dataset loaded.")
    st.stop()
X.columns = [str(c) for c in X.columns]
st.write("Dataset Shape:", X.shape)
st.write("### Dataset Summary")
st.write(X.describe())
# =============================
# TRAIN + DASHBOARD + AI
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y)
if st.button("Train Model"):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    drift = compute_drift(X_train, X_test)
    fairness = compute_fairness(preds, y_test)
    stability = system_stability_score(drift, fairness)
    st.session_state.model = model
    st.session_state.metrics = (drift, fairness, stability)
    log_run("RandomForest", drift, fairness, stability, jurisdiction)
if st.session_state.metrics:
    drift, fairness, stability = st.session_state.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Drift", drift, status_label(drift))
    c2.metric("Fairness", fairness, status_label(fairness))
    c3.metric("Stability", stability, status_label(1 - stability))
    if st.button("Generate PDF"):
        path = generate_pdf_report(drift, fairness, stability)
        with open(path, "rb") as f:
            st.download_button("Download", f, file_name="report.pdf")
# =============================
# SIMULATION
# =============================
if st.session_state.model and st.button("Start Simulation"):
    chart = st.line_chart()
    for step, current in simulate_stream(X_test):
        preds = st.session_state.model.predict(current)
        d = compute_drift(X_train, current)
        f = compute_fairness(preds, y_test)
        chart.add_rows({"Drift": [d], "Fairness": [f]})
# =============================
# LOGS
# =============================
st.subheader("Audit Logs")
logs = load_logs()
if logs:
    st.dataframe(pd.DataFrame(logs))
# =============================
# AI ASSISTANT
# =============================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
    user_input = st.chat_input("Ask about your model")
    if user_input:
        drift, fairness, stability = st.session_state.metrics or (0,0,0)
        context = f"""
        Drift: {drift}
        Fairness: {fairness}
        Stability: {stability}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ]
        )
        st.write(response.choices[0].message.content)


