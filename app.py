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

# UI SETTINGS

# =============================

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

if dark_mode:

    st.markdown("""

        <style>

        .stApp { background-color: #0e1117; color: white; }

        </style>

    """, unsafe_allow_html=True)

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

jurisdiction = st.sidebar.selectbox(

    "Regulatory Framework",

    [

        "United States (SR 11-7)",

        "EU AI Act",

        "UK Guidance",

        "APAC",

        "Custom",

    ],

)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =============================

# DATA

# =============================

if uploaded:

    df = pd.read_csv(uploaded)

else:

    X_data, y_data = make_classification(n_samples=500, n_features=6)

    df = pd.DataFrame(X_data)

    df["target"] = y_data

target_col = st.selectbox("Target Column", df.columns)

X = df.drop(columns=[target_col])

y = df[target_col]

if y.dtype == "object":

    y = LabelEncoder().fit_transform(y)

# =============================

# TRAIN

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

# =============================

# DASHBOARD

# =============================

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
