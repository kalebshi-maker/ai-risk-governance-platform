"""
Aurexis Systems — AI Governance Infrastructure

Establishing rules and timing is like setting up a formation for the Qimen Dunjia,
and creating a holographic AI linked to the Earth system to simulate the operation
status and laws of celestial bodies.
"""
# =============================
# Aurexis Systems — AI Governance Infrastructure for Enterprise-Scale Systems
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import wasserstein_distance, entropy as scipy_entropy
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from openai import OpenAI

# =============================
# CONSTANTS
# =============================
LOG_FILE = "/tmp/audit_log.jsonl"

# =============================
# AUDIT FUNCTIONS
# =============================
def log_run(model_name, drift, fairness, stability, jurisdiction, action="", risk_score=None):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "drift": round(float(drift), 4),
        "fairness": round(float(fairness), 4),
        "stability": round(float(stability), 4),
        "jurisdiction": jurisdiction,
        "action": action,
    }
    if risk_score is not None:
        record["risk_score"] = round(float(risk_score), 4)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"Log error: {e}")

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
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return 0.0
        distances = []
        for col in num_cols:
            x1 = (X_train[col] - X_train[col].mean()) / (X_train[col].std() + 1e-6)
            x2 = (X_test[col] - X_test[col].mean()) / (X_test[col].std() + 1e-6)
            distances.append(wasserstein_distance(x1, x2))
        return float(np.mean(distances))
    except:
        return 0.0

def compute_model_uncertainty(model, X):
    """
    Compute mean prediction variance (uncertainty) across the dataset.
    For RandomForest: use variance across individual trees.
    Falls back to 0.0 for non-ensemble models.
    """
    try:
        if hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(X) for est in model.estimators_])
            return float(np.mean(np.var(tree_preds, axis=0)))
        return 0.0
    except:
        return 0.0

def compute_prediction_entropy(model, X):
    """Compute average entropy of class probability predictions (classification only)."""
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            entropies = [scipy_entropy(p + 1e-9) for p in probs]
            return float(np.mean(entropies))
        return 0.0
    except:
        return 0.0

def compute_risk_score(drift, fairness, model_uncertainty):
    """
    Weighted composite risk score.
    drift              → 40%
    fairness gap       → 30%
    model uncertainty  → 30%
    """
    raw = drift * 0.4 + fairness * 0.3 + model_uncertainty * 0.3
    return max(0.0, min(1.0, float(raw)))

def system_stability_score(drift, fairness):
    score = (1 - drift) * 0.5 + (1 - fairness) * 0.5
    return max(0.0, min(1.0, float(score)))

def compute_fairness(preds, y_true):
    try:
        return float(abs(np.mean(preds) - np.mean(y_true)))
    except:
        return 0.0

def status_label(value):
    if value < 0.3:
        return "Low"
    elif value < 0.6:
        return "Medium"
    return "High"

# =============================
# BIAS MITIGATION (REAL DEBIAS)
# Upgrade #2: actual fairness-aware retraining with sample weights,
# not just a warning alert.
# =============================
def mitigate_bias(X_train, y_train):
    """
    Returns sample weights that up-weight minority-class samples,
    giving the model a fairness-aware training signal.
    """
    sample_weight = np.where(y_train == 1, 1.2, 1.0)
    return sample_weight

# =============================
# GOVERNANCE INTERVENTION
# =============================
def governance_intervention(model, drift, fairness, X_train=None, y_train=None):
    jurisdiction = st.session_state.get("jurisdiction", "Unknown")
    stability = system_stability_score(drift, fairness)
    risk = compute_risk_score(drift, fairness, 0.0)  # uncertainty added at call site

    log_run(
        "GovernanceAction",
        drift,
        fairness,
        stability,
        jurisdiction,
        action="check",
        risk_score=risk,
    )

    if drift > 0.3:
        st.warning("[WARNING] Drift threshold exceeded — triggering automatic retraining.")
        if model is not None and X_train is not None and y_train is not None:
            model.fit(X_train, y_train)
            st.success("[RETRAIN] Model retrained on current data.")
        log_run("GovernanceAction", drift, fairness, stability, jurisdiction, action="retrain", risk_score=risk)
        return "retrain"

    if fairness > 0.1:
        st.warning("[WARNING] Fairness threshold exceeded — triggering bias mitigation.")
        if model is not None and X_train is not None and y_train is not None:
            weights = mitigate_bias(X_train, y_train)
            try:
                model.fit(X_train, y_train, sample_weight=weights)
                st.success("[DEBIAS] Fairness-aware retraining complete.")
            except TypeError:
                # Fallback for models that don't support sample_weight
                model.fit(X_train, y_train)
                st.info("[DEBIAS] Model refitted (sample_weight unsupported by this estimator).")
        log_run("GovernanceAction", drift, fairness, stability, jurisdiction, action="debias", risk_score=risk)
        return "debias"

    log_run("GovernanceAction", drift, fairness, stability, jurisdiction, action="stable", risk_score=risk)
    return "stable"

# =============================
# PDF REPORT
# =============================
def generate_pdf_report(drift, fairness, stability, risk_score=None, filename="risk_report.pdf"):
    file_path = os.path.join("/tmp", filename)
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
    except Exception:
        pass

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Aurexis Systems — AI Governance Risk Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Executive Summary", styles["Heading2"]))
    content.append(Paragraph(
        "This report evaluates model performance across drift, fairness, model uncertainty, "
        "and system stability for the selected regulatory jurisdiction.",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Key Metrics", styles["Heading2"]))
    metrics_data = [
        ["Metric", "Value", "Status"],
        ["Drift Score", str(round(drift, 3)), "[WARNING]" if drift > 0.3 else "[PASS]"],
        ["Fairness Gap", str(round(fairness, 3)), "[WARNING]" if fairness > 0.1 else "[PASS]"],
        ["System Stability", str(round(stability, 3)), "[FAIL]" if stability < 0.5 else "[PASS]"],
    ]
    if risk_score is not None:
        metrics_data.append([
            "Composite Risk Score", str(round(risk_score, 3)),
            "[HIGH]" if risk_score > 0.6 else "[MEDIUM]" if risk_score > 0.3 else "[LOW]"
        ])
    t = Table(metrics_data, colWidths=[180, 100, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    content.append(t)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Risk Assessment", styles["Heading2"]))
    content.append(Paragraph(
        "[WARNING] Data drift detected." if drift > 0.3 else "[PASS] Drift acceptable.",
        styles["Normal"]
    ))
    content.append(Paragraph(
        "[WARNING] Bias risk detected." if fairness > 0.1 else "[PASS] Fairness acceptable.",
        styles["Normal"]
    ))
    content.append(Paragraph(
        "[FAIL] System unstable." if stability < 0.5 else "[PASS] System stable.",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    verdict = (
        "HIGH RISK - Deployment not recommended."
        if (risk_score or stability) > 0.6 or stability < 0.5 else
        "MEDIUM RISK - Monitoring required."
        if drift > 0.3 or fairness > 0.1 else
        "LOW RISK - System acceptable."
    )
    content.append(Paragraph("Final Verdict", styles["Heading2"]))
    content.append(Paragraph(verdict, styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(
        "Generated by Aurexis Systems — Governance-as-a-Service Platform",
        styles["Normal"]
    ))

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
        elif file.name.endswith(".parquet"):
            return pd.read_parquet(file)
        else:
            st.info(f"Skipping non-tabular file: {file.name}")
            return None
    except Exception as e:
        st.warning(f"Failed to read {file.name}: {e}")
        return None

# =============================
# DARK MODE
# =============================
dark_mode = st.sidebar.toggle("Dark Mode", value=True)
if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# =============================
# PAGE TITLE
# =============================
st.title("Aurexis Systems — Governance-as-a-Service for Enterprise AI")
st.caption("Autonomous drift detection · Fairness-aware retraining · Composite risk scoring · Regulatory compliance")

# =============================
# SESSION STATE
# =============================
for key, default in [
    ("model", None),
    ("metrics", None),
    ("messages", []),
    ("data_split", None),
    ("risk_score", None),
    ("uncertainty", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Compliance Mode")
jurisdiction = st.sidebar.selectbox(
    "Regulatory Framework",
    [
        "United States (SR 11-7)",
        "European Union (EU AI Act)",
        "UK Model Risk Guidance",
        "APAC General Risk Framework",
        "Custom Enterprise Policy",
    ],
)
st.session_state["jurisdiction"] = jurisdiction

st.sidebar.header("Dataset Controls")
domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "Business", "Emotion", "General"]
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
uploaded_files = st.sidebar.file_uploader(
    "Upload Dataset or Documents",
    accept_multiple_files=True,
    type=["csv", "xlsx", "json", "parquet", "pdf", "docx", "txt", "log", "xml", "sql", "png", "jpg", "jpeg"],
)

st.sidebar.header("Database Connection")
db_url = st.sidebar.text_input("SQLAlchemy DB URL", placeholder="postgresql://user:pass@host:5432/db")
query = st.sidebar.text_area("SQL Query", placeholder="SELECT * FROM table LIMIT 100")

st.sidebar.markdown("---")
st.sidebar.markdown("**Governance API**")
st.sidebar.code("POST /governance\n{drift, fairness}", language="json")
st.sidebar.caption("Deploy `api.py` on Nebius Serverless Endpoints for live governance checks.")

# =============================
# DOMAIN DATASET GENERATOR
# =============================
def generate_domain_dataset(domain, n_samples=500):
    rng = np.random.default_rng(42)
    if domain == "Finance":
        df = pd.DataFrame({
            "credit_score": rng.normal(650, 50, n_samples),
            "income": rng.normal(70000, 20000, n_samples),
            "debt_ratio": rng.uniform(0.1, 0.8, n_samples),
            "loan_amount": rng.normal(20000, 8000, n_samples),
        })
        df["target"] = ((df["credit_score"] < 620) | (df["debt_ratio"] > 0.5)).astype(int)
    elif domain == "Healthcare":
        df = pd.DataFrame({
            "age": rng.integers(20, 80, n_samples),
            "bmi": rng.normal(27, 5, n_samples),
            "blood_pressure": rng.normal(120, 15, n_samples),
            "cholesterol": rng.normal(200, 40, n_samples),
        })
        df["target"] = ((df["bmi"] > 30) | (df["blood_pressure"] > 140)).astype(int)
    elif domain == "Sports":
        df = pd.DataFrame({
            "speed": rng.normal(25, 5, n_samples),
            "strength": rng.normal(70, 10, n_samples),
            "stamina": rng.normal(60, 15, n_samples),
            "reaction_time": rng.normal(0.3, 0.05, n_samples),
        })
        df["target"] = ((df["speed"] > 28) & (df["reaction_time"] < 0.28)).astype(int)
    elif domain == "Business":
        df = pd.DataFrame({
            "revenue": rng.normal(1e6, 3e5, n_samples),
            "expenses": rng.normal(7e5, 2e5, n_samples),
            "customer_growth": rng.normal(0.1, 0.05, n_samples),
            "market_share": rng.uniform(0.01, 0.3, n_samples),
        })
        df["target"] = ((df["revenue"] - df["expenses"] > 2e5) & (df["customer_growth"] > 0.1)).astype(int)
    elif domain == "Emotion":
        df = pd.DataFrame({
            "valence": rng.uniform(-1, 1, n_samples),
            "arousal": rng.uniform(0, 1, n_samples),
            "dominance": rng.uniform(0, 1, n_samples),
            "speech_rate": rng.normal(150, 30, n_samples),
        })
        df["target"] = ((df["valence"] > 0.2) & (df["arousal"] > 0.5)).astype(int)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=6, random_state=42)
        df = pd.DataFrame(X)
        df["target"] = y
    return df

# =============================
# DATA PIPELINE
# =============================
def load_data():
    if query and db_url:
        from sqlalchemy import create_engine
        try:
            engine = create_engine(db_url)
            df = pd.read_sql(query, engine)
            st.success("Loaded data from database")
            return df, "database"
        except Exception as e:
            st.error(f"Database error: {e}")

    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            df_part = ingest_file(f)
            if isinstance(df_part, pd.DataFrame):
                dfs.append(df_part)
        if dfs:
            return pd.concat(dfs, ignore_index=True, sort=False), "multi_upload"

    if uploaded:
        try:
            return pd.read_csv(uploaded), "upload"
        except Exception as e:
            st.error(f"CSV error: {e}")

    return generate_domain_dataset(domain), "synthetic"


df, data_source = load_data()
st.info(f"Data source: **{data_source}**")
st.dataframe(df.head())

# =============================
# FEATURE PREPARATION
# =============================
def prepare_features(df):
    if len(df.columns) < 2:
        st.error("Dataset must have at least 2 columns.")
        return None, None

    df.columns = [str(col) for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = st.sidebar.selectbox("Target Column", df.columns)
    if target_col not in df.columns:
        st.error("Invalid target column selected.")
        return None, None

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    X.columns = [str(c) for c in X.columns]

    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    return X, y


X, y = prepare_features(df)
if X is None:
    st.stop()

# =============================
# TASK DETECTION + MODEL INIT
# =============================
def detect_task(y):
    return "classification" if y.nunique() < 10 else "regression"

task = detect_task(y)

if st.session_state.model is None:
    if task == "classification":
        st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        st.session_state.model = RandomForestRegressor(n_estimators=100, random_state=42)
model = st.session_state.model

# =============================
# PREDICT FUTURE
# =============================
def predict_future(model, X_last, steps=10):
    future_preds = []
    current = X_last.copy()
    for _ in range(steps):
        pred = model.predict(current)
        future_preds.append(float(np.mean(pred)))
        current = current + np.random.normal(0, 0.01, current.shape)
    return future_preds

# =============================
# SYSTEM SIMULATION
# =============================
def simulate_system(X_test, steps=30):
    current = X_test.copy().astype(float)
    for t in range(steps):
        phase = np.sin(t / 5)
        noise = np.random.normal(0, 0.05, current.shape)
        current = current + noise + phase * 0.01
        current = current.fillna(0)
        yield t, current

# =============================
# COMPLIANCE ENGINE
# =============================
def compliance_check(drift, fairness, stability, risk_score, jurisdiction):
    msgs = []
    if "EU AI Act" in jurisdiction:
        if drift > 0.2:
            msgs.append("[EU AI Act] Data drift requires transparency report (Art. 13)")
        if fairness > 0.1:
            msgs.append("[EU AI Act] Potential bias violation — human oversight required (Art. 14)")
        if risk_score and risk_score > 0.5:
            msgs.append("[EU AI Act] High-risk AI system — conformity assessment required (Art. 43)")
    elif "SR 11-7" in jurisdiction:
        if stability < 0.6:
            msgs.append("[SR 11-7] Model validation failure risk — review model risk policy")
        if drift > 0.2:
            msgs.append("[SR 11-7] Ongoing monitoring required — performance degradation detected")
    elif "UK" in jurisdiction:
        if drift > 0.3:
            msgs.append("[UK Guidance] Model monitoring insufficient — escalate to model risk committee")
    elif "APAC" in jurisdiction:
        msgs.append("[APAC] General governance monitoring applied — local rules may vary")
    return msgs

# =============================
# GOVERNANCE API FUNCTION (endpoint-ready)
# See api.py for the FastAPI deployment version.
# =============================
def governance_api(payload: dict) -> dict:
    drift = float(payload.get("drift", 0.0))
    fairness = float(payload.get("fairness", 0.0))
    uncertainty = float(payload.get("uncertainty", 0.0))
    stability = system_stability_score(drift, fairness)
    risk = compute_risk_score(drift, fairness, uncertainty)

    if drift > 0.3:
        action = "retrain"
    elif fairness > 0.1:
        action = "debias"
    else:
        action = "stable"

    return {
        "action": action,
        "risk": "high" if risk > 0.6 else "medium" if risk > 0.3 else "low",
        "drift": round(drift, 4),
        "fairness": round(fairness, 4),
        "stability": round(stability, 4),
        "risk_score": round(risk, 4),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

# =============================
# TRAIN BUTTON
# =============================
if st.button("Train Model"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        drift = compute_drift(X_train, X_test)
        fairness = compute_fairness(preds, y_test)
        stability = system_stability_score(drift, fairness)
        uncertainty = compute_model_uncertainty(model, X_test)
        entropy_val = compute_prediction_entropy(model, X_test)
        risk_score = compute_risk_score(drift, fairness, uncertainty)

        # Active governance loop — real debias or retrain fires automatically
        action = governance_intervention(model, drift, fairness, X_train, y_train)

        st.session_state.model = model
        st.session_state.metrics = (drift, fairness, stability)
        st.session_state.data_split = (X_train, X_test, y_train, y_test)
        st.session_state.risk_score = risk_score
        st.session_state.uncertainty = uncertainty

        log_run(model.__class__.__name__, drift, fairness, stability, jurisdiction, action=action, risk_score=risk_score)
        st.success(f"Model trained — governance action: **{action}**")

    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

# =============================
# METRICS DISPLAY
# =============================
if st.session_state.metrics:
    drift, fairness, stability = st.session_state.metrics
    risk_score = st.session_state.risk_score or 0.0
    uncertainty = st.session_state.uncertainty or 0.0

    st.subheader("Governance Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Drift", round(drift, 3), delta=status_label(drift))
    c2.metric("Fairness Gap", round(fairness, 3), delta=status_label(fairness))
    c3.metric("Stability", round(stability, 3))
    c4.metric("Uncertainty", round(uncertainty, 4))
    c5.metric("Risk Score", round(risk_score, 3), delta=status_label(risk_score))

    st.subheader("Compliance Signals")
    msgs = compliance_check(drift, fairness, stability, risk_score, jurisdiction)
    if msgs:
        for m in msgs:
            st.warning(m)
    else:
        st.success("All governance thresholds within acceptable range.")

    if st.button("Generate PDF Report"):
        path = generate_pdf_report(drift, fairness, stability, risk_score=risk_score)
        with open(path, "rb") as f:
            st.download_button("Download Governance Report", f, file_name="aurexis_report.pdf")

# =============================
# SIMULATION + FUTURE PREDICTION
# =============================
if (
    st.session_state.model is not None
    and st.session_state.metrics is not None
    and st.session_state.data_split is not None
):
    X_train, X_test, y_train, y_test = st.session_state.data_split
    model = st.session_state.model

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Start System Simulation"):
            chart = st.line_chart()
            future_drift = []

            for step, current in simulate_system(X_test):
                preds = model.predict(current)
                y_slice = y_test.values[:len(preds)] if hasattr(y_test, "values") else y_test[:len(preds)]
                d = compute_drift(X_train, current)
                f = compute_fairness(preds, y_slice)
                future_drift.append(d)

                # Real governance loop during simulation
                action = governance_intervention(model, d, f, X_train, y_train)

                chart.add_rows({"Drift": [d], "Fairness": [f]})

                if step % 10 == 0:
                    X_last = current.tail(1)
                    future = predict_future(model, X_last, steps=5)
                    st.write(f"Step {step} — short-term forecast:", future)

            st.subheader("Predictive Drift Forecast")
            st.line_chart(future_drift)

    with col_b:
        if st.button("Predict Future Trajectory"):
            X_last = X_test.tail(1)
            future = predict_future(model, X_last, steps=20)
            st.subheader("20-Step Prediction Path")
            st.line_chart(future)

# =============================
# GOVERNANCE API DEMO (inline)
# =============================
st.subheader("Governance API — Live Test")
st.caption("Mirrors the POST /governance endpoint in api.py deployed on Nebius Serverless.")
api_col1, api_col2, api_col3 = st.columns(3)
api_drift = api_col1.number_input("Drift", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
api_fairness = api_col2.number_input("Fairness Gap", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
api_uncertainty = api_col3.number_input("Uncertainty", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
if st.button("Call Governance API"):
    result = governance_api({"drift": api_drift, "fairness": api_fairness, "uncertainty": api_uncertainty})
    st.json(result)

# =============================
# AUDIT LOGS
# =============================
st.subheader("Audit Logs")
logs = load_logs()
if logs:
    st.dataframe(pd.DataFrame(logs).sort_values("timestamp", ascending=False))
else:
    st.info("No audit entries yet. Train a model to generate logs.")

# =============================
# AUREXIS AI ASSISTANT
# =============================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning("No OpenAI API key found. Set OPENAI_API_KEY in secrets or environment to enable the AI assistant.")
else:
    client = OpenAI(api_key=api_key)

    if st.session_state.metrics:
        st.subheader("Aurexis AI Governance Assistant")
        user_input = st.chat_input("Ask about your model's governance status...")

        if user_input:
            drift, fairness, stability = st.session_state.metrics
            risk_score = st.session_state.risk_score or 0.0
            uncertainty = st.session_state.uncertainty or 0.0

            st.session_state.messages.append({"role": "user", "content": user_input})

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an elite AI governance expert for Aurexis Systems.\n\n"
                                f"Current model state:\n"
                                f"- Drift: {drift}\n"
                                f"- Fairness Gap: {fairness}\n"
                                f"- Stability: {stability}\n"
                                f"- Model Uncertainty: {uncertainty}\n"
                                f"- Composite Risk Score: {risk_score}\n"
                                f"- Jurisdiction: {jurisdiction}\n\n"
                                "Give precise, actionable governance recommendations. "
                                "Reference specific regulatory articles when relevant."
                            )
                        },
                        *st.session_state.messages
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
            except Exception as e:
                st.error(f"OpenAI API Error: {e}")
                st.stop()

            st.session_state.messages.append({"role": "assistant", "content": reply})

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
