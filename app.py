"""
Aurexis Systems — AI Governance Infrastructure

Establishing rules and timing is like setting up a formation for the Qimen Dunjia,
and creating a holographic AI linked to the Earth system to simulate the operation
status and laws of celestial bodies.
"""
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

# =============================
# PAGE CONFIG (MUST BE FIRST)
# =============================
st.set_page_config(
    page_title="Aurexis Systems",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        pass  # Silently fail logging

def load_logs():
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    except:
        pass
    return logs

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
    """Compute mean prediction variance across the dataset."""
    try:
        if hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(X) for est in model.estimators_])
            return float(np.mean(np.var(tree_preds, axis=0)))
        return 0.0
    except:
        return 0.0

def compute_prediction_entropy(model, X):
    """Compute average entropy of class probability predictions."""
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            entropies = [scipy_entropy(p + 1e-9) for p in probs]
            return float(np.mean(entropies))
        return 0.0
    except:
        return 0.0

def compute_risk_score(drift, fairness, model_uncertainty):
    """Weighted composite risk score."""
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
        return "🟢 Low"
    elif value < 0.6:
        return "🟡 Medium"
    return "🔴 High"

# =============================
# BIAS MITIGATION
# =============================
def mitigate_bias(X_train, y_train):
    """Returns sample weights for fairness-aware training."""
    sample_weight = np.where(y_train == 1, 1.2, 1.0)
    return sample_weight

# =============================
# GOVERNANCE INTERVENTION
# =============================
def governance_intervention(model, drift, fairness, X_train=None, y_train=None):
    jurisdiction = st.session_state.get("jurisdiction", "Unknown")
    stability = system_stability_score(drift, fairness)
    risk = compute_risk_score(drift, fairness, 0.0)

    log_run("GovernanceAction", drift, fairness, stability, jurisdiction, action="check", risk_score=risk)

    if drift > 0.3:
        st.warning("[⚠️ WARNING] Drift threshold exceeded — triggering automatic retraining.")
        if model is not None and X_train is not None and y_train is not None:
            try:
                model.fit(X_train, y_train)
                st.success("[✅ RETRAIN] Model retrained on current data.")
            except Exception as e:
                st.error(f"Retraining failed: {e}")
        log_run("GovernanceAction", drift, fairness, stability, jurisdiction, action="retrain", risk_score=risk)
        return "retrain"

    if fairness > 0.1:
        st.warning("[⚠️ WARNING] Fairness threshold exceeded — triggering bias mitigation.")
        if model is not None and X_train is not None and y_train is not None:
            weights = mitigate_bias(X_train, y_train)
            try:
                model.fit(X_train, y_train, sample_weight=weights)
                st.success("[✅ DEBIAS] Fairness-aware retraining complete.")
            except TypeError:
                try:
                    model.fit(X_train, y_train)
                    st.info("[ℹ️ DEBIAS] Model refitted (sample_weight unsupported).")
                except Exception as e:
                    st.error(f"Debiasing failed: {e}")
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
            ["Drift Score", str(round(drift, 3)), "[⚠️ WARNING]" if drift > 0.3 else "[✅ PASS]"],
            ["Fairness Gap", str(round(fairness, 3)), "[⚠️ WARNING]" if fairness > 0.1 else "[✅ PASS]"],
            ["System Stability", str(round(stability, 3)), "[❌ FAIL]" if stability < 0.5 else "[✅ PASS]"],
        ]
        if risk_score is not None:
            metrics_data.append([
                "Composite Risk Score", str(round(risk_score, 3)),
                "[🔴 HIGH]" if risk_score > 0.6 else "[🟡 MEDIUM]" if risk_score > 0.3 else "[🟢 LOW]"
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
            "[⚠️ WARNING] Data drift detected." if drift > 0.3 else "[✅ PASS] Drift acceptable.",
            styles["Normal"]
        ))

        doc.build(content)
        return file_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

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
            return None
    except Exception as e:
        return None

# =============================
# PAGE TITLE
# =============================
st.title("⚖️ Aurexis Systems — Governance-as-a-Service for Enterprise AI")
st.caption("Autonomous drift detection · Fairness-aware retraining · Composite risk scoring · Regulatory compliance")

# =============================
# SESSION STATE INITIALIZATION
# =============================
for key, default in [
    ("model", None),
    ("metrics", None),
    ("messages", []),
    ("data_split", None),
    ("risk_score", None),
    ("uncertainty", None),
    ("jurisdiction", "United States (SR 11-7)"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Compliance Mode")
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

st.sidebar.header("📊 Dataset Controls")
domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "Business", "Emotion", "General"],
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
uploaded_files = st.sidebar.file_uploader(
    "Upload Dataset",
    accept_multiple_files=True,
    type=["csv", "xlsx", "json", "parquet"],
)

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
    else:
        X, y = make_classification(n_samples=n_samples, n_features=6, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
    return df

# =============================
# DATA PIPELINE
# =============================
def load_data():
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
            pass

    return generate_domain_dataset(domain), "synthetic"

df, data_source = load_data()
st.info(f"📁 Data source: **{data_source}** — {len(df)} rows × {len(df.columns)} columns")
st.dataframe(df.head(), use_container_width=True)

# =============================
# FEATURE PREPARATION
# =============================
def prepare_features(df):
    if len(df.columns) < 2:
        st.error("Dataset must have at least 2 columns.")
        return None, None

    df.columns = [str(col) for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = st.sidebar.selectbox("Target Column", df.columns, index=len(df.columns)-1 if len(df.columns) > 1 else 0)
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
# TRAIN BUTTON
# =============================
st.subheader("🤖 Model Training")
if st.button("🚀 Train Model", use_container_width=True):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
        
        preds = model.predict(X_test)

        drift = compute_drift(X_train, X_test)
        fairness = compute_fairness(preds, y_test)
        stability = system_stability_score(drift, fairness)
        uncertainty = compute_model_uncertainty(model, X_test)
        risk_score = compute_risk_score(drift, fairness, uncertainty)

        # Governance intervention
        action = governance_intervention(model, drift, fairness, X_train, y_train)

        st.session_state.model = model
        st.session_state.metrics = (drift, fairness, stability)
        st.session_state.data_split = (X_train, X_test, y_train, y_test)
        st.session_state.risk_score = risk_score
        st.session_state.uncertainty = uncertainty

        log_run(model.__class__.__name__, drift, fairness, stability, jurisdiction, action=action, risk_score=risk_score)
        st.success(f"✅ Model trained — governance action: **{action}**")

    except Exception as e:
        st.error(f"❌ Training failed: {e}")

# =============================
# METRICS DISPLAY
# =============================
if st.session_state.metrics:
    drift, fairness, stability = st.session_state.metrics
    risk_score = st.session_state.risk_score or 0.0
    uncertainty = st.session_state.uncertainty or 0.0

    st.subheader("📊 Governance Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Drift", round(drift, 3), delta=status_label(drift))
    c2.metric("Fairness Gap", round(fairness, 3), delta=status_label(fairness))
    c3.metric("Stability", round(stability, 3), delta=status_label(stability))
    c4.metric("Uncertainty", round(uncertainty, 4))
    c5.metric("Risk Score", round(risk_score, 3), delta=status_label(risk_score))

    st.subheader("📋 Compliance Status")
    if drift > 0.3 or fairness > 0.1:
        st.warning(f"⚠️ Compliance alert: Drift={drift:.3f}, Fairness={fairness:.3f}")
    else:
        st.success("✅ All governance thresholds within acceptable range.")

    if st.button("📄 Generate PDF Report", use_container_width=True):
        path = generate_pdf_report(drift, fairness, stability, risk_score=risk_score)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    "📥 Download Report",
                    f,
                    file_name="aurexis_report.pdf",
                    use_container_width=True
                )

# =============================
# GOVERNANCE API DEMO
# =============================
st.subheader("🔌 Governance API — Live Test")
api_col1, api_col2, api_col3, api_col4 = st.columns(4)
api_drift = api_col1.number_input("Drift", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
api_fairness = api_col2.number_input("Fairness Gap", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
api_uncertainty = api_col3.number_input("Uncertainty", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

if api_col4.button("⚡ Call API", use_container_width=True):
    stability = system_stability_score(api_drift, api_fairness)
    risk = compute_risk_score(api_drift, api_fairness, api_uncertainty)
    result = {
        "action": "retrain" if api_drift > 0.3 else "debias" if api_fairness > 0.1 else "stable",
        "risk": "high" if risk > 0.6 else "medium" if risk > 0.3 else "low",
        "drift": round(api_drift, 4),
        "fairness": round(api_fairness, 4),
        "stability": round(stability, 4),
        "risk_score": round(risk, 4),
    }
    st.json(result)

# =============================
# AUDIT LOGS
# =============================
st.subheader("📜 Audit Logs")
logs = load_logs()
if logs:
    log_df = pd.DataFrame(logs).sort_values("timestamp", ascending=False)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No audit entries yet. Train a model to generate logs.")
