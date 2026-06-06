"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              AUREXIS SYSTEMS — VERSION C                                  ║
║          AI Governance Operating System (Production-Grade)                ║
║                                                                           ║
║  Governance Frameworks:                                                   ║
║  • NIST AI Risk Management Framework                                      ║
║  • EU AI Act (High-Risk Classification)                                   ║
║  • OECD AI Principles                                                     ║
║  • UNESCO AI Ethics Recommendations                                       ║
║  • ISO/IEC 42001 AI Management Systems                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS & DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import datetime
import hashlib
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import wasserstein_distance, entropy as scipy_entropy, ks_2samp
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Optional but recommended imports
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    _FAIRLEARN_AVAILABLE = True
except ImportError:
    _FAIRLEARN_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG - MUST BE FIRST
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aurexis Systems v3",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for theming
st.markdown("""
<style>
    .governance-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-medium { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
LOG_FILE = "/tmp/audit_log.jsonl"
MODEL_CARD_DIR = "/tmp/model_cards"
os.makedirs(MODEL_CARD_DIR, exist_ok=True)

# Governance Framework Mappings
GOVERNANCE_FRAMEWORKS = {
    "NIST AI RMF": {
        "Govern": "Risk Management Framework",
        "Map": "Model Risk Monitoring",
        "Measure": "Drift, Fairness, Explainability",
        "Manage": "Governance Interventions"
    },
    "EU AI Act": {
        "High-Risk": "Healthcare, Finance, Criminal Justice",
        "Limited-Risk": "Emotion Recognition, Chatbots",
        "Minimal-Risk": "Spam Filtering, General Classification"
    },
    "OECD AI Principles": {
        "1": "Inclusive growth and sustainable development",
        "2": "Human-centered values and fairness",
        "3": "Transparency and explainability",
        "4": "Robustness and security",
        "5": "Accountability"
    },
    "ISO/IEC 42001": {
        "Context": "Organization & interested parties",
        "Planning": "Risk & opportunity mitigation",
        "Support": "Resources, competence, awareness",
        "Operation": "Control & monitoring",
        "Evaluation": "Performance & compliance"
    }
}

# AI Risk Classification Mapping
DOMAIN_RISK_MAPPING = {
    "Healthcare": ("High Risk", "🔴", "Critical domain - direct patient impact"),
    "Finance": ("High Risk", "🔴", "Critical domain - financial stability"),
    "Criminal Justice": ("High Risk", "🔴", "Critical domain - civil rights impact"),
    "Sports": ("Limited Risk", "🟡", "Non-critical but public facing"),
    "Business": ("Limited Risk", "🟡", "Operational domain - monitoring recommended"),
    "Emotion": ("Limited Risk", "🟡", "Emotion recognition - bias monitoring required"),
    "General": ("Minimal Risk", "🟢", "Low-impact classification task"),
}

# ══════════════════════════════════════════════════════════════════════════
# AUDIT & LOGGING SYSTEM
# ══════════════════════════════════════════════════════════════════════════
def log_governance_event(event_type, model_name, metrics, jurisdiction, 
                        action="", risk_class="", framework=""):
    """Log governance events to audit trail with full context."""
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "event_type": event_type,
        "model_name": model_name,
        "metrics": {
            "drift": round(float(metrics.get("drift", 0)), 4),
            "fairness": round(float(metrics.get("fairness", 0)), 4),
            "demographic_parity": round(float(metrics.get("dp", 0)), 4),
            "equalized_odds": round(float(metrics.get("eo", 0)), 4),
            "stability": round(float(metrics.get("stability", 0)), 4),
            "risk_score": round(float(metrics.get("risk_score", 0)), 4),
        },
        "governance": {
            "jurisdiction": jurisdiction,
            "action": action,
            "risk_classification": risk_class,
            "framework": framework,
        },
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        st.warning(f"Audit log write error: {e}")

def load_audit_logs():
    """Load and parse audit logs."""
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return logs

# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE METRICS - ENTERPRISE GRADE
# ══════════════════════════════════════════════════════════════════════════
def compute_drift_comprehensive(X_train, X_test):
    """Compute drift using multiple methods: Wasserstein + PSI + KS Test."""
    try:
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return {"wasserstein": 0.0, "psi": 0.0, "ks": 0.0, "overall": 0.0}
        
        wasserstein_dists = []
        psi_scores = []
        ks_stats = []
        
        for col in num_cols:
            x1 = X_train[col].values
            x2 = X_test[col].values
            
            # Normalize
            x1_norm = (x1 - x1.mean()) / (x1.std() + 1e-6)
            x2_norm = (x2 - x2.mean()) / (x2.std() + 1e-6)
            
            # Wasserstein Distance
            wasserstein_dists.append(wasserstein_distance(x1_norm, x2_norm))
            
            # Population Stability Index (PSI)
            bins = np.histogram_bin_edges(np.concatenate([x1, x2]), bins=20)
            p_train = np.histogram(x1, bins=bins)[0] / len(x1)
            p_test = np.histogram(x2, bins=bins)[0] / len(x2)
            psi = np.sum((p_test - p_train) * np.log((p_test + 1e-10) / (p_train + 1e-10)))
            psi_scores.append(abs(psi))
            
            # Kolmogorov-Smirnov Test
            ks_stat, _ = ks_2samp(x1, x2)
            ks_stats.append(ks_stat)
        
        return {
            "wasserstein": float(np.mean(wasserstein_dists)),
            "psi": float(np.mean(psi_scores)),
            "ks": float(np.mean(ks_stats)),
            "overall": float(np.mean([np.mean(wasserstein_dists), np.mean(psi_scores), np.mean(ks_stats)]))
        }
    except Exception as e:
        st.warning(f"Drift computation error: {e}")
        return {"wasserstein": 0.0, "psi": 0.0, "ks": 0.0, "overall": 0.0}

def compute_fairness_comprehensive(preds, y_true, sensitive_features=None):
    """Compute fairness using multiple metrics."""
    try:
        basic_fairness = float(abs(np.mean(preds) - np.mean(y_true)))
        
        result = {"basic": basic_fairness}
        
        # Add Fairlearn metrics if available and sensitive features provided
        if _FAIRLEARN_AVAILABLE and sensitive_features is not None:
            try:
                dp = demographic_parity_difference(y_true, preds, sensitive_features=sensitive_features)
                eo = equalized_odds_difference(y_true, preds, sensitive_features=sensitive_features)
                result["demographic_parity"] = float(abs(dp))
                result["equalized_odds"] = float(abs(eo))
            except Exception as e:
                st.warning(f"Fairlearn metric error: {e}")
        
        result["composite"] = float(np.mean(list(result.values())))
        return result
    except Exception as e:
        st.warning(f"Fairness computation error: {e}")
        return {"basic": 0.0, "composite": 0.0}

def compute_model_uncertainty(model, X):
    """Compute prediction uncertainty."""
    try:
        if hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(X) for est in model.estimators_])
            return float(np.mean(np.var(tree_preds, axis=0)))
        return 0.0
    except Exception:
        return 0.0

def compute_risk_score(drift, fairness, uncertainty):
    """Weighted composite risk score."""
    raw = drift * 0.35 + fairness * 0.35 + uncertainty * 0.30
    return max(0.0, min(1.0, float(raw)))

def system_stability_score(drift, fairness):
    """Compute system stability."""
    score = (1 - drift) * 0.5 + (1 - fairness) * 0.5
    return max(0.0, min(1.0, float(score)))

def classify_ai_risk_level(domain):
    """Classify AI risk according to EU AI Act and NIST frameworks."""
    risk_class, emoji, reason = DOMAIN_RISK_MAPPING.get(domain, ("Minimal Risk", "🟢", "Unknown domain"))
    return {
        "classification": risk_class,
        "emoji": emoji,
        "reasoning": reason,
        "requires_audit": risk_class == "High Risk",
        "monitoring_level": "Continuous" if risk_class == "High Risk" else "Periodic"
    }

def get_fairness_status(fairness_score):
    """Get fairness status label."""
    if fairness_score < 0.1:
        return "✅ Acceptable", "🟢"
    elif fairness_score < 0.2:
        return "⚠️ Warning", "🟡"
    else:
        return "🔴 Critical", "🔴"

# ══════════════════════════════════════════════════════════════════════════
# MODEL CARD GENERATION (Governance Artifact)
# ══════════════════════════════════════════════════════════════════════════
def create_model_card(model, X_train, y_train, metrics, jurisdiction, domain, risk_class):
    """Create comprehensive model card for governance documentation."""
    model_id = hashlib.md5(
        f"{model.__class__.__name__}_{datetime.datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:8]
    
    card = {
        "model_id": model_id,
        "metadata": {
            "name": f"{model.__class__.__name__}_{model_id}",
            "type": model.__class__.__name__,
            "version": "1.0",
            "created": datetime.datetime.utcnow().isoformat(),
            "domain": domain,
            "jurisdiction": jurisdiction,
        },
        "governance": {
            "ai_risk_classification": risk_class["classification"],
            "requires_human_oversight": risk_class["requires_audit"],
            "monitoring_level": risk_class["monitoring_level"],
            "governance_frameworks": list(GOVERNANCE_FRAMEWORKS.keys()),
        },
        "training_data": {
            "size": len(X_train),
            "features": len(X_train.columns),
            "feature_names": list(X_train.columns),
            "target_distribution": dict(pd.Series(y_train).value_counts()),
        },
        "model_metrics": {
            "drift": metrics.get("drift", 0),
            "fairness": metrics.get("fairness", 0),
            "stability": metrics.get("stability", 0),
            "risk_score": metrics.get("risk_score", 0),
            "uncertainty": metrics.get("uncertainty", 0),
        },
        "compliance": {
            "nist_ai_rmf": "Compliant" if metrics.get("risk_score", 0) < 0.6 else "Review Required",
            "eu_ai_act": "High-Risk" if risk_class["classification"] == "High Risk" else "Lower-Risk",
            "iso_42001": "Requires Assessment",
            "audit_timestamp": datetime.datetime.utcnow().isoformat(),
        }
    }
    
    # Save model card
    card_path = os.path.join(MODEL_CARD_DIR, f"{model_id}_card.json")
    try:
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
    except Exception as e:
        st.warning(f"Model card save error: {e}")
    
    return card, model_id

def load_model_cards():
    """Load all model cards."""
    cards = []
    try:
        for filename in os.listdir(MODEL_CARD_DIR):
            if filename.endswith("_card.json"):
                with open(os.path.join(MODEL_CARD_DIR, filename), "r") as f:
                    cards.append(json.load(f))
    except Exception:
        pass
    return cards

# ══════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE (SHAP Integration)
# ══════════════════════════════════════════════════════════════════════════
def generate_explainability_report(model, X_test, model_type="classification"):
    """Generate SHAP-based explainability report."""
    if not _SHAP_AVAILABLE:
        return None
    
    try:
        if hasattr(model, 'predict'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Get feature importance
            if isinstance(shap_values, list):
                shap_values_main = shap_values[0] if len(shap_values) > 0 else shap_values
            else:
                shap_values_main = shap_values
            
            feature_importance = np.abs(shap_values_main).mean(axis=0)
            feature_names = X_test.columns
            
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP_Importance": feature_importance
            }).sort_values("SHAP_Importance", ascending=False)
            
            return {
                "explainer": explainer,
                "shap_values": shap_values,
                "feature_importance": importance_df,
                "mean_abs_shap": np.abs(shap_values_main).mean()
            }
    except Exception as e:
        st.warning(f"SHAP computation error: {e}")
    
    return None

# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE INTERVENTION ENGINE
# ══════════════════════════════════════════════════════════════════════════
def governance_intervention(model, metrics, domain, jurisdiction, X_train=None, y_train=None):
    """Execute governance actions based on metrics and risk classification."""
    risk_class = classify_ai_risk_level(domain)
    stability = metrics.get("stability", 0)
    drift = metrics.get("drift", 0)
    fairness = metrics.get("fairness", 0)
    risk_score = metrics.get("risk_score", 0)
    
    log_governance_event(
        "governance_check",
        model.__class__.__name__,
        metrics,
        jurisdiction,
        risk_class=risk_class["classification"]
    )
    
    messages = []
    actions = []
    
    # High-risk domain checks
    if risk_class["requires_audit"]:
        messages.append(("warning", f"🔴 {risk_class['classification']} Domain - Enhanced monitoring required"))
        actions.append("continuous_monitoring")
    
    # Drift threshold
    if drift > 0.3:
        messages.append(("warning", "⚠️ Data drift detected - Retraining recommended"))
        actions.append("retrain")
        if model is not None and X_train is not None and y_train is not None:
            try:
                model.fit(X_train, y_train)
                messages.append(("success", "✅ Model retrained on current data"))
                actions.append("retrain_complete")
            except Exception as e:
                messages.append(("error", f"Retraining failed: {e}"))
    
    # Fairness threshold
    if fairness > 0.15:
        messages.append(("warning", "⚠️ Fairness gap detected - Bias mitigation recommended"))
        actions.append("debias")
    
    # Risk score threshold
    if risk_score > 0.6:
        messages.append(("error", "🔴 High risk score - Human review required"))
        actions.append("escalate")
    
    if not actions:
        messages.append(("success", "✅ System stable - Governance thresholds within acceptable range"))
        actions.append("stable")
    
    log_governance_event(
        "governance_action",
        model.__class__.__name__,
        metrics,
        jurisdiction,
        action=",".join(actions),
        risk_class=risk_class["classification"]
    )
    
    return {
        "risk_class": risk_class,
        "actions": actions,
        "messages": messages,
        "primary_action": actions[0] if actions else "stable"
    }

def render_governance_messages(gov_result):
    """Render governance intervention messages and recommendations."""
    risk_class = gov_result["risk_class"]
    messages = gov_result["messages"]
    
    # Risk classification banner
    st.markdown(f"""
    <div style="background-color: {'#ffebee' if risk_class['requires_audit'] else '#e8f5e9'}; 
                border-left: 4px solid {risk_class['emoji']}; padding: 10px;">
    <strong>{risk_class['emoji']} AI Risk Classification: {risk_class['classification']}</strong><br>
    {risk_class['reasoning']}<br>
    Monitoring Level: {risk_class['monitoring_level']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Render each message
    for level, text in messages:
        if level == "warning":
            st.warning(text)
        elif level == "success":
            st.success(text)
        elif level == "error":
            st.error(text)
        elif level == "info":
            st.info(text)

# ══════════════════════════════════════════════════════════════════════════
# ADVANCED PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════
def generate_comprehensive_pdf_report(metrics, risk_class, model_card, explainability=None, filename="governance_report.pdf"):
    """Generate comprehensive governance report with regulatory mappings."""
    file_path = os.path.join("/tmp", filename)
    try:
        doc = SimpleDocTemplate(file_path, pagesize=(8.5*inch, 11*inch), topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        content = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor("#667eea"),
            spaceAfter=30,
            alignment=1
        )
        
        # Header
        content.append(Paragraph("AUREXIS SYSTEMS", title_style))
        content.append(Paragraph("AI Governance Risk Report v3", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # Executive Summary
        content.append(Paragraph("Executive Summary", styles['Heading2']))
        content.append(Paragraph(
            f"This report provides a comprehensive governance assessment of {model_card.get('metadata', {}).get('name', 'Model')} "
            f"based on NIST AI RMF, EU AI Act, and ISO/IEC 42001 frameworks.",
            styles['Normal']
        ))
        content.append(Spacer(1, 12))
        
        # Risk Classification
        content.append(Paragraph("AI Risk Classification", styles['Heading2']))
        risk_reason = risk_class.get('reasoning', 'N/A')
        content.append(Paragraph(
            f"<b>Classification:</b> {risk_class.get('classification', 'N/A')}<br/>"
            f"<b>Reasoning:</b> {risk_reason}<br/>"
            f"<b>Monitoring Level:</b> {risk_class.get('monitoring_level', 'N/A')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 12))
        
        # Metrics
        content.append(Paragraph("Governance Metrics", styles['Heading2']))
        metrics_data = [
            ["Metric", "Value", "Status"],
            ["Drift Score", f"{metrics.get('drift', 0):.3f}", "⚠️ WARNING" if metrics.get('drift', 0) > 0.3 else "✅ PASS"],
            ["Fairness", f"{metrics.get('fairness', 0):.3f}", "⚠️ WARNING" if metrics.get('fairness', 0) > 0.15 else "✅ PASS"],
            ["Stability", f"{metrics.get('stability', 0):.3f}", "✅ PASS" if metrics.get('stability', 0) > 0.5 else "❌ FAIL"],
            ["Risk Score", f"{metrics.get('risk_score', 0):.3f}", "🔴 HIGH" if metrics.get('risk_score', 0) > 0.6 else "🟡 MEDIUM" if metrics.get('risk_score', 0) > 0.3 else "🟢 LOW"],
        ]
        
        t = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.beige, colors.white]),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        content.append(t)
        content.append(Spacer(1, 12))
        
        # Governance Frameworks
        content.append(Paragraph("Regulatory Framework Mapping", styles['Heading2']))
        content.append(Paragraph(
            "<b>NIST AI Risk Management Framework:</b> Applicable<br/>"
            "<b>EU AI Act Compliance:</b> High-Risk Category<br/>"
            "<b>ISO/IEC 42001:</b> AI Management System - Required<br/>"
            "<b>OECD AI Principles:</b> All principles apply",
            styles['Normal']
        ))
        content.append(Spacer(1, 12))
        
        # Model Information
        content.append(Paragraph("Model Card Information", styles['Heading2']))
        model_meta = model_card.get('metadata', {})
        content.append(Paragraph(
            f"<b>Model ID:</b> {model_card.get('model_id', 'N/A')}<br/>"
            f"<b>Type:</b> {model_meta.get('type', 'N/A')}<br/>"
            f"<b>Created:</b> {model_meta.get('created', 'N/A')}<br/>"
            f"<b>Training Samples:</b> {model_card.get('training_data', {}).get('size', 'N/A')}<br/>"
            f"<b>Features:</b> {model_card.get('training_data', {}).get('features', 'N/A')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 12))
        
        # Compliance Assessment
        content.append(Paragraph("Compliance Assessment", styles['Heading2']))
        compliance = model_card.get('compliance', {})
        content.append(Paragraph(
            f"<b>NIST AI RMF:</b> {compliance.get('nist_ai_rmf', 'N/A')}<br/>"
            f"<b>EU AI Act:</b> {compliance.get('eu_ai_act', 'N/A')}<br/>"
            f"<b>ISO 42001:</b> {compliance.get('iso_42001', 'N/A')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 12))
        
        # Explainability (if available)
        if explainability:
            content.append(PageBreak())
            content.append(Paragraph("Model Explainability Analysis", styles['Heading2']))
            imp_df = explainability.get('feature_importance', pd.DataFrame())
            if not imp_df.empty:
                imp_data = [["Feature", "SHAP Importance"]]
                for idx, row in imp_df.head(10).iterrows():
                    imp_data.append([str(row['Feature']), f"{row['SHAP_Importance']:.4f}"])
                
                imp_table = Table(imp_data, colWidths=[2.5*inch, 2*inch])
                imp_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#764ba2")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]))
                content.append(imp_table)
        
        # Recommendations
        content.append(PageBreak())
        content.append(Paragraph("Governance Recommendations", styles['Heading2']))
        recommendations = [
            "1. Implement continuous monitoring for drift detection",
            "2. Conduct quarterly fairness audits using Fairlearn metrics",
            "3. Maintain model card documentation updated",
            "4. Establish human-in-the-loop review for high-risk decisions",
            "5. Create incident response plan for governance violations",
            "6. Document all model changes and retraining events",
        ]
        for rec in recommendations:
            content.append(Paragraph(rec, styles['Normal']))
        
        content.append(Spacer(1, 20))
        content.append(Paragraph(
            f"Report Generated: {datetime.datetime.utcnow().isoformat()}<br/>"
            "Aurexis Systems v3 — AI Governance Operating System",
            styles['Normal']
        ))
        
        doc.build(content)
        return file_path
    
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════
# FILE INGESTION
# ══════════════════════════════════════════════════════════════════════════
def ingest_file(file):
    """Ingest data from uploaded file."""
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        elif file.name.endswith(".json"):
            return pd.read_json(file)
        elif file.name.endswith(".parquet"):
            return pd.read_parquet(file)
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════════════════════════════════
# DOMAIN DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════
def generate_domain_dataset(domain, n_samples=500):
    """Generate synthetic datasets with sensitive features for fairness testing."""
    rng = np.random.default_rng(42)
    
    if domain == "Finance":
        df = pd.DataFrame({
            "credit_score": rng.normal(650, 50, n_samples),
            "income": rng.normal(70000, 20000, n_samples),
            "debt_ratio": rng.uniform(0.1, 0.8, n_samples),
            "loan_amount": rng.normal(20000, 8000, n_samples),
            "age": rng.integers(20, 80, n_samples),
            "gender": rng.choice([0, 1], n_samples),  # Sensitive feature
        })
        df["target"] = ((df["credit_score"] < 620) | (df["debt_ratio"] > 0.5)).astype(int)
    
    elif domain == "Healthcare":
        df = pd.DataFrame({
            "age": rng.integers(20, 80, n_samples),
            "bmi": rng.normal(27, 5, n_samples),
            "blood_pressure": rng.normal(120, 15, n_samples),
            "cholesterol": rng.normal(200, 40, n_samples),
            "race": rng.choice([0, 1, 2], n_samples),  # Sensitive feature
        })
        df["target"] = ((df["bmi"] > 30) | (df["blood_pressure"] > 140)).astype(int)
    
    elif domain == "Sports":
        df = pd.DataFrame({
            "speed": rng.normal(25, 5, n_samples),
            "strength": rng.normal(70, 10, n_samples),
            "stamina": rng.normal(60, 15, n_samples),
            "reaction_time": rng.normal(0.3, 0.05, n_samples),
            "gender": rng.choice([0, 1], n_samples),
        })
        df["target"] = ((df["speed"] > 28) & (df["reaction_time"] < 0.28)).astype(int)
    
    elif domain == "Business":
        df = pd.DataFrame({
            "revenue": rng.normal(1e6, 3e5, n_samples),
            "expenses": rng.normal(7e5, 2e5, n_samples),
            "customer_growth": rng.normal(0.1, 0.05, n_samples),
            "market_share": rng.uniform(0.01, 0.3, n_samples),
            "region": rng.choice([0, 1, 2], n_samples),
        })
        df["target"] = ((df["revenue"] - df["expenses"] > 2e5) & (df["customer_growth"] > 0.1)).astype(int)
    
    elif domain == "Emotion":
        df = pd.DataFrame({
            "valence": rng.uniform(-1, 1, n_samples),
            "arousal": rng.uniform(0, 1, n_samples),
            "dominance": rng.uniform(0, 1, n_samples),
            "speech_rate": rng.normal(150, 30, n_samples),
            "demographic": rng.choice([0, 1], n_samples),  # Sensitive
        })
        df["target"] = ((df["valence"] > 0.2) & (df["arousal"] > 0.5)).astype(int)
    
    else:  # General
        X_arr, y_arr = make_classification(n_samples=n_samples, n_features=6, random_state=42)
        df = pd.DataFrame(X_arr, columns=[f"feature_{i}" for i in range(X_arr.shape[1])])
        df["target"] = y_arr
    
    return df

# ══════════════════════════════════════════════════════════════════════════
# PAGE HEADER & SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="governance-header">
<h1>⚖️ AUREXIS SYSTEMS v3</h1>
<p>Enterprise AI Governance Operating System</p>
<p>NIST AI RMF | EU AI Act | ISO/IEC 42001 | OECD Principles</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Session state
_defaults = {
    "model": None,
    "metrics": None,
    "messages": [],
    "jurisdiction": "United States (SR 11-7)",
    "model_card": None,
    "model_id": None,
    "explainability": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Governance Configuration")

jurisdiction = st.sidebar.selectbox(
    "Regulatory Framework",
    [
        "United States (SR 11-7)",
        "European Union (EU AI Act)",
        "UK Model Risk Guidance",
        "APAC General Risk Framework",
        "ISO/IEC 42001",
    ],
)
st.session_state["jurisdiction"] = jurisdiction

st.sidebar.header("📊 Dataset & Model")
domain = st.sidebar.selectbox(
    "Application Domain",
    ["Finance", "Healthcare", "Sports", "Business", "Emotion", "General"],
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Dataset",
    accept_multiple_files=True,
    type=["csv", "xlsx", "json", "parquet"],
    key="multi_uploader",
)

st.sidebar.header("📋 Framework Information")
selected_framework = st.sidebar.selectbox(
    "Select Framework Reference",
    list(GOVERNANCE_FRAMEWORKS.keys())
)

if st.sidebar.button("📖 View Framework Details"):
    st.sidebar.info(
        f"**{selected_framework}**\n\n"
        + "\n".join([f"• {k}: {v}" for k, v in GOVERNANCE_FRAMEWORKS[selected_framework].items()])
    )

# ══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def load_data():
    """Load data from uploads or generate synthetic."""
    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            part = ingest_file(f)
            if isinstance(part, pd.DataFrame):
                dfs.append(part)
        if dfs:
            return pd.concat(dfs, ignore_index=True, sort=False), "uploaded"
    return generate_domain_dataset(domain), "synthetic"

df, data_source = load_data()
st.info(f"📁 Data: **{data_source}** | {len(df):,} rows × {len(df.columns)} columns")
st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════════
def prepare_features(df):
    """Prepare features and identify sensitive attributes."""
    if len(df.columns) < 2:
        st.error("Dataset must have at least 2 columns")
        return None, None, None
    
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    
    target_col = st.sidebar.selectbox(
        "Target Column",
        df.columns,
        index=max(0, len(df.columns) - 1)
    )
    
    if target_col not in df.columns:
        st.error("Invalid target column")
        return None, None, None
    
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    X.columns = [str(c) for c in X.columns]
    
    # Auto-identify sensitive features (gender, race, age, demographic)
    sensitive_cols = [c for c in X.columns.str.lower() if any(s in c.lower() for s in ['gender', 'race', 'age', 'demographic'])]
    sensitive_features = None
    if sensitive_cols and st.checkbox("Use sensitive features for fairness testing"):
        sensitive_features = X[sensitive_cols[0]]
    
    # Encode categorical
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    
    return X, y, sensitive_features

X, y, sensitive_features = prepare_features(df)
if X is None:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════
def detect_task(y):
    """Detect classification vs regression."""
    return "classification" if y.nunique() < 10 else "regression"

task = detect_task(y)

if st.session_state.model is None:
    st.session_state.model = (
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if task == "classification" else
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )

model = st.session_state.model

# ══════════════════════════════════════════════════════════════════════════
# TRAINING SECTION
# ══════════════════════════════════════════════════════════════════════════
st.subheader("🚀 Model Training & Governance Assessment")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("Train the model and trigger comprehensive governance assessment.")

with col2:
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("🔄 Training model..."):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Compute comprehensive metrics
                drift_metrics = compute_drift_comprehensive(X_train, X_test)
                fairness_metrics = compute_fairness_comprehensive(preds, y_test, sensitive_features)
                stability = system_stability_score(drift_metrics["overall"], fairness_metrics["composite"])
                uncertainty = compute_model_uncertainty(model, X_test)
                risk_score = compute_risk_score(drift_metrics["overall"], fairness_metrics["composite"], uncertainty)
                
                # Collect all metrics
                metrics = {
                    "drift": drift_metrics["overall"],
                    "fairness": fairness_metrics["composite"],
                    "stability": stability,
                    "risk_score": risk_score,
                    "uncertainty": uncertainty,
                    "dp": fairness_metrics.get("demographic_parity", 0),
                    "eo": fairness_metrics.get("equalized_odds", 0),
                }
                
                # Create model card
                risk_class = classify_ai_risk_level(domain)
                model_card, model_id = create_model_card(
                    model, X_train, y_train, metrics, jurisdiction, domain, risk_class
                )
                
                # Generate explainability
                explainability = generate_explainability_report(model, X_test) if _SHAP_AVAILABLE else None
                
                # Governance intervention
                gov_result = governance_intervention(
                    model, metrics, domain, jurisdiction, X_train, y_train
                )
                
                # Store in session
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.model_card = model_card
                st.session_state.model_id = model_id
                st.session_state.explainability = explainability
                st.session_state.gov_result = gov_result
                
                st.success("✅ Model trained and governance assessment complete!")
                st.balloons()
            
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.metrics:
    metrics = st.session_state.metrics
    gov_result = st.session_state.get("gov_result", {})
    
    # Render governance intervention
    if gov_result:
        render_governance_messages(gov_result)
    
    st.subheader("📊 Comprehensive Metrics Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Drift Score", f"{metrics['drift']:.3f}", 
                 delta="⚠️ HIGH" if metrics['drift'] > 0.3 else "✅ OK")
    with col2:
        st.metric("Fairness Gap", f"{metrics['fairness']:.3f}",
                 delta="⚠️ HIGH" if metrics['fairness'] > 0.15 else "✅ OK")
    with col3:
        st.metric("Stability", f"{metrics['stability']:.3f}",
                 delta="✅ STABLE" if metrics['stability'] > 0.5 else "⚠️ LOW")
    with col4:
        st.metric("Uncertainty", f"{metrics['uncertainty']:.4f}")
    with col5:
        risk_status = "🟢 LOW" if metrics['risk_score'] < 0.3 else "🟡 MEDIUM" if metrics['risk_score'] < 0.6 else "🔴 HIGH"
        st.metric("Risk Score", f"{metrics['risk_score']:.3f}", delta=risk_status)
    
    # Additional fairness metrics (if available)
    if metrics.get("dp") or metrics.get("eo"):
        st.subheader("Advanced Fairness Metrics")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.metric("Demographic Parity Diff", f"{metrics.get('dp', 0):.4f}")
        with fc2:
            st.metric("Equalized Odds Diff", f"{metrics.get('eo', 0):.4f}")
    
    # Model Card Display
    if st.session_state.model_card:
        st.subheader("📋 Model Card (Governance Artifact)")
        card = st.session_state.model_card
        
        tab1, tab2, tab3 = st.tabs(["Metadata", "Governance", "Compliance"])
        
        with tab1:
            meta = card.get('metadata', {})
            st.json({
                "Model ID": card.get('model_id'),
                "Type": meta.get('type'),
                "Domain": meta.get('domain'),
                "Created": meta.get('created'),
                "Training Samples": card.get('training_data', {}).get('size'),
                "Features": card.get('training_data', {}).get('features'),
            })
        
        with tab2:
            gov = card.get('governance', {})
            st.json({
                "AI Risk Classification": gov.get('ai_risk_classification'),
                "Requires Human Oversight": gov.get('requires_human_oversight'),
                "Monitoring Level": gov.get('monitoring_level'),
                "Frameworks": gov.get('governance_frameworks'),
            })
        
        with tab3:
            compliance = card.get('compliance', {})
            st.json({
                "NIST AI RMF": compliance.get('nist_ai_rmf'),
                "EU AI Act": compliance.get('eu_ai_act'),
                "ISO 42001": compliance.get('iso_42001'),
                "Audit Timestamp": compliance.get('audit_timestamp'),
            })
    
    # Explainability Section
    if st.session_state.explainability and _SHAP_AVAILABLE:
        st.subheader("🔍 Model Explainability (SHAP Analysis)")
        
        exp = st.session_state.explainability
        imp_df = exp.get('feature_importance')
        
        if imp_df is not None and not imp_df.empty:
            st.bar_chart(imp_df.set_index('Feature').head(10))
            
            with st.expander("Feature Importance Details"):
                st.dataframe(imp_df.head(15), use_container_width=True)
    
    # PDF Report Generation
    st.subheader("📄 Governance Report Generation")
    
    if st.button("📊 Generate Comprehensive PDF Report", use_container_width=True):
        with st.spinner("Generating report..."):
            risk_class = classify_ai_risk_level(domain)
            report_path = generate_comprehensive_pdf_report(
                metrics,
                risk_class,
                st.session_state.model_card,
                st.session_state.explainability,
                filename=f"aurexis_report_{st.session_state.model_id}.pdf"
            )
            
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        "📥 Download PDF Report",
                        pdf_file,
                        file_name=f"aurexis_governance_report_{st.session_state.model_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                st.success("✅ Report generated successfully!")

# ══════════════════════════════════════════════════════════════════════════
# AUDIT LOGS & GOVERNANCE HISTORY
# ══════════════════════════════════════════════════════════════════════════
st.subheader("📜 Governance Audit Trail")

audit_logs = load_audit_logs()
if audit_logs:
    # Convert to DataFrame
    audit_df = pd.DataFrame(audit_logs)
    
    # Expandable audit view
    with st.expander("View Detailed Audit Logs", expanded=False):
        st.dataframe(audit_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(audit_logs))
    with col2:
        gov_checks = len([l for l in audit_logs if l.get('event_type') == 'governance_check'])
        st.metric("Governance Checks", gov_checks)
    with col3:
        escalations = len([l for l in audit_logs if 'escalate' in l.get('governance', {}).get('action', '')])
        st.metric("Escalations", escalations)
else:
    st.info("No audit logs yet. Train a model to generate governance events.")

# ══════════════════════════════════════════════════════════════════════════
# MODEL CARDS REPOSITORY
# ══════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Model Cards Repository")

model_cards = load_model_cards()
if model_cards:
    with st.expander(f"View {len(model_cards)} Saved Model Cards"):
        for i, card in enumerate(model_cards, 1):
            card_col1, card_col2 = st.columns([3, 1])
            with card_col1:
                st.write(f"**{i}. {card.get('model_id')}** - {card.get('metadata', {}).get('domain')}")
            with card_col2:
                if st.button("📋 View", key=f"card_{i}"):
                    st.json(card)
else:
    st.info("No model cards yet. Train models to populate the repository.")

# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE FRAMEWORKS REFERENCE
# ══════════════════════════════════════════════════════════════════════════
st.subheader("📚 Governance Frameworks Reference")

framework_tabs = st.tabs(list(GOVERNANCE_FRAMEWORKS.keys()))

for tab, framework_name in zip(framework_tabs, GOVERNANCE_FRAMEWORKS.keys()):
    with tab:
        framework_info = GOVERNANCE_FRAMEWORKS[framework_name]
        for key, value in framework_info.items():
            st.write(f"**{key}:** {value}")

# ══════════════════════════════════════════════════════════════════════════
# AI ASSISTANT (Optional - with OpenAI)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🤖 Aurexis AI Governance Advisor")

if not _OPENAI_AVAILABLE:
    st.info("💡 OpenAI package not installed. Install with: `pip install openai`")
else:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("⚠️ OpenAI API key not configured. See documentation for setup.")
    elif not st.session_state.metrics:
        st.info("ℹ️ Train a model first to enable the AI Advisor.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            
            # Display chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            user_input = st.chat_input("Ask about governance, compliance, or risk management...")
            
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                try:
                    with st.spinner("🔄 Consulting AI Governance Advisor..."):
                        metrics = st.session_state.metrics
                        risk_class = classify_ai_risk_level(domain)
                        
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert AI governance advisor for Aurexis Systems. "
                                        "You provide guidance on:\n"
                                        "- NIST AI RMF implementation\n"
                                        "- EU AI Act compliance\n"
                                        "- ISO/IEC 42001 standards\n"
                                        "- OECD AI principles\n"
                                        "- Model risk management\n"
                                        "- Fairness and bias mitigation\n"
                                        "- Governance best practices\n\n"
                                        f"Current model state:\n"
                                        f"- Risk Classification: {risk_class['classification']}\n"
                                        f"- Drift: {metrics['drift']:.3f}\n"
                                        f"- Fairness: {metrics['fairness']:.3f}\n"
                                        f"- Risk Score: {metrics['risk_score']:.3f}\n"
                                        f"- Jurisdiction: {jurisdiction}\n\n"
                                        "Provide actionable, specific governance recommendations."
                                    )
                                },
                                *st.session_state.messages
                            ],
                            temperature=0.3,
                            max_tokens=800,
                        )
                        
                        reply = response.choices[0].message.content
                
                except Exception as e:
                    reply = f"⚠️ API Error: {str(e)}"
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                with st.chat_message("assistant"):
                    st.write(reply)
        
        except Exception as e:
            st.error(f"AI Advisor Error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
<p><strong>AUREXIS SYSTEMS v3</strong> | Enterprise AI Governance Operating System</p>
<p>NIST AI RMF • EU AI Act • ISO/IEC 42001 • OECD Principles • UNESCO Ethics</p>
<p>© 2026 Aurexis Systems. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
