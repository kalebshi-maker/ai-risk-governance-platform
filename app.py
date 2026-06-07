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

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import shap

    _SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    shap = None
    _SHAP_AVAILABLE = False

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

    _FAIRLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    demographic_parity_difference = None
    equalized_odds_difference = None
    _FAIRLEARN_AVAILABLE = False

try:
    from openai import OpenAI
    from openai import RateLimitError as OpenAIRateLimitError

    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    OpenAIRateLimitError = Exception
    _OPENAI_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aurexis Systems — Version C",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .governance-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
        .risk-high { color: #d32f2f; font-weight: bold; }
        .risk-medium { color: #f57c00; font-weight: bold; }
        .risk-low { color: #388e3c; font-weight: bold; }
        .small-muted { color: #666; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
APP_VERSION = "Version C"
APP_NAME = "AUREXIS SYSTEMS"
ARTIFACT_ROOT = Path(os.getenv("AUREXIS_ARTIFACT_DIR", "/tmp/aurexis_version_c"))
MODEL_CARD_DIR = ARTIFACT_ROOT / "model_cards"
LOG_FILE = ARTIFACT_ROOT / "audit_log.jsonl"
MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)

GOVERNANCE_FRAMEWORKS = {
    "NIST AI RMF": {
        "Govern": "Risk Management Framework",
        "Map": "Model Risk Monitoring",
        "Measure": "Drift, Fairness, Explainability",
        "Manage": "Governance Interventions",
    },
    "EU AI Act": {
        "High-Risk": "Healthcare, Finance, Criminal Justice",
        "Limited-Risk": "Emotion Recognition, Chatbots",
        "Minimal-Risk": "Spam Filtering, General Classification",
    },
    "OECD AI Principles": {
        "1": "Inclusive growth and sustainable development",
        "2": "Human-centered values and fairness",
        "3": "Transparency and explainability",
        "4": "Robustness and security",
        "5": "Accountability",
    },
    "UNESCO AI Ethics": {
        "Human Rights": "Respect, protect, and promote human rights",
        "Transparency": "Explainable and auditable AI decisions",
        "Fairness": "Avoid unjust bias and discrimination",
        "Accountability": "Clear ownership and redress mechanisms",
    },
    "ISO/IEC 42001": {
        "Context": "Organization and interested parties",
        "Planning": "Risk and opportunity mitigation",
        "Support": "Resources, competence, awareness",
        "Operation": "Control and monitoring",
        "Evaluation": "Performance and compliance",
    },
}

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
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════
def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def clean_pdf_text(value: Any) -> str:
    """ReportLab's default fonts do not handle emoji reliably."""
    text = str(value)
    replacements = {
        "✅": "PASS",
        "⚠️": "WARNING",
        "⚠": "WARNING",
        "❌": "FAIL",
        "🔴": "HIGH",
        "🟡": "MEDIUM",
        "🟢": "LOW",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def safe_json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def reset_chat() -> None:
    st.session_state["messages"] = []


# ══════════════════════════════════════════════════════════════════════════
# AUDIT & LOGGING SYSTEM
# ══════════════════════════════════════════════════════════════════════════
def log_governance_event(
    event_type: str,
    model_name: str,
    metrics: Dict[str, Any],
    jurisdiction: str,
    action: str = "",
    risk_class: str = "",
    framework: str = "",
) -> None:
    record = {
        "timestamp": utc_now_iso(),
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
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=safe_json_default) + "\n")
    except Exception as exc:
        st.warning(f"Audit log write error: {exc}")


def load_audit_logs() -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    logs: List[Dict[str, Any]] = []
    try:
        with LOG_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return logs


# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE METRICS
# ══════════════════════════════════════════════════════════════════════════
def compute_drift_comprehensive(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, float]:
    try:
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0 or X_train.empty or X_test.empty:
            return {"wasserstein": 0.0, "psi": 0.0, "ks": 0.0, "overall": 0.0}

        wasserstein_dists: List[float] = []
        psi_scores: List[float] = []
        ks_stats: List[float] = []

        for col in num_cols:
            x1 = pd.to_numeric(X_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            x2 = pd.to_numeric(X_test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(x1) < 2 or len(x2) < 2:
                continue

            x1_norm = (x1 - np.mean(x1)) / (np.std(x1) + 1e-6)
            x2_norm = (x2 - np.mean(x2)) / (np.std(x2) + 1e-6)
            wasserstein_dists.append(float(wasserstein_distance(x1_norm, x2_norm)))

            bins = np.histogram_bin_edges(np.concatenate([x1, x2]), bins=10)
            if len(np.unique(bins)) > 1:
                p_train = np.histogram(x1, bins=bins)[0] / max(len(x1), 1)
                p_test = np.histogram(x2, bins=bins)[0] / max(len(x2), 1)
                psi = np.sum((p_test - p_train) * np.log((p_test + 1e-10) / (p_train + 1e-10)))
                psi_scores.append(float(abs(psi)))

            ks_stat, _ = ks_2samp(x1, x2)
            ks_stats.append(float(ks_stat))

        if not wasserstein_dists:
            return {"wasserstein": 0.0, "psi": 0.0, "ks": 0.0, "overall": 0.0}

        wasserstein_score = float(np.mean(wasserstein_dists))
        psi_score = float(np.mean(psi_scores)) if psi_scores else 0.0
        ks_score = float(np.mean(ks_stats)) if ks_stats else 0.0

        # Normalize composite so dashboard thresholds are intuitive.
        overall = float(np.mean([clamp01(wasserstein_score), clamp01(psi_score), clamp01(ks_score)]))
        return {
            "wasserstein": wasserstein_score,
            "psi": psi_score,
            "ks": ks_score,
            "overall": clamp01(overall),
        }
    except Exception as exc:
        st.warning(f"Drift computation error: {exc}")
        return {"wasserstein": 0.0, "psi": 0.0, "ks": 0.0, "overall": 0.0}


def compute_fairness_comprehensive(
    preds: np.ndarray,
    y_true: pd.Series,
    sensitive_features: Optional[pd.Series] = None,
    task: str = "classification",
) -> Dict[str, float]:
    try:
        preds_series = pd.Series(preds).reset_index(drop=True)
        y_series = pd.Series(y_true).reset_index(drop=True)

        if task == "classification":
            basic_fairness = abs(float(preds_series.mean()) - float(y_series.mean()))
        else:
            scale = float(y_series.max() - y_series.min()) or 1.0
            basic_fairness = abs(float(preds_series.mean()) - float(y_series.mean())) / scale

        result: Dict[str, float] = {"basic": clamp01(basic_fairness)}

        if (
            task == "classification"
            and _FAIRLEARN_AVAILABLE
            and sensitive_features is not None
            and len(sensitive_features) == len(y_series)
        ):
            try:
                sensitive = pd.Series(sensitive_features).reset_index(drop=True)
                dp = demographic_parity_difference(y_series, preds_series, sensitive_features=sensitive)
                eo = equalized_odds_difference(y_series, preds_series, sensitive_features=sensitive)
                result["demographic_parity"] = clamp01(abs(dp))
                result["equalized_odds"] = clamp01(abs(eo))
            except Exception as exc:
                st.warning(f"Fairlearn metric error: {exc}")

        result["composite"] = clamp01(float(np.mean(list(result.values()))))
        return result
    except Exception as exc:
        st.warning(f"Fairness computation error: {exc}")
        return {"basic": 0.0, "composite": 0.0}


def compute_model_uncertainty(model: Any, X: pd.DataFrame, task: str) -> float:
    try:
        if hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(X) for est in model.estimators_])
            variance = float(np.mean(np.var(tree_preds, axis=0)))
            if task == "regression":
                scale = float(np.var(model.predict(X))) + 1e-6
                return clamp01(variance / scale)
            return clamp01(variance)
        return 0.0
    except Exception:
        return 0.0


def compute_risk_score(drift: float, fairness: float, uncertainty: float) -> float:
    return clamp01(drift * 0.35 + fairness * 0.35 + uncertainty * 0.30)


def system_stability_score(drift: float, fairness: float) -> float:
    return clamp01((1 - drift) * 0.5 + (1 - fairness) * 0.5)


def classify_ai_risk_level(domain: str) -> Dict[str, Any]:
    risk_class, emoji, reason = DOMAIN_RISK_MAPPING.get(domain, ("Minimal Risk", "🟢", "Unknown domain"))
    return {
        "classification": risk_class,
        "emoji": emoji,
        "reasoning": reason,
        "requires_audit": risk_class == "High Risk",
        "monitoring_level": "Continuous" if risk_class == "High Risk" else "Periodic",
    }


def get_fairness_status(fairness_score: float) -> Tuple[str, str]:
    if fairness_score < 0.1:
        return "Acceptable", "🟢"
    if fairness_score < 0.2:
        return "Warning", "🟡"
    return "Critical", "🔴"


# ══════════════════════════════════════════════════════════════════════════
# MODEL CARD GENERATION
# ══════════════════════════════════════════════════════════════════════════
def create_model_card(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: Dict[str, float],
    jurisdiction: str,
    domain: str,
    risk_class: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    model_id = hashlib.sha256(
        f"{model.__class__.__name__}_{utc_now_iso()}_{domain}".encode()
    ).hexdigest()[:10]

    card = {
        "model_id": model_id,
        "metadata": {
            "name": f"{model.__class__.__name__}_{model_id}",
            "type": model.__class__.__name__,
            "version": "1.0",
            "created": utc_now_iso(),
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
            "size": int(len(X_train)),
            "features": int(len(X_train.columns)),
            "feature_names": list(X_train.columns),
            "target_distribution": {
                str(k): int(v) for k, v in pd.Series(y_train).value_counts(dropna=False).items()
            },
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
            "audit_timestamp": utc_now_iso(),
        },
    }

    card_path = MODEL_CARD_DIR / f"{model_id}_card.json"
    try:
        with card_path.open("w", encoding="utf-8") as file:
            json.dump(card, file, indent=2, default=safe_json_default)
    except Exception as exc:
        st.warning(f"Model card save error: {exc}")

    return card, model_id


def load_model_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    try:
        for path in sorted(MODEL_CARD_DIR.glob("*_card.json")):
            with path.open("r", encoding="utf-8") as file:
                cards.append(json.load(file))
    except Exception:
        return []
    return cards


# ══════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ══════════════════════════════════════════════════════════════════════════
def generate_explainability_report(model: Any, X_test: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if not _SHAP_AVAILABLE:
        return None

    try:
        sample = X_test.head(min(100, len(X_test)))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_values_main = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif getattr(shap_values, "ndim", 0) == 3:
            shap_values_main = shap_values[:, :, min(1, shap_values.shape[2] - 1)]
        else:
            shap_values_main = shap_values

        feature_importance = np.abs(shap_values_main).mean(axis=0)
        importance_df = pd.DataFrame(
            {"Feature": sample.columns, "SHAP_Importance": feature_importance}
        ).sort_values("SHAP_Importance", ascending=False)

        return {
            "feature_importance": importance_df,
            "mean_abs_shap": float(np.abs(shap_values_main).mean()),
        }
    except Exception as exc:
        st.warning(f"SHAP computation error: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE INTERVENTION ENGINE
# ══════════════════════════════════════════════════════════════════════════
def governance_intervention(
    model: Any,
    metrics: Dict[str, float],
    domain: str,
    jurisdiction: str,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    risk_class = classify_ai_risk_level(domain)
    drift = metrics.get("drift", 0)
    fairness = metrics.get("fairness", 0)
    risk_score = metrics.get("risk_score", 0)

    log_governance_event(
        "governance_check",
        model.__class__.__name__,
        metrics,
        jurisdiction,
        risk_class=risk_class["classification"],
    )

    messages: List[Tuple[str, str]] = []
    actions: List[str] = []

    if risk_class["requires_audit"]:
        messages.append(("warning", f"{risk_class['emoji']} {risk_class['classification']} domain - enhanced monitoring required"))
        actions.append("continuous_monitoring")

    if drift > 0.3:
        messages.append(("warning", "Data drift detected - retraining recommended"))
        actions.append("retrain")
        if model is not None and X_train is not None and y_train is not None:
            try:
                model.fit(X_train, y_train)
                messages.append(("success", "Model retrained on current data"))
                actions.append("retrain_complete")
            except Exception as exc:
                messages.append(("error", f"Retraining failed: {exc}"))

    if fairness > 0.15:
        messages.append(("warning", "Fairness gap detected - bias mitigation recommended"))
        actions.append("debias")

    if risk_score > 0.6:
        messages.append(("error", "High risk score - human review required"))
        actions.append("escalate")

    if not actions:
        messages.append(("success", "System stable - governance thresholds are within acceptable range"))
        actions.append("stable")

    log_governance_event(
        "governance_action",
        model.__class__.__name__,
        metrics,
        jurisdiction,
        action=",".join(actions),
        risk_class=risk_class["classification"],
    )

    return {
        "risk_class": risk_class,
        "actions": actions,
        "messages": messages,
        "primary_action": actions[0] if actions else "stable",
    }


def render_governance_messages(gov_result: Dict[str, Any]) -> None:
    risk_class = gov_result["risk_class"]
    border_color = "#d32f2f" if risk_class["requires_audit"] else "#388e3c"
    bg_color = "#ffebee" if risk_class["requires_audit"] else "#e8f5e9"
    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 12px; border-radius: 4px;">
        <strong>{risk_class['emoji']} AI Risk Classification: {risk_class['classification']}</strong><br>
        {risk_class['reasoning']}<br>
        Monitoring Level: {risk_class['monitoring_level']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for level, text in gov_result["messages"]:
        if level == "warning":
            st.warning(text)
        elif level == "success":
            st.success(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)


# ══════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════
def generate_comprehensive_pdf_report(
    metrics: Dict[str, float],
    risk_class: Dict[str, Any],
    model_card: Dict[str, Any],
    explainability: Optional[Dict[str, Any]] = None,
    filename: str = "governance_report.pdf",
) -> Optional[Path]:
    file_path = ARTIFACT_ROOT / filename
    try:
        doc = SimpleDocTemplate(str(file_path), pagesize=(8.5 * inch, 11 * inch), topMargin=0.5 * inch)
        styles = getSampleStyleSheet()
        content: List[Any] = []

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#667eea"),
            spaceAfter=30,
            alignment=1,
        )

        content.append(Paragraph("AUREXIS SYSTEMS", title_style))
        content.append(Paragraph("AI Governance Risk Report - Version C", styles["Heading2"]))
        content.append(Spacer(1, 12))

        content.append(Paragraph("Executive Summary", styles["Heading2"]))
        content.append(
            Paragraph(
                f"This report provides a governance assessment of "
                f"{clean_pdf_text(model_card.get('metadata', {}).get('name', 'Model'))} "
                "based on NIST AI RMF, EU AI Act, OECD, UNESCO, and ISO/IEC 42001 frameworks.",
                styles["Normal"],
            )
        )
        content.append(Spacer(1, 12))

        content.append(Paragraph("AI Risk Classification", styles["Heading2"]))
        content.append(
            Paragraph(
                f"<b>Classification:</b> {clean_pdf_text(risk_class.get('classification', 'N/A'))}<br/>"
                f"<b>Reasoning:</b> {clean_pdf_text(risk_class.get('reasoning', 'N/A'))}<br/>"
                f"<b>Monitoring Level:</b> {clean_pdf_text(risk_class.get('monitoring_level', 'N/A'))}",
                styles["Normal"],
            )
        )
        content.append(Spacer(1, 12))

        metrics_data = [
            ["Metric", "Value", "Status"],
            ["Drift Score", f"{metrics.get('drift', 0):.3f}", "WARNING" if metrics.get("drift", 0) > 0.3 else "PASS"],
            ["Fairness", f"{metrics.get('fairness', 0):.3f}", "WARNING" if metrics.get("fairness", 0) > 0.15 else "PASS"],
            ["Stability", f"{metrics.get('stability', 0):.3f}", "PASS" if metrics.get("stability", 0) > 0.5 else "FAIL"],
            ["Risk Score", f"{metrics.get('risk_score', 0):.3f}", "HIGH" if metrics.get("risk_score", 0) > 0.6 else "MEDIUM" if metrics.get("risk_score", 0) > 0.3 else "LOW"],
        ]

        table = Table(metrics_data, colWidths=[2 * inch, 1.5 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.beige, colors.white]),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        content.append(Paragraph("Governance Metrics", styles["Heading2"]))
        content.append(table)
        content.append(Spacer(1, 12))

        content.append(Paragraph("Regulatory Framework Mapping", styles["Heading2"]))
        content.append(
            Paragraph(
                "<b>NIST AI Risk Management Framework:</b> Applicable<br/>"
                "<b>EU AI Act Compliance:</b> Risk-tier assessment applied<br/>"
                "<b>ISO/IEC 42001:</b> AI management system controls recommended<br/>"
                "<b>OECD and UNESCO:</b> Fairness, accountability, and transparency principles apply",
                styles["Normal"],
            )
        )
        content.append(Spacer(1, 12))

        metadata = model_card.get("metadata", {})
        content.append(Paragraph("Model Card Information", styles["Heading2"]))
        content.append(
            Paragraph(
                f"<b>Model ID:</b> {clean_pdf_text(model_card.get('model_id', 'N/A'))}<br/>"
                f"<b>Type:</b> {clean_pdf_text(metadata.get('type', 'N/A'))}<br/>"
                f"<b>Created:</b> {clean_pdf_text(metadata.get('created', 'N/A'))}<br/>"
                f"<b>Training Samples:</b> {model_card.get('training_data', {}).get('size', 'N/A')}<br/>"
                f"<b>Features:</b> {model_card.get('training_data', {}).get('features', 'N/A')}",
                styles["Normal"],
            )
        )
        content.append(Spacer(1, 12))

        compliance = model_card.get("compliance", {})
        content.append(Paragraph("Compliance Assessment", styles["Heading2"]))
        content.append(
            Paragraph(
                f"<b>NIST AI RMF:</b> {clean_pdf_text(compliance.get('nist_ai_rmf', 'N/A'))}<br/>"
                f"<b>EU AI Act:</b> {clean_pdf_text(compliance.get('eu_ai_act', 'N/A'))}<br/>"
                f"<b>ISO 42001:</b> {clean_pdf_text(compliance.get('iso_42001', 'N/A'))}",
                styles["Normal"],
            )
        )

        if explainability:
            content.append(PageBreak())
            content.append(Paragraph("Model Explainability Analysis", styles["Heading2"]))
            importance_df = explainability.get("feature_importance", pd.DataFrame())
            if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
                importance_data = [["Feature", "SHAP Importance"]]
                for _, row in importance_df.head(10).iterrows():
                    importance_data.append([clean_pdf_text(row["Feature"]), f"{row['SHAP_Importance']:.4f}"])
                importance_table = Table(importance_data, colWidths=[2.5 * inch, 2 * inch])
                importance_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#764ba2")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                        ]
                    )
                )
                content.append(importance_table)

        content.append(PageBreak())
        content.append(Paragraph("Governance Recommendations", styles["Heading2"]))
        recommendations = [
            "1. Implement continuous monitoring for drift detection.",
            "2. Conduct recurring fairness audits with sensitive-feature review where legally appropriate.",
            "3. Keep model card documentation updated for every material model change.",
            "4. Establish human-in-the-loop review for high-risk decisions.",
            "5. Create an incident response plan for governance violations.",
            "6. Document retraining events, approval decisions, and deployment controls.",
        ]
        for recommendation in recommendations:
            content.append(Paragraph(recommendation, styles["Normal"]))

        content.append(Spacer(1, 20))
        content.append(
            Paragraph(
                f"Report Generated: {utc_now_iso()}<br/>Aurexis Systems Version C - AI Governance Operating System",
                styles["Normal"],
            )
        )

        doc.build(content)
        return file_path
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# DATA INGESTION AND DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════
def ingest_file(file: Any) -> Optional[pd.DataFrame]:
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file)
        if name.endswith(".xlsx"):
            return pd.read_excel(file)
        if name.endswith(".json"):
            return pd.read_json(file)
        if name.endswith(".parquet"):
            return pd.read_parquet(file)
    except Exception as exc:
        st.warning(f"Could not read {getattr(file, 'name', 'file')}: {exc}")
    return None


def generate_domain_dataset(domain: str, n_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    if domain == "Finance":
        df = pd.DataFrame(
            {
                "credit_score": rng.normal(650, 50, n_samples),
                "income": rng.normal(70000, 20000, n_samples),
                "debt_ratio": rng.uniform(0.1, 0.8, n_samples),
                "loan_amount": rng.normal(20000, 8000, n_samples),
                "age": rng.integers(20, 80, n_samples),
                "gender": rng.choice([0, 1], n_samples),
            }
        )
        df["target"] = ((df["credit_score"] < 620) | (df["debt_ratio"] > 0.5)).astype(int)
    elif domain == "Healthcare":
        df = pd.DataFrame(
            {
                "age": rng.integers(20, 80, n_samples),
                "bmi": rng.normal(27, 5, n_samples),
                "blood_pressure": rng.normal(120, 15, n_samples),
                "cholesterol": rng.normal(200, 40, n_samples),
                "race": rng.choice([0, 1, 2], n_samples),
            }
        )
        df["target"] = ((df["bmi"] > 30) | (df["blood_pressure"] > 140)).astype(int)
    elif domain == "Sports":
        df = pd.DataFrame(
            {
                "speed": rng.normal(25, 5, n_samples),
                "strength": rng.normal(70, 10, n_samples),
                "stamina": rng.normal(60, 15, n_samples),
                "reaction_time": rng.normal(0.3, 0.05, n_samples),
                "gender": rng.choice([0, 1], n_samples),
            }
        )
        df["target"] = ((df["speed"] > 28) & (df["reaction_time"] < 0.28)).astype(int)
    elif domain == "Business":
        df = pd.DataFrame(
            {
                "revenue": rng.normal(1e6, 3e5, n_samples),
                "expenses": rng.normal(7e5, 2e5, n_samples),
                "customer_growth": rng.normal(0.1, 0.05, n_samples),
                "market_share": rng.uniform(0.01, 0.3, n_samples),
                "region": rng.choice([0, 1, 2], n_samples),
            }
        )
        df["target"] = ((df["revenue"] - df["expenses"] > 2e5) & (df["customer_growth"] > 0.1)).astype(int)
    elif domain == "Emotion":
        df = pd.DataFrame(
            {
                "valence": rng.uniform(-1, 1, n_samples),
                "arousal": rng.uniform(0, 1, n_samples),
                "dominance": rng.uniform(0, 1, n_samples),
                "speech_rate": rng.normal(150, 30, n_samples),
                "demographic": rng.choice([0, 1], n_samples),
            }
        )
        df["target"] = ((df["valence"] > 0.2) & (df["arousal"] > 0.5)).astype(int)
    else:
        X_arr, y_arr = make_classification(n_samples=n_samples, n_features=6, random_state=42)
        df = pd.DataFrame(X_arr, columns=[f"feature_{i}" for i in range(X_arr.shape[1])])
        df["target"] = y_arr

    return df


def load_data(uploaded_files: Optional[List[Any]], domain: str) -> Tuple[pd.DataFrame, str]:
    if uploaded_files:
        frames = []
        for uploaded_file in uploaded_files:
            frame = ingest_file(uploaded_file)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True, sort=False), "uploaded"
    return generate_domain_dataset(domain), "synthetic"


def prepare_features(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    if len(df.columns) < 2:
        st.error("Dataset must have at least 2 columns.")
        return None, None, None, None

    working_df = df.copy()
    working_df.columns = [str(col) for col in working_df.columns]
    working_df = working_df.loc[:, ~working_df.columns.duplicated()]

    default_target_index = max(0, len(working_df.columns) - 1)
    target_col = st.sidebar.selectbox("Target Column", working_df.columns, index=default_target_index)
    if target_col not in working_df.columns:
        st.error("Invalid target column.")
        return None, None, None, None

    X = working_df.drop(columns=[target_col]).copy()
    y = working_df[target_col].copy()

    lower_map = {col: str(col).lower() for col in X.columns}
    sensitive_candidates = [
        col for col, lower in lower_map.items()
        if any(token in lower for token in ("gender", "race", "age", "demographic", "region"))
    ]

    sensitive_features = None
    if sensitive_candidates:
        use_sensitive = st.sidebar.checkbox("Use sensitive features for fairness testing", value=True)
        if use_sensitive:
            selected_sensitive = st.sidebar.selectbox("Sensitive Feature", sensitive_candidates)
            sensitive_features = X[selected_sensitive].copy()

    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            try:
                X[col] = pd.to_numeric(X[col])
            except Exception:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        try:
            y = pd.to_numeric(y)
        except Exception:
            y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    else:
        y = pd.to_numeric(y, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    if sensitive_features is not None:
        if sensitive_features.dtype == "object" or str(sensitive_features.dtype).startswith("category"):
            sensitive_features = pd.Series(LabelEncoder().fit_transform(sensitive_features.astype(str)), index=sensitive_features.index)
        else:
            sensitive_features = pd.to_numeric(sensitive_features, errors="coerce").fillna(0)

    return X, y, sensitive_features, target_col


def detect_task(y: pd.Series) -> str:
    unique_count = y.nunique(dropna=True)
    if unique_count <= 20 and pd.api.types.is_integer_dtype(y.astype(float)):
        return "classification"
    return "regression"


def make_model(task: str) -> Any:
    if task == "classification":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_features: Optional[pd.Series],
    task: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    stratify = None
    if task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y

    if sensitive_features is not None:
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive_features, test_size=0.3, random_state=42, stratify=stratify
        )
        return X_train, X_test, y_train, y_test, s_train, s_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, None, None


# ══════════════════════════════════════════════════════════════════════════
# AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════
def local_governance_advice(prompt: str, metrics: Optional[Dict[str, float]], risk_class: Dict[str, Any], jurisdiction: str) -> str:
    prompt = prompt.strip() or "Provide governance guidance."
    if not metrics:
        return (
            "Hello. Train a model first so I can analyze drift, fairness, stability, and risk. "
            "In the meantime, configure the jurisdiction, domain, target column, and optional sensitive feature in the sidebar."
        )

    recommendations = []
    if metrics.get("risk_score", 0) > 0.6:
        recommendations.append("Escalate to human review before deployment.")
    if metrics.get("drift", 0) > 0.3:
        recommendations.append("Investigate data drift and consider retraining or data-quality controls.")
    if metrics.get("fairness", 0) > 0.15:
        recommendations.append("Run a fairness audit and document bias mitigation actions.")
    if metrics.get("stability", 1) < 0.5:
        recommendations.append("Hold deployment until stability improves.")
    if not recommendations:
        recommendations.append("Current metrics are within default thresholds; continue periodic monitoring.")

    return (
        f"Local Aurexis Advisor response for: {prompt}\n\n"
        f"Risk classification: {risk_class['classification']} ({risk_class['monitoring_level']} monitoring)\n"
        f"Jurisdiction/framework: {jurisdiction}\n"
        f"Current risk score: {metrics.get('risk_score', 0):.3f}\n\n"
        "Recommended next steps:\n- " + "\n- ".join(recommendations)
    )


def get_openai_api_key() -> Optional[str]:
    try:
        secret_value = st.secrets.get("OPENAI_API_KEY")
        if secret_value:
            return secret_value
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY") or None


def render_ai_advisor(domain: str, jurisdiction: str) -> None:
    st.markdown("---")
    st.subheader("🤖 Aurexis AI Governance Advisor")

    metrics = st.session_state.get("metrics")
    risk_class = classify_ai_risk_level(domain)
    api_key = get_openai_api_key()
    use_openai = _OPENAI_AVAILABLE and bool(api_key)

    if not _OPENAI_AVAILABLE:
        st.info("OpenAI package is not installed. The local advisor is active.")
    elif not api_key:
        st.info("OpenAI API key is not configured. The local advisor is active.")
    else:
        st.caption("OpenAI advisor enabled. If quota or billing is unavailable, Aurexis will automatically use local guidance.")

    if st.button("Clear Advisor Chat"):
        reset_chat()
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask about governance, compliance, risk, drift, or fairness...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    reply = ""
    if use_openai:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI governance advisor for Aurexis Systems. "
                            "Give concise, actionable guidance for NIST AI RMF, EU AI Act, ISO/IEC 42001, "
                            "OECD principles, UNESCO ethics, model risk management, drift, fairness, and audit readiness.\n\n"
                            f"Domain: {domain}\n"
                            f"Risk classification: {risk_class['classification']}\n"
                            f"Jurisdiction: {jurisdiction}\n"
                            f"Metrics: {json.dumps(metrics or {}, default=safe_json_default)}"
                        ),
                    },
                    *st.session_state.messages,
                ],
                temperature=0.3,
                max_tokens=800,
            )
            reply = response.choices[0].message.content or ""
        except OpenAIRateLimitError:
            st.warning(
                "OpenAI quota or billing is unavailable for this API key. "
                "Using the built-in local Aurexis Advisor instead."
            )
            reply = local_governance_advice(user_input, metrics, risk_class, jurisdiction)
        except Exception as exc:
            st.warning(f"OpenAI advisor unavailable ({exc}). Using local guidance instead.")
            reply = local_governance_advice(user_input, metrics, risk_class, jurisdiction)
    else:
        reply = local_governance_advice(user_input, metrics, risk_class, jurisdiction)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)


# ══════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="governance-header">
        <h1>⚖️ AUREXIS SYSTEMS — VERSION C</h1>
        <p>AI Governance Operating System (Production-Grade)</p>
        <p>NIST AI RMF | EU AI Act | ISO/IEC 42001 | OECD Principles | UNESCO Ethics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for key, value in {
    "model": None,
    "model_task": None,
    "metrics": None,
    "messages": [],
    "jurisdiction": "United States (SR 11-7)",
    "model_card": None,
    "model_id": None,
    "explainability": None,
    "gov_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

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
selected_framework = st.sidebar.selectbox("Select Framework Reference", list(GOVERNANCE_FRAMEWORKS.keys()))
if st.sidebar.button("📖 View Framework Details"):
    st.sidebar.info(
        f"**{selected_framework}**\n\n"
        + "\n".join([f"• {key}: {value}" for key, value in GOVERNANCE_FRAMEWORKS[selected_framework].items()])
    )

with st.expander("Dependency Status", expanded=False):
    st.write(
        {
            "SHAP explainability": "available" if _SHAP_AVAILABLE else "not installed - app still works",
            "Fairlearn advanced fairness": "available" if _FAIRLEARN_AVAILABLE else "not installed - basic fairness active",
            "OpenAI advisor": "available" if _OPENAI_AVAILABLE and get_openai_api_key() else "local advisor active",
        }
    )

df, data_source = load_data(uploaded_files, domain)
st.info(f"📁 Data: **{data_source}** | {len(df):,} rows × {len(df.columns)} columns")
st.dataframe(df.head(10), use_container_width=True)

X, y, sensitive_features, target_col = prepare_features(df)
if X is None or y is None:
    st.stop()

task = detect_task(y)
if st.session_state.model is None or st.session_state.model_task != task:
    st.session_state.model = make_model(task)
    st.session_state.model_task = task

st.caption(f"Detected task: **{task}** | Target column: **{target_col}**")

st.subheader("🚀 Model Training & Governance Assessment")
col1, col2 = st.columns([3, 1])
with col1:
    st.write("Train the model and trigger comprehensive governance assessment.")
with col2:
    train_clicked = st.button("🚀 Train Model", use_container_width=True)

if train_clicked:
    with st.spinner("Training model and evaluating governance controls..."):
        try:
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = split_dataset(
                X, y, sensitive_features, task
            )
            model = make_model(task)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            drift_metrics = compute_drift_comprehensive(X_train, X_test)
            fairness_metrics = compute_fairness_comprehensive(preds, y_test, sensitive_test, task)
            stability = system_stability_score(drift_metrics["overall"], fairness_metrics["composite"])
            uncertainty = compute_model_uncertainty(model, X_test, task)
            risk_score = compute_risk_score(drift_metrics["overall"], fairness_metrics["composite"], uncertainty)

            metrics = {
                "drift": drift_metrics["overall"],
                "wasserstein": drift_metrics["wasserstein"],
                "psi": drift_metrics["psi"],
                "ks": drift_metrics["ks"],
                "fairness": fairness_metrics["composite"],
                "stability": stability,
                "risk_score": risk_score,
                "uncertainty": uncertainty,
                "dp": fairness_metrics.get("demographic_parity", 0),
                "eo": fairness_metrics.get("equalized_odds", 0),
            }

            risk_class = classify_ai_risk_level(domain)
            model_card, model_id = create_model_card(model, X_train, y_train, metrics, jurisdiction, domain, risk_class)
            explainability = generate_explainability_report(model, X_test) if _SHAP_AVAILABLE else None
            gov_result = governance_intervention(model, metrics, domain, jurisdiction, X_train, y_train)

            st.session_state.model = model
            st.session_state.model_task = task
            st.session_state.metrics = metrics
            st.session_state.model_card = model_card
            st.session_state.model_id = model_id
            st.session_state.explainability = explainability
            st.session_state.gov_result = gov_result

            st.success("Model trained and governance assessment completed.")
            st.balloons()
        except Exception as exc:
            st.error(f"Training failed: {exc}")

if st.session_state.metrics:
    metrics = st.session_state.metrics
    gov_result = st.session_state.get("gov_result")

    if gov_result:
        render_governance_messages(gov_result)

    st.subheader("📊 Comprehensive Metrics Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Drift Score", f"{metrics['drift']:.3f}", delta="HIGH" if metrics["drift"] > 0.3 else "OK")
    col2.metric("Fairness Gap", f"{metrics['fairness']:.3f}", delta="HIGH" if metrics["fairness"] > 0.15 else "OK")
    col3.metric("Stability", f"{metrics['stability']:.3f}", delta="STABLE" if metrics["stability"] > 0.5 else "LOW")
    col4.metric("Uncertainty", f"{metrics['uncertainty']:.4f}")
    risk_status = "LOW" if metrics["risk_score"] < 0.3 else "MEDIUM" if metrics["risk_score"] < 0.6 else "HIGH"
    col5.metric("Risk Score", f"{metrics['risk_score']:.3f}", delta=risk_status)

    with st.expander("Detailed Drift Metrics"):
        st.json(
            {
                "wasserstein": metrics.get("wasserstein", 0),
                "population_stability_index": metrics.get("psi", 0),
                "kolmogorov_smirnov": metrics.get("ks", 0),
            }
        )

    if metrics.get("dp") or metrics.get("eo"):
        st.subheader("Advanced Fairness Metrics")
        fc1, fc2 = st.columns(2)
        fc1.metric("Demographic Parity Diff", f"{metrics.get('dp', 0):.4f}")
        fc2.metric("Equalized Odds Diff", f"{metrics.get('eo', 0):.4f}")

    if st.session_state.model_card:
        st.subheader("📋 Model Card (Governance Artifact)")
        card = st.session_state.model_card
        tab1, tab2, tab3 = st.tabs(["Metadata", "Governance", "Compliance"])
        with tab1:
            metadata = card.get("metadata", {})
            st.json(
                {
                    "Model ID": card.get("model_id"),
                    "Type": metadata.get("type"),
                    "Domain": metadata.get("domain"),
                    "Created": metadata.get("created"),
                    "Training Samples": card.get("training_data", {}).get("size"),
                    "Features": card.get("training_data", {}).get("features"),
                }
            )
        with tab2:
            st.json(card.get("governance", {}))
        with tab3:
            st.json(card.get("compliance", {}))

    if st.session_state.explainability and _SHAP_AVAILABLE:
        st.subheader("🔍 Model Explainability (SHAP Analysis)")
        importance_df = st.session_state.explainability.get("feature_importance")
        if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
            st.bar_chart(importance_df.set_index("Feature").head(10))
            with st.expander("Feature Importance Details"):
                st.dataframe(importance_df.head(15), use_container_width=True)

    st.subheader("📄 Governance Report Generation")
    if st.button("📊 Generate Comprehensive PDF Report", use_container_width=True):
        with st.spinner("Generating report..."):
            report_path = generate_comprehensive_pdf_report(
                metrics,
                classify_ai_risk_level(domain),
                st.session_state.model_card,
                st.session_state.explainability,
                filename=f"aurexis_report_{st.session_state.model_id}.pdf",
            )
            if report_path and report_path.exists():
                with report_path.open("rb") as pdf_file:
                    st.download_button(
                        "📥 Download PDF Report",
                        pdf_file,
                        file_name=f"aurexis_governance_report_{st.session_state.model_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                st.success("Report generated successfully.")

st.subheader("📜 Governance Audit Trail")
audit_logs = load_audit_logs()
if audit_logs:
    audit_df = pd.DataFrame(audit_logs)
    with st.expander("View Detailed Audit Logs", expanded=False):
        st.dataframe(audit_df, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(audit_logs))
    col2.metric("Governance Checks", len([log for log in audit_logs if log.get("event_type") == "governance_check"]))
    col3.metric("Escalations", len([log for log in audit_logs if "escalate" in log.get("governance", {}).get("action", "")]))
else:
    st.info("No audit logs yet. Train a model to generate governance events.")

st.subheader("🗂️ Model Cards Repository")
model_cards = load_model_cards()
if model_cards:
    with st.expander(f"View {len(model_cards)} Saved Model Cards"):
        for index, card in enumerate(model_cards, 1):
            card_col1, card_col2 = st.columns([3, 1])
            with card_col1:
                st.write(f"**{index}. {card.get('model_id')}** - {card.get('metadata', {}).get('domain')}")
            with card_col2:
                if st.button("📋 View", key=f"card_{index}"):
                    st.json(card)
else:
    st.info("No model cards yet. Train models to populate the repository.")

st.subheader("📚 Governance Frameworks Reference")
framework_tabs = st.tabs(list(GOVERNANCE_FRAMEWORKS.keys()))
for tab, framework_name in zip(framework_tabs, GOVERNANCE_FRAMEWORKS.keys()):
    with tab:
        for key, value in GOVERNANCE_FRAMEWORKS[framework_name].items():
            st.write(f"**{key}:** {value}")

render_ai_advisor(domain, jurisdiction)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 12px;">
    <p><strong>AUREXIS SYSTEMS — VERSION C</strong> | AI Governance Operating System</p>
    <p>NIST AI RMF • EU AI Act • ISO/IEC 42001 • OECD Principles • UNESCO Ethics</p>
    <p>© 2026 Aurexis Systems. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
