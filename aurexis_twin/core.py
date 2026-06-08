from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.stats import ks_2samp, wasserstein_distance


ARTIFACT_ROOT = Path(os.getenv("AUREXIS_ARTIFACT_DIR", "/tmp/aurexis_serverless_twin"))
AUDIT_LOG = ARTIFACT_ROOT / "audit" / "governance_events.jsonl"
REPORT_DIR = ARTIFACT_ROOT / "reports"
DATA_DIR = ARTIFACT_ROOT / "data"

THRESHOLDS = {
    "drift": float(os.getenv("AUREXIS_DRIFT_THRESHOLD", "0.30")),
    "fairness": float(os.getenv("AUREXIS_FAIRNESS_THRESHOLD", "0.15")),
    "risk": float(os.getenv("AUREXIS_RISK_THRESHOLD", "0.60")),
    "stability": float(os.getenv("AUREXIS_STABILITY_THRESHOLD", "0.50")),
}


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_artifact_dirs() -> None:
    for path in (ARTIFACT_ROOT, AUDIT_LOG.parent, REPORT_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def safe_json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass
class EvaluationResult:
    model_name: str
    jurisdiction: str
    drift: float
    fairness: float
    stability: float
    risk_score: float
    intervention: str
    compliance: str
    timestamp: str
    details: Dict[str, Any]


@dataclass
class ForecastResult:
    model_name: str
    threshold: float
    projected_drift: List[Dict[str, float]]
    days_until_threshold: Optional[int]
    recommended_action: str
    timestamp: str


@dataclass
class ModelSpec:
    name: str
    jurisdiction: str = "EU AI Act"
    domain: str = "Finance"
    risk_weight: float = 1.0


@dataclass
class BatchAssessmentResult:
    timestamp: str
    models: List[EvaluationResult]
    report_path: Optional[str]
    summary: Dict[str, Any]


class AuditStore:
    def __init__(self, path: Path = AUDIT_LOG):
        self.path = path
        ensure_artifact_dirs()

    def append(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "event_id": hashlib.sha256(f"{event_type}:{utc_now_iso()}:{json.dumps(payload, default=safe_json_default)}".encode()).hexdigest()[:16],
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=safe_json_default) + "\n")

    def read(self, limit: int = 1000) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-limit:]


def load_reference_data(n_samples: int = 500, n_features: int = 6, seed: int = 42) -> pd.DataFrame:
    """Load or synthesize reference training behavior for the governance twin."""
    data_path = Path(os.getenv("AUREXIS_REFERENCE_DATA", ""))
    if data_path.exists() and data_path.is_file():
        return _read_dataframe(data_path)

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(0, 1, size=(n_samples, n_features)),
        columns=[f"feature_{idx}" for idx in range(n_features)],
    )
    frame["label"] = ((frame["feature_0"] + frame["feature_1"] * 0.4 + rng.normal(0, 0.2, n_samples)) > 0).astype(int)
    return frame


def load_production_data(n_samples: int = 500, n_features: int = 6, drift: float = 0.08, seed: int = 101) -> pd.DataFrame:
    """Load or synthesize current production behavior."""
    data_path = Path(os.getenv("AUREXIS_PRODUCTION_DATA", ""))
    if data_path.exists() and data_path.is_file():
        return _read_dataframe(data_path)

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(drift, 1 + drift, size=(n_samples, n_features)),
        columns=[f"feature_{idx}" for idx in range(n_features)],
    )
    frame["label"] = ((frame["feature_0"] + frame["feature_1"] * 0.4 + rng.normal(0, 0.25, n_samples)) > 0).astype(int)
    return frame


def _read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file: {path}")


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric


def compute_drift(reference: pd.DataFrame, production: pd.DataFrame) -> Dict[str, float]:
    ref = _numeric_features(reference)
    prod = _numeric_features(production)
    common_cols = [col for col in ref.columns if col in prod.columns and col != "label"]
    if not common_cols:
        return {"wasserstein": 0.0, "ks": 0.0, "overall": 0.0}

    wasserstein_scores: List[float] = []
    ks_scores: List[float] = []
    for col in common_cols:
        x1 = ref[col].to_numpy()
        x2 = prod[col].to_numpy()
        if len(x1) < 2 or len(x2) < 2:
            continue
        x1_norm = (x1 - np.mean(x1)) / (np.std(x1) + 1e-6)
        x2_norm = (x2 - np.mean(x2)) / (np.std(x2) + 1e-6)
        wasserstein_scores.append(float(wasserstein_distance(x1_norm, x2_norm)))
        ks_scores.append(float(ks_2samp(x1, x2)[0]))

    wasserstein_score = float(np.mean(wasserstein_scores)) if wasserstein_scores else 0.0
    ks_score = float(np.mean(ks_scores)) if ks_scores else 0.0
    return {
        "wasserstein": wasserstein_score,
        "ks": ks_score,
        "overall": clamp01(np.mean([clamp01(wasserstein_score), clamp01(ks_score)])),
    }


def compute_fairness(predictions: Sequence[int], labels: Optional[Sequence[int]] = None) -> float:
    preds = np.asarray(predictions, dtype=float)
    if preds.size == 0:
        return 0.0
    if labels is None:
        return clamp01(abs(float(preds.mean()) - 0.5))
    y = np.asarray(labels, dtype=float)
    if y.size == 0:
        return 0.0
    min_len = min(len(preds), len(y))
    return clamp01(abs(float(preds[:min_len].mean()) - float(y[:min_len].mean())))


def system_stability_score(drift: float, fairness: float) -> float:
    return clamp01((1.0 - drift) * 0.55 + (1.0 - fairness) * 0.45)


def compute_risk_score(drift: float, fairness: float, stability: float) -> float:
    return clamp01(drift * 0.40 + fairness * 0.35 + (1.0 - stability) * 0.25)


def choose_intervention(drift: float, fairness: float, stability: float, risk_score: float) -> str:
    if risk_score >= THRESHOLDS["risk"]:
        return "suspend_and_human_review"
    if drift >= THRESHOLDS["drift"]:
        return "retrain"
    if fairness >= THRESHOLDS["fairness"]:
        return "debias"
    if stability <= THRESHOLDS["stability"]:
        return "stability_review"
    return "stable"


def compliance_verdict(intervention: str, jurisdiction: str) -> str:
    if intervention in {"suspend_and_human_review", "retrain", "debias"}:
        return "review_required"
    if jurisdiction.lower().startswith("eu"):
        return "pass_with_monitoring"
    return "pass"


def evaluate_predictions(
    predictions: Sequence[int],
    labels: Optional[Sequence[int]] = None,
    reference_data: Optional[pd.DataFrame] = None,
    production_data: Optional[pd.DataFrame] = None,
    model_name: str = "external_model",
    jurisdiction: str = "EU AI Act",
) -> EvaluationResult:
    reference_data = reference_data if reference_data is not None else load_reference_data()
    production_data = production_data if production_data is not None else load_production_data()
    drift_metrics = compute_drift(reference_data, production_data)
    fairness = compute_fairness(predictions, labels)
    stability = system_stability_score(drift_metrics["overall"], fairness)
    risk_score = compute_risk_score(drift_metrics["overall"], fairness, stability)
    intervention = choose_intervention(drift_metrics["overall"], fairness, stability, risk_score)
    verdict = compliance_verdict(intervention, jurisdiction)
    return EvaluationResult(
        model_name=model_name,
        jurisdiction=jurisdiction,
        drift=drift_metrics["overall"],
        fairness=fairness,
        stability=stability,
        risk_score=risk_score,
        intervention=intervention,
        compliance=verdict,
        timestamp=utc_now_iso(),
        details={"drift_metrics": drift_metrics, "thresholds": THRESHOLDS},
    )


def predict_labels_from_features(frame: pd.DataFrame) -> np.ndarray:
    numeric = _numeric_features(frame)
    feature_cols = [col for col in numeric.columns if col != "label"]
    if not feature_cols:
        return np.zeros(len(frame), dtype=int)
    score = numeric[feature_cols].sum(axis=1)
    return (score > score.median()).astype(int).to_numpy()


def run_governance_cycle(
    model_name: str = "RF",
    jurisdiction: str = "EU AI Act",
    audit_store: Optional[AuditStore] = None,
) -> EvaluationResult:
    reference = load_reference_data()
    production = load_production_data()
    labels = production["label"].to_numpy() if "label" in production.columns else None
    predictions = predict_labels_from_features(production)
    result = evaluate_predictions(predictions, labels, reference, production, model_name, jurisdiction)
    audit = audit_store or AuditStore()
    audit.append("scheduled_governance_evaluation", asdict(result))
    report_path = generate_compliance_report(result, audit.read(limit=50), jurisdiction=jurisdiction)
    audit.append("compliance_report_generated", {"model_name": model_name, "report_path": str(report_path)})
    return result


def simulate_drift_forecast(
    reference: Optional[pd.DataFrame] = None,
    base_production: Optional[pd.DataFrame] = None,
    model_name: str = "RF",
    horizon_days: int = 45,
    step_days: int = 7,
    threshold: float = THRESHOLDS["drift"],
) -> ForecastResult:
    reference = reference if reference is not None else load_reference_data()
    base_production = base_production if base_production is not None else load_production_data()
    numeric = _numeric_features(base_production)
    projected: List[Dict[str, float]] = []
    days_until_threshold: Optional[int] = None

    for day in range(step_days, horizon_days + 1, step_days):
        drift_factor = day / max(horizon_days, 1)
        simulated = numeric.copy()
        for idx, col in enumerate([col for col in simulated.columns if col != "label"]):
            simulated[col] = simulated[col] + drift_factor * (idx + 1) * 0.08
        if "label" in base_production.columns:
            simulated["label"] = base_production["label"].to_numpy()
        drift_value = compute_drift(reference, simulated)["overall"]
        projected.append({"day": float(day), "drift": float(drift_value)})
        if days_until_threshold is None and drift_value >= threshold:
            days_until_threshold = day

    action = "retrain" if days_until_threshold is not None else "continue_monitoring"
    return ForecastResult(
        model_name=model_name,
        threshold=threshold,
        projected_drift=projected,
        days_until_threshold=days_until_threshold,
        recommended_action=action,
        timestamp=utc_now_iso(),
    )


def load_model_specs() -> List[ModelSpec]:
    config_path = Path(os.getenv("AUREXIS_MODEL_REGISTRY", ""))
    if config_path.exists() and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return [ModelSpec(**item) for item in raw.get("models", raw if isinstance(raw, list) else [])]
    return [
        ModelSpec(name="credit_model", jurisdiction="EU AI Act", domain="Finance", risk_weight=1.0),
        ModelSpec(name="fraud_model", jurisdiction="SR 11-7", domain="Finance", risk_weight=1.1),
        ModelSpec(name="aml_model", jurisdiction="UK Model Risk Guidance", domain="Finance", risk_weight=1.2),
        ModelSpec(name="loan_approval_model", jurisdiction="EU AI Act", domain="Finance", risk_weight=1.0),
    ]


def batch_risk_assessment(model_specs: Optional[Iterable[ModelSpec]] = None) -> BatchAssessmentResult:
    specs = list(model_specs or load_model_specs())
    audit = AuditStore()
    results: List[EvaluationResult] = []
    for idx, spec in enumerate(specs):
        reference = load_reference_data(seed=42 + idx)
        production = load_production_data(drift=0.08 * spec.risk_weight + idx * 0.03, seed=101 + idx)
        labels = production["label"].to_numpy() if "label" in production.columns else None
        predictions = predict_labels_from_features(production)
        result = evaluate_predictions(predictions, labels, reference, production, spec.name, spec.jurisdiction)
        results.append(result)
        audit.append("batch_model_assessment", asdict(result))

    report_path = generate_enterprise_report(results)
    summary = {
        "total_models": len(results),
        "review_required": sum(result.compliance != "pass" and result.compliance != "pass_with_monitoring" for result in results),
        "max_risk_score": max((result.risk_score for result in results), default=0.0),
        "generated_at": utc_now_iso(),
    }
    audit.append("batch_risk_assessment_complete", {"summary": summary, "report_path": str(report_path)})
    return BatchAssessmentResult(timestamp=utc_now_iso(), models=results, report_path=str(report_path), summary=summary)


def generate_compliance_report(
    result: EvaluationResult,
    audit_events: Optional[List[Dict[str, Any]]] = None,
    jurisdiction: str = "EU AI Act",
    filename: Optional[str] = None,
) -> Path:
    ensure_artifact_dirs()
    filename = filename or f"{result.model_name}_{jurisdiction.replace(' ', '_')}_Report.pdf"
    path = REPORT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=(8.5 * inch, 11 * inch), topMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    story: List[Any] = [
        Paragraph("AUREXIS SERVERLESS GOVERNANCE TWIN", styles["Title"]),
        Paragraph(f"{jurisdiction} Compliance Evidence Report", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph("Executive Summary", styles["Heading2"]),
        Paragraph(
            f"Model {result.model_name} was autonomously evaluated at {result.timestamp}. "
            f"Final verdict: {result.compliance}. Recommended intervention: {result.intervention}.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]
    rows = [
        ["Metric", "Value", "Threshold"],
        ["Drift", f"{result.drift:.3f}", f"{THRESHOLDS['drift']:.3f}"],
        ["Fairness", f"{result.fairness:.3f}", f"{THRESHOLDS['fairness']:.3f}"],
        ["Stability", f"{result.stability:.3f}", f"> {THRESHOLDS['stability']:.3f}"],
        ["Risk Score", f"{result.risk_score:.3f}", f"{THRESHOLDS['risk']:.3f}"],
        ["Intervention", result.intervention, ""],
        ["Compliance", result.compliance, ""],
    ]
    table = Table(rows, colWidths=[2 * inch, 2 * inch, 2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.extend([table, Spacer(1, 12), Paragraph("Audit Trail", styles["Heading2"])])
    for event in (audit_events or [])[-10:]:
        story.append(Paragraph(f"{event.get('timestamp')} - {event.get('event_type')}", styles["Normal"]))
    doc.build(story)
    return path


def generate_enterprise_report(results: List[EvaluationResult], filename: str = "Enterprise_Batch_Risk_Report.pdf") -> Path:
    ensure_artifact_dirs()
    path = REPORT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=(8.5 * inch, 11 * inch), topMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    rows = [["Model", "Jurisdiction", "Drift", "Fairness", "Stability", "Risk", "Intervention"]]
    for result in results:
        rows.append([
            result.model_name,
            result.jurisdiction,
            f"{result.drift:.3f}",
            f"{result.fairness:.3f}",
            f"{result.stability:.3f}",
            f"{result.risk_score:.3f}",
            result.intervention,
        ])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#764ba2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story = [
        Paragraph("AUREXIS ENTERPRISE AI GOVERNANCE REPORT", styles["Title"]),
        Paragraph("Batch Risk Assessment Across Enterprise AI Systems", styles["Heading2"]),
        Spacer(1, 12),
        table,
    ]
    doc.build(story)
    return path


def write_json_artifact(name: str, payload: Dict[str, Any]) -> Path:
    ensure_artifact_dirs()
    path = ARTIFACT_ROOT / f"{name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=safe_json_default)
    return path
