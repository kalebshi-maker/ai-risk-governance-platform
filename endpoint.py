"""Aurexis Governance Twin Endpoint.

Run locally:
    uvicorn endpoint:app --host 0.0.0.0 --port 8000

Nebius Endpoint target:
    endpoint:app
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from aurexis_twin import AuditStore, evaluate_predictions


class EvaluationRequest(BaseModel):
    predictions: List[int] = Field(..., description="Model predictions.")
    labels: Optional[List[int]] = Field(default=None, description="Ground-truth labels when available.")
    jurisdiction: str = Field(default="EU AI Act", description="Regulatory context.")
    model_name: str = Field(default="external_model", description="Name of the AI system.")
    reference_rows: Optional[List[dict]] = Field(default=None, description="Optional reference feature rows.")
    production_rows: Optional[List[dict]] = Field(default=None, description="Optional production feature rows.")


class EvaluationResponse(BaseModel):
    model_name: str
    jurisdiction: str
    drift: float
    fairness: float
    stability: float
    risk_score: float
    intervention: str
    compliance: str
    timestamp: str
    details: dict


app = FastAPI(
    title="Aurexis Serverless Governance Twin Endpoint",
    version="1.0.0",
    description="Autonomous governance endpoint for production AI outputs.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "aurexis-governance-twin"}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest) -> dict:
    reference = pd.DataFrame(request.reference_rows) if request.reference_rows else None
    production = pd.DataFrame(request.production_rows) if request.production_rows else None
    result = evaluate_predictions(
        predictions=request.predictions,
        labels=request.labels,
        reference_data=reference,
        production_data=production,
        model_name=request.model_name,
        jurisdiction=request.jurisdiction,
    )
    AuditStore().append("endpoint_evaluation", asdict(result))
    return asdict(result)

