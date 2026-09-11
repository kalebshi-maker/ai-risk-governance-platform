from datetime import datetime, timezone

from typing import Dict, Any

from fastapi import FastAPI, HTTPException

from backend.governance import (

    APP_NAME,

    APP_VERSION,

    classify_ai_risk_level,

    governance_intervention,

    list_frameworks,

)

from backend.schemas import (

    AdvisorRequest,

    AdvisorResponse,

    AuditEventResponse,

    GovernanceAssessmentRequest,

    GovernanceAssessmentResponse,

    HealthResponse,

    RiskClassificationResponse,

)

from backend.storage import read_audit_events, write_audit_event

app = FastAPI(

    title="AUREXIS SYSTEMS",

    description=(

        "AI Governance Operating System — Version C"

    ),

    version=APP_VERSION,

)

@app.get(

    "/",

    tags=["System"],

)

def root() -> Dict[str, str]:

    return {

        "service": APP_NAME,

        "version": APP_VERSION,

        "status": "operational",

    }

@app.get(

    "/api/v1/health",

    response_model=HealthResponse,

    tags=["System"],

)

def health() -> HealthResponse:

    return HealthResponse(

        status="healthy",

        service=APP_NAME,

        version=APP_VERSION,

    )

@app.get(

    "/api/v1/governance/frameworks",

    tags=["Governance"],

)

def frameworks():

    return {

        "version": APP_VERSION,

        "frameworks": list_frameworks(),

    }

@app.get(

    "/api/v1/governance/risk/{domain}",

    response_model=RiskClassificationResponse,

    tags=["Governance"],

)

def governance_risk(domain: str):

    return classify_ai_risk_level(domain)

@app.post(

    "/api/v1/governance/assess",

    response_model=GovernanceAssessmentResponse,

    tags=["Governance"],

)

def governance_assess(

    request: GovernanceAssessmentRequest,

):

    result = governance_intervention(

        domain=request.domain,

        jurisdiction=request.jurisdiction,

        metrics=request.metrics.model_dump(),

    )

    write_audit_event(

        {

            "timestamp": datetime.now(

                timezone.utc

            ).isoformat(),

            "event_type": "governance_assessment",

            "model_name": None,

            "metrics": result["metrics"],

            "governance": {

                "domain": request.domain,

                "jurisdiction": request.jurisdiction,

                "risk_class": result["risk_class"],

                "primary_action": result["primary_action"],

            },

        }

    )

    return result

@app.get(

    "/api/v1/audit/events",

    response_model=list[AuditEventResponse],

    tags=["Audit"],

)

def audit_events():

    return read_audit_events()

@app.post(

    "/api/v1/advisor",

    response_model=AdvisorResponse,

    tags=["Advisor"],

)

def advisor(

    request: AdvisorRequest,

):

    """

    Backend-safe advisor endpoint.

    This intentionally does not require OpenAI.

    The existing Streamlit Version C AI Advisor remains unchanged.

    Future versions can connect this endpoint to an external LLM

    provider without changing the governance engine.

    """

    risk_text = ""

    if request.metrics is not None:

        metrics = request.metrics.model_dump()

        result = governance_intervention(

            domain=request.domain,

            jurisdiction=request.jurisdiction,

            metrics=metrics,

        )

        risk_text = (

            f"\n\nCurrent governance assessment:\n"

            f"- Primary action: {result['primary_action']}\n"

            f"- Risk score: {result['metrics']['risk_score']:.3f}\n"

            f"- Stability: {result['metrics']['stability']:.3f}\n"

        )

    response = (

        "AUREXIS SYSTEMS Governance Advisor\n\n"

        f"Domain: {request.domain}\n"

        f"Jurisdiction: {request.jurisdiction}\n\n"

        f"Question: {request.prompt}"

        f"{risk_text}\n\n"

        "This API endpoint currently provides deterministic "

        "governance context. The Version C Streamlit application "

        "continues to provide the existing multilingual AI Advisor."

    )

    return AdvisorResponse(

        response=response,

        provider="aurexis-governance-engine",

        domain=request.domain,

        jurisdiction=request.jurisdiction,

    )

@app.get(

    "/api/v1/audit/events/{event_index}",

    tags=["Audit"],

)

def audit_event(event_index: int):

    events = read_audit_events()

    if event_index < 0 or event_index >= len(events):

        raise HTTPException(

            status_code=404,

            detail="Audit event not found.",

        )

    return events[event_index]
