from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class HealthResponse(BaseModel):

    status: str

    service: str

    version: str

class RiskClassificationResponse(BaseModel):

    classification: str

    emoji: str

    reasoning: str

    requires_audit: bool

    monitoring_level: str

class GovernanceMetrics(BaseModel):

    drift: float = Field(default=0.0, ge=0.0, le=1.0)

    wasserstein: float = Field(default=0.0, ge=0.0, le=1.0)

    psi: float = Field(default=0.0, ge=0.0, le=1.0)

    ks: float = Field(default=0.0, ge=0.0, le=1.0)

    fairness: float = Field(default=0.0, ge=0.0, le=1.0)

    dp: float = Field(default=0.0, ge=0.0, le=1.0)

    eo: float = Field(default=0.0, ge=0.0, le=1.0)

    stability: float = Field(default=0.0, ge=0.0, le=1.0)

    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)

    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)

class GovernanceAssessmentRequest(BaseModel):

    domain: str = Field(default="General")

    jurisdiction: str = Field(

        default="United States (SR 11-7)"

    )

    metrics: GovernanceMetrics

class GovernanceAssessmentResponse(BaseModel):

    domain: str

    jurisdiction: str

    risk_class: RiskClassificationResponse

    metrics: GovernanceMetrics

    actions: List[str]

    primary_action: str

    messages: List[Dict[str, str]]

class AdvisorRequest(BaseModel):

    prompt: str

    domain: str = "General"

    jurisdiction: str = "United States (SR 11-7)"

    metrics: Optional[GovernanceMetrics] = None

class AdvisorResponse(BaseModel):

    response: str

    provider: str

    domain: str

    jurisdiction: str

class FrameworkResponse(BaseModel):

    name: str

    description: str

    principles: List[str]

class AuditEventResponse(BaseModel):

    timestamp: str

    event_type: str

    model_name: Optional[str] = None

    metrics: Dict[str, Any] = Field(default_factory=dict)

    governance: Dict[str, Any] = Field(default_factory=dict)
