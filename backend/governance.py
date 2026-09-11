from typing import Any, Dict, List

APP_VERSION = "Version C"

APP_NAME = "AUREXIS SYSTEMS"

GOVERNANCE_FRAMEWORKS = {

    "NIST AI Risk Management Framework": {

        "description": (

            "Risk management framework for governing, mapping, "

            "measuring, and managing AI risks."

        ),

        "principles": [

            "Govern",

            "Map",

            "Measure",

            "Manage",

        ],

    },

    "EU AI Act": {

        "description": (

            "Risk-based regulatory framework for artificial "

            "intelligence systems in the European Union."

        ),

        "principles": [

            "Risk classification",

            "Transparency",

            "Human oversight",

            "Technical documentation",

            "Monitoring",

        ],

    },

    "OECD AI Principles": {

        "description": (

            "Principles supporting trustworthy and human-centered "

            "AI development and deployment."

        ),

        "principles": [

            "Inclusive growth",

            "Human-centered values",

            "Transparency",

            "Robustness",

            "Accountability",

        ],

    },

    "UNESCO AI Ethics Recommendations": {

        "description": (

            "Ethical framework emphasizing human rights, "

            "fairness, transparency, and sustainability."

        ),

        "principles": [

            "Human rights",

            "Fairness",

            "Transparency",

            "Human oversight",

            "Sustainability",

        ],

    },

    "ISO/IEC 42001 AI Management Systems": {

        "description": (

            "Management-system framework for responsible "

            "governance of artificial intelligence."

        ),

        "principles": [

            "AI governance",

            "Risk management",

            "Impact assessment",

            "Operational controls",

            "Continual improvement",

        ],

    },

}

DOMAIN_RISK_MAPPING = {

    "Healthcare": {

        "classification": "High Risk",

        "emoji": "🔴",

        "reasoning": "Critical direct patient impact",

    },

    "Finance": {

        "classification": "High Risk",

        "emoji": "🔴",

        "reasoning": "Financial stability",

    },

    "Criminal Justice": {

        "classification": "High Risk",

        "emoji": "🔴",

        "reasoning": "Civil rights impact",

    },

    "Sports": {

        "classification": "Limited Risk",

        "emoji": "🟡",

        "reasoning": "Potential consequential decision support",

    },

    "Business": {

        "classification": "Limited Risk",

        "emoji": "🟡",

        "reasoning": "Business decision support",

    },

    "Emotion": {

        "classification": "Limited Risk",

        "emoji": "🟡",

        "reasoning": "Potentially sensitive inference",

    },

    "General": {

        "classification": "Minimal Risk",

        "emoji": "🟢",

        "reasoning": "General-purpose AI application",

    },

}

def clamp01(value: float) -> float:

    return max(0.0, min(1.0, float(value)))

def classify_ai_risk_level(domain: str) -> Dict[str, Any]:

    """

    Classify AI risk using the Version C domain mapping.

    """

    result = DOMAIN_RISK_MAPPING.get(

        domain,

        DOMAIN_RISK_MAPPING["General"],

    )

    classification = result["classification"]

    return {

        "classification": classification,

        "emoji": result["emoji"],

        "reasoning": result["reasoning"],

        "requires_audit": classification == "High Risk",

        "monitoring_level": (

            "Continuous"

            if classification == "High Risk"

            else "Periodic"

        ),

    }

def compute_risk_score(

    drift: float,

    fairness: float,

    uncertainty: float,

) -> float:

    """

    Version C risk formula:

    drift       = 35%

    fairness    = 35%

    uncertainty = 30%

    """

    return clamp01(

        drift * 0.35

        + fairness * 0.35

        + uncertainty * 0.30

    )

def system_stability_score(

    drift: float,

    fairness: float,

) -> float:

    """

    Version C stability formula.

    """

    return clamp01(

        (1.0 - drift) * 0.5

        + (1.0 - fairness) * 0.5

    )

def get_fairness_status(

    fairness: float,

) -> str:

    if fairness > 0.15:

        return "Needs attention"

    return "Acceptable"

def governance_intervention(

    domain: str,

    jurisdiction: str,

    metrics: Dict[str, float],

) -> Dict[str, Any]:

    """

    API-safe equivalent of the Version C governance intervention

    decision layer.

    Important:

    The server calculates the risk and intervention decision.

    """

    drift = clamp01(metrics.get("drift", 0.0))

    fairness = clamp01(metrics.get("fairness", 0.0))

    uncertainty = clamp01(metrics.get("uncertainty", 0.0))

    stability = clamp01(

        metrics.get(

            "stability",

            system_stability_score(

                drift,

                fairness,

            ),

        )

    )

    risk_score = compute_risk_score(

        drift,

        fairness,

        uncertainty,

    )

    risk_class = classify_ai_risk_level(domain)

    actions: List[str] = []

    messages: List[Dict[str, str]] = []

    # High-risk domains

    if risk_class["classification"] == "High Risk":

        actions.append(

            "Continuous monitoring required for high-risk AI deployment."

        )

        messages.append(

            {

                "level": "warning",

                "message": (

                    "High-risk domain detected. "

                    "Maintain continuous monitoring and human oversight."

                ),

            }

        )

    # Drift

    if drift > 0.30:

        actions.append(

            "Retraining or data-quality controls recommended."

        )

        messages.append(

            {

                "level": "warning",

                "message": (

                    "Significant data drift detected. "

                    "Review production data and consider retraining."

                ),

            }

        )

    # Fairness

    if fairness > 0.15:

        actions.append(

            "Fairness audit and bias-mitigation controls recommended."

        )

        messages.append(

            {

                "level": "error",

                "message": (

                    "Fairness risk exceeds the Version C threshold. "

                    "Conduct a fairness audit and mitigation assessment."

                ),

            }

        )

    # Stability

    if stability < 0.50:

        actions.append(

            "Deployment should be held pending stability review."

        )

        messages.append(

            {

                "level": "error",

                "message": (

                    "System stability is below the deployment threshold."

                ),

            }

        )

    # Risk score

    if risk_score > 0.60:

        actions.append(

            "Human review and governance escalation required."

        )

        messages.append(

            {

                "level": "error",

                "message": (

                    "Overall AI governance risk exceeds the escalation "

                    "threshold. Human review is required."

                ),

            }

        )

    if not actions:

        actions.append(

            "System is currently stable under the configured governance thresholds."

        )

        messages.append(

            {

                "level": "success",

                "message": (

                    "No governance intervention is currently required."

                ),

            }

        )

    if risk_score > 0.60:

        primary_action = "Human review / escalation"

    elif stability < 0.50:

        primary_action = "Hold deployment"

    elif fairness > 0.15:

        primary_action = "Fairness audit / mitigation"

    elif drift > 0.30:

        primary_action = "Retraining / data-quality controls"

    elif risk_class["classification"] == "High Risk":

        primary_action = "Continuous monitoring"

    else:

        primary_action = "Stable"

    return {

        "domain": domain,

        "jurisdiction": jurisdiction,

        "risk_class": risk_class,

        "metrics": {

            **metrics,

            "stability": stability,

            "risk_score": risk_score,

        },

        "actions": actions,

        "primary_action": primary_action,

        "messages": messages,

    }

def list_frameworks() -> List[Dict[str, Any]]:

    results = []

    for name, data in GOVERNANCE_FRAMEWORKS.items():

        results.append(

            {

                "name": name,

                "description": data["description"],

                "principles": data["principles"],

            }

        )

    return results
