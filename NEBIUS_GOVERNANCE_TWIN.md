# Aurexis Serverless Governance Twin

Aurexis now has two modes:

1. **Human oversight dashboard** - `streamlit run app.py`
2. **Autonomous governance infrastructure** - FastAPI endpoint + Nebius Serverless Jobs

The serverless twin is a digital twin of production AI behavior. It receives
production outputs, evaluates drift/fairness/stability, forecasts governance
degradation, recommends interventions, emits audit logs, and generates
regulatory evidence reports.

## Architecture

```text
Production AI System
        ↓
Nebius Endpoint (/evaluate)
        ↓
Aurexis Governance Twin
        ↓
Nebius Serverless Jobs
        ↓
Scheduled Evaluation
Drift Simulation
Batch Risk Assessment
Compliance Reports
        ↓
Audit Logs + PDF Evidence
        ↓
Human Oversight Dashboard
```

## Local Reproducibility

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
python3 -m streamlit run app.py
```

Run the endpoint:

```bash
uvicorn endpoint:app --host 0.0.0.0 --port 8000
```

Evaluate a model:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  --data @examples/evaluate_request.json
```

Expected response:

```json
{
  "model_name": "credit_approval_model",
  "jurisdiction": "EU AI Act",
  "drift": 0.21,
  "fairness": 0.05,
  "stability": 0.87,
  "risk_score": 0.18,
  "intervention": "stable",
  "compliance": "pass_with_monitoring"
}
```

Run the autonomous jobs locally:

```bash
python jobs/governance_job.py
python jobs/drift_simulation_job.py
python jobs/batch_risk_job.py
python jobs/compliance_report_job.py
```

Artifacts are written to:

```bash
/tmp/aurexis_serverless_twin
```

Override with:

```bash
export AUREXIS_ARTIFACT_DIR=/path/to/artifacts
```

## Nebius Serverless Jobs

Use `nebius/aurexis_jobs.yaml` as the job manifest blueprint. It defines:

- `aurexis-scheduled-governance-evaluation`
- `aurexis-drift-simulation-forecast`
- `aurexis-batch-risk-assessment`
- `aurexis-compliance-report-generator`

Each job is represented as a command:

```bash
python jobs/governance_job.py
python jobs/drift_simulation_job.py
python jobs/batch_risk_job.py
python jobs/compliance_report_job.py
```

Map those commands into Nebius Serverless Jobs with your container image,
project, service account, and schedule.

## Nebius Endpoint

The endpoint target is:

```text
endpoint:app
```

Container default command:

```bash
uvicorn endpoint:app --host 0.0.0.0 --port 8000
```

Public API:

- `GET /health`
- `POST /evaluate`

## Autonomous Intervention

The governance twin returns interventions:

- `stable`
- `retrain`
- `debias`
- `stability_review`
- `suspend_and_human_review`

These can be wired to downstream Nebius jobs:

```python
if intervention == "retrain":
    launch_retraining_job()
elif intervention == "debias":
    launch_bias_mitigation_job()
```

The current implementation records the intervention and produces regulatory
evidence. Production deployments can connect these actions to model training or
bias-mitigation workflows.

