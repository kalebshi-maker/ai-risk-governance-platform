# AGENTS.md

## Cursor Cloud specific instructions

This repo is a single Streamlit application (no backend service or database required to run).

### Services
- **`app.py`** — primary "AUREXIS SYSTEMS" AI governance dashboard. Loads synthetic data by default (no upload/API key needed) and runs model training, drift/fairness/risk scoring.
- **`demo_app.py`** — small standalone "Break This AI Model" demo. Run only one app per port.

### Running
- Always use the venv: `. .venv/bin/activate`.
- Run: `streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0`
- Health check: `curl http://localhost:8501/_stcore/health` returns `ok`.

### Notes / gotchas
- `requirements.txt` lists `fastapi`/`uvicorn`/`sqlalchemy`/`psycopg2-binary`, but `app.py` is Streamlit-only and does not start a FastAPI server or connect to a DB at runtime.
- `shap` and `fairlearn` are optional; they are not in `requirements.txt` and the app degrades gracefully ("Dependency Status" expander shows what's active).
- OpenAI features are optional and gated behind `OPENAI_API_KEY` (env var or `st.secrets`); a local advisor is used when absent.
- There are no automated tests; validate with `python -m py_compile app.py demo_app.py` and a manual run.
- Lint: no linter is configured in the repo.
