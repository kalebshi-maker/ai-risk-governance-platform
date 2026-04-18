# JTYYLSPH — Enterprise AI Governance Platform

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![Build](https://img.shields.io/github/actions/workflow/status/hs5738-design/jtyylsph-ml-dashboard/ci.yml)
![Coverage](https://img.shields.io/codecov/c/github/hs5738-design/jtyylsph-ml-dashboard)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

Streamlit App:  
(https://ai-governance-platform-jtyylsph.streamlit.app/)

---

## 📂 Repository

GitHub:  (https://github.com/kalebshi-maker/ai-risk-governance-platform/tree/main)

---

## 🎯 Key Features

### Core ML
- AutoML model training (RandomForest, GradientBoosting, LogisticRegression)
- Classification & regression tasks
- Risk prediction and decision analytics
- Multi-domain dataset support (Finance, Healthcare, Sports, General)

### Governance
- SQL-based model registry with versioning
- Model approval workflow
- Model logging and reproducibility
- End-to-end ML pipeline management

### Monitoring
- Real-time drift detection (Wasserstein + KS test)
- Bias/fairness detection across sensitive features
- Alerts and metric logging

### Explainability
- SHAP global & local explanations
- LLM-based explainers for natural language model interpretations
- Feature importance & correlation analysis

### API & Deployment
- REST API with FastAPI
- Authentication support
- Docker-ready for production deployment
- Streamlit web dashboard

**Folder Overview:**  
- `automl/` - automated ML training engine  
- `explainability/` - SHAP and LLM model explanations  
- `monitoring/` - drift and bias monitoring utilities  
- `governance/` - model approval workflows and risk assessment  
- `registry/` - SQL-based production model registry  
- `dashboard/` - Streamlit front-end UI  

---

## 🧠 Architecture Overview

**Frontend:** Streamlit dashboard  
**Backend:** FastAPI service  
**Storage:** SQL model registry  
**Monitoring:** Drift + bias detection  
**Deployment:** Docker & Docker Compose 
JTYYLSPH/
│
├── api/                  # FastAPI REST API
├── automl/               # Automated ML engines
├── explainability/       # SHAP & LLM explainers
├── monitoring/           # Drift & bias monitoring
├── governance/           # Approval workflow & risk rating
├── registry/             # SQL model registry
├── dashboard/            # Streamlit UI
├── tests/                # Unit & integration tests
├── docker/               # Dockerfiles & docker-compose
├── scripts/              # Training & deployment scripts
├── models/               # Trained model storage
├── requirements.txt
├── README.md
├── LICENSE
└── .github/              # GitHub CI/CD workflows
---

## ⚙️ Installation

### Clone repository

```bash
git clone https://github.com/hs5738-design/jtyylsph-ml-dashboard.git
cd jtyylsph-ml-dashboard
Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Run locally
# Start API backend
uvicorn api.main:app --reload

# Start Streamlit dashboard
streamlit run dashboard/streamlit_app.py
Docker (Production-ready)
docker-compose up --build

🔬 Example Use Cases
Financial risk prediction and scoring


Healthcare classification & diagnosis prediction


Policy compliance & governance analytics


Supply chain & operational risk modeling


Experimental ML research and teaching

📊 Model Capabilities
Domain
Accuracy
F1 Score
Notes
Financial Risk
88%
0.87
LogisticRegression baseline
Synthetic Classification
90%+
0.91
AutoML-selected RandomForest
Healthcare
85%
0.84
GradientBoosting
General Multi-domain
88%+
0.88
Ensemble models


📈 Evaluation Metrics
Accuracy, Precision, Recall, F1 Score


Confusion Matrix


ROC Curve & AUC


Feature importance & correlation heatmaps


SHAP global and local explanations

📌 Engineering Highlights
Modular, extensible architecture


Explainable ML & LLM design principles


Production-oriented logging & versioning


Multi-domain adaptability


Interactive visualization dashboard


Reproducible end-to-end ML training pipeline


SQL-based persistent model registry


Model approval workflow for enterprise compliance

🤝 Contributing
Contributions are welcome! Please follow the standard GitHub workflow:
Fork repository


Create a feature branch (git checkout -b feature/my-feature)


Commit changes (git commit -am 'Add feature')


Push branch (git push origin feature/my-feature)


Open a Pull Request

📜 License
MIT License. See LICENSE file for details.

👨‍💻 Author
Kaleb Carter Shi
---


