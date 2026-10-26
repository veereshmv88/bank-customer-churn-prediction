# Predictive Modeling and Risk Scoring for Bank Customer Churn

An enterprise-grade full-stack machine learning web application and analytics platform designed to predict retail bank customer attrition, calculate operational risk scores (0–100), explain prediction drivers using Explainable AI (SHAP), and provide real-time scenario simulation.

---

## 🏛️ Project Overview

- **Academic Title:** Predictive Modeling and Risk Scoring for Bank Customer Churn
- **Target Variable:** `Exited` (1 = Customer Churned, 0 = Customer Retained)
- **Frontend:** Responsive HTML5, CSS3 (Enterprise Navy Theme), Vanilla JavaScript, Chart.js 4.4
- **Backend:** Python 3.10 and Flask RESTful API
- **Machine Learning:** Scikit-Learn, XGBoost, SHAP, Pandas, NumPy
- **Persistence:** Joblib
- **Testing:** Pytest (30 Automated Tests, 100% Pass Rate)

---

## 📁 Repository Architecture

```
bank-customer-churn-prediction/
│
├── app.py                      # Production Flask application & REST API
├── streamlit_app.py            # Streamlit evaluation interface
├── train.py                    # End-to-end ML training & persistence pipeline
├── evaluate.py                 # Standalone model test set evaluation
├── run_pipeline.py             # Master orchestrator script
├── requirements.txt            # Project dependencies
├── README.md                   # Complete system documentation
├── .gitignore                  # Git exclusion rules
│
├── data/
│   ├── raw/
│   │   └── European_Bank.csv   # Canonical 10,000 customer dataset
│   └── processed/
│       └── scored_customers.csv# Batch customer risk scored dataset
│
├── models/
│   ├── best_model.joblib       # Persisted scikit-learn Pipeline
│   ├── model_metadata.json     # Hyperparameters, features, and selection rationale
│   ├── model_metrics.json      # CV benchmarks, test metrics, and curve points
│   └── feature_importance.json # Sorted global feature importances
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Safe dataset discovery & loading
│   ├── data_audit.py           # Data hygiene and quality verification
│   ├── feature_engineering.py  # Domain-specific transformer
│   ├── preprocessing.py        # ColumnTransformer & stratified splitting
│   ├── model_training.py       # Candidate models & hyperparameter tuning
│   ├── model_evaluation.py     # Metrics, ROC/PR curves, and calibration
│   ├── risk_scoring.py         # 100-point risk score engine & tier mapping
│   ├── prediction.py           # Unified inference service & validation
│   ├── explainability.py       # SHAP and feature attribution
│   └── generate_pdp.py         # Partial Dependence Plot generator
│
├── templates/
│   └── index.html              # Standalone single page web application
│
├── static/
│   ├── css/
│   │   └── style.css           # Enterprise responsive styling
│   └── js/
│       └── app.js              # Vanilla JavaScript UI logic & API handlers
│
├── reports/
│   ├── figures/                # EDA & Partial Dependence figures
│   ├── tables/                 # Audit & comparison tables (JSON & CSV)
│   ├── research_report.md      # Comprehensive 26-section academic report
│   ├── executive_summary.md    # C-suite executive briefing
│   ├── research_report.pdf     # Publication-ready PDF report
│   └── final_verification.md   # Requirements checklist & verification
│
└── tests/
    ├── __init__.py
    ├── test_data.py            # Dataset integrity & schema tests
    ├── test_model.py           # Pipeline steps & probability bounds tests
    ├── test_prediction.py      # Feature engineering & risk score tests
    ├── test_api.py             # Flask REST API integration tests
    ├── test_explainability.py  # TreeSHAP additivity & direction tests
    └── test_streamlit_and_pdp.py # Streamlit AppTest & PDP tests
```

---

## ⚡ Quick Start & Execution

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Train Models & Generate Artifacts
Executes data auditing, EDA figure generation, 5-fold cross-validation of 5 models, hyperparameter tuning, and artifact serialization:
```bash
python train.py
```

### 3. Evaluate Model on Held-Out Test Data
Evaluates the trained pipeline on the untouched 2,000-sample test set:
```bash
python evaluate.py
```

### 4. Run Complete Orchestrated Pipeline
Runs training, evaluation, PDP generation, and test suite sequentially:
```bash
python run_pipeline.py
```

### 5. Generate Partial Dependence Plots (PDP)
Generates marginal feature effect plots for Age, NumOfProducts, Balance, and IsActiveMember:
```bash
python src/generate_pdp.py
```

### 6. Launch the Application

#### Option A: Full-Stack Flask Platform (Production UI)
```bash
python app.py
```
Open your browser to: **`http://127.0.0.1:5000`**

#### Option B: Streamlit Evaluation Interface (Internship Benchmark)
```bash
streamlit run streamlit_app.py
```
Open your browser to the local Streamlit port (typically **`http://localhost:8501`**).

### 7. Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## 📊 Model Performance Benchmarks

### 5-Fold Stratified Cross-Validation (8,000 Training Samples)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting (Selected)** | **86.29%** | **75.12%** | 48.22% | 0.5869 | **0.8607** | **0.6924** |
| **XGBoost** | 80.44% | 51.13% | **73.37%** | 0.6025 | 0.8602 | 0.6938 |
| **Random Forest** | 83.36% | 56.95% | 67.06% | **0.6158** | 0.8577 | 0.6835 |
| **Logistic Regression** | 75.96% | 43.57% | 77.67% | 0.5583 | 0.8308 | 0.6025 |
| **Decision Tree** | 75.46% | 43.02% | 79.32% | 0.5583 | 0.8231 | 0.6073 |

### Final Test Set Evaluation (2,000 Isolated Records)
- **ROC-AUC:** `0.8690`
- **PR-AUC:** `0.7184`
- **Accuracy (Threshold 0.50):** `86.85%` (Precision: 78.80%, Recall: 48.40%, F1: 0.5997)
- **Calibrated F1 Threshold (0.28):** **Recall: 67.32%** (Captures +77 additional churners, F1: 0.6402)
- **Brier Score (Probability Calibration):** `0.0994`

---

## 🎯 Operational Risk Scoring Framework

$$\text{Risk Score} = \text{Churn Probability} \times 100$$

- **Low Risk (0.0 – 24.9% Probability):** Score < 25. Standard retention touchpoints.
- **Medium Risk (25.0 – 49.9% Probability):** Score 25 – 49.9. Relationship outreach.
- **High Risk (≥ 50.0% Probability):** Score ≥ 50. Immediate executive intervention.

---

## 🌐 Flask REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the Single Page Application UI |
| `GET` | `/api/health` | Service health status and loaded model information |
| `GET` | `/api/metadata` | Model hyperparameters, risk tiers, and feature names |
| `GET` | `/api/metrics` | Model comparison table, ROC/PR curves, confusion matrix |
| `GET` | `/api/analytics` | Portfolio statistics and category distributions |
| `POST` | `/api/predict` | Predict individual churn probability & factor attribution |
| `POST` | `/api/what-if` | Compare current vs hypothetical profile scenarios |
| `GET` | `/api/explain` | Global feature importances from trained model |
| `GET` | `/api/customers` | Sample customer profiles for testing presets |

---

## 📑 Research Reports & Documentation

- 📄 **Academic Research Paper:** [reports/research_report.md](reports/research_report.md) (26 comprehensive sections covering EDA, methodology, benchmarks, and XAI)
- 📑 **Publication-Ready PDF:** [reports/research_report.pdf](reports/research_report.pdf) (Formally compiled academic PDF report)
- 🏛️ **Executive Summary:** [reports/executive_summary.md](reports/executive_summary.md) (Strategic briefing for bank executives, CROs, and governance committees)
- ✅ **Final Compliance Checklist:** [reports/final_verification.md](reports/final_verification.md) (Full audit verification against requirements)

---

## ⚖️ Responsible AI Notice
Predictions are statistical estimations derived from historical customer patterns. This platform is designed as an advisory decision-support system. It must not be used to automatically deny services or enforce punitive pricing without human institutional review.
