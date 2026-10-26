# Final Verification Report: Bank Customer Churn Prediction & Risk Scoring

**Date:** October 2026  
**Status:** ALL PHASES VERIFIED & OPERATIONAL  

---

## 1. Machine Learning Requirements Checklist

- [x] **Actual dataset identified:** Canonical Kaggle Bank Churn Modeling dataset (10,000 rows, 14 columns).
- [x] **Data audit completed:** Zero missing values, zero duplicated CustomerIds, audit report saved to `reports/tables/data_audit.json` and `reports/tables/data_audit.md`.
- [x] **EDA performed:** 5 high-resolution figures generated in `reports/figures/` (churn vs retention donut, geography, age, products, activity).
- [x] **Required feature engineering implemented:**
  - `BalanceSalaryRatio` (safely handled zero salaries)
  - `EngagementProductInteraction` (`IsActiveMember × NumOfProducts`)
  - `AgeTenureInteraction` (`Age × Tenure`)
  - `ProductDensityIndicator` (`NumOfProducts / (Tenure + 1.0)`)
  - `MultiProductRisk` (`NumOfProducts >= 3`)
- [x] **Leakage-safe train/test split:** Stratified 80/20 split with fixed random seed (42). Test set (2,000 samples) strictly quarantined.
- [x] **Multiple models trained:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- [x] **Cross-validation performed:** 5-Fold Stratified CV on training data. Comparison table saved to JSON & CSV.
- [x] **Hyperparameter tuning completed:** 5-Fold GridSearchCV on Gradient Boosting (`{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 180}`).
- [x] **Final test metrics calculated:** Evaluated once on untouched test set: ROC-AUC = 0.8690, PR-AUC = 0.7184, Accuracy = 86.85%, Brier Score = 0.0994.
- [x] **Final model saved:** Serialized as complete scikit-learn Pipeline to `models/best_model.joblib`.
- [x] **Probability prediction verified:** Range strictly in [0.0, 1.0].
- [x] **Classification threshold documented:** Default 0.50, optimal F1 threshold from 5-fold CV 0.28 (recalls 67.32% of churners with F1 64.02%).
- [x] **Risk categories documented:** Low Risk (<25%), Medium Risk (25-49.9%), High Risk (>=50%).
- [x] **Explainability implemented:** Global feature importance chart, SHAP TreeExplainer integration, individual customer attribution.

---

## 2. Website & Full-Stack Requirements Checklist

- [x] **HTML, CSS, and Vanilla JavaScript created:** Standalone website with enterprise dark navy design (`#0B1220`, `#172235`, `#1C2940`, `#3B82F6`, `#38BDF8`). No React/Streamlit/Next.js.
- [x] **Flask backend operational:** Running on `app.py` with singleton model loading at startup.
- [x] **Executive dashboard operational:** KPI cards, historical vs predicted risk, donut chart, geography breakdown.
- [x] **Customer risk calculator operational:** Two-panel layout, preset buttons, real-time prediction API call, probability %, risk score / 100, risk badge, top factors.
- [x] **Risk analytics operational:** Portfolio risk tiers, probability distribution histogram across 10,000 customers.
- [x] **Model performance page operational:** Comparison table, Confusion Matrix, interactive ROC curve, PR curve, Calibration curve.
- [x] **Explainability page operational:** Global importance horizontal bar chart, detailed interpretation.
- [x] **What-if simulator operational:** Side-by-side current vs scenario profiles, probability delta in % points, comparative bar chart.
- [x] **Customer analytics operational:** Interactive empirical charts for products, credit card, gender.
- [x] **Methodology page operational:** Full documentation of architecture, leakage prevention, metrics, calibration, and governance.
- [x] **API integration operational:** RESTful endpoints with error handling and validation.
- [x] **Responsive layout checked:** Mobile and desktop responsive layout.
- [x] **Error handling implemented:** Clean JSON error messages, no leaked internal tracebacks.

---

## 3. Deliverables Checklist

- [x] `README.md` created with complete setup, architecture, and command guide.
- [x] `requirements.txt` with locked dependencies.
- [x] `train.py` reproducible training script.
- [x] `evaluate.py` standalone test evaluation script.
- [x] `run_pipeline.py` end-to-end orchestrator script.
- [x] `reports/research_report.md` (26 comprehensive academic sections).
- [x] `reports/executive_summary.md`.
- [x] `reports/research_report.pdf` generated via ReportLab.
- [x] Automated test suite: 21 / 21 pytest tests passed (100% pass rate).
