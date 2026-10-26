# Predictive Modeling and Explainable Risk Scoring for Bank Customer Churn

**Author:** Antigravity AI & Senior Machine Learning Research Team  
**Institution:** Advanced Financial Analytics Laboratory  
**Date:** October 2026  
**Document Classification:** Academic Research & Implementation Report  

---

## 1. Abstract
Customer attrition (churn) represents a primary risk to the revenue stability and portfolio longevity of retail commercial banks. This study implements an end-to-end, enterprise-grade machine learning architecture designed to estimate individual customer churn probability and generate actionable operational risk scores. Utilizing a canonical corpus of 10,000 retail banking accounts across France, Germany, and Spain, we systematically develop, cross-validate, and evaluate five diverse classification algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Extreme Gradient Boosting (XGBoost). Our end-to-end pipeline features rigorous data hygiene (excluding customer identifiers to prevent target leakage), domain-driven feature engineering (incorporating debt-to-salary ratios, engagement-product interactions, and product tenure density), probability calibration via Brier score minimization, and Explainable AI (SHAP / TreeExplainer). The final tuned Gradient Boosting pipeline achieves a 5-fold cross-validated ROC-AUC of 0.8636 and an isolated test set ROC-AUC of 0.8690, PR-AUC of 0.7184, Accuracy of 86.85%, and Brier score of 0.0994. An optimal F1 decision threshold (0.28), calibrated strictly on training cross-validation, increases minority churn recall to 67.32% with an F1-score of 64.02%. The model is deployed as a responsive, real-time web application with interactive what-if simulation, global explainability, and customer analytics.

---

## 2. Introduction
Retail banking markets operate under competitive pressure characterized by diminished customer switching friction, digital neobank alternatives, and escalating customer acquisition costs (CAC). Empirical banking studies demonstrate that acquiring a replacement depositor costs 5 to 7 times more than retaining an existing client. Consequently, early identification of churn signals empowers proactive relationship interventions, tailored loyalty structures, and deposit preservation.

---

## 3. Problem Statement
The objective is binary classification of bank customer churn (`Exited` ∈ {0, 1}):
- `Exited = 0`: Retained account holder.
- `Exited = 1`: Account holder closed relationship / terminated deposits.

The challenge is exacerbated by moderate class imbalance (20.37% churn rate), multi-attribute interactions between tenure and product adoption, and the institutional requirement that model predictions be explainable, calibrated, and auditable under banking compliance guidelines.

---

## 4. Objectives
1. Perform rigorous data auditing and exploratory data analysis (EDA) on real historical records.
2. Formulate leakage-free feature engineering capturing balance-salary alignment, product velocity, and engagement interactions.
3. Train, benchmark, and cross-validate five machine learning families under identical 5-fold stratified validation splits.
4. Tune hyperparameters and calibrate classification thresholds for optimal F1 and recall performance.
5. Provide localized and global feature attribution via Explainable AI (XAI).
6. Deploy a full-stack, responsive web application with what-if scenario testing and portfolio analytics.

---

## 5. Dataset Description
The dataset contains 10,000 real customer records spanning three European markets.

| Variable | Type | Description |
|---|---|---|
| `RowNumber` | Integer | Record row index (excluded from modeling) |
| `CustomerId` | Integer | Unique customer identifier (excluded from modeling) |
| `Surname` | Categorical | Customer family name (excluded from modeling) |
| `CreditScore` | Integer | Credit bureau score (Range: 350 – 850) |
| `Geography` | Categorical | Country of residence (France, Germany, Spain) |
| `Gender` | Categorical | Female, Male |
| `Age` | Integer | Customer age (Range: 18 – 92) |
| `Tenure` | Integer | Years of relationship with the financial institution (0 – 10) |
| `Balance` | Float | Deposit balance in account (€) |
| `NumOfProducts` | Integer | Number of active bank services utilized (1 – 4) |
| `HasCrCard` | Binary | 1 = Holds active credit card, 0 = No credit card |
| `IsActiveMember` | Binary | 1 = Actively engaged customer, 0 = Inactive member |
| `EstimatedSalary` | Float | Modeled annual income (€) |
| `Exited` | Binary | Target variable (0 = Retained, 1 = Churned) |

---

## 6. Data Audit & Quality Verification
- **Total Records:** 10,000 rows, 14 columns.
- **Completeness:** 0 missing values across all columns.
- **Uniqueness:** 0 duplicate rows; 10,000 unique `CustomerId` records.
- **Target Distribution:**
  - Class 0 (Retained): 7,963 customers (79.63%)
  - Class 1 (Churned): 2,037 customers (20.37%)
  - Imbalance Ratio: 3.91 : 1
- **Leakage Prevention:** `RowNumber`, `CustomerId`, and `Surname` are strictly quarantined and removed from all model inputs.

---

## 7. Exploratory Data Analysis (EDA)
Comprehensive analysis revealed distinctive behavioral drivers:
1. **Geography:** German clients churn at 32.44%, compared to 16.15% in France and 16.67% in Spain.
2. **Gender:** Female accounts demonstrate a 25.07% churn rate versus 16.46% for male accounts.
3. **Age Curve:** Churn increases monotonically between ages 40 and 60, peaking at 56.2% in the 51–60 cohort.
4. **Product Paradox:** Customers with 1 product churn at 27.71%; 2 products represent the most stable segment (7.58% churn); holding 3 products (82.71% churn) or 4 products (100.0% churn) exhibits severe churn signals.
5. **Membership Activity:** Inactive members churn at 26.85%, whereas active members churn at 14.27%.

---

## 8. Preprocessing
To prevent data leakage, all transformations are nested within a Scikit-Learn `ColumnTransformer` fitted solely on training partitions:
- **Numerical Features:** Median imputation followed by `StandardScaler` standard normal transformation.
- **Categorical Features:** Most frequent imputation followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

---

## 9. Feature Engineering
We formulated five domain-specific features:
1. **`BalanceSalaryRatio`**: `np.where(EstimatedSalary > 0, Balance / EstimatedSalary, 0.0)` clipped to [0, 100]. Represents liquid deposit concentration relative to annual earnings.
2. **`EngagementProductInteraction`**: `IsActiveMember × NumOfProducts`. Quantifies whether multi-product adoption is accompanied by active digital/branch engagement.
3. **`AgeTenureInteraction`**: `Age × Tenure`. Captures cumulative lifecycle tenure maturity.
4. **`ProductDensityIndicator`**: `NumOfProducts / (Tenure + 1.0)`. Measures product cross-sell velocity relative to account duration.
5. **`MultiProductRisk`**: Binary flag indicating `NumOfProducts >= 3`.

---

## 10. Machine Learning Methodology
Five candidate algorithms representing diverse inductive biases were developed:
1. **Logistic Regression (L2 Penalty):** Balanced class weights, maximum 1000 iterations.
2. **Decision Tree:** Maximum depth 6, balanced class weights, Gini criterion.
3. **Random Forest:** Ensemble of 150 bagged de-correlated trees, balanced subsampling.
4. **Gradient Boosting:** 150 sequential boosting estimators, deviance loss, learning rate 0.08.
5. **Extreme Gradient Boosting (XGBoost):** Scale positive weight 3.9, depth 4, learning rate 0.08.

---

## 11. Train/Test Strategy
The 10,000-record dataset was partitioned using a Stratified 80/20 train/test split:
- **Training Set:** 8,000 records (6,370 Retained, 1,630 Churned). Used exclusively for cross-validation and hyperparameter tuning.
- **Test Set:** 2,000 records (1,593 Retained, 407 Churned). Completely isolated and evaluated once at the conclusion of training.

---

## 12. Model Comparison & 5-Fold Cross-Validation
Cross-validation results across the 8,000 training samples (mean ± standard deviation):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8629 ± 0.007** | **0.7512 ± 0.021** | 0.4822 ± 0.019 | 0.5869 ± 0.016 | **0.8607 ± 0.009** | **0.6924 ± 0.015** |
| **XGBoost** | 0.8044 ± 0.009 | 0.5113 ± 0.015 | **0.7337 ± 0.024** | 0.6025 ± 0.014 | 0.8602 ± 0.008 | 0.6938 ± 0.014 |
| **Random Forest** | 0.8336 ± 0.006 | 0.5695 ± 0.013 | 0.6706 ± 0.018 | **0.6158 ± 0.011** | 0.8577 ± 0.010 | 0.6835 ± 0.016 |
| **Logistic Regression** | 0.7596 ± 0.008 | 0.4357 ± 0.011 | 0.7767 ± 0.022 | 0.5583 ± 0.012 | 0.8308 ± 0.008 | 0.6025 ± 0.017 |
| **Decision Tree** | 0.7546 ± 0.011 | 0.4302 ± 0.014 | 0.7932 ± 0.026 | 0.5583 ± 0.014 | 0.8231 ± 0.011 | 0.6073 ± 0.018 |

---

## 13. Hyperparameter Tuning
Hyperparameter tuning was executed via 5-Fold Stratified `GridSearchCV` on the training set for the top performing architecture (Gradient Boosting).
- Parameter space explored: `learning_rate` ∈ [0.05, 0.1], `max_depth` ∈ [3, 5], `n_estimators` ∈ [100, 180].
- Optimal configuration: `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 180}`.
- Refitted CV ROC-AUC: **0.8636**.

---

## 14. Evaluation Metrics Formulation
- **ROC-AUC (Receiver Operating Characteristic):** Area under TPR vs FPR curve across all operational thresholds.
- **PR-AUC (Precision-Recall Average Precision):** Emphasizes positive class ranking performance in imbalanced contexts.
- **Brier Score:** Mean squared discrepancy between predicted probabilities and binary outcomes:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (P_i - Y_i)^2$$

---

## 15. Final Test Results
Evaluated on the isolated 2,000-sample test set:

| Metric | Default Threshold (0.50) | Calibrated Optimal Threshold (0.28) |
|---|---|---|
| **Accuracy** | 86.85% | 84.60% |
| **Precision** | 78.80% | 61.02% |
| **Recall** | 48.40% | **67.32%** |
| **F1-Score** | 59.97% | **64.02%** |
| **ROC-AUC** | **0.8690** | **0.8690** |
| **PR-AUC** | **0.7184** | **0.7184** |
| **Brier Score** | **0.0994** | **0.0994** |

### Confusion Matrix on Test Set
- At Threshold = 0.50: TN = 1540, FP = 53, FN = 210, TP = 197.
- At Threshold = 0.28: TN = 1418, FP = 175, FN = 133, TP = 274 (Identifies +77 additional churning accounts).

---

## 16. Probability Calibration
Probability calibration was assessed via 10-bin calibration curves and Brier loss. The model's Brier score of **0.0994** (substantially below the naive baseline of 0.162) demonstrates that predicted probabilities closely match empirical attrition frequencies.

---

## 17. Risk Scoring Methodology
Operational Risk Scores translate model probabilities into a standardized 100-point index:
$$\text{Risk Score} = \text{Churn Probability} \times 100$$

Configured Enterprise Risk Tiers:
- **Low Risk (0.0 – 24.9% Probability):** Standard automated marketing and quarterly check-ins.
- **Medium Risk (25.0 – 49.9% Probability):** Relationship manager outreach, service fee waivers, loyalty enhancements.
- **High Risk (≥ 50.0% Probability):** Immediate retention specialist escalation and executive relationship reviews.

---

## 18. Explainability & Model Inspection (XAI)
Global and local explanations are computed via TreeSHAP and Partial Dependence Plots (PDP):

### 18.1 SHAP Feature Attribution
1. **Age:** Strongest global predictor (+0.24 mean importance).
2. **NumOfProducts & MultiProductRisk:** Secondary drivers; accounts with 3+ products dramatically raise churn probability.
3. **Geography_Germany:** Geographic origin adds substantial positive attribution toward churn risk.
4. **IsActiveMember:** Active engagement serves as the strongest protective (negative churn) factor.

### 18.2 Partial Dependence Analysis (PDP)
Partial Dependence Plots (generated via scikit-learn's `PartialDependenceDisplay` and saved under `reports/figures/pdp_analysis.png`) evaluate the marginal effect of individual features on the model's predicted churn probability, holding all other features at their empirical joint distribution:
- **Age:** Exhibits a pronounced non-linear trajectory. For customers between ages 18 and 38, marginal churn probability remains low and stable (~10.5%–11.0%). Beginning around age 40, predicted risk accelerates sharply, peaking between ages 50 and 60 (>45%), before plateauing in retirement.
- **NumOfProducts:** Demonstrates a distinct non-linear U-curve. A single-product holding reflects a baseline risk of ~26.6%. Holding two products represents the optimal retention point, lowering marginal churn probability to ~11.3%. However, holding 3 or 4 products triggers a dramatic surge in predicted risk to ~76.0% and ~86.6%, respectively (reflecting the empirical multi-product churn paradox).
- **Balance:** Displays a gradual upward gradient. Customers with zero balances average a marginal churn probability of ~20.6%, whereas accounts maintaining balances between €100,000 and €200,000 rise toward ~25%–28%.
- **IsActiveMember:** Displays a discrete downward shift. Active engagement reduces the average marginal predicted churn probability from ~26.4% (inactive) to ~16.1% (active), representing an average marginal risk reduction of ~10.3 percentage points.
- *Methodological Note on Causality:* These partial dependence profiles reflect marginal associations learned by the ensemble trees from historical observations; they must not be interpreted as counterfactual causal proofs or absolute guarantees that artificially altering an attribute will unilaterally alter customer behavior.

---

## 19. What-If Simulation
The application incorporates a comparative scenario engine allowing relationship managers to observe score deltas when customer attributes are hypothetically modified (e.g. converting an inactive customer to active, or reducing multi-product friction). The system computes exact probability differences in percentage points.

---

## 20. Web Application Architecture
Built without heavy front-end frameworks:
- **Backend:** Python 3.10 and Flask RESTful API.
- **Frontend:** Responsive HTML5, CSS3 with enterprise dark theme, and Vanilla JavaScript.
- **Visualization:** Chart.js 4.4 interactive charts for ROC curves, PR curves, calibration curves, and categorical distributions.

---

## 21. Results and Discussion
The tuned Gradient Boosting pipeline delivers balanced discrimination and calibrated uncertainty estimates. By operating at the tuned threshold of 0.28, the bank captures over 67.3% of impending churn events while maintaining an acceptable false positive rate (11.0%).

---

## 22. Limitations
1. Historical dataset represents a snapshot without longitudinal monthly transactional time-series data.
2. Market macro-factors (interest rate shifts, regional inflation) are unrepresented.
3. Feature interactions reflect statistical patterns rather than counterfactual causal guarantees.

---

## 23. Responsible AI & Institutional Compliance
1. **Decision Support Only:** Predictions must never automatically trigger punitive terms, fee surcharges, or account restrictions.
2. **Fair Lending & Non-Discrimination:** Gender and demographic variables must not be leveraged in consequential credit decisions.
3. **Model Monitoring:** Model weights require quarterly drift auditing against covariate shifts.

---

## 24. Future Work
1. Incorporate time-to-event survival analysis (Cox proportional hazards).
2. Integrate transactional log embeddings using sequential neural networks.
3. Implement automated A/B testing feedback loops for retention offer success.

---

## 25. Conclusion
This project successfully establishes a full-stack, enterprise-grade predictive modeling and explainable risk scoring platform for bank customer churn. Through disciplined feature engineering, leakage prevention, multi-model benchmarking, and threshold calibration, the system equips financial institutions with an accurate, auditable, and actionable tool for deposit retention.

---

## 26. References
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM KDD.
4. Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. ICML.
