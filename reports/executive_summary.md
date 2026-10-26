# Executive Summary: Bank Customer Churn Prediction & Risk Scoring Platform

**Prepared for:** Bank Executive Committee, Retail Banking Division, Chief Risk Officer  
**Platform Release:** Version 1.0.0 (Production Verified)  
**Date:** October 2026  

---

### Executive Overview
Customer retention is one of the highest leverage opportunities in modern retail banking. Losing an existing depositor requires significant marketing capital to replace, while directly diminishing net interest margins and cross-sell momentum. 

This platform delivers an enterprise-grade artificial intelligence solution that identifies at-risk depositors before they depart, scores each customer on a transparent 0–100 operational risk scale, and diagnoses the specific factors driving churn risk.

---

### Key Financial & Behavioral Findings
Across our audit of 10,000 retail customer relationships:
1. **The Core Attrition Rate:** The historical churn rate across the customer base is **20.37%** (2,037 accounts lost).
2. **Regional Disparity:** Customers in Germany experience more than double the churn rate (**32.4%**) compared to France (**16.1%**) and Spain (**16.7%**), highlighting the need for targeted regional service reviews.
3. **The Multi-Product Paradox:** While 2-product customers represent the most loyal cohort (only **7.6%** churn), accounts holding 3 or 4 products exhibit catastrophic attrition (**82.7%** and **100.0%**), indicating potential product bundling friction, unexpected account maintenance fees, or poorly integrated cross-selling.
4. **Age Cohort Vulnerability:** Churn probability accelerates dramatically for depositors aged 45 to 60 (surpassing **56%**), making pre-retirement retention strategies a high priority.

---

### Verified Model Performance
Our research team benchmarked five machine learning architectures (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost) through 5-fold cross-validation. 

The finalized **Gradient Boosting Pipeline** was evaluated on an isolated held-out test set of 2,000 untouched accounts:
- **ROC-AUC (Discriminant Power):** **0.8690** (exceptional class separation)
- **PR-AUC (Precision-Recall Performance):** **0.7184**
- **Overall Accuracy:** **86.85%**
- **Minority Churn Capture (Recall at Calibrated Threshold 0.28):** **67.32%** (capturing over two-thirds of all departing accounts with F1-score 64.02%)
- **Probability Calibration (Brier Score):** **0.0994** (indicating high confidence in predicted percentages)

---

### Operational Risk Scoring Framework
To convert raw statistical probabilities into intuitive operational workflows for branch managers and relationship teams, the platform translates model output into a 100-point index:

$$\text{Risk Score} = \text{Churn Probability} \times 100$$

| Operational Tier | Score Band | Portfolio % | Recommended Action |
|---|---|---|---|
| **Low Risk** | 0 – 24.9 | 69.8% | Standard digital engagement, regular automated touchpoints. |
| **Medium Risk** | 25 – 49.9 | 18.1% | Proactive relationship outreach, service fee review, survey. |
| **High Risk** | 50 – 100 | 12.1% | Senior manager intervention, personalized retention package. |

---

### Platform Capabilities
1. **Executive Dashboard:** Live KPIs, portfolio churn distribution, and regional breakdown.
2. **Real-Time Customer Risk Calculator:** Instantaneous risk scoring with top positive and negative driver attribution.
3. **Portfolio Risk Analytics:** Full distribution breakdown across 10,000 accounts with dynamic filtering.
4. **Model Performance Hub:** Complete transparency into cross-validation benchmarks, ROC/PR curves, and confusion matrices.
5. **Explainability (XAI):** Transparent feature rankings powered by SHAP (TreeExplainer).
6. **What-If Simulator:** Interactive sandbox allowing managers to simulate the retention impact of fee waivers, active engagement, and product consolidation.

---

### Governance & Responsible Use Guidelines
- **Decision Support:** The platform is engineered to support, not replace, human institutional review.
- **Fair Treatment:** In compliance with fair lending and banking guidelines, predictions must never be used to enforce punitive terms or deny services to customers.
- **Auditable & Reproducible:** Zero data leakage, version-controlled artifacts, and full automated test suite coverage (100% pytest pass rate).
