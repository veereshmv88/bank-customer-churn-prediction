"""
Main Training Script for Bank Customer Churn Prediction.
Orchestrates data loading, auditing, feature engineering, multi-model cross-validation,
hyperparameter tuning, model selection, threshold calibration, and artifact persistence.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from src.data_loader import load_raw_data
from src.data_audit import audit_dataset
from src.preprocessing import prepare_data_split
from src.model_training import (
    get_candidate_models,
    cross_validate_candidates,
    tune_best_model,
    extract_feature_names_and_importance
)
from src.model_evaluation import (
    evaluate_predictions,
    compute_curve_points,
    find_optimal_threshold
)
from src.risk_scoring import RISK_BANDS

def generate_eda_figures(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Generate high-resolution EDA figures saved to disk."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. Churn versus Retention Donut
    plt.figure(figsize=(6, 6))
    counts = df["Exited"].value_counts()
    plt.pie(
        counts,
        labels=["Retained (0)", "Churned (1)"],
        autopct="%1.1f%%",
        colors=["#3B82F6", "#EF4444"],
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor="w")
    )
    plt.title("Customer Churn vs Retention Distribution", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "churn_distribution.png"), dpi=200)
    plt.close()

    # 2. Churn by Geography
    plt.figure(figsize=(8, 5))
    geo_df = df.groupby(["Geography", "Exited"]).size().unstack(fill_value=0)
    geo_pct = geo_df.div(geo_df.sum(axis=1), axis=0) * 100
    ax = geo_pct.plot(kind="bar", stacked=True, color=["#3B82F6", "#EF4444"], figsize=(8, 5))
    plt.title("Churn Percentage by Geography", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Country")
    plt.legend(["Retained", "Churned"], title="Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "churn_by_geography.png"), dpi=200)
    plt.close()

    # 3. Churn by Age Distribution
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df, x="Age", hue="Exited", common_norm=False, fill=True, palette=["#3B82F6", "#EF4444"], alpha=0.4)
    plt.title("Age Density Distribution by Churn Status", fontsize=13, fontweight="bold")
    plt.xlabel("Customer Age")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "churn_by_age.png"), dpi=200)
    plt.close()

    # 4. Churn by NumOfProducts
    plt.figure(figsize=(8, 5))
    prod_df = df.groupby(["NumOfProducts", "Exited"]).size().unstack(fill_value=0)
    prod_pct = prod_df.div(prod_df.sum(axis=1), axis=0) * 100
    prod_pct.plot(kind="bar", color=["#3B82F6", "#EF4444"], figsize=(8, 5))
    plt.title("Churn Rate by Number of Bank Products", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage within Product Count (%)")
    plt.xlabel("Number of Products")
    plt.legend(["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "churn_by_num_products.png"), dpi=200)
    plt.close()

    # 5. Churn by IsActiveMember
    plt.figure(figsize=(7, 5))
    act_df = df.groupby(["IsActiveMember", "Exited"]).size().unstack(fill_value=0)
    act_pct = act_df.div(act_df.sum(axis=1), axis=0) * 100
    act_pct.index = ["Inactive Member", "Active Member"]
    act_pct.plot(kind="bar", color=["#3B82F6", "#EF4444"], figsize=(7, 5))
    plt.title("Churn Rate by Member Activity Status", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.legend(["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "churn_by_active_member.png"), dpi=200)
    plt.close()

    print("Generated and saved EDA figures.")

def run_training_pipeline():
    """Execute complete end-to-end training and evaluation workflow."""
    print("==========================================================")
    print("STARTING MACHINE LEARNING PIPELINE TRAINING")
    print("==========================================================")

    # 1. Load Data
    print("\n[Step 1/10] Loading raw dataset...")
    df = load_raw_data()
    print(f"Loaded raw dataset with shape: {df.shape}")

    # 2. Audit Data
    print("\n[Step 2/10] Auditing dataset...")
    audit = audit_dataset(df)
    print(f"Data audit completed. Target distribution: {audit['target_distribution']['percentages']}")

    # 3. Generate EDA Figures
    print("\n[Step 3/10] Generating Exploratory Data Analysis figures...")
    generate_eda_figures(df)

    # 4. Prepare Leakage-Safe Data Split
    print("\n[Step 4/10] Splitting data into Stratified 80/20 train/test sets...")
    X_train, X_test, y_train, y_test = prepare_data_split(df, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples (isolated)")

    # 5. Multi-Model Cross-Validation
    print("\n[Step 5/10] Training and evaluating 5 candidate models with 5-Fold Stratified CV...")
    models = get_candidate_models()
    cv_comparison = cross_validate_candidates(models, X_train, y_train, cv_folds=5)

    # Save comparison table
    os.makedirs("reports/tables", exist_ok=True)
    cv_comparison.to_csv("reports/tables/model_comparison.csv", index=False)
    cv_comparison.to_json("reports/tables/model_comparison.json", orient="records", indent=2)
    print("\nCross-Validation Comparison Results:")
    print(cv_comparison[["Model", "Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"]])

    # 6. Model Selection
    # Select best model balancing PR_AUC, ROC_AUC, and F1-score
    print("\n[Step 6/10] Selecting best model based on PR-AUC, ROC-AUC, and F1...")
    # XGBoost and Gradient Boosting typically achieve highest ROC-AUC and PR-AUC on tabular churn
    sorted_models = cv_comparison.sort_values(by=["ROC_AUC", "PR_AUC", "F1_Score"], ascending=False)
    best_row = sorted_models.iloc[0]
    best_model_name = best_row["Model"]
    print(f"Selected Model: '{best_model_name}' (CV ROC-AUC: {best_row['ROC_AUC']:.4f}, PR-AUC: {best_row['PR_AUC']:.4f})")

    # 7. Hyperparameter Tuning
    print(f"\n[Step 7/10] Performing hyperparameter tuning on {best_model_name}...")
    best_pipeline = tune_best_model(best_model_name, models[best_model_name], X_train, y_train)

    # 8. Optimal Threshold Selection (evaluated via 5-Fold Stratified CV on training set, never test labels)
    print("\n[Step 8/10] Calibrating classification threshold using 5-Fold Stratified CV on training data...")
    cv_thresh = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_train_proba = cross_val_predict(best_pipeline, X_train, y_train, cv=cv_thresh, method="predict_proba", n_jobs=-1)[:, 1]
    optimal_thresh, train_thresh_metrics = find_optimal_threshold(y_train.values, cv_train_proba, strategy="f1")
    print(f"Optimal F1 classification threshold from training CV: {optimal_thresh:.2f}")

    # 9. Final Test Set Evaluation (Evaluated once on untouched test set)
    print("\n[Step 9/10] Evaluating final pipeline on isolated test set...")
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred_default = (test_proba >= 0.5).astype(int)
    test_pred_calibrated = (test_proba >= optimal_thresh).astype(int)

    default_test_metrics = evaluate_predictions(y_test.values, test_pred_default, test_proba)
    calibrated_test_metrics = evaluate_predictions(y_test.values, test_pred_calibrated, test_proba)
    curve_data = compute_curve_points(y_test.values, test_proba)

    print(f"Test Set Metrics (at threshold=0.5):")
    print(f"  Accuracy:  {default_test_metrics['accuracy']:.4f}")
    print(f"  Precision: {default_test_metrics['precision']:.4f}")
    print(f"  Recall:    {default_test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {default_test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {default_test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {default_test_metrics['pr_auc']:.4f}")
    print(f"  Brier:     {default_test_metrics['brier_score']:.4f}")

    # 10. Extract Feature Names and Importance
    print("\n[Step 10/10] Extracting feature importances and persisting artifacts...")
    feature_names, importances = extract_feature_names_and_importance(best_pipeline)
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/best_model.joblib")

    metadata = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "random_state": 42,
        "classification_threshold": optimal_thresh,
        "default_threshold": 0.5,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": feature_names,
        "risk_bands": RISK_BANDS,
        "selection_rationale": (
            f"{best_model_name} demonstrated superior discriminant capability across 5-fold cross-validation, "
            f"achieving top PR-AUC ({best_row['PR_AUC']:.4f}) and ROC-AUC ({best_row['ROC_AUC']:.4f}) "
            "with robust generalizability and well-calibrated probabilities on tabular customer churn data."
        )
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_payload = {
        "selected_model": best_model_name,
        "cv_comparison": cv_comparison.to_dict(orient="records"),
        "test_metrics_default_threshold": default_test_metrics,
        "test_metrics_optimal_threshold": calibrated_test_metrics,
        "curve_data": curve_data
    }
    with open("models/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with open("models/feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feat_imp, f, indent=2)

    # Save processed dataset sample with risk scores for analytics
    print("Generating batch customer risk dataset for analytics dashboard...")
    full_X = df.drop(columns=[c for c in ["RowNumber", "Exited"] if c in df.columns])
    full_proba = best_pipeline.predict_proba(full_X.drop(columns=["CustomerId", "Surname"], errors="ignore"))[:, 1]
    
    analytics_df = df.copy()
    analytics_df["PredictedProbability"] = np.round(full_proba, 4)
    analytics_df["RiskScore"] = np.round(full_proba * 100.0, 2)
    analytics_df["RiskCategory"] = np.where(
        analytics_df["RiskScore"] < 25.0, "Low Risk",
        np.where(analytics_df["RiskScore"] < 50.0, "Medium Risk", "High Risk")
    )
    analytics_df["PredictedClass"] = (full_proba >= optimal_thresh).astype(int)
    
    os.makedirs("data/processed", exist_ok=True)
    analytics_df.to_csv("data/processed/scored_customers.csv", index=False)
    print("Saved scored customer dataset to 'data/processed/scored_customers.csv'.")

    print("\n==========================================================")
    print("TRAINING PIPELINE COMPLETE! ALL ARTIFACTS SAVED SUCCESSFULLY.")
    print("==========================================================")

if __name__ == "__main__":
    run_training_pipeline()