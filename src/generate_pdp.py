"""
Partial Dependence Plot (PDP) Generation Script.
Computes and visualizes marginal effects of top predictive features on predicted churn probability
using scikit-learn's PartialDependenceDisplay and the production Gradient Boosting pipeline.
"""
import os
import sys

# Ensure repository root is on sys.path for joblib unpickling
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def generate_partial_dependence_plots():
    print("=" * 60)
    print("Generating Partial Dependence Plots (PDP)")
    print("=" * 60)

    # 1. Load Data
    data_path = "European_Bank.csv" if os.path.exists("European_Bank.csv") else "data/raw/European_Bank.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2. Load Model Pipeline
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipeline = joblib.load(model_path)
    print(f"Loaded pipeline: {type(pipeline).__name__} with steps: {list(pipeline.named_steps.keys())}")

    # 3. Top 4 Features for PDP Analysis
    features = ["Age", "NumOfProducts", "Balance", "IsActiveMember"]
    print(f"Computing PDP for features: {features}")

    # 4. Create Matplotlib Figure with Enterprise Styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    display = PartialDependenceDisplay.from_estimator(
        estimator=pipeline,
        X=X,
        features=features,
        kind="average",
        ax=axes,
        line_kw={"color": "#2563EB", "linewidth": 2.5},
        pd_line_kw={"color": "#1D4ED8", "linewidth": 2.5}
    )

    # Polish axes aesthetics
    feature_titles = {
        "Age": "Age (Years) - Marginal Effect",
        "NumOfProducts": "Number of Products - Non-Linear Risk Dynamic",
        "Balance": "Account Balance (€) - Marginal Effect",
        "IsActiveMember": "Active Membership Status - Effect on Churn Risk"
    }

    for ax in axes:
        current_label = ax.get_xlabel()
        if current_label in feature_titles:
            ax.set_title(feature_titles[current_label], fontsize=11, fontweight="bold", color="#1E293B", pad=10)
        ax.set_ylabel("Marginal Predicted Churn Probability", fontsize=9, color="#475569")
        ax.tick_params(colors="#475569", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        "Partial Dependence Plots: Gradient Boosting Model\nMarginal Relationship Between Key Features and Predicted Churn Probability",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 5. Save Artifacts
    os.makedirs("reports/figures", exist_ok=True)
    out_paths = [
        "reports/figures/pdp_analysis.png",
        "reports/figures/partial_dependence_plots.png"
    ]
    for p in out_paths:
        fig.savefig(p, bbox_inches="tight")
        print(f"  -> Saved PDP figure to: {p}")

    plt.close(fig)
    print("PDP generation complete.")

if __name__ == "__main__":
    generate_partial_dependence_plots()
