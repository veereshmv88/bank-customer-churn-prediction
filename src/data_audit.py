"""
Data Audit Module.
Performs thorough data quality verification and generates structured audit summaries.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

def audit_dataset(df: pd.DataFrame, output_dir: str = "reports/tables") -> Dict[str, Any]:
    """
    Perform a comprehensive data audit on the bank churn dataset.
    Saves audit results to JSON and Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_rows = len(df)
    total_cols = len(df.columns)

    # Missing values
    missing = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / total_rows * 100).round(2).to_dict()

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    dup_cust = int(df["CustomerId"].duplicated().sum()) if "CustomerId" in df.columns else 0

    # Target distribution
    target_counts = {}
    target_pct = {}
    if "Exited" in df.columns:
        counts = df["Exited"].value_counts()
        target_counts = {str(k): int(v) for k, v in counts.items()}
        target_pct = {str(k): round(float(v) / total_rows * 100, 2) for k, v in counts.items()}

    # Categorical summary
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_summary = {}
    for col in categorical_cols:
        cat_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "values": df[col].value_counts().to_dict()
        }

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_summary = {}
    for col in numeric_cols:
        num_summary[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
            "std": round(float(df[col].std()), 2)
        }

    # Potential leakage check: columns with 100% correlation with target or identifier leaks
    leakage_flags = []
    if "Exited" in df.columns:
        for col in numeric_cols:
            if col != "Exited" and col not in ["RowNumber", "CustomerId", "Year"] and df[col].std() > 0:
                corr = abs(df[col].corr(df["Exited"]))
                if corr > 0.9:
                    leakage_flags.append(f"High correlation with Exited: {col} ({corr:.3f})")

    audit_report = {
        "dataset_shape": {"rows": total_rows, "columns": total_cols},
        "column_names": list(df.columns),
        "missing_values": missing,
        "missing_percentages": missing_pct,
        "duplicate_rows": dup_rows,
        "duplicate_customer_ids": dup_cust,
        "target_distribution": {
            "counts": target_counts,
            "percentages": target_pct,
            "imbalance_ratio": round(target_counts.get("0", 1) / max(1, target_counts.get("1", 1)), 2)
        },
        "categorical_columns": cat_summary,
        "numeric_summaries": num_summary,
        "data_leakage_risks": leakage_flags if leakage_flags else ["None detected. CustomerId and Surname must be excluded from feature vectors."]
    }

    # Save to JSON
    json_path = os.path.join(output_dir, "data_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    # Save to Markdown
    md_path = os.path.join(output_dir, "data_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Bank Customer Churn Dataset Audit Report\n\n")
        f.write(f"- **Total Rows:** {total_rows:,}\n")
        f.write(f"- **Total Columns:** {total_cols}\n")
        f.write(f"- **Duplicate Records:** {dup_rows}\n")
        f.write(f"- **Duplicate Customer IDs:** {dup_cust}\n")
        f.write(f"- **Target Imbalance Ratio (0:1):** {audit_report['target_distribution']['imbalance_ratio']}:1\n\n")
        f.write("## Target Variable Distribution (`Exited`)\n\n")
        f.write("| Class | Meaning | Count | Percentage |\n")
        f.write("|-------|---------|-------|------------|\n")
        f.write(f"| 0 | Retained Customer | {target_counts.get('0', 0):,} | {target_pct.get('0', 0)}% |\n")
        f.write(f"| 1 | Churned Customer | {target_counts.get('1', 0):,} | {target_pct.get('1', 0)}% |\n\n")
        f.write("## Categorical Feature Breakdown\n\n")
        for col, info in cat_summary.items():
            f.write(f"### {col} ({info['unique_count']} unique values)\n")
            for val, cnt in info["values"].items():
                f.write(f"- **{val}:** {cnt:,} ({cnt/total_rows*100:.1f}%)\n")
            f.write("\n")

    return audit_report

if __name__ == "__main__":
    from data_loader import load_raw_data
    raw_df = load_raw_data()
    rep = audit_dataset(raw_df)
    print("Data audit completed successfully!")
