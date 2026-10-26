"""
Data Loader Module for Bank Customer Churn Prediction.
Discovers and loads the official attached European_Bank.csv dataset.
"""
import os
import pandas as pd
from typing import Optional, List

EXPECTED_COLUMNS: List[str] = [
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

def find_dataset_file() -> str:
    """Locate the official attached dataset European_Bank.csv."""
    candidates = [
        "European_Bank.csv",
        os.path.join("data", "raw", "European_Bank.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Search in data directory
    if os.path.exists("data"):
        for root, _, files in os.walk("data"):
            for fname in files:
                if fname.lower().endswith(".csv") and "european" in fname.lower():
                    return os.path.join(root, fname)

    raise FileNotFoundError(
        "Could not locate the official attached dataset 'European_Bank.csv'."
    )

def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw CSV data into a pandas DataFrame.
    Validates that required customer attributes and target 'Exited' are present.
    """
    if filepath is None:
        filepath = find_dataset_file()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Official attached dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Validate essential columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{filepath}' is missing expected columns: {missing_cols}"
        )

    return df
