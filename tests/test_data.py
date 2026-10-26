"""
Automated Tests for Dataset Loading and Schema Integrity.
"""
import pytest
import os
import pandas as pd
from src.data_loader import load_raw_data, EXPECTED_COLUMNS

def test_raw_data_file_exists():
    path = "European_Bank.csv"
    assert os.path.exists(path) or os.path.exists(os.path.join("data", "raw", "European_Bank.csv")), f"Official dataset file missing at {path}"

def test_raw_data_loading():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    assert df.shape[1] >= 13

def test_expected_columns_present():
    df = load_raw_data()
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column '{col}' is missing"

def test_target_values_and_distribution():
    df = load_raw_data()
    unique_targets = set(df["Exited"].unique())
    assert unique_targets == {0, 1}, f"Target variable has unexpected classes: {unique_targets}"
    assert df["Exited"].isnull().sum() == 0, "Target column contains missing values"

def test_customer_id_uniqueness():
    df = load_raw_data()
    assert df["CustomerId"].duplicated().sum() == 0, "Duplicate CustomerId entries found"
