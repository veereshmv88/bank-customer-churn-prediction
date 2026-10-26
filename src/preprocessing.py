"""
Preprocessing Module for Bank Customer Churn Prediction.
Builds scikit-learn ColumnTransformers and data splits.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Any

from src.feature_engineering import BankFeatureEngineer

# Identifiers and non-predictive metadata to exclude from predictive features
EXCLUDED_COLUMNS = ["Year", "RowNumber", "CustomerId", "Surname", "Exited"]

# Base raw feature groups
CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERICAL_RAW_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]
ENGINEERED_NUMERICAL_FEATURES = [
    "BalanceSalaryRatio",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
    "ProductDensityIndicator",
    "MultiProductRisk"
]

ALL_NUMERICAL_FEATURES = NUMERICAL_RAW_FEATURES + ENGINEERED_NUMERICAL_FEATURES

def get_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for numerical and categorical features.
    Applied after feature engineering.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor

def build_full_preprocessing_pipeline() -> Pipeline:
    """
    Combines feature engineering and column transformations into a single scikit-learn Pipeline.
    """
    return Pipeline(steps=[
        ("feature_engineer", BankFeatureEngineer()),
        ("preprocessor", get_preprocessor())
    ])

def prepare_data_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into stratified train/test sets, excluding identifiers from X.
    """
    if "Exited" not in df.columns:
        raise ValueError("DataFrame must contain target column 'Exited'")

    y = df["Exited"].astype(int)
    
    # Feature columns excluding target and identifiers
    drop_cols = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
