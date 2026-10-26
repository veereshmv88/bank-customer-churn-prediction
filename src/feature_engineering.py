"""
Feature Engineering Module for Bank Customer Churn Prediction.
Implements domain-specific engineered features consistently for both training and inference.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs engineered features.
    
    Engineered Features:
    1. BalanceSalaryRatio: Balance / EstimatedSalary (safely handling zero salary).
    2. EngagementProductInteraction: IsActiveMember * NumOfProducts.
    3. AgeTenureInteraction: Age * Tenure.
    4. ProductDensityIndicator: NumOfProducts / (Tenure + 1.0)
       - Definition: Product adoption rate relative to customer relationship duration (in years, smoothed by +1).
    5. MultiProductRisk: Indicator if NumOfProducts >= 3 (empirically associated with sharp churn surges).
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Avoid modifying original dataframe
        df = X.copy()

        # 1. BalanceSalaryRatio
        # Handle zero or negative salary safely
        salary = df["EstimatedSalary"].fillna(0.0)
        balance = df["Balance"].fillna(0.0)
        df["BalanceSalaryRatio"] = np.where(salary > 0, balance / salary, 0.0)
        # Clip extreme ratios to prevent numerical instability
        df["BalanceSalaryRatio"] = np.clip(df["BalanceSalaryRatio"], 0.0, 100.0)

        # 2. EngagementProductInteraction
        active = df["IsActiveMember"].fillna(0).astype(int)
        products = df["NumOfProducts"].fillna(1).astype(int)
        df["EngagementProductInteraction"] = active * products

        # 3. AgeTenureInteraction
        age = df["Age"].fillna(35).astype(float)
        tenure = df["Tenure"].fillna(0).astype(float)
        df["AgeTenureInteraction"] = age * tenure

        # 4. ProductDensityIndicator
        # Definition: NumOfProducts divided by (Tenure + 1). Measures product intensity over account age.
        df["ProductDensityIndicator"] = products / (tenure + 1.0)

        # 5. MultiProductRisk: NumOfProducts >= 3
        df["MultiProductRisk"] = (products >= 3).astype(int)

        return df

    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        if input_features is None:
            input_features = []
        new_cols = [
            "BalanceSalaryRatio",
            "EngagementProductInteraction",
            "AgeTenureInteraction",
            "ProductDensityIndicator",
            "MultiProductRisk"
        ]
        return list(input_features) + new_cols

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper to apply engineering to a pandas DataFrame."""
    fe = BankFeatureEngineer()
    return fe.transform(df)
