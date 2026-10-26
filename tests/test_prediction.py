"""
Automated Tests for Feature Engineering, Prediction Engine, and Risk Scoring.
"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import BankFeatureEngineer, add_engineered_features
from src.risk_scoring import calculate_risk_score, assign_risk_category, format_risk_assessment
from src.prediction import PredictionService, validate_customer_input

def test_feature_engineering_columns():
    df = pd.DataFrame([{
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 4,
        "Balance": 80000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000.0
    }])
    transformed = add_engineered_features(df)
    
    assert "BalanceSalaryRatio" in transformed.columns
    assert "EngagementProductInteraction" in transformed.columns
    assert "AgeTenureInteraction" in transformed.columns
    assert "ProductDensityIndicator" in transformed.columns
    assert "MultiProductRisk" in transformed.columns

    # Verify mathematical values
    assert np.isclose(transformed["BalanceSalaryRatio"].iloc[0], 0.8)
    assert transformed["EngagementProductInteraction"].iloc[0] == 2
    assert transformed["AgeTenureInteraction"].iloc[0] == 160.0
    assert np.isclose(transformed["ProductDensityIndicator"].iloc[0], 2.0 / 5.0)
    assert transformed["MultiProductRisk"].iloc[0] == 0

def test_feature_engineering_zero_salary_safety():
    df = pd.DataFrame([{
        "CreditScore": 600, "Geography": "France", "Gender": "Female",
        "Age": 40, "Tenure": 0, "Balance": 50000.0, "NumOfProducts": 1,
        "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 0.0
    }])
    transformed = add_engineered_features(df)
    assert np.isfinite(transformed["BalanceSalaryRatio"].iloc[0])
    assert transformed["BalanceSalaryRatio"].iloc[0] == 0.0

def test_risk_scoring_bounds():
    assert calculate_risk_score(0.0) == 0.0
    assert calculate_risk_score(1.0) == 100.0
    assert calculate_risk_score(0.045) == 4.5
    assert calculate_risk_score(0.7891) == 78.91
    # Check clamping
    assert calculate_risk_score(-0.5) == 0.0
    assert calculate_risk_score(1.5) == 100.0

def test_risk_category_assignment():
    assert assign_risk_category(12.0) == "Low Risk"
    assert assign_risk_category(24.9) == "Low Risk"
    assert assign_risk_category(25.0) == "Medium Risk"
    assert assign_risk_category(49.9) == "Medium Risk"
    assert assign_risk_category(50.0) == "High Risk"
    assert assign_risk_category(85.0) == "High Risk"

def test_customer_input_validation():
    valid_profile = {
        "CreditScore": 650, "Geography": "France", "Gender": "Male",
        "Age": 35, "Tenure": 5, "Balance": 50000.0, "NumOfProducts": 1,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000.0
    }
    valid, err = validate_customer_input(valid_profile)
    assert valid is True
    assert err is None

    # Invalid Geography
    invalid_geo = valid_profile.copy()
    invalid_geo["Geography"] = "Atlantis"
    valid, err = validate_customer_input(invalid_geo)
    assert valid is False
    assert "Geography" in err

    # Out of bounds Credit Score
    invalid_cs = valid_profile.copy()
    invalid_cs["CreditScore"] = 999
    valid, err = validate_customer_input(invalid_cs)
    assert valid is False
    assert "Credit Score" in err
