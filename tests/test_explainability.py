"""
Unit and Integration Tests for Explainable AI (SHAP TreeExplainer).
Verifies:
1. SHAP TreeExplainer initialization without background data on tree models.
2. Alignment of feature names with preprocessed column space.
3. Proper mathematical direction: positive SHAP values increase predicted churn risk.
4. Correctness of TreeSHAP additivity: sum of SHAP values + base value = model decision function.
5. Error and failure handling: fallback behavior when explainer is unavailable or simulated failure occurs.
6. Attribution method metadata labeling.
"""
import pytest
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from src.explainability import ChurnExplainer
from src.prediction import PredictionService

@pytest.fixture(scope="module")
def model_pipeline():
    return joblib.load("models/best_model.joblib")

@pytest.fixture
def sample_customer_df():
    return pd.DataFrame([{
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88
    }])

def test_shap_tree_explainer_initialization(model_pipeline):
    """Confirm ChurnExplainer automatically initializes TreeExplainer for tree pipelines."""
    explainer = ChurnExplainer(model_pipeline)
    assert explainer.explainer is not None
    assert explainer.shap_init_error is None
    assert hasattr(explainer.explainer, "expected_value")

def test_feature_names_alignment(model_pipeline, sample_customer_df):
    """Ensure transformed matrix shape matches the explainer's feature_names count exactly."""
    explainer = ChurnExplainer(model_pipeline)
    X_trans = explainer._transform_data(sample_customer_df)
    assert X_trans.shape[1] == len(explainer.feature_names)
    assert len(explainer.feature_names) == 18
    assert "CreditScore" in explainer.feature_names
    assert "Geography_Germany" in explainer.feature_names

def test_shap_additivity_and_prediction_consistency(model_pipeline, sample_customer_df):
    """
    Verify exact TreeSHAP mathematical additivity:
    sum(SHAP values) + base_value == decision_function(X)
    expit(sum(SHAP values) + base_value) == predict_proba(X)[:, 1]
    """
    explainer = ChurnExplainer(model_pipeline)
    X_trans = explainer._transform_data(sample_customer_df)
    
    raw_shap = explainer.explainer.shap_values(X_trans)[0]
    base_val = explainer.explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)

    classifier = model_pipeline.named_steps["classifier"]
    margin = float(classifier.decision_function(X_trans)[0])
    prob = float(classifier.predict_proba(X_trans)[0, 1])

    calculated_margin = base_val + float(np.sum(raw_shap))
    calculated_prob = float(expit(calculated_margin))

    assert np.isclose(margin, calculated_margin, atol=1e-5)
    assert np.isclose(prob, calculated_prob, atol=1e-5)

def test_shap_contribution_direction_consistency(model_pipeline, sample_customer_df):
    """
    Verify that positive SHAP impacts are tagged 'Increases Churn Risk'
    and negative impacts are tagged 'Decreases Churn Risk'.
    """
    explainer = ChurnExplainer(model_pipeline)
    factors = explainer.explain_instance(sample_customer_df)

    assert len(factors) > 0
    for factor in factors:
        assert factor["attribution_method"] == "TreeSHAP"
        if factor["impact"] > 0:
            assert factor["direction"] == "Increases Churn Risk"
        else:
            assert factor["direction"] == "Decreases Churn Risk"
        assert np.isclose(factor["abs_impact"], abs(factor["impact"]))

def test_prediction_service_explainability_integration():
    """Verify PredictionService end-to-end includes TreeSHAP top_factors."""
    service = PredictionService()
    profile = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88
    }
    result = service.predict_single(profile)
    assert "top_factors" in result
    assert len(result["top_factors"]) > 0
    assert result["top_factors"][0]["attribution_method"] == "TreeSHAP"

def test_shap_failure_handling_fallback(model_pipeline, sample_customer_df):
    """
    Ensure that if explainer is None or fails, explain_instance returns
    explicitly labeled 'Fallback Heuristic' and does not crash.
    """
    explainer = ChurnExplainer(model_pipeline)
    # Simulate explainer unavailable
    explainer.explainer = None
    factors = explainer.explain_instance(sample_customer_df)

    assert len(factors) > 0
    for factor in factors:
        assert factor["attribution_method"] == "Fallback Heuristic"
        assert "feature" in factor
        assert "impact" in factor
        assert "direction" in factor
