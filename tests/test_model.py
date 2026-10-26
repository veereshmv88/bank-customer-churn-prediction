"""
Automated Tests for Machine Learning Model Pipeline and Inference.
"""
import pytest
import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import prepare_data_split
from src.data_loader import load_raw_data

MODEL_PATH = "models/best_model.joblib"

@pytest.fixture(scope="module")
def trained_pipeline():
    assert os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    return pipeline

def test_pipeline_steps(trained_pipeline):
    step_names = [step[0] for step in trained_pipeline.steps]
    assert "feature_engineer" in step_names, "Pipeline missing 'feature_engineer' step"
    assert "preprocessor" in step_names, "Pipeline missing 'preprocessor' step"
    assert "classifier" in step_names, "Pipeline missing 'classifier' step"

def test_model_predict_proba_range(trained_pipeline):
    df = load_raw_data()
    _, X_test, _, _ = prepare_data_split(df, test_size=0.1, random_state=42)
    proba = trained_pipeline.predict_proba(X_test)
    
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0.0), "Negative probability detected"
    assert np.all(proba <= 1.0), "Probability exceeding 1.0 detected"
    assert np.allclose(np.sum(proba, axis=1), 1.0, atol=1e-5), "Probabilities do not sum to 1.0"
