"""
Automated Integration Tests for Flask REST API Endpoints.
"""
import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"BANK CUSTOMER CHURN" in res.data
    assert b"Risk Calculator" in res.data

def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_name" in data

def test_api_metadata(client):
    res = client.get("/api/metadata")
    assert res.status_code == 200
    data = res.get_json()
    assert "classification_threshold" in data
    assert "features" in data
    assert "risk_bands" in data

def test_api_metrics(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "cv_comparison" in data
    assert "test_metrics_default_threshold" in data
    assert "curve_data" in data

def test_api_analytics(client):
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.get_json()
    assert "kpis" in data
    assert data["kpis"]["total_customers"] == 10000
    assert "geography" in data
    assert "products" in data

def test_api_predict_valid(client):
    payload = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 38,
        "Tenure": 6,
        "Balance": 60000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert "predicted_class" in data

def test_api_predict_invalid_input(client):
    # Missing required field Age
    payload = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
    data = res.get_json()
    assert "error" in data

def test_api_what_if(client):
    current = {
        "CreditScore": 650, "Geography": "France", "Gender": "Female",
        "Age": 45, "Tenure": 3, "Balance": 70000.0, "NumOfProducts": 3,
        "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 80000.0
    }
    scenario = {
        "CreditScore": 650, "Geography": "France", "Gender": "Female",
        "Age": 45, "Tenure": 3, "Balance": 70000.0, "NumOfProducts": 1,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 80000.0
    }
    res = client.post("/api/what-if", json={"current": current, "scenario": scenario})
    assert res.status_code == 200
    data = res.get_json()
    assert "current" in data
    assert "scenario" in data
    assert "probability_difference_pp" in data
    assert "risk_score_difference" in data
    assert "disclaimer" in data

def test_api_customers(client):
    res = client.get("/api/customers")
    assert res.status_code == 200
    data = res.get_json()
    assert "customers" in data
    assert len(data["customers"]) > 0
