"""
Bank Customer Churn Prediction and Risk Scoring Application.
Production-ready Flask Web Application and REST API.
"""
import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

from src.prediction import PredictionService, validate_customer_input
from src.risk_scoring import RISK_BANDS

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

# Initialize Prediction Service at startup
prediction_service = PredictionService.get_instance()

# Cache analytics data in memory
_analytics_cache = None

def get_analytics_data():
    global _analytics_cache
    if _analytics_cache is not None:
        return _analytics_cache

    scored_path = "data/processed/scored_customers.csv"
    raw_path = "European_Bank.csv" if os.path.exists("European_Bank.csv") else "data/raw/European_Bank.csv"
    
    if os.path.exists(scored_path):
        df = pd.read_csv(scored_path)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        return {}

    total_customers = len(df)
    churned_customers = int((df["Exited"] == 1).sum()) if "Exited" in df.columns else 0
    retained_customers = total_customers - churned_customers
    churn_rate = round((churned_customers / total_customers * 100), 2) if total_customers > 0 else 0.0

    # Risk counts if scored
    low_risk = int((df["RiskCategory"] == "Low Risk").sum()) if "RiskCategory" in df.columns else 0
    med_risk = int((df["RiskCategory"] == "Medium Risk").sum()) if "RiskCategory" in df.columns else 0
    high_risk = int((df["RiskCategory"] == "High Risk").sum()) if "RiskCategory" in df.columns else 0

    avg_prob = round(float(df["PredictedProbability"].mean() * 100), 2) if "PredictedProbability" in df.columns else 0.0
    median_prob = round(float(df["PredictedProbability"].median() * 100), 2) if "PredictedProbability" in df.columns else 0.0

    # Probability histogram (bands: 0-10, 10-25, 25-50, 50-75, 75-100)
    prob_bands = {
        "0 - 10%": 0,
        "10 - 25%": 0,
        "25 - 50%": 0,
        "50 - 75%": 0,
        "75 - 100%": 0
    }
    if "PredictedProbability" in df.columns:
        p = df["PredictedProbability"] * 100
        prob_bands["0 - 10%"] = int(((p >= 0) & (p < 10)).sum())
        prob_bands["10 - 25%"] = int(((p >= 10) & (p < 25)).sum())
        prob_bands["25 - 50%"] = int(((p >= 25) & (p < 50)).sum())
        prob_bands["50 - 75%"] = int(((p >= 50) & (p < 75)).sum())
        prob_bands["75 - 100%"] = int((p >= 75).sum())

    # Geography churn stats
    geo_stats = {}
    for geo in df["Geography"].unique():
        sub = df[df["Geography"] == geo]
        tot = len(sub)
        ch = int((sub["Exited"] == 1).sum()) if "Exited" in sub.columns else 0
        geo_stats[geo] = {
            "total": tot,
            "churned": ch,
            "retained": tot - ch,
            "churn_rate": round(ch / tot * 100, 2) if tot > 0 else 0
        }

    # Gender churn stats
    gender_stats = {}
    for gen in df["Gender"].unique():
        sub = df[df["Gender"] == gen]
        tot = len(sub)
        ch = int((sub["Exited"] == 1).sum()) if "Exited" in sub.columns else 0
        gender_stats[gen] = {
            "total": tot,
            "churned": ch,
            "retained": tot - ch,
            "churn_rate": round(ch / tot * 100, 2) if tot > 0 else 0
        }

    # Age groups
    age_bins = [18, 30, 40, 50, 60, 100]
    age_labels = ["18-30", "31-40", "41-50", "51-60", "60+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
    age_stats = {}
    for grp in age_labels:
        sub = df[df["AgeGroup"] == grp]
        tot = len(sub)
        ch = int((sub["Exited"] == 1).sum()) if "Exited" in sub.columns else 0
        age_stats[grp] = {
            "total": tot,
            "churned": ch,
            "retained": tot - ch,
            "churn_rate": round(ch / tot * 100, 2) if tot > 0 else 0
        }

    # Products churn stats
    product_stats = {}
    for p_num in sorted(df["NumOfProducts"].unique()):
        sub = df[df["NumOfProducts"] == p_num]
        tot = len(sub)
        ch = int((sub["Exited"] == 1).sum()) if "Exited" in sub.columns else 0
        product_stats[str(p_num)] = {
            "total": tot,
            "churned": ch,
            "retained": tot - ch,
            "churn_rate": round(ch / tot * 100, 2) if tot > 0 else 0
        }

    # Active member stats
    act_stats = {}
    for act_val, label in [(1, "Active Member"), (0, "Inactive Member")]:
        sub = df[df["IsActiveMember"] == act_val]
        tot = len(sub)
        ch = int((sub["Exited"] == 1).sum()) if "Exited" in sub.columns else 0
        act_stats[label] = {
            "total": tot,
            "churned": ch,
            "retained": tot - ch,
            "churn_rate": round(ch / tot * 100, 2) if tot > 0 else 0
        }

    # Credit card stats
    cc_stats = {}
    for cc_val, label in [(1, "Has Credit Card"), (0, "No Credit Card")]:
        sub = df[df["HasCrCard"] == cc_val]
        tot = len(sub)
        ch = int((sub["Exited"] == 1).sum()) if "Exited" in sub.columns else 0
        cc_stats[label] = {
            "total": tot,
            "churned": ch,
            "retained": tot - ch,
            "churn_rate": round(ch / tot * 100, 2) if tot > 0 else 0
        }

    _analytics_cache = {
        "kpis": {
            "total_customers": total_customers,
            "churn_rate": churn_rate,
            "churned_customers": churned_customers,
            "retained_customers": retained_customers,
            "predicted_high_risk": high_risk,
            "predicted_medium_risk": med_risk,
            "predicted_low_risk": low_risk,
            "avg_predicted_probability": avg_prob,
            "median_predicted_probability": median_prob
        },
        "risk_bands": prob_bands,
        "geography": geo_stats,
        "gender": gender_stats,
        "age_groups": age_stats,
        "products": product_stats,
        "activity": act_stats,
        "credit_card": cc_stats
    }
    return _analytics_cache

# ==================== WEB ROUTES ====================

@app.route("/")
def index():
    """Serve the primary Single Page Application."""
    return render_template("index.html")

# ==================== API ROUTES ====================

@app.route("/healthz", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """Application health and model status."""
    return jsonify({
        "status": "healthy",
        "model_loaded": prediction_service.is_loaded(),
        "model_name": prediction_service.model_name,
        "threshold": prediction_service.threshold
    })

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return model metadata, operational risk bands, and features."""
    return jsonify({
        "model_name": prediction_service.model_name,
        "model_version": prediction_service.metadata.get("model_version", "1.0.0"),
        "classification_threshold": prediction_service.threshold,
        "features": prediction_service.metadata.get("features", []),
        "risk_bands": RISK_BANDS,
        "selection_rationale": prediction_service.metadata.get("selection_rationale", "")
    })

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return actual model comparison and evaluation metrics."""
    if not prediction_service.metrics:
        # Try loading directly
        metrics_path = "models/model_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"error": "Model metrics not found. Please run train.py."}), 404
    return jsonify(prediction_service.metrics)

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Return aggregate dataset analytics and distributions."""
    data = get_analytics_data()
    return jsonify(data)

@app.route("/api/explain", methods=["GET"])
def get_global_explanations():
    """Return global feature importances and explanations."""
    feat_path = "models/feature_importance.json"
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            features = json.load(f)
            return jsonify({
                "model_name": prediction_service.model_name,
                "global_importance": features
            })
    return jsonify({"error": "Feature importance data not found."}), 404

@app.route("/api/predict", methods=["POST"])
def predict():
    """Score an individual customer profile."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        result = prediction_service.predict_single(data)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 503
    except Exception as e:
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

@app.route("/api/what-if", methods=["POST"])
def what_if():
    """Simulate what-if scenario comparison."""
    try:
        data = request.get_json(force=True)
        if not data or "current" not in data or "scenario" not in data:
            return jsonify({"error": "Request must contain 'current' and 'scenario' customer profiles."}), 400

        result = prediction_service.simulate_what_if(data["current"], data["scenario"])
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 503
    except Exception as e:
        return jsonify({"error": f"Internal simulation error: {str(e)}"}), 500

@app.route("/api/customers", methods=["GET"])
def get_sample_customers():
    """Return sample real customer profiles categorized by risk level for interactive testing."""
    samples = [
        {
            "label": "Low Risk Customer (Long Tenure, Single Product, Active)",
            "CreditScore": 750,
            "Geography": "France",
            "Gender": "Female",
            "Age": 32,
            "Tenure": 8,
            "Balance": 45000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 85000.0
        },
        {
            "label": "Medium Risk Customer (Moderate Age, Inactive, Spain)",
            "CreditScore": 620,
            "Geography": "Spain",
            "Gender": "Male",
            "Age": 45,
            "Tenure": 3,
            "Balance": 78000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 65000.0
        },
        {
            "label": "High Risk Customer (Older Age, 3 Products, High Balance, Inactive, Germany)",
            "CreditScore": 580,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 54,
            "Tenure": 2,
            "Balance": 125000.0,
            "NumOfProducts": 3,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 115000.0
        }
    ]
    return jsonify({"customers": samples})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
