"""
Risk Scoring Engine for Bank Customer Churn Prediction.
Translates model-predicted churn probabilities into standardized operational risk scores and tiers.
"""
from typing import Dict, Any

# Operational Risk Categories (Configurable provisional risk bands)
RISK_BANDS = {
    "Low Risk": {"min": 0.0, "max": 25.0, "color": "#10B981", "badge": "bg-emerald-500/20 text-emerald-400"},
    "Medium Risk": {"min": 25.0, "max": 50.0, "color": "#F59E0B", "badge": "bg-amber-500/20 text-amber-400"},
    "High Risk": {"min": 50.0, "max": 100.0, "color": "#EF4444", "badge": "bg-rose-500/20 text-rose-400"}
}

def calculate_risk_score(probability: float) -> float:
    """
    Standard formula: Risk Score = Churn Probability * 100.
    Clamped strictly between 0.0 and 100.0.
    """
    prob = max(0.0, min(1.0, float(probability)))
    return round(prob * 100.0, 2)

def assign_risk_category(risk_score: float) -> str:
    """
    Assign operational risk band based on calculated risk score.
    """
    if risk_score < 25.0:
        return "Low Risk"
    elif risk_score < 50.0:
        return "Medium Risk"
    else:
        return "High Risk"

def format_risk_assessment(
    probability: float,
    threshold: float = 0.5,
    model_name: str = "Trained Pipeline"
) -> Dict[str, Any]:
    """
    Format standard risk assessment dictionary from churn probability.
    """
    prob = max(0.0, min(1.0, float(probability)))
    score = calculate_risk_score(prob)
    category = assign_risk_category(score)
    pred_class = 1 if prob >= threshold else 0

    return {
        "churn_probability": round(prob, 4),
        "churn_percentage": round(prob * 100.0, 1),
        "risk_score": score,
        "risk_category": category,
        "predicted_class": pred_class,
        "classification_threshold": threshold,
        "model_name": model_name,
        "disclaimer": "Predictions are estimates derived from historical data. They should support, not replace, human institutional review."
    }
