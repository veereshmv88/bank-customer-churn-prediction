"""
Unified Prediction Service for Bank Customer Churn.
Validates input, evaluates models, computes risk scores, and generates explainability factors.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.risk_scoring import format_risk_assessment
from src.explainability import ChurnExplainer

# Required input fields and acceptable domains
VALID_GEOGRAPHIES = ["France", "Germany", "Spain"]
VALID_GENDERS = ["Female", "Male"]

def validate_customer_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate customer profile inputs.
    Returns (is_valid, error_message).
    """
    required_fields = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            return False, f"Missing required field: {field}"

    try:
        cs = float(data["CreditScore"])
        if not (300 <= cs <= 850):
            return False, f"Credit Score must be between 300 and 850 (received {cs})"

        age = float(data["Age"])
        if not (18 <= age <= 100):
            return False, f"Age must be between 18 and 100 (received {age})"

        tenure = float(data["Tenure"])
        if not (0 <= tenure <= 10):
            return False, f"Tenure must be between 0 and 10 years (received {tenure})"

        bal = float(data["Balance"])
        if bal < 0:
            return False, f"Balance cannot be negative (received {bal})"

        nop = int(data["NumOfProducts"])
        if not (1 <= nop <= 4):
            return False, f"Number of Products must be between 1 and 4 (received {nop})"

        salary = float(data["EstimatedSalary"])
        if salary < 0:
            return False, f"Estimated Salary cannot be negative (received {salary})"

        geo = str(data["Geography"]).capitalize()
        if geo not in VALID_GEOGRAPHIES:
            return False, f"Geography must be one of {VALID_GEOGRAPHIES} (received {data['Geography']})"

        gender = str(data["Gender"]).capitalize()
        if gender not in VALID_GENDERS:
            return False, f"Gender must be one of {VALID_GENDERS} (received {data['Gender']})"

    except (ValueError, TypeError) as e:
        return False, f"Invalid data format or numeric conversion error: {str(e)}"

    return True, None

class PredictionService:
    """
    Singleton-style service that loads model artifacts once at startup and serves predictions.
    """
    _instance = None

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.metadata = {}
        self.metrics = {}
        self.explainer = None
        self.threshold = 0.5
        self.model_name = "Gradient Boosting"
        self._load_artifacts()

    @classmethod
    def get_instance(cls, model_dir: str = "models") -> "PredictionService":
        if cls._instance is None:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def _load_artifacts(self):
        model_path = os.path.join(self.model_dir, "best_model.joblib")
        meta_path = os.path.join(self.model_dir, "model_metadata.json")
        metrics_path = os.path.join(self.model_dir, "model_metrics.json")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded trained pipeline from {model_path}")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.model_name = self.metadata.get("model_name", "Best Model")
                self.threshold = self.metadata.get("classification_threshold", 0.5)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        if self.model is not None:
            # Initialize explainability module
            self.explainer = ChurnExplainer(self.model)

    def is_loaded(self) -> bool:
        return self.model is not None

    def format_input_df(self, profile: Dict[str, Any]) -> pd.DataFrame:
        """Construct standard DataFrame row matching training feature schema."""
        row = {
            "CreditScore": [float(profile["CreditScore"])],
            "Geography": [str(profile["Geography"]).capitalize()],
            "Gender": [str(profile["Gender"]).capitalize()],
            "Age": [float(profile["Age"])],
            "Tenure": [float(profile["Tenure"])],
            "Balance": [float(profile["Balance"])],
            "NumOfProducts": [int(profile["NumOfProducts"])],
            "HasCrCard": [1 if profile.get("HasCrCard") in [1, True, "1", "true", "True"] else 0],
            "IsActiveMember": [1 if profile.get("IsActiveMember") in [1, True, "1", "true", "True"] else 0],
            "EstimatedSalary": [float(profile["EstimatedSalary"])]
        }
        return pd.DataFrame(row)

    def predict_single(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single customer profile, returning churn probability, risk score, category, and factors.
        """
        if not self.is_loaded():
            raise RuntimeError("Model pipeline is not loaded. Please train models first.")

        valid, err = validate_customer_input(profile)
        if not valid:
            raise ValueError(err)

        df = self.format_input_df(profile)
        proba = float(self.model.predict_proba(df)[0, 1])

        result = format_risk_assessment(
            probability=proba,
            threshold=self.threshold,
            model_name=self.model_name
        )

        # Explain top individual factors
        if self.explainer:
            factors = self.explainer.explain_instance(df)
            result["top_factors"] = factors
        else:
            result["top_factors"] = []

        return result

    def simulate_what_if(self, current_profile: Dict[str, Any], scenario_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current and scenario profiles through the identical backend model pipeline.
        Calculates exact probability difference in percentage points.
        """
        curr_res = self.predict_single(current_profile)
        scen_res = self.predict_single(scenario_profile)

        # Delta in percentage points: (Scenario - Current) * 100
        prob_diff = round((scen_res["churn_probability"] - curr_res["churn_probability"]) * 100.0, 2)
        risk_score_diff = round(scen_res["risk_score"] - curr_res["risk_score"], 2)

        direction = "increased" if prob_diff > 0 else ("decreased" if prob_diff < 0 else "unchanged")

        return {
            "current": curr_res,
            "scenario": scen_res,
            "probability_difference_pp": prob_diff,
            "risk_score_difference": risk_score_diff,
            "impact_direction": direction,
            "summary": f"Hypothetical profile changes resulted in a {abs(prob_diff):.2f}% point {direction} in estimated churn probability.",
            "disclaimer": "This is a model-based scenario projection, not a causal experiment. Changing customer attributes does not guarantee real-world retention."
        }
