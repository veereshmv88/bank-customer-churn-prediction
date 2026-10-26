"""
Explainable AI (XAI) Module for Bank Customer Churn Prediction.
Computes SHAP explanations and localized feature contributions.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import shap
from sklearn.pipeline import Pipeline

class ChurnExplainer:
    """
    Manages global and localized explanations for customer churn predictions.
    Uses SHAP TreeExplainer when applicable with graceful fallbacks.
    """

    def __init__(self, pipeline: Pipeline, background_data: Optional[pd.DataFrame] = None):
        self.pipeline = pipeline
        self.feature_engineer = pipeline.named_steps["feature_engineer"]
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        
        # Get feature names
        from src.preprocessing import ALL_NUMERICAL_FEATURES, CATEGORICAL_FEATURES
        cat_step = self.preprocessor.named_transformers_["cat"]
        onehot = cat_step.named_steps["onehot"]
        cat_names = list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = ALL_NUMERICAL_FEATURES + cat_names

        self.explainer = None
        self._init_shap(background_data)

    def _transform_data(self, df: pd.DataFrame) -> np.ndarray:
        df_eng = self.feature_engineer.transform(df)
        return self.preprocessor.transform(df_eng)

    def _init_shap(self, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize genuine SHAP explainer on classifier.
        For tree-based estimators (GradientBoosting, RandomForest, XGBoost),
        TreeExplainer computes exact tree-path dependent SHAP values without requiring background data.
        If background data is provided, interventional feature perturbation can be utilized.
        """
        self.explainer = None
        self.shap_init_error = None
        try:
            clf_name = type(self.classifier).__name__
            is_tree = any(k in clf_name for k in ["Tree", "Forest", "XGB", "Gradient"])

            if is_tree:
                # TreeExplainer calculates exact tree-path marginals without background data
                if background_data is not None:
                    bg_trans = self._transform_data(background_data)
                    if len(bg_trans) > 100:
                        bg_trans = shap.sample(bg_trans, 100, random_state=42)
                    self.explainer = shap.TreeExplainer(self.classifier, data=bg_trans)
                else:
                    self.explainer = shap.TreeExplainer(self.classifier)
            elif background_data is not None:
                bg_trans = self._transform_data(background_data)
                if len(bg_trans) > 100:
                    bg_trans = shap.sample(bg_trans, 100, random_state=42)
                self.explainer = shap.Explainer(self.classifier.predict_proba, bg_trans)
            else:
                self.shap_init_error = f"Model type {clf_name} requires background data for SHAP Explainer."
        except Exception as e:
            self.shap_init_error = str(e)
            self.explainer = None
            print(f"Warning: SHAP Explainer initialization error: {e}.")

    def explain_instance(self, customer_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Explain a single customer prediction using genuine SHAP values.
        Returns top positive and negative contributing factors.
        Positive impact indicates the feature increases churn risk (log-odds).
        Negative impact indicates the feature decreases churn risk (log-odds).
        """
        X_trans = self._transform_data(customer_df)
        contributions = []

        if self.explainer is not None:
            try:
                raw_shap = self.explainer.shap_values(X_trans)

                # Handle various SHAP return structures across versions and model types:
                # 1. Binary classification list [shap_class_0, shap_class_1]
                if isinstance(raw_shap, list) and len(raw_shap) >= 2:
                    vals = np.array(raw_shap[1][0])
                # 2. 3D array (n_samples, n_features, n_classes)
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    vals = raw_shap[0, :, 1]
                # 3. 2D array (n_samples, n_features) - e.g. GradientBoosting log-odds
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
                    vals = raw_shap[0]
                # 4. 1D array
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 1:
                    vals = raw_shap
                else:
                    vals = np.array(raw_shap)[0]

                for name, val in zip(self.feature_names, vals):
                    val_float = float(val)
                    impact = round(val_float, 4)
                    contributions.append({
                        "feature": name,
                        "impact": impact,
                        "direction": "Increases Churn Risk" if impact > 0 else "Decreases Churn Risk",
                        "abs_impact": abs(impact),
                        "attribution_method": "TreeSHAP"
                    })
            except Exception as e:
                print(f"Warning: SHAP computation error during explain_instance: {e}")
                contributions = []

        # If SHAP is unavailable or failed, provide clearly labeled fallback
        if not contributions:
            if hasattr(self.classifier, "feature_importances_"):
                importances = self.classifier.feature_importances_
            else:
                importances = [1.0 / len(self.feature_names)] * len(self.feature_names)

            for i, name in enumerate(self.feature_names):
                val = float(X_trans[0, i])
                impact = round(val * importances[i] * 0.1, 4)
                contributions.append({
                    "feature": name,
                    "impact": impact,
                    "direction": "Increases Churn Risk" if impact > 0 else "Decreases Churn Risk",
                    "abs_impact": abs(impact),
                    "attribution_method": "Fallback Heuristic"
                })

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        return contributions[:8]

    def get_global_importance(self) -> List[Dict[str, Any]]:
        """Return sorted global feature importances."""
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
        elif hasattr(self.classifier, "coef_"):
            importances = np.abs(self.classifier.coef_[0])
        else:
            importances = [1.0 / len(self.feature_names)] * len(self.feature_names)

        results = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(self.feature_names, importances)
        ]
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results
