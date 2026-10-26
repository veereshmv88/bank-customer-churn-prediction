"""
Model Training & Selection Pipeline for Bank Customer Churn Prediction.
Trains, cross-validates, tunes, and evaluates multiple candidate models.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.preprocessing import (
    build_full_preprocessing_pipeline,
    prepare_data_split,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.model_evaluation import (
    evaluate_predictions,
    compute_curve_points,
    find_optimal_threshold
)

def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            scale_pos_weight=3.9,
            random_state=42,
            eval_metric="logloss"
        )
    }

def cross_validate_candidates(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Perform Stratified K-Fold CV on training data for all candidate models.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    results = []

    for name, clf in models.items():
        print(f"  Cross-validating {name}...")
        pipe = build_full_preprocessing_pipeline()
        pipe.steps.append(("classifier", clf))

        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "Model": name,
            "Accuracy": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "Accuracy_Std": round(float(np.std(cv_scores["test_accuracy"])), 4),
            "Precision": round(float(np.mean(cv_scores["test_precision"])), 4),
            "Precision_Std": round(float(np.std(cv_scores["test_precision"])), 4),
            "Recall": round(float(np.mean(cv_scores["test_recall"])), 4),
            "Recall_Std": round(float(np.std(cv_scores["test_recall"])), 4),
            "F1_Score": round(float(np.mean(cv_scores["test_f1"])), 4),
            "F1_Std": round(float(np.std(cv_scores["test_f1"])), 4),
            "ROC_AUC": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "ROC_AUC_Std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "PR_AUC": round(float(np.mean(cv_scores["test_pr_auc"])), 4),
            "PR_AUC_Std": round(float(np.std(cv_scores["test_pr_auc"])), 4),
        })

    return pd.DataFrame(results)

def tune_best_model(
    model_name: str,
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Perform Hyperparameter Tuning on the training data using 5-fold Stratified CV.
    """
    print(f"Tuning hyperparameters for {model_name}...")
    pipe = build_full_preprocessing_pipeline()
    pipe.steps.append(("classifier", base_model))

    param_grid = {}
    if "Random Forest" in model_name:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [8, 12],
            "classifier__min_samples_split": [5, 10]
        }
    elif "XGBoost" in model_name:
        param_grid = {
            "classifier__n_estimators": [120, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__subsample": [0.8, 1.0]
        }
    elif "Gradient" in model_name:
        param_grid = {
            "classifier__n_estimators": [100, 180],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1]
        }

    if param_grid:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
        return grid.best_estimator_

    # Default refit
    pipe.fit(X_train, y_train)
    return pipe

def extract_feature_names_and_importance(pipeline: Pipeline) -> Tuple[list, list]:
    """
    Extract transformed feature names and corresponding feature importances.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Retrieve categorical encoded names
    cat_step = preprocessor.named_transformers_["cat"]
    onehot = cat_step.named_steps["onehot"]
    cat_names = list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))

    transformed_feature_names = ALL_NUMERICAL_FEATURES + cat_names

    importances = []
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_.tolist()
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0]).tolist()
    else:
        importances = [1.0 / len(transformed_feature_names)] * len(transformed_feature_names)

    return transformed_feature_names, importances
