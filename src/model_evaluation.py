"""
Model Evaluation Module for Bank Customer Churn Prediction.
Calculates classification metrics, ROC curves, PR curves, calibration curves, and confusion matrices.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for positive class (Exited = 1).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "true_negatives": cm[0][0],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_positives": cm[1][1]
    }

def compute_curve_points(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_size: int = 50
) -> Dict[str, Any]:
    """
    Generate decimated curve points for ROC, PR, and Calibration for frontend Chart.js visualization.
    """
    # 1. ROC Curve
    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    # Downsample points smoothly if length exceeds sample_size
    idx = np.linspace(0, len(fpr) - 1, min(sample_size, len(fpr)), dtype=int)
    roc_points = [{"x": round(float(fpr[i]), 4), "y": round(float(tpr[i]), 4)} for i in idx]

    # 2. Precision-Recall Curve
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_proba)
    idx_pr = np.linspace(0, len(rec) - 1, min(sample_size, len(rec)), dtype=int)
    pr_points = [{"x": round(float(rec[i]), 4), "y": round(float(prec[i]), 4)} for i in idx_pr]

    # 3. Calibration Curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    calib_points = [
        {"x": round(float(mean_pred[i]), 4), "y": round(float(fraction_pos[i]), 4)}
        for i in range(len(mean_pred))
    ]

    return {
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "calibration_curve": calib_points
    }

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1"
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal classification threshold evaluated on validation data.
    Strategies: 'f1' (maximize harmonic mean) or 'balanced' (balance precision and recall).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if strategy == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            rec = recall_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        if score > best_score:
            best_score = score
            best_thresh = float(round(t, 2))
            best_metrics = evaluate_predictions(y_true, preds, y_proba)

    return best_thresh, best_metrics
