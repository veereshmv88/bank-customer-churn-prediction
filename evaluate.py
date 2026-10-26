"""
Standalone Model Evaluation Script.
Loads persisted model artifact and evaluates performance on held-out test data.
"""
import os
import json
import joblib
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import prepare_data_split
from src.model_evaluation import evaluate_predictions

def evaluate_saved_model():
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    print("Loading saved pipeline from:", model_path)
    pipeline = joblib.load(model_path)

    print("Loading test data...")
    df = load_raw_data()
    _, X_test, _, y_test = prepare_data_split(df, test_size=0.2, random_state=42)

    proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Load threshold from metadata if available
    threshold = 0.5
    if os.path.exists("models/model_metadata.json"):
        with open("models/model_metadata.json", "r") as f:
            meta = json.load(f)
            threshold = meta.get("classification_threshold", 0.5)

    preds = (proba >= threshold).astype(int)
    metrics = evaluate_predictions(y_test.values, preds, proba)

    print(f"\nEvaluation on Held-Out Test Set (Threshold: {threshold:.2f}):")
    print("-" * 45)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v}")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  {metrics['confusion_matrix']}")

    return metrics

if __name__ == "__main__":
    evaluate_saved_model()
