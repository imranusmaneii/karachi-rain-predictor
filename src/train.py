"""Train the baseline Logistic Regression pipeline and save it.

This is the training entry point for the baseline model (Phase 5).
Later phases extend this script with more models and hyperparameter tuning.

The saved artifact is the FULL pipeline (preprocessing + classifier), so
predictions always use the exact same transformations as training.

Usage:
    python src/train.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from split_data import FEATURE_COLUMNS, load_splits

BASE_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "karachi_rain_baseline.pkl"

NUMERIC_FEATURES: list[str] = [
    "MaxTemperature", "MinTemperature", "MeanTemperature",
    "Pressure", "Humidity", "CloudCoverage",
    "WindSpeed", "WindDirection", "Rainfall", "WeatherCode",
    "PreviousRainfall", "PreviousDayTemperature", "PreviousDayHumidity",
    "Month", "DayOfYear",
]
CATEGORICAL_FEATURES: list[str] = ["Season"]


def build_baseline_pipeline() -> Pipeline:
    """Build the full baseline pipeline (preprocessing + LogisticRegression)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return pipeline


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the core classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return metrics


def main() -> None:
    """Train the baseline, report metrics, and save the pipeline."""
    X_train, X_test, y_train, y_test = load_splits(SPLITS_DIR)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    print("\nBaseline metrics on the chronological test set:")
    for name, value in metrics.items():
        print(f"  {name:10s}: {value:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved baseline pipeline -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
