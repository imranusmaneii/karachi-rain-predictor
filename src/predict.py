"""Make predictions with the champion model (Phase 10).

The champion artifact carries the FULL preprocessing pipeline (scaling + one-hot
Season) and its decision threshold, so predicting only needs the 16 raw features
listed in FEATURE_COLUMNS.

Usage:
    python src/predict.py                       # self-test on an example day
    from predict import load_model, predict

Example:
    >>> result = predict({"MaxTemperature": 33.0, ..., "Season": "Monsoon"})
    {'model_name': 'GradientBoosting', 'threshold': 0.25,
     'probability': 0.81, 'prediction': 1, 'label': 'Rain'}
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from split_data import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "karachi_rain_model.pkl"

RAIN_THRESHOLD: float = 0.25  # fallback if the artifact has no threshold

# A plausible monsoon evening in Karachi (used by the self-test and the app's
# "load example" button).
EXAMPLE_FEATURES: dict[str, float | str] = {
    "MaxTemperature": 33.0,
    "MinTemperature": 27.0,
    "MeanTemperature": 30.0,
    "Pressure": 1002.0,
    "Humidity": 78.0,
    "CloudCoverage": 70.0,
    "WindSpeed": 25.0,
    "WindDirection": 220.0,
    "Rainfall": 0.0,
    "WeatherCode": 3.0,
    "PreviousRainfall": 2.8,
    "PreviousDayTemperature": 31.0,
    "PreviousDayHumidity": 72.0,
    "Month": 7,
    "DayOfYear": 200,
    "Season": "Monsoon",
}


def load_model() -> dict:
    """Load the saved champion {model_name, threshold, pipeline}."""
    return joblib.load(MODEL_PATH)


def missing_features(features: dict) -> list[str]:
    """Which of the 16 required features are absent from `features`?"""
    return [column for column in FEATURE_COLUMNS if column not in features]


def predict(features: dict, model: dict | None = None) -> dict:
    """Predict rain probability for one day's features.

    Returns a plain dict ready for JSON serialization:
    {model_name, threshold, probability, prediction, label}.
    """
    if model is None:
        model = load_model()

    absent = missing_features(features)
    if absent:
        raise ValueError(f"Missing feature(s): {absent}")

    row = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
    probability = float(model["pipeline"].predict_proba(row)[0, 1])
    threshold = float(model.get("threshold", RAIN_THRESHOLD))
    is_rain = probability >= threshold

    return {
        "model_name": str(model.get("model_name", "unknown")),
        "threshold": threshold,
        "probability": round(probability, 4),
        "prediction": int(is_rain),
        "label": "Rain" if is_rain else "No rain",
    }


if __name__ == "__main__":
    result = predict(EXAMPLE_FEATURES)
    print(f"Model      : {result['model_name']}")
    print(f"Threshold  : {result['threshold']}")
    print(f"P(rain)    : {result['probability']:.2%}")
    print(f"Prediction : {result['label']}")
