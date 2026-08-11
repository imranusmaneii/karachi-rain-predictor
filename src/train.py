"""Train and compare models for the Karachi rain predictor.

Phase 5: baseline Logistic Regression pipeline.
Phase 7: extended with Random Forest, Gradient Boosting and XGBoost, plus an
         honest comparison (each model judged at its own best decision
         threshold, chosen on training out-of-fold predictions only).

The saved artifact is the FULL pipeline (preprocessing + classifier), so
predictions always use the exact same transformations as training.

Usage:
    python src/train.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import evaluate_model, find_best_threshold
from split_data import load_splits

BASE_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODEL_DIR = BASE_DIR / "models"
BASELINE_PATH = MODEL_DIR / "karachi_rain_baseline.pkl"
BEST_PATH = MODEL_DIR / "karachi_rain_model.pkl"

NUMERIC_FEATURES: list[str] = [
    "MaxTemperature", "MinTemperature", "MeanTemperature",
    "Pressure", "Humidity", "CloudCoverage",
    "WindSpeed", "WindDirection", "Rainfall", "WeatherCode",
    "PreviousRainfall", "PreviousDayTemperature", "PreviousDayHumidity",
    "Month", "DayOfYear",
]
CATEGORICAL_FEATURES: list[str] = ["Season"]


def build_preprocessor() -> ColumnTransformer:
    """Shared preprocessing: scale numerics, one-hot encode Season."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Phase 5 baseline: linear model."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Random Forest: bagged deep trees, automatic feature interactions."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300, max_depth=12, min_samples_leaf=5,
                    class_weight="balanced", n_jobs=-1, random_state=42,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline() -> Pipeline:
    """Gradient Boosting: sequential additive trees."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float | None = None) -> Pipeline:
    """XGBoost: regularized gradient boosting with imbalance weighting."""
    params: dict = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", __import__("xgboost").XGBClassifier(**params)),
        ]
    )


def train_comparison() -> tuple[pd.DataFrame, dict]:
    """Train all four models, tune each threshold honestly, return (table, models)."""
    X_train, X_test, y_train, y_test = load_splits(SPLITS_DIR)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    scale_pos_weight = round((len(y_train) - y_train.sum()) / y_train.sum())
    builders = {
        "LogisticRegression": build_baseline_pipeline(),
        "RandomForest": build_random_forest_pipeline(),
        "GradientBoosting": build_gradient_boosting_pipeline(),
        "XGBoost": build_xgboost_pipeline(scale_pos_weight=scale_pos_weight),
    }

    rows = []
    for name, pipeline in builders.items():
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        default = evaluate_model(pipeline, X_test, y_test)  # threshold = 0.5
        tuned_threshold = find_best_threshold(pipeline, X_train, y_train)
        tuned = evaluate_model(pipeline, X_test, y_test, threshold=tuned_threshold)

        rows.append(
            {
                "model": name,
                "fit_s": round(fit_s, 1),
                "ROC-AUC": round(roc_auc, 3),
                # default 0.5 threshold
                "acc@0.5": round(default["accuracy"], 3),
                "prec@0.5": round(default["precision"], 3),
                "rec@0.5": round(default["recall"], 3),
                "f1@0.5": round(default["f1"], 3),
                # best threshold (OOF training predictions only)
                "threshold": tuned_threshold,
                "accuracy": round(tuned["accuracy"], 3),
                "precision": round(tuned["precision"], 3),
                "recall": round(tuned["recall"], 3),
                "f1": round(tuned["f1"], 3),
            }
        )
        print(f"  {name:20s} fitted in {fit_s:5.1f}s | OOF threshold {tuned_threshold:.2f}")

    table = pd.DataFrame(rows).set_index("model")
    table["threshold"] = table["threshold"].round(2)
    return table, builders


def main() -> None:
    """Train the comparison, report the table, save the winning model."""
    table, fitted = train_comparison()

    print("\nDefault 0.5 threshold:")
    print(table[["acc@0.5", "prec@0.5", "rec@0.5", "f1@0.5"]])

    print("\nEach model at its own best (F1-max) threshold:")
    print(table[["fit_s", "ROC-AUC", "threshold", "accuracy", "precision", "recall", "f1"]])

    best_name = table["f1"].idxmax()
    best_row = table.loc[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "threshold": float(best_row["threshold"]),
            "pipeline": fitted[best_name],
        },
        BEST_PATH,
    )
    print(f"\nWinner: {best_name} (F1 {best_row['f1']:.3f} @ threshold {best_row['threshold']:.2f})")
    print(f"Saved -> {BEST_PATH}")


if __name__ == "__main__":
    main()
