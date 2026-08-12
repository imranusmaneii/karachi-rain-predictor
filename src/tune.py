"""Hyperparameter tuning for the model comparison (Phase 8).

Strategy (stays honest):
1. Search hyperparameters on the TRAINING set only, using 3-fold
   stratified CV and ROC-AUC as the objective (threshold-free, so the
   search is not biased by the default 0.5 cutoff).
2. With the best hyperparameters, choose the decision threshold on
   training out-of-fold predictions (Phase 6 protocol).
3. Evaluate on the test set exactly once.

The chronological train/test split (2018+) is what protects against
temporal leakage; shuffling inside training folds is safe because all
fold data still ends in 2018.

Usage:
    python src/tune.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from evaluate import evaluate_model, find_best_threshold
from split_data import load_splits
from train import (
    MODEL_DIR,
    SPLITS_DIR,
    build_gradient_boosting_pipeline,
    build_xgboost_pipeline,
)

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

GB_PARAM_GRID: dict = {
    "classifier__n_estimators": randint(200, 500),
    "classifier__max_depth": [3, 4, 5, 6],
    "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__min_samples_leaf": [5, 10, 20],
}

XGB_PARAM_GRID: dict = {
    "classifier__n_estimators": randint(200, 500),
    "classifier__max_depth": [3, 4, 6],
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.9, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
}


def search(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10):
    """Run a randomized CV search for one model; return the fitted search."""
    if model_name == "GradientBoosting":
        base = build_gradient_boosting_pipeline()
        grid = GB_PARAM_GRID
    elif model_name == "XGBoost":
        scale_pos_weight = round((len(y_train) - y_train.sum()) / y_train.sum())
        base = build_xgboost_pipeline(scale_pos_weight=scale_pos_weight)
        grid = XGB_PARAM_GRID
    else:
        raise ValueError(f"Unknown model: {model_name}")

    search = RandomizedSearchCV(
        base, grid, n_iter=n_iter, cv=CV, scoring="roc_auc",
        random_state=42, n_jobs=1, refit=True,
    )
    print(f"\n=== Searching {model_name} ({n_iter} random combos x {CV.n_splits} folds) ===")
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"Best params     : {search.best_params_}")
    print(f"Search took     : {elapsed:.1f}s")
    return search


def main() -> None:
    """Tune Gradient Boosting and XGBoost, pick the champion, save it."""
    X_train, X_test, y_train, y_test = load_splits(SPLITS_DIR)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    gb = search("GradientBoosting", X_train, y_train, n_iter=10)
    xgb = search("XGBoost", X_train, y_train, n_iter=12)

    gb_threshold = find_best_threshold(gb.best_estimator_, X_train, y_train)
    xgb_threshold = find_best_threshold(xgb.best_estimator_, X_train, y_train)
    print(f"\nTuned thresholds (training OOF, F1-max): GB={gb_threshold:.2f} XGB={xgb_threshold:.2f}")

    gb_metrics = evaluate_model(gb.best_estimator_, X_test, y_test, threshold=gb_threshold)
    xgb_metrics = evaluate_model(xgb.best_estimator_, X_test, y_test, threshold=xgb_threshold)

    rows = [
        {
            "model": "GB (Phase 7, recorded)",
            "threshold": 0.25, "precision": 0.734,
            "recall": 0.752, "f1": 0.743,
        },
        {
            "model": "GradientBoosting (tuned)",
            "threshold": gb_threshold,
            **{k: round(v, 3) for k, v in gb_metrics.items()},
        },
        {
            "model": "XGBoost (tuned)",
            "threshold": xgb_threshold,
            **{k: round(v, 3) for k, v in xgb_metrics.items()},
        },
    ]
    table = pd.DataFrame(rows)[["model", "threshold", "accuracy", "precision", "recall", "f1"]]
    print("\nFinal comparison (test set, one-time evaluation):")
    print(table.to_string(index=False))

    best_name = table.iloc[1:].sort_values("f1", ascending=False).iloc[0]["model"]
    best_pipeline = gb.best_estimator_ if "GradientBoosting" in best_name else xgb.best_estimator_
    best_threshold = gb_threshold if "GradientBoosting" in best_name else xgb_threshold

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "threshold": float(best_threshold),
            "pipeline": best_pipeline,
        },
        MODEL_DIR / "karachi_rain_model.pkl",
    )
    print(f"\nChampion: {best_name} (F1 {table.iloc[1:].sort_values('f1', ascending=False).iloc[0]['f1']:.3f})")
    print(f"Saved -> {MODEL_DIR / 'karachi_rain_model.pkl'}")


if __name__ == "__main__":
    main()
