"""Feature importance and error analysis for the champion model (Phase 9).

Turns "why does the model work / fail" questions into reproducible code:

- transformed_feature_names(): the real feature names the model sees after the
  ColumnTransformer (numeric columns + one-hot expanded Season).
- impurity_importance(): GradientBoosting's built-in (mean-decrease-impurity)
  importances. Cheap, but computed on training fit and skewed toward
  high-frequency splits.
- permutation_importance_table(): honest "how much does this feature matter"
  by shuffling it and watching test metrics drop. More trustworthy, slower.
  Works on the RAW feature columns (16), the same unit the user thinks in —
  sklearn can only permute columns the estimator receives, i.e. before the
  preprocessor expands Season into four one-hot columns.
- error_frame(): tag every test row as hit / miss / false alarm / correct
  no-rain at the champion's threshold.
- error_breakdown(): error rates grouped by any column (Season, Month, Year...).

Usage:
    from analyze import error_frame, error_breakdown
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "karachi_rain_model.pkl"
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"

ERROR_LABELS: dict[str, str] = {
    (0, 0): "correct no-rain",
    (1, 1): "hit (rain caught)",
    (1, 0): "miss (FN)",
    (0, 1): "false alarm (FP)",
}


def transformed_feature_names(pipe) -> list[str]:
    """Names of the columns the model actually sees (post-preprocessing)."""
    pre = pipe.named_steps["preprocessor"]
    num_cols: list[str] = list(pre.transformers_[0][2])
    cat_encoder = pre.named_transformers_["cat"]
    cat_names: list[str] = []
    if hasattr(cat_encoder, "categories_"):
        for cats in cat_encoder.categories_:
            cat_names.extend(str(c) for c in cats)
    return num_cols + [f"Season_{c}" for c in cat_names]


def impurity_importance(pipe, feature_names: list[str]) -> pd.DataFrame:
    """GB's built-in mean-decrease-impurity importances (training fit)."""
    classifier = pipe.named_steps["classifier"]
    values = classifier.feature_importances_
    table = pd.DataFrame({"feature": feature_names, "importance": values})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    pipe, X: pd.DataFrame, y, n_repeats: int = 10,
    scoring: str = "roc_auc", random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance: drop in score when a raw feature is shuffled.

    Permutation happens on the raw input columns (X.columns), because the
    estimator receives those and sklearn permutes what it is handed.
    """
    result = permutation_importance(
        pipe, X, y, n_repeats=n_repeats, scoring=scoring,
        random_state=random_state, n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def error_frame(pipe, X: pd.DataFrame, y, threshold: float) -> pd.DataFrame:
    """Tag every row with its prediction and outcome type at a threshold."""
    prob = pipe.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    truth = y.reset_index(drop=True).astype(int).values
    pairs = list(zip(truth, pred))
    return pd.DataFrame(
        {
            "y_true": truth,
            "probability": prob,
            "prediction": pred,
            "type": [ERROR_LABELS[p] for p in pairs],
        }
    )


def error_breakdown(
    X: pd.DataFrame, err: pd.DataFrame, by: str
) -> pd.DataFrame:
    """Error rates of the model grouped by a column in X (Season, Month...)."""
    df = X.reset_index(drop=True).copy()
    df["y_true"] = err["y_true"].values
    df["prediction"] = err["prediction"].values

    rows = []
    for key, group in df.groupby(by, dropna=False):
        n = len(group)
        rainy = int((group["y_true"] == 1).sum())
        predicted_rain = int((group["prediction"] == 1).sum())
        misses = int(((group["y_true"] == 1) & (group["prediction"] == 0)).sum())
        false_alarms = int(((group["y_true"] == 0) & (group["prediction"] == 1)).sum())
        rows.append(
            {
                by: key,
                "days": n,
                "rainy": rainy,
                "rain_rate": rainy / n,
                "predicted_rain": predicted_rain,
                "misses": misses,
                "miss_rate_of_rain": misses / rainy if rainy else np.nan,
                "false_alarms": false_alarms,
                "fp_per_day": false_alarms / n,
            }
        )
    return pd.DataFrame(rows).set_index(by)


if __name__ == "__main__":
    import joblib

    import pandas as pd

    from evaluate import evaluate_model
    from split_data import load_splits

    model = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test = load_splits(SPLITS_DIR)
    threshold = model["threshold"]
    print("Champion:", model["model_name"], "@ threshold", threshold)
    print("Test metrics:", evaluate_model(model["pipeline"], X_test, y_test))
    print()
    print(impurity_importance(model["pipeline"], transformed_feature_names(model["pipeline"])))
    print()
    err = error_frame(model["pipeline"], X_test, y_test, threshold)
    print(err["type"].value_counts())
