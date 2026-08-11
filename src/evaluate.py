"""Evaluation helpers for the Karachi rain predictor (Phase 6).

Codifies the evaluation logic from notebooks/06_model_evaluation.ipynb:

- Full metric set (not accuracy alone)
- Honest threshold selection via out-of-fold training predictions
  (the test set is touched only once, at the very end)
- Plots: confusion matrix, ROC curve, PR curve, threshold trade-off

Usage:
    from evaluate import evaluate_model, find_best_threshold
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import cross_val_predict


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return the probability of class 1 (Rain)."""
    return model.predict_proba(X)[:, 1]


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> dict:
    """Compute the full metric set at a given decision threshold."""
    y_prob = predict_proba(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def find_best_threshold(
    model, X_train: pd.DataFrame, y_train: pd.Series,
    thresholds: np.ndarray | None = None,
) -> float:
    """Choose the decision threshold that maximizes F1.

    Uses out-of-fold predictions on the TRAINING set (via cross_val_predict)
    so the test set is not used for any model decision.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    oof_prob = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]

    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (oof_prob >= t).astype(int)
        f1 = f1_score(y_train, pred)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def plot_confusion_matrix(
    y_test: pd.Series, y_pred: np.ndarray, save_path: str | Path | None = None
) -> None:
    """Plot a labelled confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", square=True,
        xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"],
    )
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    y_test: pd.Series, y_prob: np.ndarray, save_path: str | Path | None = None
) -> None:
    """Plot the ROC curve with its AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"Model (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_pr_curve(
    y_test: pd.Series, y_prob: np.ndarray, save_path: str | Path | None = None
) -> None:
    """Plot the precision-recall curve with the no-skill baseline."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2, label="Model")
    plt.axhline(y_test.mean(), color="k", ls="--", label=f"No-skill ({y_test.mean():.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # quick self-check on the saved baseline
    import joblib

    base = Path(__file__).resolve().parents[1]
    splits = base / "data" / "processed" / "splits"
    model = joblib.load(base / "models" / "karachi_rain_baseline.pkl")
    X_test = pd.read_csv(splits / "X_test.csv")
    y_test = pd.read_csv(splits / "y_test.csv").iloc[:, 0]
    print(evaluate_model(model, X_test, y_test))
