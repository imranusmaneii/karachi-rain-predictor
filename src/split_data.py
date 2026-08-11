"""Chronological (time-based) train/test split for the Karachi dataset.

Why chronological and not random?
    Weather is a time series: nearby days are similar (autocorrelation).
    A random shuffle can put near-identical days in both training and test,
    inflating scores (a subtle form of leakage). A chronological split mimics
    real use: train on the past, evaluate on the recent past.

This module codifies Phase 4 (see notebooks/04_train_test_split.ipynb).

Functions:
    get_feature_columns(): the 16 features chosen for modeling
    split_chronological(df, test_fraction=0.2): returns X/y train+test
    save_splits(...): persist the split CSVs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET: str = "RainTomorrow"

FEATURE_COLUMNS: list[str] = [
    # today's observations (known by the end of today)
    "MaxTemperature", "MinTemperature", "MeanTemperature",
    "Pressure", "Humidity", "CloudCoverage",
    "WindSpeed", "WindDirection", "Rainfall", "WeatherCode",
    # yesterday's observations (lag features, always in the past)
    "PreviousRainfall", "PreviousDayTemperature", "PreviousDayHumidity",
    # calendar features (known in advance)
    "Month", "DayOfYear", "Season",
]


def split_chronological(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a date-sorted DataFrame into train (old) and test (recent).

    Returns X_train, X_test, y_train, y_test.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    X_train = df.loc[: split_idx - 1, FEATURE_COLUMNS].reset_index(drop=True)
    X_test = df.loc[split_idx:, FEATURE_COLUMNS].reset_index(drop=True)
    y_train = df.loc[: split_idx - 1, TARGET].reset_index(drop=True)
    y_test = df.loc[split_idx:, TARGET].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: str | Path,
) -> None:
    """Save the four split arrays to CSV files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)
    print(f"Saved splits -> {out_dir}")


def load_splits(split_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load previously saved splits (keeps every experiment comparable)."""
    split_dir = Path(split_dir)
    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(split_dir / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    features_path = base / "data" / "processed" / "karachi_weather_features.csv"
    splits_dir = base / "data" / "processed" / "splits"

    data = pd.read_csv(features_path, parse_dates=["Date"])
    result = split_chronological(data)
    X_train, X_test, y_train, y_test = result
    save_splits(X_train, X_test, y_train, y_test, splits_dir)

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
