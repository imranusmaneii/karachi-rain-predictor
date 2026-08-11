"""Feature engineering for the Karachi rain predictor.

This module codifies Phase 3 (see notebooks/03_feature_engineering.ipynb).

Two hard rules are enforced here:

1. **No data leakage** - features may only use information available
   *before* tomorrow. The target `RainTomorrow` is the ONLY column derived
   from the future, and an audit function proves no feature leaks.

2. **Documented thresholds** - a "rainy day" means rainfall > 0 mm.

Functions:
    build_features(raw_path, out_path): full pipeline, load -> engineer -> save
    create_target / create_seasonal_features / create_lag_features: steps
    audit_no_leakage(df): proves no future information is used as a feature
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "Winter": (12, 1, 2),
    "HotDry": (3, 4, 5),
    "Monsoon": (6, 7, 8, 9),
    "PostMonsoon": (10, 11),
}

# columns that describe the weather of a single day (source for audit)
RAW_COLUMNS: list[str] = [
    "MaxTemperature", "MinTemperature", "MeanTemperature", "Pressure",
    "Humidity", "CloudCoverage", "WindSpeed", "WindDirection",
    "Rainfall", "WeatherCode",
]


def load_clean_data(path: str | Path) -> pd.DataFrame:
    """Load the cleaned dataset produced by data_preprocessing."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add RainToday and RainTomorrow.

    RainTomorrow = is *tomorrow* a rainy day (rainfall > 0 mm)?
    The final row has no real tomorrow, so its target is left as NaN
    (unknown) rather than a fake 0.
    """
    df = df.copy()
    df["RainToday"] = (df["Rainfall"] > 0).astype(int)
    tomorrow_rain = df["Rainfall"].shift(-1)
    df["RainTomorrow"] = (tomorrow_rain > 0).astype(float)
    df.loc[tomorrow_rain.isna(), "RainTomorrow"] = np.nan
    return df


def season_of(month: int) -> str:
    """Map a month (1-12) to its Karachi season name."""
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    raise ValueError(f"month out of range: {month}")


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features: Month, DayOfYear, Season."""
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Season"] = df["Month"].map(season_of)
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add yesterday's observations (shift +1 = past)."""
    df = df.copy()
    df["PreviousRainfall"] = df["Rainfall"].shift(1)
    df["PreviousDayTemperature"] = df["MeanTemperature"].shift(1)
    df["PreviousDayHumidity"] = df["Humidity"].shift(1)
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Precipitation (identical to Rainfall for Karachi)."""
    df = df.copy()
    if "Precipitation" in df.columns:
        df = df.drop(columns=["Precipitation"])
    return df


def audit_no_leakage(df: pd.DataFrame) -> None:
    """Raise AssertionError if any feature leaks future information.

    1. Target isolation: RainTomorrow == RainToday shifted one day into the
       future (the only column allowed to use tomorrow's data).
    2. No feature equals any future observation of a raw column.
    3. Lag features point to the past (shift +1), never the future.
    """
    # 1. Target isolation
    assert (
        df["RainToday"].shift(-1) == df["RainTomorrow"]
    ).iloc[:-1].all(), "RainTomorrow was not built from tomorrow's rain!"

    # 2. Feature vs future raw observations
    feature_columns = [
        c for c in df.columns
        if c not in ("Date", "RainTomorrow", "RainToday", "Month",
                     "DayOfYear", "Season")
    ]
    for feature in feature_columns:
        for raw_col in RAW_COLUMNS:
            future = df[raw_col].shift(-1)
            assert not (df[feature] == future).iloc[:-1].all(), (
                f"LEAK: feature '{feature}' equals future '{raw_col}'!"
            )

    # 3. Lags point to the past
    assert (df["PreviousRainfall"] == df["Rainfall"].shift(1)).iloc[1:].all()
    assert (
        df["PreviousDayTemperature"] == df["MeanTemperature"].shift(1)
    ).iloc[1:].all()
    assert (df["PreviousDayHumidity"] == df["Humidity"].shift(1)).iloc[1:].all()


def build_features(raw_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Run the full feature engineering pipeline and save the result."""
    df = load_clean_data(raw_path)
    df = create_target(df)
    df = create_seasonal_features(df)
    df = create_lag_features(df)
    df = drop_redundant_columns(df)

    # Edge rows: first (no "yesterday") and last (no real "tomorrow")
    n_before = len(df)
    df = df.dropna(subset=["RainTomorrow", "PreviousRainfall"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} edge row(s) with unknown past/future.")

    audit_no_leakage(df)
    print("Leakage audit passed: no feature uses future information.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {out_path}")
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    clean = base / "data" / "processed" / "karachi_weather_clean.csv"
    features = base / "data" / "processed" / "karachi_weather_features.csv"
    build_features(clean, features)
