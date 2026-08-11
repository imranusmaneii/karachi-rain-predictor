"""Reusable data loading, validation and cleaning for the Karachi rain project.

This module codifies the decisions made in Phase 2 (see
notebooks/02_data_cleaning.ipynb). Key principles:

- Never silently drop rows: any decision is reported to the caller.
- Extreme weather values are kept unless proven wrong (they are real events).
- All validation thresholds are physically justified and documented.

Usage:
    from data_preprocessing import load_clean_data
    df = load_clean_data("data/raw/karachi_weather.csv")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Physically sensible ranges for each variable.
# Source: weather science + known Karachi climate.
# (None means no upper bound is enforced.)
SENSIBLE_RANGES: dict[str, tuple[float, float | None]] = {
    "Humidity": (0, 100),
    "CloudCoverage": (0, 100),
    "WindDirection": (0, 360),
    "WindSpeed": (0, None),
    "Pressure": (850, 1100),        # hPa, sea level
    "MaxTemperature": (-20, 60),    # deg C
    "MinTemperature": (-20, 60),    # deg C
    "Rainfall": (0, None),          # mm, cannot be negative
    "WeatherCode": (0, 99),         # WMO codes
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV with correct data types (Date as datetime)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check every column against its sensible range.

    Returns a report DataFrame (one row per checked column). Raises a
    RuntimeError if any value is physically impossible, because such data
    would be an error we must inspect rather than quietly fix.
    """
    report_rows = []
    for col, (low, high) in SENSIBLE_RANGES.items():
        values = df[col]
        violations = values < low if low is not None else pd.Series(False, index=values.index)
        if high is not None:
            violations = violations | (values > high)
        n_violations = int(violations.sum())
        report_rows.append(
            {
                "column": col,
                "min": df[col].min(),
                "max": df[col].max(),
                "violations": n_violations,
            }
        )
    report = pd.DataFrame(report_rows)

    total = int(report["violations"].sum())
    if total > 0:
        raise RuntimeError(
            f"Validation failed: {total} out-of-range values found.\n{report}"
        )
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the DataFrame.

    For this dataset cleaning is validation: the raw data already has no
    missing values, no duplicates, correct types and a complete date range.
    Extreme rainfall days are KEPT (they are real flood events).

    If future data contains problems, this function should be extended to
    handle them explicitly - never silently.
    """
    # 1. Missing values
    if df.isna().sum().sum() > 0:
        print(f"WARNING: {df.isna().sum().sum()} missing values remain in the data.")

    # 2. Duplicate rows / dates
    n_duplicates = int(df.duplicated().sum())
    n_dup_dates = int(df["Date"].duplicated().sum())
    if n_duplicates or n_dup_dates:
        raise RuntimeError(
            f"Duplicate rows: {n_duplicates}, duplicate dates: {n_dup_dates}. "
            "Remove duplicates before proceeding."
        )

    # 3. Physical plausibility
    report = validate_data(df)
    print("Validation passed - all values are physically plausible:")
    print(report.to_string(index=False))

    # 4. Chronological order (time series must be sorted)
    if not df["Date"].is_monotonic_increasing:
        df = df.sort_values("Date").reset_index(drop=True)
        print("Dates were re-sorted.")

    return df.copy()


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save the cleaned dataset to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows -> {path}")


def load_clean_data(raw_path: str | Path) -> pd.DataFrame:
    """Convenience wrapper: load raw, clean, and return the clean data."""
    df = load_data(raw_path)
    df = clean_data(df)
    return df


if __name__ == "__main__":
    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "karachi_weather.csv"
    out = Path(__file__).resolve().parents[1] / "data" / "processed" / "karachi_weather_clean.csv"
    clean = load_clean_data(raw)
    save_processed(clean, out)
