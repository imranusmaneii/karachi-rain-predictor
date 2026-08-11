"""Download historical daily weather observations for Karachi, Pakistan.

Data source: Open-Meteo Historical Archive API
    https://open-meteo.com/en/docs/historical-weather-api

Why this source?
    - Free, no API key required
    - Real observations (ERA5 reanalysis) for Karachi, not invented data
    - Daily observations back to 1940
    - Provides temperature, humidity, pressure, wind, cloud and rainfall

We fetch daily data from 1985-01-01 until today.
This gives ~15,000 days of Karachi weather, including many monsoon cycles.

Output: data/raw/karachi_weather.csv
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

# Karachi, Pakistan coordinates
KARACHI_LAT: float = 24.8607
KARACHI_LON: float = 67.0011

# Fetch from 1985 until today (plenty of history for training)
START_DATE: str = "1985-01-01"

# Daily weather variables available from Open-Meteo.
# NOTE: Open-Meteo does NOT provide a visibility variable in its daily
# archive, so we document that feature as unavailable instead of faking it.
DAILY_VARIABLES: list[str] = [
    "temperature_2m_max",     # max temperature (deg C)
    "temperature_2m_min",     # min temperature (deg C)
    "temperature_2m_mean",    # mean temperature (deg C)
    "precipitation_sum",      # total precipitation in mm
    "rain_sum",               # rain amount in mm (Karachi: ~precipitation_sum)
    "pressure_msl_mean",      # mean sea-level pressure (hPa)
    "relative_humidity_2m_mean",  # mean relative humidity (%)
    "cloud_cover_mean",       # mean cloud cover (%)
    "wind_speed_10m_max",     # max wind speed 10 m above ground (km/h)
    "wind_direction_10m_dominant",  # dominant wind direction (deg)
    "weather_code",           # WMO weather condition code
]

# Map API variable names to clean, beginner-friendly feature names.
# The new names are what the rest of the project will use.
RENAME: dict[str, str] = {
    "time": "Date",
    "temperature_2m_max": "MaxTemperature",
    "temperature_2m_min": "MinTemperature",
    "temperature_2m_mean": "MeanTemperature",
    "precipitation_sum": "Precipitation",
    "rain_sum": "Rainfall",
    "pressure_msl_mean": "Pressure",
    "relative_humidity_2m_mean": "Humidity",
    "cloud_cover_mean": "CloudCoverage",
    "wind_speed_10m_max": "WindSpeed",
    "wind_direction_10m_dominant": "WindDirection",
    "weather_code": "WeatherCode",
}


def build_url() -> str:
    """Construct the Open-Meteo archive API URL for Karachi."""
    end_date: str = dt.date.today().isoformat()
    variables: str = ",".join(DAILY_VARIABLES)
    return (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={KARACHI_LAT}"
        f"&longitude={KARACHI_LON}"
        f"&start_date={START_DATE}"
        f"&end_date={end_date}"
        f"&daily={variables}"
        f"&timezone=Asia%2FKarachi"  # use Karachi local days
    )


def download_data() -> pd.DataFrame:
    """Download daily weather data and return it as a clean DataFrame."""
    url: str = build_url()
    print(f"Requesting data from:\n{url}\n")

    response: requests.Response = requests.get(url, timeout=60)
    response.raise_for_status()  # fail loudly if the API call fails
    payload = response.json()

    if "daily" not in payload:
        raise RuntimeError(f"Unexpected API response: {payload.get('reason') or payload}")

    df: pd.DataFrame = pd.DataFrame(payload["daily"])

    # Keep only the variables we asked for, renamed to clean feature names
    df = df.rename(columns=RENAME)

    # Convert to proper data types
    df["Date"] = pd.to_datetime(df["Date"])

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the dataset to CSV and report basic stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows -> {output_path}")


def main() -> None:
    """Download the Karachi weather dataset and save it to disk."""
    raw_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw"
    output_path: Path = raw_dir / "karachi_weather.csv"

    df: pd.DataFrame = download_data()
    save_data(df, output_path)

    # Quick sanity check of what we just downloaded
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Missing values per column:\n{df.isna().sum()}")


if __name__ == "__main__":
    main()
