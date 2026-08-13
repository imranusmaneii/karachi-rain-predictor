"""FastAPI service for the Karachi rain predictor (Phase 10).

Run from the repo root:

    .venv\\Scripts\\python -m uvicorn api.main:app --reload

Endpoints:
    GET  /health   -> model name + decision threshold
    POST /predict  -> {"probability": 0.45, "prediction": 1, "label": "Rain", ...}

The request body carries the 16 features the model needs (the champion pipeline
handles scaling and Season one-hot encoding internally).
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from predict import load_model, predict  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    MODEL = load_model()
    yield


app = FastAPI(title="Karachi Rain Predictor", version="1.0.0", lifespan=lifespan)
MODEL: dict | None = None


class WeatherFeatures(BaseModel):
    """Today's observations + yesterday's lags + calendar (all 16 features)."""

    MaxTemperature: float = Field(description="Today's max temperature (deg C)")
    MinTemperature: float = Field(description="Today's min temperature (deg C)")
    MeanTemperature: float = Field(description="Today's mean temperature (deg C)")
    Pressure: float = Field(description="Today's mean sea-level pressure (hPa)")
    Humidity: float = Field(description="Today's mean relative humidity (%)")
    CloudCoverage: float = Field(description="Today's cloud cover (%)")
    WindSpeed: float = Field(description="Today's mean wind speed (km/h)")
    WindDirection: float = Field(description="Today's mean wind direction (deg)")
    Rainfall: float = Field(description="Today's rainfall (mm)")
    WeatherCode: float = Field(description="Open-Meteo weather code for today")
    PreviousRainfall: float = Field(description="Yesterday's rainfall (mm)")
    PreviousDayTemperature: float = Field(description="Yesterday's mean temperature (deg C)")
    PreviousDayHumidity: float = Field(description="Yesterday's mean humidity (%)")
    Month: int = Field(ge=1, le=12, description="Month of tomorrow's date (1-12)")
    DayOfYear: int = Field(ge=1, le=366, description="Day of year of tomorrow's date")
    Season: str = Field(description="Karachi season: Winter, HotDry, Monsoon, PostMonsoon")


class PredictionResponse(BaseModel):
    model_name: str
    threshold: float
    probability: float
    prediction: int
    label: str


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Report the loaded model and its decision threshold."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL["model_name"], "threshold": MODEL["threshold"]}


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict_endpoint(features: WeatherFeatures) -> PredictionResponse:
    """Predict the chance of rain tomorrow from one day's features."""
    try:
        result = predict(features.model_dump(), model=MODEL)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PredictionResponse(**result)
