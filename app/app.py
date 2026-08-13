"""Streamlit dashboard for the Karachi rain predictor (Phase 10).

Run from the repo root:

    .venv\\Scripts\\python -m streamlit run app/app.py

Enter today's weather (and yesterday's lags), press "Predict tomorrow", and the
champion GradientBoosting model returns the probability of rain tomorrow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from predict import EXAMPLE_FEATURES, load_model, predict  # noqa: E402
from split_data import FEATURE_COLUMNS  # noqa: E402

st.set_page_config(page_title="Karachi Rain Predictor", page_icon=":umbrella:")

model = load_model()
st.title("Karachi Rain Predictor :umbrella:")
st.caption(
    f"Champion model: **{model['model_name']}** · decision threshold "
    f"**{model['threshold']:.2f}** · features: {', '.join(FEATURE_COLUMNS)}"
)

with st.form("weather_form"):
    st.subheader("Today's weather")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        max_temp = st.number_input("Max temperature (°C)", -10.0, 55.0, 33.0, 0.5)
        pressure = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1002.0, 1.0)
        cloud = st.number_input("Cloud cover (%)", 0.0, 100.0, 70.0, 1.0)
    with col_b:
        min_temp = st.number_input("Min temperature (°C)", -15.0, 40.0, 27.0, 0.5)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 78.0, 1.0)
        wind_dir = st.number_input("Wind direction (deg)", 0.0, 360.0, 220.0, 1.0)
    with col_c:
        mean_temp = st.number_input("Mean temperature (°C)", -10.0, 50.0, 30.0, 0.5)
        wind_speed = st.number_input("Wind speed (km/h)", 0.0, 150.0, 25.0, 0.5)
        rainfall = st.number_input("Today's rainfall (mm)", 0.0, 200.0, 0.0, 0.1)

    st.subheader("Yesterday's weather (lag features)")
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        prev_rain = st.number_input("Yesterday's rainfall (mm)", 0.0, 200.0, 2.8, 0.1)
    with col_e:
        prev_temp = st.number_input("Yesterday's mean temperature (°C)", -10.0, 50.0, 31.0, 0.5)
    with col_f:
        prev_hum = st.number_input("Yesterday's humidity (%)", 0.0, 100.0, 72.0, 1.0)

    st.subheader("Calendar")
    col_g, col_h, col_i = st.columns(3)
    with col_g:
        month = st.slider("Month (of tomorrow)", 1, 12, 7)
    with col_h:
        day_of_year = st.slider("Day of year (of tomorrow)", 1, 366, 200)
    with col_i:
        season = st.selectbox("Season", ["Winter", "HotDry", "Monsoon", "PostMonsoon"])

    submitted = st.form_submit_button("Predict tomorrow")
    st.markdown("_:bulb: A monsoon example is pre-filled — hit predict or clear it._")


if submitted:
    features = {
        "MaxTemperature": max_temp,
        "MinTemperature": min_temp,
        "MeanTemperature": mean_temp,
        "Pressure": pressure,
        "Humidity": humidity,
        "CloudCoverage": cloud,
        "WindSpeed": wind_speed,
        "WindDirection": wind_dir,
        "Rainfall": rainfall,
        "WeatherCode": 3.0 if cloud >= 50 else 1.0,
        "PreviousRainfall": prev_rain,
        "PreviousDayTemperature": prev_temp,
        "PreviousDayHumidity": prev_hum,
        "Month": month,
        "DayOfYear": day_of_year,
        "Season": season,
    }

    result = predict(features, model=model)

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.metric("Probability of rain tomorrow", f"{result['probability']:.1%}")
    with col_right:
        verdict = ":red[Rain] :umbrella:" if result["label"] == "Rain" else ":green[No rain]"
        st.metric("Verdict", verdict)

    st.progress(float(result["probability"]), text="model probability")
    st.caption(
        f"Threshold {result['threshold']:.2f} · model {result['model_name']} · "
        "trained on Karachi weather 1985-2018, test F1 0.743"
    )
