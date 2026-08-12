# Karachi Rain Predictor — Project Todo / Progress Log

Repo: `https://github.com/imranusmaneii/karachi-rain-predictor` (branch `main`)
Work dir: `G:\Machine Learning Models\1st Model` (Windows, PowerShell)

## Goal
Predict whether it will rain in Karachi tomorrow (Rain/No Rain + probability) from real
historical weather. Educational, phase-by-phase, classical ML only (no TF/PyTorch).

## Environment notes
- Project venv: `.venv\Scripts\python.exe`. Jupyter kernel: `karachi-rain`.
- Python ssl/urllib3 fails on large HTTPS downloads (`SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`);
  `curl.exe` works. New pip packages must be installed offline from wheels in
  `C:\Users\IMRAN\AppData\Local\Temp\opencode\wheels` (see `fetch_wheels.py` + `requirements_pinned.txt`).
- xgboost 3.4.0 was installed this way (wheel was 48.9 MB).
- Notebook execution: `& ".venv\Scripts\python.exe" -m jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`
  Notebook builders live in `C:\Users\IMRAN\AppData\Local\Temp\opencode\build_*.py`.
- Model files `models/*.pkl` and `reports/figures/*.png` are gitignored (not pushed).

## Data
- Source: Open-Meteo Historical Archive API, Karachi (24.8607, 67.0011), tz Asia/Karachi,
  1985-01-01 to 2026-08-11.
- Raw: `data/raw/karachi_weather.csv` — 15,198 rows x 12 cols, no nulls/duplicates.
  No visibility variable exists in Open-Meteo daily archive (documented unavailable).
- Cleaned: `data/processed/karachi_weather_clean.csv` (0 range violations).
- Features: `data/processed/karachi_weather_features.csv` — 15,196 rows x 19 cols.
  Rainy day = Rainfall > 0 mm. Target `RainTomorrow` from shift(-1); last-row target NaN dropped.
  Leakage audit (`src/feature_engineering.py`) passes 3 checks.
- Split (chronological, no shuffle): train 1985-01-02..2018-04-14 (12,156 rows, rain 10.4%),
  test 2018-04-15..2026-08-10 (3,040 rows, rain 24.8%). split_idx = 12156.
  Saved: `data/processed/splits/{X_train,X_test,y_train,y_test}.csv`.

## Phases completed (notebooks 01-09 pushed to GitHub)
1. Data exploration — `notebooks/01_data_exploration.ipynb`
2. Data cleaning — `notebooks/02_data_cleaning.ipynb` + `src/data_preprocessing.py`
3. Feature engineering — `notebooks/03_feature_engineering.ipynb` + `src/feature_engineering.py`
4. Train/test split — `notebooks/04_train_test_split.ipynb` + `src/split_data.py`
5. Baseline Logistic Regression — `notebooks/05_baseline_logistic_regression.ipynb` + `src/train.py`
   (Acc 0.848 / Prec 0.828 / Rec 0.491 / F1 0.616 / AUC 0.917 at default 0.5)
6. Deep evaluation — `notebooks/06_model_evaluation.ipynb` + `src/evaluate.py`
   Threshold tuning via training OOF predictions: best threshold 0.25 →
   test Acc 0.870 / Prec 0.749 / Rec 0.719 / F1 0.733.
7. Model comparison — `notebooks/07_compare_models.ipynb` (updated `src/train.py`, added xgboost to requirements).
   Each model at its own tuned threshold:
   - GradientBoosting: AUC 0.920, thr 0.25, Prec 0.734 / Rec 0.752 / F1 0.743  <-- winner
   - RandomForest:     AUC 0.924, thr 0.60, Prec 0.719 / Rec 0.756 / F1 0.737
   - LogisticRegression: AUC 0.917, thr 0.25, Prec 0.749 / Rec 0.719 / F1 0.733
   - XGBoost:          AUC 0.922, thr 0.70, Prec 0.736 / Rec 0.725 / F1 0.731
   Champion saved: `models/karachi_rain_model.pkl` = {model_name, threshold, pipeline} (gitignored).

## Phase 8 — COMPLETE (hyperparameter tuning, committed)
- `src/tune.py` — randomized CV search (ROC-AUC, 3-fold StratifiedKFold) for GradientBoosting
  and XGBoost, OOF threshold tuning (Phase 6 protocol), one-time test evaluation.
- `notebooks/08_hyperparameter_tuning.ipynb` executed (nbconvert, kernel karachi-rain).
  Earlier silent failures were just the tool's 30-min timeout — the run needs ~32 min;
  rerun with `--ExecutePreprocessor.timeout=1800` and a >45 min shell timeout.
- Results (one-time test eval, F1-max thresholds):
  - GB best CV ROC-AUC 0.8891, params {lr 0.03, depth 5, leaf 20, trees 369, subsample 1.0},
    thr 0.25 → test F1 0.729
  - XGB best CV ROC-AUC 0.8928, params {lr 0.03, depth 6, mcw 5, trees 271, subsample 0.7,
    colsample 1.0}, thr 0.75 → test F1 0.711
- Honest finding: tuning did NOT beat the Phase 7 defaults (tuned GB 0.729 vs Phase 7 GB 0.743).
  Champion was therefore restored to the Phase 7 GradientBoosting via `python src/train.py`
  (test F1 0.743 @ threshold 0.25) → `models/karachi_rain_model.pkl`. Noted in notebook 08.
- Phase 4 split CSVs (`data/processed/splits/*.csv`) now committed (were untracked).

## Phase 9 — COMPLETE (feature importance & error analysis, committed)
- `src/analyze.py` — feature names (post-transform), impurity & permutation importance,
  error tagging (`error_frame`), grouped error rates (`error_breakdown`).
- `notebooks/09_feature_importance_error_analysis.ipynb` executed (15 code cells, 4 figures).
- Champion recap (GB @ thr 0.25): Acc 0.871 / Prec 0.734 / Rec 0.752 / F1 0.743.
- Feature importance — the two measures DISAGREE (the key lesson):
  - Impurity: WeatherCode 0.296, Rainfall 0.229, CloudCoverage 0.135 dominate.
  - Permutation (test, 5 reps): MinTemperature (AUC drop 0.060), CloudCoverage (0.057),
    Rainfall (0.016), Pressure (0.015). WeatherCode falls to #7 (0.010) — its impurity
    score is inflated by being a many-valued code; temperature/cloud/pressure carry
    the same signal. Month & Season are near-useless (Season_* all < 0.002).
- Error analysis (test, 3,040 days; 754 rainy / 2,286 dry → 567 TP / 187 FN / 206 FP / 2080 TN):
  - Monsoon months: low miss rate (Jul 8% / Aug 13%) but MOST false alarms (Jul 17.6%,
    Aug 14.7%, Jun+Sep ~15% of days). Over-alerting exactly when it rains a lot.
  - Dry months: rain is rare and the model MISSES most of it (Mar-Apr miss 75-80% of
    rainy days; Nov misses 100% of its 5). Off-season rain looks like ordinary days.
  - Missed days look like *partly* cloudy (CloudCoverage 40 vs 69 for caught) — no clear
    signal; false alarms look monsoon-like (humidity 76, cloud 62) but rain never fell.
  - No strong temporal drift: miss rate 24% (2018) ... 36% (2026, partial year), no trend.

## Next phases (planned, not started)
- Phase 10+: src/predict.py, FastAPI `api/main.py` (POST /predict), Streamlit `app/app.py`,
  README.md.

## Git log (pushed to main)
e8b9416 .. c7d5337 (12 commits) then:
- 6d72062 feat: deep evaluation notebook + evaluate.py
- 5f25d14 feat: 4-model comparison (GradientBoosting wins), train.py + xgboost dep
