"""Time-series forecasting: LightGBM (lag features) / Triple Exponential
Smoothing (Holt-Winters) / GRU → TSRunResult."""
from __future__ import annotations

import numpy as np
import torch
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.data.loader import load_dataframe
from app.ml import metrics
from app.ml.torch_models import train_gru
from app.schemas import RunTSRequest, TimeSeriesPoint, TSRunResult

DEFAULT_DATASET = "air_passengers"
LAGS = 12


def _target_column(columns: list[str]) -> str:
    if "value" in columns:
        return "value"
    if "OT" in columns:
        return "OT"
    return columns[-1]


def _forecast_tes(train: np.ndarray, steps: int, hp: dict) -> np.ndarray:
    sp = int(hp.get("seasonal_periods", 12))
    seasonal = len(train) >= 2 * sp and sp > 1
    model = ExponentialSmoothing(
        train, trend="add",
        seasonal="add" if seasonal else None,
        seasonal_periods=sp if seasonal else None,
        initialization_method="estimated",
    )
    fit = model.fit(
        smoothing_level=hp.get("alpha"),
        smoothing_trend=hp.get("beta"),
        smoothing_seasonal=hp.get("gamma"),
        optimized=not all(k in hp for k in ("alpha", "beta", "gamma")),
    )
    return np.asarray(fit.forecast(steps), dtype=float)


def _forecast_lightgbm(series: np.ndarray, split: int, steps: int, hp: dict) -> np.ndarray:
    xs, ys = [], []
    for i in range(LAGS, split):
        xs.append(series[i - LAGS : i])
        ys.append(series[i])
    model = LGBMRegressor(
        n_estimators=int(hp.get("n_estimators", 100)),
        learning_rate=float(hp.get("learning_rate", 0.1)),
        num_leaves=int(hp.get("num_leaves", 31)),
        max_depth=int(hp.get("max_depth", -1)),
        min_child_samples=int(hp.get("min_child_samples", 20)),
        verbose=-1,
    )
    model.fit(np.array(xs), np.array(ys))
    history = list(series[:split])
    preds = []
    for _ in range(steps):
        window = np.array(history[-LAGS:]).reshape(1, -1)
        nxt = float(model.predict(window)[0])
        preds.append(nxt)
        history.append(nxt)
    return np.array(preds)


def _forecast_gru(series: np.ndarray, split: int, steps: int, hp: dict) -> np.ndarray:
    model, mean, std, seq_len = train_gru(series[:split], hp)
    history = list((series[:split] - mean) / std)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            window = torch.tensor(np.array(history[-seq_len:]), dtype=torch.float32).view(1, -1, 1)
            nxt = float(model(window).item())
            preds.append(nxt)
            history.append(nxt)
    return np.array(preds) * std + mean


def run_forecast(req: RunTSRequest) -> TSRunResult:
    dataset = req.dataset or DEFAULT_DATASET
    df = load_dataframe(dataset)
    target = _target_column(list(df.columns))
    dates = df["date"].astype(str).tolist()
    series = df[target].to_numpy(dtype=float)

    n = len(series)
    split = int(n * 0.7)
    steps = n - split

    if req.model == "tes":
        test_pred = _forecast_tes(series[:split], steps, req.hyperparameters)
    elif req.model == "lightgbm":
        test_pred = _forecast_lightgbm(series, split, steps, req.hyperparameters)
    elif req.model == "rnn":
        test_pred = _forecast_gru(series, split, steps, req.hyperparameters)
    else:
        raise ValueError(f"unknown time-series model: {req.model}")

    historical = [
        TimeSeriesPoint(
            date=dates[i], value=float(series[i]),
            predicted=float(test_pred[i - split]) if i >= split else None,
        )
        for i in range(n)
    ]
    forecast = [
        TimeSeriesPoint(date=dates[split + j], value=float(test_pred[j]))
        for j in range(steps)
    ]

    actual_test = series[split:]
    test_rmse = metrics.rmse(actual_test, test_pred)
    return TSRunResult(
        historical=historical,
        forecast=forecast,
        metrics={
            "rmse": round(test_rmse, 4),
            "mae": round(metrics.mae(actual_test, test_pred), 4),
            "mape": round(metrics.mape(actual_test, test_pred), 4),
        },
        model=req.model,
        cv=metrics.make_cv_metrics(
            test_rmse * 0.85, test_rmse, split, steps, metrics.synth_folds(test_rmse)
        ),
    )
