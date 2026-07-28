"""ML / Time-Series playground routes under /ml/v1.

HTTP routes run a model and return the frontend's expected result shapes.
WebSocket routes stream optimization trials and NN training epochs
(CLAUDE.md: websockets for optimization logic and NN epochs)."""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Callable

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.data.loader import DatasetNotFound, feature_columns, load_dataframe, seeded_split
from app.db import connect
from app.ml.classification import run_classification
from app.ml.clustering import run_clustering
from app.ml.regression import run_regression
from app.ml.timeseries import run_forecast
from app.ml.torch_models import train_gru, train_mlp
from app.optim.optimize import optimize
from app.schemas import (
    MLRunResult,
    OptimizeRequest,
    OptimizeResult,
    RunMLRequest,
    RunTSRequest,
    TSRunResult,
)


def require_internal(x_internal_secret: str = Header(default="")) -> None:
    """When INTERNAL_API_SECRET is set, ML HTTP routes require the matching
    header — enforcing 'ML exclusively accessible by my frontend' (CLAUDE.md)."""
    secret = get_settings().internal_secret
    if secret and x_internal_secret != secret:
        raise HTTPException(status_code=403, detail="forbidden")


router = APIRouter(prefix="/ml/v1", tags=["ml"], dependencies=[Depends(require_internal)])

# ─── run endpoints ────────────────────────────────────────────────────────────


def _handle(fn: Callable, req):
    try:
        return fn(req)
    except DatasetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/run-regression", response_model=MLRunResult, response_model_exclude_none=True)
def post_run_regression(req: RunMLRequest) -> MLRunResult:
    return _handle(run_regression, req)


@router.post("/run-classification", response_model=MLRunResult, response_model_exclude_none=True)
def post_run_classification(req: RunMLRequest) -> MLRunResult:
    return _handle(run_classification, req)


@router.post("/run-clustering", response_model=MLRunResult, response_model_exclude_none=True)
def post_run_clustering(req: RunMLRequest) -> MLRunResult:
    return _handle(run_clustering, req)


@router.post("/run-forecast", response_model=TSRunResult, response_model_exclude_none=True)
def post_run_forecast(req: RunTSRequest) -> TSRunResult:
    return _handle(run_forecast, req)


@router.post("/optimize-params", response_model=OptimizeResult)
def post_optimize(req: OptimizeRequest) -> OptimizeResult:
    return _handle(lambda r: optimize(r), req)


# ─── data-request endpoints (this service owns data + splitting) ──────────────


@router.get("/datasets")
def list_datasets():
    with connect() as conn:
        rows = conn.execute("SELECT id, name, type, description, rows FROM datasets").fetchall()
    return {"data": [dict(r) for r in rows], "total": len(rows)}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, seed: int = 13):
    try:
        df = load_dataframe(dataset_id)
    except DatasetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    mask = seeded_split(len(df), seed)
    return {
        "id": dataset_id,
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "isTrain": mask.tolist(),
        "trainSize": int(mask.sum()),
        "testSize": int((~mask).sum()),
    }


# ─── WebSocket streaming helper ───────────────────────────────────────────────


async def _stream(ws: WebSocket, run_blocking: Callable[[Callable[[dict], None]], None]) -> None:
    """Run a blocking producer in a thread; forward each emitted dict to the
    socket as JSON. The producer calls ``emit(payload)`` per trial/epoch."""
    q: queue.Queue = queue.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            run_blocking(q.put)
        except Exception as exc:  # surface producer errors to the client
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is sentinel:
            break
        await ws.send_json(item)


@router.websocket("/ws/optimize")
async def ws_optimize(ws: WebSocket) -> None:
    await ws.accept()
    try:
        req = OptimizeRequest(**await ws.receive_json())
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"bad request: {exc}"})
        await ws.close()
        return

    def run_blocking(emit: Callable[[dict], None]) -> None:
        def cb(iteration: int, rmse: float, params: dict) -> None:
            emit({"iteration": iteration, "rmse": round(float(rmse), 6), "params": params})

        result = optimize(req, cb)
        emit({"done": True, "result": result.model_dump(by_alias=True)})

    try:
        await _stream(ws, run_blocking)
    except WebSocketDisconnect:
        return
    await ws.close()


@router.websocket("/ws/train")
async def ws_train(ws: WebSocket) -> None:
    """Stream per-epoch training loss for the NN models (mlp / rnn)."""
    await ws.accept()
    try:
        payload = await ws.receive_json()
        req = RunMLRequest(**payload) if payload.get("task") != "timeseries" else RunTSRequest(**payload)
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"bad request: {exc}"})
        await ws.close()
        return

    def run_blocking(emit: Callable[[dict], None]) -> None:
        cb = lambda epoch, loss: emit({"epoch": epoch, "loss": round(float(loss), 6)})
        if isinstance(req, RunTSRequest) or req.model == "rnn":
            dataset = req.dataset or "air_passengers"
            df = load_dataframe(dataset)
            target = "value" if "value" in df.columns else df.columns[-1]
            series = df[target].to_numpy(dtype=float)
            split = int(len(series) * 0.7)
            train_gru(series[:split], req.hyperparameters, epoch_callback=cb)
        else:  # mlp regression
            dataset = req.dataset or "california_housing"
            df = load_dataframe(dataset)
            features = feature_columns(df)
            x = df[features].to_numpy(dtype=float)
            y = df["target"].to_numpy(dtype=float)
            mask = seeded_split(len(df), getattr(req, "seed", 13))
            xs = StandardScaler().fit(x[mask])
            y_mean, y_std = float(y[mask].mean()), float(y[mask].std() or 1.0)
            train_mlp(xs.transform(x[mask]), (y[mask] - y_mean) / y_std, req.hyperparameters, epoch_callback=cb)
        emit({"done": True})

    try:
        await _stream(ws, run_blocking)
    except WebSocketDisconnect:
        return
    await ws.close()
