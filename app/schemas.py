"""Pydantic request/response models.

Fields use snake_case in Python (per CLAUDE.md) but serialize to the camelCase
JSON the frontend contract expects, via explicit aliases. Routes serialize with
``by_alias=True`` (FastAPI default) and ``exclude_none=True``.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# ─── ML / TS shared ───────────────────────────────────────────────────────────


class ScatterPoint(APIModel):
    x: float
    y: float
    label: Optional[str] = None
    is_train: Optional[bool] = Field(default=None, alias="isTrain")
    features: Optional[dict[str, float]] = None
    correct: Optional[bool] = None
    predicted_label: Optional[str] = Field(default=None, alias="predictedLabel")
    predicted_prob: Optional[float] = Field(default=None, alias="predictedProb")
    real_value: Optional[float] = Field(default=None, alias="realValue")


class TimeSeriesPoint(APIModel):
    date: str  # ISO-8601 string; the frontend parses into a Date
    value: float
    predicted: Optional[float] = None


class CVMetrics(APIModel):
    train_rmse: float = Field(alias="trainRMSE")
    test_rmse: float = Field(alias="testRMSE")
    train_size: int = Field(alias="trainSize")
    test_size: int = Field(alias="testSize")
    cv_folds: list[float] = Field(alias="cvFolds")


# ─── ML requests/responses ────────────────────────────────────────────────────


class RunMLRequest(APIModel):
    task: Optional[str] = None  # regression | classification | clustering
    model: str
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    dataset: Optional[str] = None
    seed: int = 13


class MLRunResult(APIModel):
    scatter: list[ScatterPoint]
    test_predictions: Optional[list[ScatterPoint]] = Field(default=None, alias="testPredictions")
    metrics: dict[str, float]
    model: str
    cv: CVMetrics


class RunTSRequest(APIModel):
    model: str  # lightgbm | tes | rnn
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    dataset: Optional[str] = None
    seed: int = 13


class TSRunResult(APIModel):
    historical: list[TimeSeriesPoint]
    forecast: list[TimeSeriesPoint]
    metrics: dict[str, float]
    model: str
    cv: CVMetrics


class OptimizeRequest(APIModel):
    task: str  # regression | classification | clustering | timeseries
    model: str
    method: str  # gridsearch | tpe | genetic
    config: dict[str, Any] = Field(default_factory=dict)


class OptimizeResult(APIModel):
    best_params: dict[str, float]
    best_score: float
    n_trials: int
    duration_seconds: float


# ─── LLM pipeline ─────────────────────────────────────────────────────────────


class TokenizeRequest(APIModel):
    prompt: str


class TokenizeResult(APIModel):
    tokens: list[str]


class EncodeRequest(APIModel):
    tokens: list[str]


class EncodeResult(APIModel):
    ids: list[int]


class EmbedRequest(APIModel):
    tokens: list[str]
    ids: list[int]


class EmbedPoint(APIModel):
    x: float
    y: float
    label: str


class EmbedResult(APIModel):
    points: list[EmbedPoint]


class GenerateRequest(APIModel):
    prompt: str
    tokens: list[str] = Field(default_factory=list)
    ids: list[int] = Field(default_factory=list)


class GenerateResult(APIModel):
    output_tokens: list[str] = Field(alias="outputTokens")
    output_ids: list[int] = Field(default_factory=list, alias="outputIds")
    output_points: list[EmbedPoint] = Field(default_factory=list, alias="outputPoints")


class ClusterRequest(APIModel):
    points: list[EmbedPoint]
    n_clusters: Optional[int] = None


class ClusterGroup(APIModel):
    id: int
    tokens: list[str]
    description: str


class ClusterResult(APIModel):
    n_groups: int = Field(alias="nGroups")
    groups: list[ClusterGroup]
    # Cluster id per input point (aligned to request order); -1 for the excluded marker.
    assignments: list[int] = Field(default_factory=list)
