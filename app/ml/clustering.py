"""Clustering task: KMeans / DBSCAN → MLRunResult (unsupervised)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from app.data.loader import feature_columns, load_dataframe
from app.ml import metrics
from app.schemas import MLRunResult, RunMLRequest, ScatterPoint

DEFAULT_DATASET = "blobs"


def _inertia(x: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in set(labels):
        if c == -1:  # DBSCAN noise
            continue
        pts = x[labels == c]
        if len(pts):
            total += float(np.sum((pts - pts.mean(axis=0)) ** 2))
    return total


def run_clustering(req: RunMLRequest) -> MLRunResult:
    dataset = req.dataset or DEFAULT_DATASET
    df = load_dataframe(dataset)
    features = feature_columns(df)  # x, y
    x = df[features].to_numpy(dtype=float)
    hp = req.hyperparameters

    if req.model == "kmeans":
        model = KMeans(
            n_clusters=int(hp.get("n_clusters", 3)), max_iter=int(hp.get("max_iter", 300)),
            tol=float(hp.get("tol", 1e-4)), n_init=int(hp.get("n_init", 10)), random_state=42,
        )
        labels = model.fit_predict(x)
        inertia = float(model.inertia_)
    elif req.model == "dbscan":
        model = DBSCAN(
            eps=float(hp.get("eps", 0.5)), min_samples=int(hp.get("min_samples", 5)),
            leaf_size=int(hp.get("leaf_size", 30)),
        )
        labels = model.fit_predict(x)
        inertia = _inertia(x, labels)
    else:
        raise ValueError(f"unknown clustering model: {req.model}")

    def cluster_label(c: int) -> str:
        return "Noise" if c == -1 else f"Cluster {c + 1}"

    scatter = [
        ScatterPoint(x=float(x[i, 0]), y=float(x[i, 1]), label=cluster_label(int(labels[i])))
        for i in range(len(df))
    ]

    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    if n_clusters > 1 and non_noise.sum() > n_clusters:
        sil = float(silhouette_score(x[non_noise], labels[non_noise]))
    else:
        sil = 0.0

    error = 1.0 - sil  # CV card proxy for unsupervised runs
    return MLRunResult(
        scatter=scatter,
        test_predictions=None,
        metrics={"silhouette": round(sil, 4), "inertia": round(inertia, 4)},
        model=req.model,
        cv=metrics.make_cv_metrics(
            error, error, len(df), 0, metrics.synth_folds(error)
        ),
    )
