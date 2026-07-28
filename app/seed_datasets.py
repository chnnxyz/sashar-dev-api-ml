"""Seed the shared sqlite ``data`` table with the playground datasets.

Sources are sklearn built-ins (offline) plus synthetic generators for the
clustering/time-series sets — no network required. California Housing is fetched
from sklearn if available and falls back to a synthetic generator otherwise.
Every dataset is stratified-truncated to <=1000 rows (CLAUDE.md) and written in
long format: one row per (dataset_id, row_index, field).

Run:  poetry run python -m app.seed_datasets [--force]
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_blobs,
    make_moons,
)

from app.db import DATA_TABLE, connect, dataset_exists, ensure_data_table

MAX_ROWS = 1000
RNG = np.random.default_rng(42)

# Monthly international airline passengers 1949–1960 (Box & Jenkins). Verbatim.
_AIR_PASSENGERS = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


def _stratified_sample(df: pd.DataFrame, strat: pd.Series, n: int) -> pd.DataFrame:
    """Return at most ``n`` rows, sampling proportionally within each stratum."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _, idx in df.groupby(strat.values).groups.items():
        sub = df.loc[idx]
        take = max(1, int(round(len(sub) * frac)))
        parts.append(sub.sample(n=min(take, len(sub)), random_state=42))
    out = pd.concat(parts).sample(frac=1, random_state=42).head(n)
    return out.reset_index(drop=True)


def _insert_long(
    conn: sqlite3.Connection,
    dataset_id: str,
    df: pd.DataFrame,
    types: dict[str, str],
) -> None:
    """Replace all rows for ``dataset_id`` with the long-format contents of df."""
    conn.execute(f"DELETE FROM {DATA_TABLE} WHERE dataset_id = ?", (dataset_id,))
    records: list[tuple] = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        for field, value in row.items():
            records.append(
                (dataset_id, row_index, str(field), types.get(str(field), "float"), str(value))
            )
    conn.executemany(
        f"INSERT INTO {DATA_TABLE} (dataset_id, row_index, field, field_type, value) "
        f"VALUES (?, ?, ?, ?, ?)",
        records,
    )
    print(f"  seeded {dataset_id}: {len(df)} rows × {len(df.columns)} fields")


# ─── individual dataset builders ──────────────────────────────────────────────


def _iris() -> tuple[pd.DataFrame, dict[str, str]]:
    ds = load_iris(as_frame=True)
    df = ds.frame.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target_idx"]
    df["target"] = [ds.target_names[i] for i in df.pop("target_idx")]
    types = {c: "float" for c in df.columns}
    types["target"] = "string"
    return df, types


def _breast_cancer() -> tuple[pd.DataFrame, dict[str, str]]:
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    target_col = df.columns[-1]
    df["target"] = [ds.target_names[i] for i in df.pop(target_col)]
    df.columns = [c.replace(" ", "_") for c in df.columns]
    types = {c: "float" for c in df.columns}
    types["target"] = "string"
    return df, types


def _diabetes() -> tuple[pd.DataFrame, dict[str, str]]:
    ds = load_diabetes(as_frame=True)
    df = ds.frame.copy()
    df = df.rename(columns={"target": "target"})
    types = {c: "float" for c in df.columns}
    return df, types


def _california_housing() -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        from sklearn.datasets import fetch_california_housing

        ds = fetch_california_housing(as_frame=True)
        df = ds.frame.rename(columns={"MedHouseVal": "target"})
    except Exception:  # offline / download blocked → synthetic fallback
        n = 2000
        med_inc = RNG.uniform(0.5, 13.0, n)
        house_age = RNG.uniform(5, 52, n)
        ave_rooms = RNG.uniform(1.5, 10.0, n)
        ave_bedrms = RNG.uniform(0.8, 1.5, n)
        population = RNG.uniform(200, 5200, n)
        ave_occup = RNG.uniform(1.5, 6.0, n)
        latitude = RNG.uniform(32.5, 42.0, n)
        longitude = RNG.uniform(-124.3, -114.3, n)
        target = np.clip(
            0.33 * med_inc + 0.008 * house_age + 0.04 * ave_rooms + RNG.normal(0, 0.45, n),
            0.15, 5.0,
        )
        df = pd.DataFrame(
            {
                "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
                "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
                "Latitude": latitude, "Longitude": longitude, "target": target,
            }
        )
    types = {c: "float" for c in df.columns}
    return df, types


def _blobs() -> tuple[pd.DataFrame, dict[str, str]]:
    x, y = make_blobs(n_samples=300, centers=4, cluster_std=1.1, random_state=42)
    df = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "target": [f"Cluster {i + 1}" for i in y]})
    return df, {"x": "float", "y": "float", "target": "string"}


def _moons() -> tuple[pd.DataFrame, dict[str, str]]:
    x, y = make_moons(n_samples=200, noise=0.08, random_state=42)
    df = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "target": [f"Crescent {'AB'[i]}" for i in y]})
    return df, {"x": "float", "y": "float", "target": "string"}


def _air_passengers() -> tuple[pd.DataFrame, dict[str, str]]:
    dates = pd.date_range("1949-01-01", periods=len(_AIR_PASSENGERS), freq="MS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": _AIR_PASSENGERS})
    return df, {"date": "string", "value": "float"}


def _etth1() -> tuple[pd.DataFrame, dict[str, str]]:
    # Synthetic hourly oil-temperature + load series (~1000 points) mirroring the
    # frontend's ETTh1 stand-in: daily seasonality + slow trend + noise.
    n = MAX_ROWS
    dates = pd.date_range("2016-07-01", periods=n, freq="h")
    i = np.arange(n)
    seasonal = 8 * np.sin(2 * np.pi * (i % 24) / 24)
    trend = i * 0.002
    ot = np.clip(22 + seasonal + trend + RNG.normal(0, 2.0, n), 8, 42)
    loads = {
        name: np.clip(base + 6 * np.sin(2 * np.pi * (i % 24) / 24 + ph) + RNG.normal(0, 1.5, n), 0, None)
        for name, base, ph in [
            ("HUFL", 30, 0.0), ("HULL", 8, 0.3), ("MUFL", 20, 0.6),
            ("MULL", 5, 0.9), ("LUFL", 15, 1.2), ("LULL", 3, 1.5),
        ]
    }
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), **loads, "OT": ot})
    types = {c: "float" for c in df.columns}
    types["date"] = "string"
    return df, types


# dataset_id → (builder, stratify-column-or-None)
BUILDERS = {
    "iris": (_iris, "target"),
    "breast_cancer": (_breast_cancer, "target"),
    "diabetes": (_diabetes, None),
    "california_housing": (_california_housing, None),
    "blobs": (_blobs, "target"),
    "moons": (_moons, "target"),
    "air_passengers": (_air_passengers, None),
    "etth1": (_etth1, None),
}


def seed(force: bool = False) -> None:
    with connect() as conn:
        ensure_data_table(conn)
        for dataset_id, (builder, strat_col) in BUILDERS.items():
            if not force and dataset_exists(conn, dataset_id):
                print(f"  {dataset_id}: already seeded (use --force to rebuild)")
                continue
            df, types = builder()
            if len(df) > MAX_ROWS:
                strat = df[strat_col] if strat_col else pd.qcut(
                    df.get("target", pd.Series(range(len(df)))), q=10, labels=False, duplicates="drop"
                )
                df = _stratified_sample(df, strat, MAX_ROWS)
            _insert_long(conn, dataset_id, df, types)
    print("done.")


if __name__ == "__main__":
    seed(force="--force" in sys.argv)
