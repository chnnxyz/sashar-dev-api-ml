"""Read seeded datasets back out of the shared sqlite ``data`` table and provide
train/test splitting. This service is the single owner of data access and
splitting for the frontend (CLAUDE.md)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.db import DATA_TABLE, connect

# Which stored column is the supervised target / the time index, per dataset.
TARGET_COLUMN = "target"
DATE_COLUMN = "date"
VALUE_COLUMN = "value"


class DatasetNotFound(Exception):
    pass


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    """Pivot the long-format rows for ``dataset_id`` back into a wide DataFrame,
    casting each column by its stored field_type."""
    with connect() as conn:
        rows = conn.execute(
            f"SELECT row_index, field, field_type, value FROM {DATA_TABLE} "
            f"WHERE dataset_id = ? ORDER BY row_index",
            (dataset_id,),
        ).fetchall()
    if not rows:
        raise DatasetNotFound(f"dataset {dataset_id!r} has no rows — run app.seed_datasets")

    records: dict[int, dict[str, object]] = {}
    types: dict[str, str] = {}
    for r in rows:
        types[r["field"]] = r["field_type"]
        records.setdefault(r["row_index"], {})[r["field"]] = r["value"]

    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    for col, ftype in types.items():
        if ftype in ("float", "int"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def dataset_meta(dataset_id: str) -> dict:
    """Return the dataset metadata row (type, features, ...) seeded by Go."""
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, type, description, rows, features FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
    if row is None:
        raise DatasetNotFound(f"no metadata for dataset {dataset_id!r}")
    features = json.loads(row["features"]) if row["features"] else []
    return {
        "id": row["id"], "name": row["name"], "type": row["type"],
        "description": row["description"], "rows": row["rows"], "features": features,
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns = everything except target/date/value bookkeeping columns."""
    drop = {TARGET_COLUMN, DATE_COLUMN, VALUE_COLUMN}
    return [c for c in df.columns if c not in drop]


def seeded_split(n: int, seed: int, train_frac: float = 0.7) -> np.ndarray:
    """Boolean mask (len n) marking the train set, using the same LCG Fisher-Yates
    shuffle as the frontend's ``applySeededSplit`` so splits are reproducible and
    consistent across the stack."""
    s = seed & 0xFFFFFFFF
    indices = list(range(n))

    def lcg() -> float:
        nonlocal s
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        return s / 0x100000000

    for i in range(n - 1, 0, -1):
        j = int(lcg() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]

    train_count = int(n * train_frac)
    train_idx = set(indices[:train_count])
    return np.array([i in train_idx for i in range(n)], dtype=bool)
