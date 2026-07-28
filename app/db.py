"""Shared sqlite access for the ML service.

The Go backend owns schema creation for CV/Models/Datasets/Emails via gorm
AutoMigrate. This service owns the ``data`` table (actual dataset points) — it
creates it if missing (so the seeder can run before/independently of Go) and
reads dataset metadata + rows back out. Long format: one row per
(dataset, row_index, field).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import get_settings

# gorm's default table name for the Data model is the pluralized snake_case
# "data" (already plural / uncountable → stays "data").
DATA_TABLE = "data"

_CREATE_DATA_SQL = f"""
CREATE TABLE IF NOT EXISTS {DATA_TABLE} (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT    NOT NULL,
    row_index  INTEGER NOT NULL,
    field      TEXT    NOT NULL,
    field_type TEXT,
    value      TEXT
);
CREATE INDEX IF NOT EXISTS idx_data_dataset ON {DATA_TABLE} (dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_row     ON {DATA_TABLE} (dataset_id, row_index);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a connection to the shared sqlite file with WAL + busy timeout."""
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_data_table(conn: sqlite3.Connection) -> None:
    """Create the ``data`` table if it does not yet exist."""
    conn.executescript(_CREATE_DATA_SQL)


def dataset_exists(conn: sqlite3.Connection, dataset_id: str) -> bool:
    cur = conn.execute(
        f"SELECT 1 FROM {DATA_TABLE} WHERE dataset_id = ? LIMIT 1", (dataset_id,)
    )
    return cur.fetchone() is not None
