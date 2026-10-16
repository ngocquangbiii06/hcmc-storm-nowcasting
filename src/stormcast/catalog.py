"""Parquet catalog: one file per (table, day) so daily partitions can run in parallel safely.

Layout: data/catalog/<table>/date=YYYY-MM-DD.parquet
Query everything with DuckDB:  SELECT * FROM 'data/catalog/<table>/*.parquet'
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def partition_path(catalog_root: Path, table: str, day: date | str) -> Path:
    return Path(catalog_root) / table / f"date={day}.parquet"


def write_partition(df: pd.DataFrame, catalog_root: Path, table: str, day: date | str) -> Path:
    """Overwrite one day of a table (idempotent: re-running a day replaces it)."""
    path = partition_path(catalog_root, table, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_table(catalog_root: Path, table: str, days: list | None = None) -> pd.DataFrame:
    folder = Path(catalog_root) / table
    if days is None:
        files = sorted(folder.glob("date=*.parquet"))
    else:
        files = [p for d in days if (p := partition_path(catalog_root, table, d)).exists()]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def to_rel(path: str | Path, root: Path) -> str:
    """Store paths relative to data_root so a catalog built on Colab still resolves locally."""
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def resolve(path: str | Path, root: Path) -> Path:
    return Path(root) / path  # pathlib keeps absolute paths unchanged
