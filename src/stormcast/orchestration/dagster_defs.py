"""Dagster assets: thin wrappers around stormcast.pipeline with daily partitions.

    dagster dev -f src/stormcast/orchestration/dagster_defs.py

Backfill August in the UI (Assets -> Materialize -> select partitions), and re-run any
single failed day without touching the others.
"""
from __future__ import annotations

import os
from datetime import date

from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    Definitions,
    MaterializeResult,
    MetadataValue,
    asset,
)

from stormcast import pipeline
from stormcast.config import load_config

CONFIG_PATH = os.environ.get("STORMCAST_CONFIG", "configs/default.yaml")
daily = DailyPartitionsDefinition(start_date="2025-08-01", end_date="2025-09-01")


def _day(context: AssetExecutionContext) -> date:
    return date.fromisoformat(context.partition_key)


def _summary(df) -> MaterializeResult:
    meta = {"rows": len(df)}
    if not df.empty and "status" in df:
        meta["status_counts"] = MetadataValue.json(df["status"].value_counts().to_dict())
    return MaterializeResult(metadata=meta)


@asset(partitions_def=daily, group_name="bronze")
def himawari_raw(context: AssetExecutionContext) -> MaterializeResult:
    return _summary(pipeline.himawari_download(load_config(CONFIG_PATH), _day(context)))


@asset(group_name="bronze")
def radar_index() -> MaterializeResult:
    return _summary(pipeline.radar_index(load_config(CONFIG_PATH)))


@asset(partitions_def=daily, deps=[himawari_raw], group_name="silver")
def himawari_frames(context: AssetExecutionContext) -> MaterializeResult:
    return _summary(pipeline.himawari_process(load_config(CONFIG_PATH), _day(context)))


@asset(partitions_def=daily, deps=[radar_index], group_name="silver")
def radar_frames(context: AssetExecutionContext) -> MaterializeResult:
    return _summary(pipeline.radar_process(load_config(CONFIG_PATH), _day(context)))


@asset(partitions_def=daily, deps=[himawari_frames, radar_frames], group_name="gold")
def pairs(context: AssetExecutionContext) -> MaterializeResult:
    return _summary(pipeline.align(load_config(CONFIG_PATH), _day(context)))


@asset(deps=[pairs], group_name="gold")
def gold_zarr() -> MaterializeResult:
    path = pipeline.build(load_config(CONFIG_PATH))
    return MaterializeResult(metadata={"path": str(path)})


@asset(deps=[himawari_frames], group_name="gold")
def gold_sat_zarr() -> MaterializeResult:
    path = pipeline.build_sat(load_config(CONFIG_PATH))
    return MaterializeResult(metadata={"path": str(path)})


@asset(deps=[himawari_frames], group_name="gold")
def data_quality_report() -> MaterializeResult:
    from stormcast.report import build_report

    path = build_report(load_config(CONFIG_PATH))
    return MaterializeResult(metadata={"report": MetadataValue.md(path.read_text(encoding="utf-8"))})


defs = Definitions(
    assets=[
        himawari_raw, radar_index, himawari_frames, radar_frames, pairs, gold_zarr, gold_sat_zarr, data_quality_report
    ]
)
