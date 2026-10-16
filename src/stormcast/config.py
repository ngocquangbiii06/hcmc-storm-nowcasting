"""Typed access to the YAML pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PeriodConfig:
    start: str = "2025-08-01"
    end: str = "2025-08-31"

    def days(self) -> list[date]:
        import pandas as pd

        return [d.date() for d in pd.date_range(self.start, self.end, freq="D")]


@dataclass(frozen=True)
class GridConfig:
    center_lat: float = 10.6596
    center_lon: float = 106.7283
    size: int = 256
    resolution_m: float = 1000.0

    @property
    def half_extent_m(self) -> float:
        return self.size * self.resolution_m / 2


@dataclass(frozen=True)
class PathsConfig:
    data_root: str = "data"
    reports: str = "reports"

    @property
    def root(self) -> Path:
        return Path(self.data_root)

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def catalog(self) -> Path:
        return self.root / "catalog"

    @property
    def cache(self) -> Path:
        return self.root / "cache"


@dataclass(frozen=True)
class HimawariConfig:
    bucket: str = "noaa-himawari9"
    product: str = "AHI-L1b-FLDK"
    bands: tuple[str, ...] = ("B08", "B13", "B14", "B15", "B16")
    resolution: str = "R20"
    segments: Any = "auto"
    keep_raw: bool = True
    resampler: str = "bilinear"
    download_workers: int = 16


@dataclass(frozen=True)
class RadarConfig:
    station: str = "NhaBe"
    raw_dir: str = "data/bronze/radar"
    reader: str = "auto"
    reflectivity_fields: tuple[str, ...] = ("reflectivity", "DBZH", "DBZ")
    cappi_height_m: float = 2000.0
    min_dbz: float = -10.0
    no_echo_dbz: float = -10.0
    filename_time_regex: str = r"(\d{14})"
    filename_time_format: str = "%Y%m%d%H%M%S"
    filename_utc_offset_hours: float = 0.0


@dataclass(frozen=True)
class AlignConfig:
    tolerance_minutes: float = 5.0


@dataclass(frozen=True)
class QualityConfig:
    bt_range_k: tuple[float, float] = (170.0, 340.0)
    dbz_range: tuple[float, float] = (-35.0, 80.0)
    max_nan_fraction: float = 0.2
    himawari_freq_minutes: int = 10
    radar_max_gap_minutes: int = 20


@dataclass(frozen=True)
class ProcessingConfig:
    workers: int = 4


@dataclass(frozen=True)
class MLConfig:
    task: str = "sat_nowcast"  # sat_nowcast | sat_to_radar
    sat_zarr_path: str = "data/gold/himawari_aug2025.zarr"
    zarr_path: str = "data/gold/hcmc_aug2025.zarr"
    split: dict = field(default_factory=dict)
    history_steps: int = 0
    max_dbz: float = 70.0
    thresholds_dbz: tuple[float, ...] = (20.0, 35.0, 45.0)
    input_steps: int = 4
    lead_steps: int = 6
    target_band: str = "B13"
    preload: bool = True
    bt_thresholds_k: tuple[float, ...] = (235.0, 220.0)
    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    base_channels: int = 32
    num_workers: int = 0
    output_dir: str = "runs"


@dataclass(frozen=True)
class Config:
    period: PeriodConfig = field(default_factory=PeriodConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    himawari: HimawariConfig = field(default_factory=HimawariConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ml: MLConfig = field(default_factory=MLConfig)


def _build(cls, raw: dict | None):
    raw = raw or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        # YAML lists -> tuples so the config stays hashable/frozen
        kwargs[name] = tuple(value) if isinstance(value, list) and name != "segments" else value
    return cls(**kwargs)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    sections = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(
        period=_build(PeriodConfig, raw.get("period")),
        grid=_build(GridConfig, raw.get("grid")),
        paths=_build(PathsConfig, raw.get("paths")),
        himawari=_build(HimawariConfig, raw.get("himawari")),
        radar=_build(RadarConfig, raw.get("radar")),
        align=_build(AlignConfig, raw.get("align")),
        quality=_build(QualityConfig, raw.get("quality")),
        processing=_build(ProcessingConfig, raw.get("processing")),
        ml=_build(MLConfig, raw.get("ml")),
    )
