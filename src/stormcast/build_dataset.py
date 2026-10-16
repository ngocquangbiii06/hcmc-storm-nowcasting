"""Gold layer: silver frames -> one chunked, ML-ready Zarr store.

Two products share one writer:
  * build_sat_zarr  - Himawari only (satellite nowcasting; needs no radar)
  * build_zarr      - Himawari + radar pairs (satellite -> radar retrieval)
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from stormcast.catalog import resolve
from stormcast.config import Config
from stormcast.grid import axis_centers, lonlats, proj_dict

log = logging.getLogger(__name__)

TIME_ENCODING = {"units": "seconds since 2000-01-01", "dtype": "int64"}
TIME_CHUNK = 16


def _load_stack(cfg: Config, paths, var: str, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty((len(paths), *shape), dtype="float32")
    for i, p in enumerate(paths):
        with xr.open_dataset(resolve(p, cfg.paths.root)) as ds:
            out[i] = ds[var].values
    return out


def _sat_batch(cfg: Config, frames: pd.DataFrame) -> xr.Dataset:
    n, size = len(cfg.himawari.bands), cfg.grid.size
    bt = _load_stack(cfg, frames["silver_path"], "bt", (n, size, size))
    x, y = axis_centers(cfg.grid)
    return xr.Dataset(
        {"bt": (("time", "band", "y", "x"), bt, {"units": "K"})},
        coords={
            "time": pd.to_datetime(frames["nominal_time"]).values,
            "sat_obs_time": ("time", pd.to_datetime(frames["obs_time"]).values),
            "band": list(cfg.himawari.bands),
            "x": x,
            "y": y,
        },
    )


def _pair_batch(cfg: Config, pairs: pd.DataFrame) -> xr.Dataset:
    n, size = len(cfg.himawari.bands), cfg.grid.size
    bt = _load_stack(cfg, pairs["sat_path"], "bt", (n, size, size))
    dbz = _load_stack(cfg, pairs["radar_path"], "dbz", (size, size))
    x, y = axis_centers(cfg.grid)
    return xr.Dataset(
        {
            "bt": (("time", "band", "y", "x"), bt, {"units": "K"}),
            "dbz": (("time", "y", "x"), dbz, {"units": "dBZ", "cappi_height_m": cfg.radar.cappi_height_m}),
        },
        coords={
            "time": pd.to_datetime(pairs["time"]).values,
            "sat_obs_time": ("time", pd.to_datetime(pairs["sat_obs_time"]).values),
            "radar_time": ("time", pd.to_datetime(pairs["radar_time"]).values),
            "band": list(cfg.himawari.bands),
            "x": x,
            "y": y,
        },
    )


def _daily(df: pd.DataFrame, time_col: str) -> Iterator[tuple[object, pd.DataFrame]]:
    df = df.sort_values(time_col)
    yield from df.groupby(pd.to_datetime(df[time_col]).dt.date)


def _write(cfg: Config, out: Path, batches: Iterator[tuple[object, xr.Dataset]], title: str) -> Path:
    """Rebuild the store from scratch, appending one day at a time to bound memory."""
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for day, ds in batches:
        if first:
            lons, lats = lonlats(cfg.grid)
            ds = ds.assign_coords(lon=(("y", "x"), lons), lat=(("y", "x"), lats))
            ds.attrs.update(
                {
                    "title": title,
                    "grid_projection": str(proj_dict(cfg.grid)),
                    "grid_resolution_m": cfg.grid.resolution_m,
                    "row_order": "row 0 = north",
                }
            )
            encoding = {name: TIME_ENCODING for name, c in ds.coords.items() if np.issubdtype(c.dtype, np.datetime64)}
            encoding |= {name: {"chunks": (TIME_CHUNK, *v.shape[1:])} for name, v in ds.data_vars.items()}
            ds.to_zarr(out, mode="w", encoding=encoding, zarr_format=2)
            first = False
        else:
            ds.to_zarr(out, append_dim="time")
        log.info("gold: wrote %s (%d frames)", day, ds.sizes["time"])
    if first:
        raise RuntimeError("nothing to write")
    return out


def build_sat_zarr(cfg: Config, frames: pd.DataFrame, out: Path | None = None) -> Path:
    """frames: himawari_frames catalog rows with status == ok."""
    out = Path(out or cfg.ml.sat_zarr_path)
    batches = ((day, _sat_batch(cfg, chunk)) for day, chunk in _daily(frames, "nominal_time"))
    return _write(cfg, out, batches, "Himawari-9 IR bands on the HCMC 256x256 grid")


def build_zarr(cfg: Config, pairs: pd.DataFrame, out: Path | None = None) -> Path:
    """pairs: output of align.pair_frames."""
    out = Path(out or cfg.ml.zarr_path)
    batches = ((day, _pair_batch(cfg, chunk)) for day, chunk in _daily(pairs, "time"))
    return _write(cfg, out, batches, "HCMC radar + Himawari-9 paired dataset")
