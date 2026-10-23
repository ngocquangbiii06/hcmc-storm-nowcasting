"""Silver layer for Himawari: HSD segments -> brightness temperature of 5 IR bands on the shared grid (SatPy)."""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stormcast.catalog import to_rel
from stormcast.config import Config
from stormcast.grid import axis_centers, target_area
from stormcast.quality import check_frame

log = logging.getLogger(__name__)


def observation_time(attrs: dict) -> pd.Timestamp:
    """Mid-scan time of the loaded segments.

    A full-disk scan takes ~10 min, north to south, so the HCMC strip is observed a few
    minutes after the nominal timestamp in the file name. That offset matters when pairing
    with radar scans that are only 5-10 min apart.
    """
    tp = attrs.get("time_parameters") or {}
    start = tp.get("observation_start_time") or attrs.get("start_time")
    end = tp.get("observation_end_time") or attrs.get("end_time") or start
    return pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2


def regrid_timestep(cfg: Config, files: list[str]) -> tuple[np.ndarray, pd.Timestamp]:
    import dask
    import satpy
    from satpy import Scene

    bands = list(cfg.himawari.bands)
    tmp_dir = satpy_tmp_dir(cfg)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # off-disk (space) pixels emit "invalid value encountered in log" thousands of times per day,
    # which floods hosted log viewers; they become NaN and are counted by QC anyway
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*PROJ string.*")
    warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")
    # parallelism comes from the process pool; bz2 segments are decompressed into our own tmp dir
    with dask.config.set(scheduler="synchronous"), satpy.config.set(tmp_dir=str(tmp_dir)):
        scn = Scene(reader="ahi_hsd", filenames=files)
        scn.load(bands, calibration="brightness_temperature")
        local = scn.resample(
            target_area(cfg.grid),
            resampler=cfg.himawari.resampler,
            # one cache per worker process: pyresample's cache is not safe for concurrent writers
            cache_dir=str(cfg.paths.cache / "resample" / f"pid{os.getpid()}"),
        )
        bt = np.stack([local[b].values.astype("float32") for b in bands])
    return bt, observation_time(local[bands[0]].attrs)


def satpy_tmp_dir(cfg: Config) -> Path:
    """Where SatPy writes decompressed HSD segments (~6 MB each); wiped after every day."""
    return cfg.paths.cache / "satpy_tmp"


def silver_path(cfg: Config, nominal_time: pd.Timestamp) -> Path:
    return cfg.paths.silver / "himawari" / f"{nominal_time:%Y-%m-%d}" / f"him_{nominal_time:%Y%m%d_%H%M}.nc"


def process_timestep(cfg: Config, nominal_time: pd.Timestamp, files: list[str]) -> dict:
    """Worker: one 10-minute timestep -> one silver NetCDF + QC record. Skips if output exists."""
    import xarray as xr

    out = silver_path(cfg, nominal_time)
    record = {"nominal_time": nominal_time, "silver_path": to_rel(out, cfg.paths.root), "n_files": len(files)}
    expected = len(cfg.himawari.bands) * len(_segments(cfg))
    try:
        if len(files) < expected:
            raise RuntimeError(f"incomplete timestep: {len(files)}/{expected} files")
        if out.exists():
            with xr.open_dataset(out) as ds:
                bt, obs = ds["bt"].values, pd.Timestamp(ds.attrs["obs_time"])
        else:
            bt, obs = regrid_timestep(cfg, files)
            x, y = axis_centers(cfg.grid)
            ds = xr.Dataset(
                {"bt": (("band", "y", "x"), bt, {"units": "K", "long_name": "brightness temperature"})},
                coords={"band": list(cfg.himawari.bands), "x": x, "y": y},
                attrs={"nominal_time": str(nominal_time), "obs_time": str(obs), "resampler": cfg.himawari.resampler},
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out.with_suffix(".tmp"), encoding={"bt": {"zlib": True, "complevel": 4}})
            out.with_suffix(".tmp").replace(out)
            if not cfg.himawari.keep_raw:
                for f in files:
                    Path(f).unlink(missing_ok=True)
        record["obs_time"] = obs
        record.update(check_frame(bt, cfg.quality.bt_range_k, cfg.quality.max_nan_fraction))
    except Exception as exc:  # noqa: BLE001
        log.warning("himawari %s failed: %s", nominal_time, exc)
        record.update({"obs_time": pd.NaT, "status": "error", "issues": str(exc)[:500]})
    return record


def _segments(cfg: Config) -> list[int]:
    from stormcast.ingest.himawari import wanted_segments

    return wanted_segments(cfg)
