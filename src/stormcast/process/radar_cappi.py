"""Silver layer for radar: raw volume -> cleaned CAPPI on the shared 256x256 grid (Py-ART)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stormcast.catalog import to_rel
from stormcast.config import Config, RadarConfig
from stormcast.grid import axis_centers, pyart_grid_limits
from stormcast.quality import check_frame

log = logging.getLogger(__name__)


def read_radar(path: Path, reader: str = "auto"):
    import pyart

    readers = {
        "auto": pyart.io.read,
        "sigmet": pyart.io.read_sigmet,
        "cfradial": pyart.io.read_cfradial,
        "nexrad": pyart.io.read_nexrad_archive,
        "uf": pyart.io.read_uf,
        "odim": lambda p: pyart.aux_io.read_odim_h5(p),
        "rainbow": lambda p: pyart.aux_io.read_rainbow_wrl(p),  # needs wradlib
    }
    return readers[reader](str(path))


def pick_field(radar, candidates) -> str:
    for name in candidates:
        if name in radar.fields:
            return name
    raise KeyError(f"No reflectivity field among {list(candidates)}; file has {list(radar.fields)}")


def clean_reflectivity(radar, field: str, rcfg: RadarConfig) -> str:
    """Add a 'dbz_clean' field: noise/speckle removed and *set to no-echo* (not masked).

    Masked gates would be treated as "no data" by the gridder, which makes clear sky
    indistinguishable from outside-radar-range. Filling them keeps that distinction:
    after gridding, NaN == no radar coverage, no_echo_dbz == covered but no rain.
    """
    import pyart

    gf = pyart.filters.GateFilter(radar)
    gf.exclude_invalid(field)
    gf.exclude_masked(field)
    gf.exclude_below(field, rcfg.min_dbz)
    gf = pyart.correct.despeckle_field(radar, field, gatefilter=gf, size=10)

    clean = np.ma.filled(radar.fields[field]["data"].astype("float32"), rcfg.no_echo_dbz)
    clean[gf.gate_excluded] = rcfg.no_echo_dbz
    radar.add_field_like(field, "dbz_clean", clean, replace_existing=True)
    return "dbz_clean"


def radar_to_cappi(radar, cfg: Config) -> np.ndarray:
    """Grid one volume to a CAPPI at cappi_height_m. Returns (y, x) float32, row 0 = north."""
    import pyart

    field = clean_reflectivity(radar, pick_field(radar, cfg.radar.reflectivity_fields), cfg.radar)
    lo, hi = pyart_grid_limits(cfg.grid)
    z = float(cfg.radar.cappi_height_m)
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(1, cfg.grid.size, cfg.grid.size),
        grid_limits=((z, z), (lo, hi), (lo, hi)),
        grid_origin=(cfg.grid.center_lat, cfg.grid.center_lon),
        fields=[field],
        weighting_function="Barnes2",
    )
    data = np.ma.filled(grid.fields[field]["data"][0].astype("float32"), np.nan)
    return data[::-1].copy()  # Py-ART rows go south->north; flip to north-up


def silver_path(cfg: Config, scan_time: pd.Timestamp) -> Path:
    return cfg.paths.silver / "radar" / f"{scan_time:%Y-%m-%d}" / f"radar_{scan_time:%Y%m%d_%H%M%S}.nc"


def process_file(cfg: Config, path: str, scan_time: pd.Timestamp) -> dict:
    """Worker: one radar volume -> one silver NetCDF + QC record. Skips if output exists."""
    import xarray as xr

    out = silver_path(cfg, scan_time)
    record = {"scan_time": scan_time, "raw_path": path, "silver_path": to_rel(out, cfg.paths.root)}
    try:
        if out.exists():
            with xr.open_dataset(out) as ds:
                arr = ds["dbz"].values
        else:
            arr = radar_to_cappi(read_radar(Path(path), cfg.radar.reader), cfg)
            x, y = axis_centers(cfg.grid)
            ds = xr.Dataset(
                {"dbz": (("y", "x"), arr, {"units": "dBZ", "long_name": "CAPPI reflectivity"})},
                coords={"x": x, "y": y},
                attrs={
                    "scan_time": str(scan_time),
                    "cappi_height_m": cfg.radar.cappi_height_m,
                    "station": cfg.radar.station,
                    "source_file": Path(path).name,
                },
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out.with_suffix(".tmp"), encoding={"dbz": {"zlib": True, "complevel": 4}})
            out.with_suffix(".tmp").replace(out)
        record.update(check_frame(arr, cfg.quality.dbz_range, cfg.quality.max_nan_fraction, check_flat=False))
    except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the month
        log.warning("radar %s failed: %s", path, exc)
        record.update({"status": "error", "issues": str(exc)[:500]})
    return record
