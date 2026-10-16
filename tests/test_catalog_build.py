from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from stormcast import catalog
from stormcast.build_dataset import build_zarr
from stormcast.config import GridConfig, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.grid.size == 256
    assert cfg.himawari.bands == ("B08", "B13", "B14", "B15", "B16")
    assert len(cfg.period.days()) == 31


def test_catalog_partitions_are_idempotent(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    catalog.write_partition(df, tmp_path, "t", date(2025, 8, 1))
    catalog.write_partition(df, tmp_path, "t", date(2025, 8, 1))  # rerun replaces, not appends
    catalog.write_partition(df, tmp_path, "t", date(2025, 8, 2))
    assert len(catalog.read_table(tmp_path, "t")) == 4
    assert len(catalog.read_table(tmp_path, "t", [date(2025, 8, 2)])) == 2


def _write_silver(path, name, dims, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({name: (dims, arr)}).to_netcdf(path)


def test_build_zarr_appends_days(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, grid=GridConfig(size=8, resolution_m=1000))
    rows = []
    for i, t in enumerate(pd.to_datetime(["2025-08-01 23:50", "2025-08-02 00:00", "2025-08-02 00:10"])):
        sp, rp = tmp_path / f"s{i}.nc", tmp_path / f"r{i}.nc"
        _write_silver(sp, "bt", ("band", "y", "x"), np.full((5, 8, 8), 250.0 + i, dtype="float32"))
        _write_silver(rp, "dbz", ("y", "x"), np.full((8, 8), 10.0 * i, dtype="float32"))
        rows.append({"time": t, "sat_obs_time": t + pd.Timedelta(minutes=4), "radar_time": t,
                     "sat_path": str(sp), "radar_path": str(rp), "dt_seconds": -240.0})
    out = build_zarr(cfg, pd.DataFrame(rows), tmp_path / "gold.zarr")

    ds = xr.open_zarr(out)
    assert ds["bt"].shape == (3, 5, 8, 8)
    assert ds["dbz"].values[:, 0, 0].tolist() == [0.0, 10.0, 20.0]
    assert ds["lat"].shape == (8, 8)
    assert ds["lat"].values[0, 0] > ds["lat"].values[-1, 0]  # row 0 = north
