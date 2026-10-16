"""Satellite nowcasting: windows, baselines, Himawari-only gold store, and an end-to-end training smoke run."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from stormcast.build_dataset import build_sat_zarr
from stormcast.config import GridConfig, load_config
from stormcast.ml.baselines import persistence

torch = pytest.importorskip("torch")
from stormcast.ml.nowcast import window_indices  # noqa: E402


def test_windows_stay_inside_split_and_skip_gaps():
    times = pd.date_range("2025-08-01", "2025-08-02 23:50", freq="10min").delete(30)  # one missing frame
    idx = window_indices(times.values, "2025-08-01", "2025-08-01", input_steps=4, lead_steps=6)
    t0 = times[idx]
    assert t0.min() == pd.Timestamp("2025-08-01 00:30")  # needs 00:00 ... 00:30 as inputs
    assert t0.max() == pd.Timestamp("2025-08-01 22:50")  # +60 min must stay on Aug 1
    missing = times[29] + pd.Timedelta(minutes=10)
    assert not any((t0 - pd.Timedelta(minutes=30) <= missing) & (t0 + pd.Timedelta(minutes=60) >= missing))


def _blob(h, w, cy, cx, r=4.0, cold=210.0, warm=290.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r, cold, warm).astype("float32")


def test_persistence_repeats_last_frame():
    last = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(persistence(last, 3)[2], last)


def test_optical_flow_moves_cloud_along_motion():
    pytest.importorskip("cv2")
    from stormcast.ml.baselines import optical_flow_forecast

    prev, last = _blob(64, 64, 32, 20), _blob(64, 64, 32, 23)  # moving east 3 px / step
    fc = optical_flow_forecast(prev, last, lead_steps=3)
    cols = np.nonzero(fc[2] <= 235)[1]
    assert abs(cols.mean() - 32) <= 2  # 23 + 3 * 3
    truth = _blob(64, 64, 32, 32)
    hit = lambda f: ((f <= 235) & (truth <= 235)).sum() / ((f <= 235) | (truth <= 235)).sum()  # noqa: E731
    assert hit(fc[2]) > hit(persistence(last, 3)[2])


def _write_sat_silver(tmp_path, cfg, times):
    rows = []
    size = cfg.grid.size
    for i, t in enumerate(times):
        bt = np.stack([_blob(size, size, size // 2, 4 + (i % 20), cold=210 + b, warm=290 - b) for b in range(5)])
        path = tmp_path / "silver" / f"him_{t:%Y%m%d_%H%M}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"bt": (("band", "y", "x"), bt)}).to_netcdf(path)
        rows.append({"nominal_time": t, "obs_time": t + pd.Timedelta(minutes=4), "silver_path": str(path)})
    return pd.DataFrame(rows)


def test_build_sat_zarr_and_train_smoke(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, grid=GridConfig(size=32, resolution_m=1000))
    times = pd.date_range("2025-08-01", "2025-08-03 23:50", freq="10min")
    frames = _write_sat_silver(tmp_path, cfg, times)
    store = build_sat_zarr(cfg, frames, tmp_path / "sat.zarr")
    ds = xr.open_zarr(store)
    assert ds["bt"].shape == (len(times), 5, 32, 32)
    assert "dbz" not in ds

    raw = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    raw["grid"]["size"] = 32
    raw["ml"].update(
        sat_zarr_path=str(store),
        split={"train": ["2025-08-01", "2025-08-01"], "val": ["2025-08-02", "2025-08-02"],
               "test": ["2025-08-03", "2025-08-03"]},
        batch_size=8, epochs=1, base_channels=8, output_dir=str(tmp_path / "runs"),
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    from stormcast.ml.train_nowcast import main

    main(["--config", str(cfg_path), "--max-train-batches", "3"])
    metrics = json.loads(next((tmp_path / "runs").glob("nowcast-*/metrics.json")).read_text())
    assert set(metrics["test"]) == {"unet", "persistence", "optical_flow"}
    assert set(metrics["test"]["unet"]) == {f"+{10 * k}min" for k in range(1, 7)}
    assert all(np.isfinite(v["mae_k"]) for v in metrics["test"]["persistence"].values())
