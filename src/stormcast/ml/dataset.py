"""PyTorch dataset over the gold Zarr store: satellite BT (t, t-10, ...) -> radar CAPPI at t."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

FRAME_STEP = pd.Timedelta(minutes=10)


def split_indices(times: np.ndarray, start: str, end: str, history_steps: int) -> np.ndarray:
    """Indices whose date is in [start, end] and whose `history_steps` previous frames exist.

    Splits are contiguous blocks of days: neighbouring 10-min frames are nearly identical,
    so a random frame split would leak test weather into training.
    """
    t = pd.DatetimeIndex(times)
    in_split = (t.normalize() >= pd.Timestamp(start)) & (t.normalize() <= pd.Timestamp(end))
    available = set(t)
    keep = []
    for i in np.flatnonzero(in_split):
        if all((t[i] - j * FRAME_STEP) in available for j in range(1, history_steps + 1)):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def band_stats(
    ds: xr.Dataset, indices: np.ndarray, max_frames: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std of brightness temperature on (a sample of) the training frames."""
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(indices, size=min(max_frames, len(indices)), replace=False))
    bt = ds["bt"].isel(time=sample).values  # (n, band, y, x)
    return np.nanmean(bt, axis=(0, 2, 3)), np.nanstd(bt, axis=(0, 2, 3))


class SatToRadarDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        history_steps: int = 0,
        max_dbz: float = 70.0,
    ):
        self.ds = xr.open_zarr(zarr_path)
        self.times = pd.DatetimeIndex(self.ds["time"].values)
        self.pos = {t: i for i, t in enumerate(self.times)}
        self.indices = indices
        self.mean = mean.astype("float32")[:, None, None]
        self.std = std.astype("float32")[:, None, None]
        self.history_steps = history_steps
        self.max_dbz = max_dbz
        self.bands = [str(b) for b in self.ds["band"].values]

    def __len__(self) -> int:
        return len(self.indices)

    def frame_indices(self, i: int) -> list[int]:
        t = self.times[i]
        return [self.pos[t - j * FRAME_STEP] for j in range(self.history_steps, -1, -1)]  # oldest -> current

    def __getitem__(self, k: int) -> dict:
        i = int(self.indices[k])
        bt = self.ds["bt"].isel(time=self.frame_indices(i)).values.astype("float32")  # (T, band, y, x)
        x = np.nan_to_num((bt - self.mean) / self.std, nan=0.0).reshape(-1, *bt.shape[-2:])
        dbz = self.ds["dbz"].isel(time=i).values.astype("float32")
        valid = np.isfinite(dbz)
        y = np.clip(np.nan_to_num(dbz, nan=0.0), 0.0, self.max_dbz) / self.max_dbz
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y[None]),
            "valid": torch.from_numpy(valid[None]),
            "bt13": torch.from_numpy(bt[-1, self.bands.index("B13")]),
            "dbz": torch.from_numpy(dbz),
        }
