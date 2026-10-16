"""Satellite nowcasting dataset: past Himawari frames (all bands) -> future B13 brightness temperature."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

FRAME_STEP = pd.Timedelta(minutes=10)


def window_indices(times: np.ndarray, start: str, end: str, input_steps: int, lead_steps: int) -> np.ndarray:
    """Indices t0 (last input frame) whose whole window [t0-(in-1)*10min, t0+lead*10min] exists
    and lies inside [start, end]. Keeping full windows inside one split prevents a training
    target from being a validation input. Windows crossing a missing frame are dropped.
    """
    t = pd.DatetimeIndex(times)
    available = set(t)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    offsets = range(-(input_steps - 1), lead_steps + 1)
    keep = [
        i
        for i, t0 in enumerate(t)
        if t0 - (input_steps - 1) * FRAME_STEP >= lo
        and t0 + lead_steps * FRAME_STEP < hi
        and all(t0 + k * FRAME_STEP in available for k in offsets)
    ]
    return np.asarray(keep, dtype=int)


class SatNowcastDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        input_steps: int = 4,
        lead_steps: int = 6,
        target_band: str = "B13",
        bt: np.ndarray | None = None,
    ):
        """`bt`: optional in-memory copy of the whole (time, band, y, x) array, shared between splits.
        Reading 10 frames per sample straight from Zarr is IO-bound; preloading as float16 halves RAM."""
        self.ds = xr.open_zarr(zarr_path)
        self.bt = bt
        self.times = pd.DatetimeIndex(self.ds["time"].values)
        self.pos = {t: i for i, t in enumerate(self.times)}
        self.indices = indices
        self.mean = mean.astype("float32")[:, None, None]
        self.std = std.astype("float32")[:, None, None]
        self.input_steps = input_steps
        self.lead_steps = lead_steps
        self.band = [str(b) for b in self.ds["band"].values].index(target_band)

    def __len__(self) -> int:
        return len(self.indices)

    def frames(self, i: int) -> list[int]:
        t0 = self.times[i]
        return [self.pos[t0 + k * FRAME_STEP] for k in range(-(self.input_steps - 1), self.lead_steps + 1)]

    def __getitem__(self, k: int) -> dict:
        i = int(self.indices[k])
        idx = self.frames(i)
        raw = self.bt[idx] if self.bt is not None else self.ds["bt"].isel(time=idx).values
        bt = raw.astype("float32")  # (in + lead, band, y, x)
        inputs, b = bt[: self.input_steps], self.band
        x = np.nan_to_num((inputs - self.mean) / self.std, nan=0.0).reshape(-1, *bt.shape[-2:])
        target = bt[self.input_steps :, b]
        m, s = self.mean[b], self.std[b]
        return {
            "x": torch.from_numpy(x),
            "last_norm": torch.from_numpy(np.nan_to_num((inputs[-1, b] - m) / s, nan=0.0)),
            "y_norm": torch.from_numpy(np.nan_to_num((target - m) / s, nan=0.0)),
            "valid": torch.from_numpy(np.isfinite(target)),
            "target": torch.from_numpy(target),
            "last": torch.from_numpy(inputs[-1, b]),
            "prev": torch.from_numpy(inputs[-2, b]),
            "t0": str(self.times[i]),
        }
