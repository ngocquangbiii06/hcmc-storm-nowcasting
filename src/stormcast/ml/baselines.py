"""Baselines the model has to beat.

Cold-cloud baseline: deep convection has very cold cloud tops, so "BT(10.4 um, B13) below
T => reflectivity above X dBZ". T is tuned on the training split for each dBZ threshold.
"""
from __future__ import annotations

import numpy as np

from stormcast.ml.metrics import Contingency


def tune_bt_threshold(
    bt13: np.ndarray,
    dbz: np.ndarray,
    dbz_threshold: float,
    candidates: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick the BT13 threshold (K) maximising CSI on the given pixels. Returns (threshold, csi)."""
    candidates = np.arange(190.0, 262.0, 2.0) if candidates is None else candidates
    valid = np.isfinite(bt13) & np.isfinite(dbz)
    obs = dbz >= dbz_threshold
    best = (float(candidates[0]), -1.0)
    for t in candidates:
        c = Contingency()
        c.update_events(bt13 <= t, obs, valid)
        if np.isfinite(c.csi) and c.csi > best[1]:
            best = (float(t), c.csi)
    return best


def cold_cloud_events(bt13: np.ndarray, bt_threshold: float) -> np.ndarray:
    return np.nan_to_num(bt13, nan=np.inf) <= bt_threshold


# --- satellite nowcasting baselines -------------------------------------------------------------

def persistence(last: np.ndarray, lead_steps: int) -> np.ndarray:
    """Nothing changes: repeat the last observed frame for every lead time."""
    return np.repeat(last[None], lead_steps, axis=0)


def optical_flow_forecast(
    prev: np.ndarray, last: np.ndarray, lead_steps: int, vmin: float = 180.0, vmax: float = 310.0
) -> np.ndarray:
    """Advect `last` along the motion between `prev` and `last` (Farneback flow, semi-Lagrangian).

    Assumes the motion stays constant, so clouds keep moving but never grow or decay -
    exactly the part a learned model has to improve on.
    """
    import cv2

    def to_u8(a: np.ndarray) -> np.ndarray:
        scaled = (np.nan_to_num(a, nan=vmax) - vmin) / (vmax - vmin) * 255
        return np.clip(scaled, 0, 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        to_u8(prev), to_u8(last), None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
        poly_sigma=1.2, flags=0,
    )
    h, w = last.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src = np.nan_to_num(last, nan=float(np.nanmean(last))).astype(np.float32)
    frames = [
        cv2.remap(src, gx - k * flow[..., 0], gy - k * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for k in range(1, lead_steps + 1)
    ]
    return np.stack(frames)
