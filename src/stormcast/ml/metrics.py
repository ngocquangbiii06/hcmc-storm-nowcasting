"""Verification scores used in radar nowcasting (numpy only, no torch needed).

Accumulate the contingency table over the whole test set and compute scores once;
averaging per-frame CSI over-weights frames with almost no rain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Contingency:
    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    correct_negatives: int = 0

    def update(self, pred: np.ndarray, obs: np.ndarray, threshold: float, valid: np.ndarray | None = None) -> None:
        valid = np.isfinite(obs) & np.isfinite(pred) if valid is None else valid & np.isfinite(obs)
        self.update_events(pred >= threshold, obs >= threshold, valid)

    def update_events(self, pred_event: np.ndarray, obs_event: np.ndarray, valid: np.ndarray) -> None:
        p = pred_event & valid
        o = obs_event & valid
        self.hits += int((p & o).sum())
        self.misses += int((~p & o).sum())
        self.false_alarms += int((p & ~o).sum())
        self.correct_negatives += int((~p & ~o & valid).sum())

    @property
    def csi(self) -> float:
        d = self.hits + self.misses + self.false_alarms
        return self.hits / d if d else float("nan")

    @property
    def pod(self) -> float:
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    @property
    def far(self) -> float:
        d = self.hits + self.false_alarms
        return self.false_alarms / d if d else float("nan")

    def as_dict(self) -> dict:
        return {"csi": self.csi, "pod": self.pod, "far": self.far}


def fss(pred: np.ndarray, obs: np.ndarray, threshold: float, window: int) -> float:
    """Fractions Skill Score for one 2-D field (Roberts & Lean 2008). NaNs count as no event."""
    from scipy.ndimage import uniform_filter

    pb = (np.nan_to_num(pred, nan=-np.inf) >= threshold).astype(float)
    ob = (np.nan_to_num(obs, nan=-np.inf) >= threshold).astype(float)
    pf = uniform_filter(pb, size=window, mode="constant")
    of = uniform_filter(ob, size=window, mode="constant")
    mse = np.mean((pf - of) ** 2)
    ref = np.mean(pf**2) + np.mean(of**2)
    return 1.0 - mse / ref if ref > 0 else float("nan")
