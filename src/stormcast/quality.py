"""Data-quality rules applied to every processed frame, plus time-coverage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

OK, BAD = "ok", "bad"


def check_frame(
    arr: np.ndarray,
    valid_range: tuple[float, float],
    max_nan_fraction: float,
    check_flat: bool = True,
) -> dict:
    """Return QC metrics and a status for one gridded frame (any shape).

    check_flat=False for radar: a clear-sky CAPPI is legitimately all "no echo".
    """
    arr = np.asarray(arr, dtype="float64")
    finite = np.isfinite(arr)
    nan_fraction = 1.0 - finite.mean() if arr.size else 1.0
    result = {
        "nan_fraction": float(nan_fraction),
        "vmin": float(arr[finite].min()) if finite.any() else np.nan,
        "vmax": float(arr[finite].max()) if finite.any() else np.nan,
        "out_of_range_fraction": 0.0,
        "issues": "",
        "status": OK,
    }
    issues = []
    if not finite.any():
        issues.append("empty")
    else:
        lo, hi = valid_range
        vals = arr[finite]
        oor = float(((vals < lo) | (vals > hi)).mean())
        result["out_of_range_fraction"] = oor
        if oor > 0.001:
            issues.append("out_of_range")
        if check_flat and np.nanstd(vals) == 0:
            issues.append("flat")
    if nan_fraction > max_nan_fraction:
        issues.append("too_many_nan")
    if issues:
        result["issues"] = ",".join(issues)
        result["status"] = BAD
    return result


def missing_timesteps(times: pd.Series, start: str, end: str, freq_minutes: int) -> pd.DatetimeIndex:
    """Expected regular timesteps in [start, end-of-day] that never appear in `times`."""
    expected = pd.date_range(
        pd.Timestamp(start),
        pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=freq_minutes),
        freq=f"{freq_minutes}min",
    )
    present = pd.DatetimeIndex(pd.to_datetime(times)).floor(f"{freq_minutes}min")
    return expected.difference(present)


def gaps(times: pd.Series, max_gap_minutes: float) -> pd.DataFrame:
    """Intervals between consecutive timestamps that exceed max_gap_minutes."""
    t = pd.Series(pd.to_datetime(times)).sort_values().reset_index(drop=True)
    delta = t.diff()
    mask = delta > pd.Timedelta(minutes=max_gap_minutes)
    return pd.DataFrame(
        {
            "gap_start": t.shift(1)[mask].values,
            "gap_end": t[mask].values,
            "gap_minutes": (delta[mask].dt.total_seconds() / 60).values,
        }
    )
