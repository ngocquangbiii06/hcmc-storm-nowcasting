"""Pair each satellite frame with the radar scan closest to its *observation* time."""
from __future__ import annotations

import pandas as pd


def pair_frames(sat: pd.DataFrame, radar: pd.DataFrame, tolerance_minutes: float) -> pd.DataFrame:
    """Nearest-time join.

    sat:   columns nominal_time, obs_time, silver_path (status == ok rows only)
    radar: columns scan_time, silver_path              (status == ok rows only)
    Returns one row per satellite frame that has a radar scan within the tolerance.
    """
    if sat.empty or radar.empty:
        return pd.DataFrame(columns=["time", "sat_obs_time", "radar_time", "sat_path", "radar_path", "dt_seconds"])
    left = (
        sat[["nominal_time", "obs_time", "silver_path"]]
        .rename(columns={"silver_path": "sat_path"})
        .assign(obs_time=lambda d: pd.to_datetime(d["obs_time"]))
        .sort_values("obs_time")
    )
    right = (
        radar[["scan_time", "silver_path"]]
        .rename(columns={"silver_path": "radar_path"})
        .assign(scan_time=lambda d: pd.to_datetime(d["scan_time"]))
        .sort_values("scan_time")
    )
    pairs = pd.merge_asof(
        left,
        right,
        left_on="obs_time",
        right_on="scan_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    ).dropna(subset=["radar_path"])
    pairs = pairs.rename(columns={"nominal_time": "time", "obs_time": "sat_obs_time", "scan_time": "radar_time"})
    pairs["dt_seconds"] = (pairs["radar_time"] - pairs["sat_obs_time"]).dt.total_seconds()
    return pairs.sort_values("time").reset_index(drop=True)
