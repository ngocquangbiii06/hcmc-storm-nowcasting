import numpy as np
import pandas as pd

from stormcast.align import pair_frames
from stormcast.quality import check_frame, gaps, missing_timesteps


def test_check_frame_ok():
    arr = np.random.default_rng(0).uniform(200, 300, (5, 8, 8))
    assert check_frame(arr, (170, 340), 0.2)["status"] == "ok"


def test_check_frame_flags_problems():
    assert check_frame(np.full((4, 4), np.nan), (0, 1), 0.2)["issues"].startswith("empty")
    assert "out_of_range" in check_frame(np.full((4, 4), 999.0), (170, 340), 0.2)["issues"]
    half_nan = np.where(np.arange(16).reshape(4, 4) < 8, np.nan, 250.0)
    assert "too_many_nan" in check_frame(half_nan, (170, 340), 0.2)["issues"]


def test_clear_sky_radar_is_not_flat_error():
    clear = np.full((4, 4), -10.0)
    assert check_frame(clear, (-35, 80), 0.2, check_flat=False)["status"] == "ok"
    assert check_frame(clear, (-35, 80), 0.2)["status"] == "bad"


def test_missing_timesteps_and_gaps():
    times = pd.Series(pd.date_range("2025-08-01", periods=144, freq="10min").delete([5, 6]))
    missing = missing_timesteps(times, "2025-08-01", "2025-08-01", 10)
    assert list(missing) == [pd.Timestamp("2025-08-01 00:50"), pd.Timestamp("2025-08-01 01:00")]
    g = gaps(times, 20)
    assert len(g) == 1 and g["gap_minutes"].iloc[0] == 30


def test_pairing_uses_observation_time_and_tolerance():
    sat = pd.DataFrame(
        {
            "nominal_time": pd.to_datetime(["2025-08-01 00:00", "2025-08-01 00:10", "2025-08-01 00:20"]),
            "obs_time": pd.to_datetime(["2025-08-01 00:04", "2025-08-01 00:14", "2025-08-01 00:24"]),
            "silver_path": ["s0", "s1", "s2"],
        }
    )
    radar = pd.DataFrame(
        {
            # 00:05 is closest to obs 00:04 (but 5 min from nominal 00:00); nothing near 00:24
            "scan_time": pd.to_datetime(["2025-08-01 00:05", "2025-08-01 00:12", "2025-08-01 00:40"]),
            "silver_path": ["r0", "r1", "r2"],
        }
    )
    pairs = pair_frames(sat, radar, tolerance_minutes=5)
    assert pairs["sat_path"].tolist() == ["s0", "s1"]
    assert pairs["radar_path"].tolist() == ["r0", "r1"]
    assert pairs["dt_seconds"].tolist() == [60.0, -120.0]
