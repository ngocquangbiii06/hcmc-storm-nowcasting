"""Synthetic radar volume -> CAPPI: checks orientation, no-echo filling and coverage NaNs."""
import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from stormcast.config import load_config  # noqa: E402
from stormcast.process.radar_cappi import radar_to_cappi  # noqa: E402


def make_volume(storm_azimuth: float, storm_range_m: float):
    elevations = [0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0]
    radar = pyart.testing.make_empty_ppi_radar(240, 360, len(elevations))
    radar.range["data"] = np.arange(240, dtype="float32") * 500.0 + 250.0  # 0-120 km
    radar.fixed_angle["data"] = np.array(elevations, dtype="float32")
    radar.elevation["data"] = np.repeat(elevations, 360).astype("float32")
    radar.azimuth["data"] = np.tile(np.arange(360), len(elevations)).astype("float32")
    radar.latitude["data"] = np.array([10.6596])
    radar.longitude["data"] = np.array([106.7283])
    radar.altitude["data"] = np.array([10.0])
    radar.init_gate_x_y_z()
    radar.init_gate_longitude_latitude()
    radar.init_gate_altitude()

    az = radar.azimuth["data"][:, None]
    rng = radar.range["data"][None, :]
    dist = np.hypot(rng * np.sin(np.deg2rad(az)) - storm_range_m * np.sin(np.deg2rad(storm_azimuth)),
                    rng * np.cos(np.deg2rad(az)) - storm_range_m * np.cos(np.deg2rad(storm_azimuth)))
    dbz = np.ma.masked_where(dist > 8000, np.full(dist.shape, 50.0, dtype="float32"))  # masked = no echo
    radar.add_field("reflectivity", {"data": dbz, "units": "dBZ"})
    return radar


def test_storm_north_of_radar_appears_in_top_half():
    cfg = load_config("configs/default.yaml")
    cappi = radar_to_cappi(make_volume(storm_azimuth=0, storm_range_m=40_000), cfg)

    assert cappi.shape == (256, 256)
    rows, cols = np.nonzero(np.nan_to_num(cappi, nan=-99) >= 40)  # the storm is a flat 50 dBZ disc
    assert len(rows) > 0
    # centre pixel is index 127.5 (between 127 and 128); 40 km north => row ~87.5
    assert abs(rows.mean() - (127.5 - 40)) <= 2  # row 0 = north
    assert abs(cols.mean() - 127.5) <= 2


def test_clear_air_is_no_echo_and_far_range_is_nan():
    cfg = load_config("configs/default.yaml")
    cappi = radar_to_cappi(make_volume(storm_azimuth=90, storm_range_m=40_000), cfg)

    assert cappi[128, 128 - 30] == pytest.approx(cfg.radar.no_echo_dbz, abs=0.5)  # 30 km west: covered, no rain
    assert np.nanmax(cappi[:, 128 + 30 : 128 + 50]) > 40  # storm 40 km east
    assert np.isnan(cappi[0, 0])  # corner is ~180 km away, beyond the 120 km synthetic range
