"""The shared 256x256 target grid that radar and satellite are both mapped onto.

Convention used everywhere in this repo: arrays are (y, x) with row 0 = NORTH edge
(image / satpy convention). Py-ART grids come out south-up and are flipped on write.
"""
from __future__ import annotations

import math

import numpy as np

from stormcast.config import GridConfig

# Py-ART's default 'pyart_aeqd' projection uses a spherical earth of this radius.
# Using the same sphere for the satpy AreaDefinition keeps both grids identical.
PYART_EARTH_RADIUS_M = 6370997.0
METERS_PER_DEG_LAT = math.pi * PYART_EARTH_RADIUS_M / 180


def axis_centers(grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates in metres: x ascending (west->east), y descending (north->south)."""
    h, r, n = grid.half_extent_m, grid.resolution_m, grid.size
    x = -h + r / 2 + r * np.arange(n)
    return x, x[::-1].copy()


def pyart_grid_limits(grid: GridConfig) -> tuple[float, float]:
    """(min, max) pixel-centre coordinate so Py-ART's spacing is exactly resolution_m."""
    edge = grid.half_extent_m - grid.resolution_m / 2
    return -edge, edge


def approx_bbox(grid: GridConfig, margin: float = 0.1) -> tuple[float, float, float, float]:
    """Approximate (lat_min, lat_max, lon_min, lon_max) of the grid, padded by `margin` (fraction)."""
    half = grid.half_extent_m * (1 + margin)
    dlat = half / METERS_PER_DEG_LAT
    dlon = half / (METERS_PER_DEG_LAT * math.cos(math.radians(grid.center_lat)))
    return (
        grid.center_lat - dlat,
        grid.center_lat + dlat,
        grid.center_lon - dlon,
        grid.center_lon + dlon,
    )


def proj_dict(grid: GridConfig) -> dict:
    return {
        "proj": "aeqd",
        "lat_0": grid.center_lat,
        "lon_0": grid.center_lon,
        "R": PYART_EARTH_RADIUS_M,
        "units": "m",
    }


def target_area(grid: GridConfig):
    """pyresample AreaDefinition for the grid (row 0 = north)."""
    from pyresample import create_area_def

    h = grid.half_extent_m
    return create_area_def(
        "nhabe_radar_grid",
        proj_dict(grid),
        width=grid.size,
        height=grid.size,
        area_extent=(-h, -h, h, h),
        description="Nha Be radar-centred aeqd grid",
    )


def lonlats(grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    lons, lats = target_area(grid).get_lonlats()
    return np.asarray(lons, dtype="float32"), np.asarray(lats, dtype="float32")
