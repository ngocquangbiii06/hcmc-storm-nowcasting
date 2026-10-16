"""Himawari AHI fixed-grid navigation.

Maps lat/lon to full-disk line/column using the normalized geostationary projection
from the JMA Himawari Standard Data (HSD) User's Guide. Used to download only the
segments that cover our target grid instead of the whole disk.
"""
from __future__ import annotations

import math

SUB_LON = 140.7  # Himawari-8/9 sub-satellite longitude
SAT_DISTANCE_KM = 42164.0
EARTH_POL_RADIUS_KM = 6356.7523
RPOL2_REQ2 = 0.993305616  # (r_pol / r_eq)^2
ECC2 = 0.00669438444  # first eccentricity squared
N_SEGMENTS = 10

# resolution code -> (CFAC == LFAC, COFF == LOFF, lines per segment)
RESOLUTIONS = {
    "R05": (81865099, 11000.5, 2200),
    "R10": (40932549, 5500.5, 1100),
    "R20": (20466275, 2750.5, 550),
}


def latlon_to_linecol(lat: float, lon: float, resolution: str = "R20") -> tuple[float, float]:
    """Return the 1-based (line, column) of a lat/lon point on the AHI full disk."""
    cfac, off, _ = RESOLUTIONS[resolution]
    c_lat = math.atan(RPOL2_REQ2 * math.tan(math.radians(lat)))
    dlon = math.radians(lon - SUB_LON)
    rl = EARTH_POL_RADIUS_KM / math.sqrt(1 - ECC2 * math.cos(c_lat) ** 2)
    r1 = SAT_DISTANCE_KM - rl * math.cos(c_lat) * math.cos(dlon)
    r2 = -rl * math.cos(c_lat) * math.sin(dlon)
    r3 = rl * math.sin(c_lat)
    rn = math.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    x = math.degrees(math.atan(-r2 / r1))
    y = math.degrees(math.asin(-r3 / rn))
    scale = cfac / 2**16
    return off + y * scale, off + x * scale


def segment_of_line(line: float, resolution: str = "R20") -> int:
    lines_per_segment = RESOLUTIONS[resolution][2]
    return min(N_SEGMENTS, max(1, int((line - 1) // lines_per_segment) + 1))


def segments_for_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution: str = "R20",
    margin_lines: int = 10,
) -> list[int]:
    """Segments needed to cover a lat/lon box (plus a few lines of margin for resampling)."""
    lines = [
        latlon_to_linecol(lat, lon, resolution)[0]
        for lat in (lat_min, lat_max)
        for lon in (lon_min, (lon_min + lon_max) / 2, lon_max)
    ]
    first = segment_of_line(min(lines) - margin_lines, resolution)
    last = segment_of_line(max(lines) + margin_lines, resolution)
    return list(range(first, last + 1))
