import numpy as np
import pytest

from stormcast.config import GridConfig
from stormcast.geos import latlon_to_linecol, segment_of_line, segments_for_bbox
from stormcast.grid import approx_bbox, axis_centers, pyart_grid_limits


def test_sub_satellite_point_is_disk_centre():
    line, col = latlon_to_linecol(0.0, 140.7)
    assert line == pytest.approx(2750.5)
    assert col == pytest.approx(2750.5)


def test_north_is_up_and_west_is_left():
    line_n, _ = latlon_to_linecol(20.0, 140.7)
    _, col_w = latlon_to_linecol(0.0, 120.0)
    assert line_n < 2750.5
    assert col_w < 2750.5


def test_hcmc_needs_segments_4_and_5():
    line, _ = latlon_to_linecol(10.6596, 106.7283)
    assert segment_of_line(line) == 4
    assert segments_for_bbox(*approx_bbox(GridConfig())) == [4, 5]


def test_axis_centres_are_1km_and_symmetric():
    x, y = axis_centers(GridConfig())
    assert len(x) == 256
    assert np.allclose(np.diff(x), 1000.0)
    assert np.allclose(np.diff(y), -1000.0)  # row 0 = north
    assert x[0] == -127500.0 and x[-1] == 127500.0


def test_pyart_limits_match_axis_centres():
    lo, hi = pyart_grid_limits(GridConfig())
    x, _ = axis_centers(GridConfig())
    assert (lo, hi) == (x[0], x[-1])
    assert np.isclose((hi - lo) / 255, 1000.0)
