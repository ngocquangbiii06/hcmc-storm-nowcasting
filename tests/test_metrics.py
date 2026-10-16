import numpy as np
import pytest

from stormcast.ml.baselines import tune_bt_threshold
from stormcast.ml.metrics import Contingency, fss


def test_contingency_scores():
    obs = np.array([40, 40, 10, 10, np.nan])
    pred = np.array([40, 10, 40, 10, 40])
    c = Contingency()
    c.update(pred, obs, 35)
    assert (c.hits, c.misses, c.false_alarms, c.correct_negatives) == (1, 1, 1, 1)
    assert c.csi == pytest.approx(1 / 3)
    assert c.pod == pytest.approx(0.5)
    assert c.far == pytest.approx(0.5)


def test_fss_perfect_and_displaced():
    obs = np.zeros((32, 32))
    obs[10:14, 10:14] = 50
    assert fss(obs, obs, 35, 5) == pytest.approx(1.0)
    shifted = np.roll(obs, 3, axis=1)
    assert 0 < fss(shifted, obs, 35, 1) < fss(shifted, obs, 35, 9) < 1


def test_cold_cloud_threshold_recovers_synthetic_rule():
    rng = np.random.default_rng(0)
    bt13 = rng.uniform(190, 300, 20000)
    dbz = np.where(bt13 < 220, 45.0, 5.0)
    threshold, csi = tune_bt_threshold(bt13, dbz, 35)
    assert 218 <= threshold <= 220
    assert csi > 0.95
