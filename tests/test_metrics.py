import numpy as np
import pytest

from src.metrics import block_bootstrap_mae, evaluate


def test_metrics_known_values():
    y = np.array([600.0, 1200.0, 1800.0])
    p = np.array([600.0, 1800.0, 1200.0])      # errors: 0, +600, -600
    m = evaluate(y, p)
    assert m["mae_min"] == pytest.approx(400 / 60)
    assert m["rmse_min"] == pytest.approx(np.sqrt((600**2 + 600**2) / 3) / 60)
    assert m["within_10min"] == pytest.approx(1.0)
    assert m["bias_min"] == pytest.approx(0.0)
    assert m["over_pct"] == pytest.approx(1 / 3)


def test_within_tolerance_is_inclusive_at_ten_minutes_only():
    m = evaluate([0.0, 0.0], [600.0, 601.0])
    assert m["within_10min"] == pytest.approx(0.5)
    

def test_block_bootstrap_resamples_whole_blocks():
    abs_err = {"a": [1, 1, 3, 3], "b": [2, 2, 2, 2]}
    boot = block_bootstrap_mae(abs_err, [0, 0, 1, 1], n_boot=200, seed=1)
    assert boot.shape == (200, 2)
    assert (boot["b"] == 2).all()                          # constant error -> no spread
    assert boot["a"].between(1, 3).all()
    assert boot["a"].nunique() > 1                          # blocks of mean 1 and 3 do get mixed