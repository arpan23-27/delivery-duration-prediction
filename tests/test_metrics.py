import numpy as np
import pytest

from src.metrics import evaluate


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