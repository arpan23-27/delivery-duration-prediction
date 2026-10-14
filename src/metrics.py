"""Evaluation metrics. Inputs are seconds; outputs are labeled in minutes or fractions."""
import numpy as np

TOLERANCE_S = 10 * 60


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true                     # > 0 means over-prediction
    return {
        "mae_min": np.mean(np.abs(err)) / 60,
        "rmse_min": np.sqrt(np.mean(err ** 2)) / 60,
        "within_10min": np.mean(np.abs(err) <= TOLERANCE_S),
        "bias_min": np.mean(err) / 60,
        "over_pct": np.mean(err > 0),
    }