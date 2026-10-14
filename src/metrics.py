"""Evaluation metrics. Inputs are seconds; outputs are labeled in minutes or fractions."""
import numpy as np
import pandas as pd

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
    

def block_bootstrap_mae(abs_err, blocks, n_boot=2000, seed=0):
    """abs_err: dict of model name -> absolute errors (min); blocks: block id per order.
    Resamples whole blocks; each draw is shared by all models, so comparisons are paired.
    Returns a DataFrame of bootstrapped MAEs (one column per model)."""
    names = list(abs_err)
    frame = pd.DataFrame({n: np.asarray(a, dtype=float) for n, a in abs_err.items()})
    frame["_block"] = np.asarray(blocks)
    frame["_n"] = 1
    per_block = frame.groupby("_block").sum()
    sums = per_block[names].to_numpy()                    # (n_blocks, n_models)
    counts = per_block["_n"].to_numpy()                   # (n_blocks,)
    idx = np.random.default_rng(seed).integers(0, len(per_block), size=(n_boot, len(per_block)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]
    return pd.DataFrame(boot, columns=names)