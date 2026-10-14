"""Smoothed target encoding of store_id, learned from the rows it is fit on."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SMOOTHING = 20   # pseudo-count: a store needs about this many orders before its own mean dominates
N_FOLDS = 5
OUTPUT_COLS = ["store_te", "store_share"]


class StoreEncoder:
    """store_id -> (smoothed mean target, share of fit rows)."""

    def __init__(self, smoothing=SMOOTHING):
        self.smoothing = smoothing

    def fit(self, df, y):
        tmp = pd.DataFrame({"store_id": df["store_id"].to_numpy(), "y": np.asarray(y)})
        self.global_mean_ = tmp["y"].mean()
        stats = tmp.groupby("store_id")["y"].agg(["mean", "count"])
        k = self.smoothing
        stats["store_te"] = (stats["count"] * stats["mean"] + k * self.global_mean_) / (stats["count"] + k)
        stats["store_share"] = stats["count"] / len(tmp)
        self.stats_ = stats[OUTPUT_COLS]
        return self

    def transform(self, df):
        out = df[["store_id"]].join(self.stats_, on="store_id")
        out["store_te"] = out["store_te"].fillna(self.global_mean_)
        out["store_share"] = out["store_share"].fillna(0.0)
        return out[OUTPUT_COLS]


def oof_encode(df, y, groups, n_splits=N_FOLDS, smoothing=SMOOTHING):
    """Encode each row using stats from the other folds, so no row sees its own target."""
    out = pd.DataFrame(index=df.index, columns=OUTPUT_COLS, dtype="float64")
    for fit_idx, enc_idx in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        enc = StoreEncoder(smoothing).fit(df.iloc[fit_idx], y.iloc[fit_idx])
        out.iloc[enc_idx] = enc.transform(df.iloc[enc_idx]).to_numpy()
    return out