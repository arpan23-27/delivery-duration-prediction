import numpy as np
import pandas as pd
import pytest

from src.encoders import StoreEncoder, oof_encode


def test_smoothing_shrinks_small_stores_toward_global_mean():
    df = pd.DataFrame({"store_id": ["a"] * 2 + ["b"] * 4})
    y = pd.Series([100.0, 100.0, 300.0, 300.0, 300.0, 300.0])
    enc = StoreEncoder(smoothing=2).fit(df, y)
    out = enc.transform(df)
    global_mean = y.mean()                               # 233.33
    assert out["store_te"].iloc[0] == pytest.approx((2 * 100 + 2 * global_mean) / 4)
    assert out["store_share"].iloc[0] == pytest.approx(2 / 6)


def test_unseen_store_gets_global_mean_and_zero_share():
    df = pd.DataFrame({"store_id": ["a", "a"]})
    enc = StoreEncoder().fit(df, pd.Series([10.0, 30.0]))
    out = enc.transform(pd.DataFrame({"store_id": ["zzz"]}))
    assert out["store_te"].iloc[0] == pytest.approx(20.0)
    assert out["store_share"].iloc[0] == 0.0


def test_oof_encoding_ignores_own_target():
    df = pd.DataFrame({"store_id": ["s1"] * 20})
    groups = np.repeat(np.arange(10), 2)
    y1 = pd.Series(np.arange(20, dtype=float))
    y2 = y1.copy()
    y2.iloc[0] = 10_000.0                                # change only row 0's target
    a = oof_encode(df, y1, groups, n_splits=5)
    b = oof_encode(df, y2, groups, n_splits=5)
    assert a["store_te"].iloc[0] == b["store_te"].iloc[0]          # unaffected by its own y
    naive = StoreEncoder().fit(df, y2).transform(df)
    assert naive["store_te"].iloc[0] != a["store_te"].iloc[0]       # in-sample would leak it