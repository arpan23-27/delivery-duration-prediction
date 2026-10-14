import numpy as np
import pandas as pd
import pytest

from src.modeling import (LGBM_CATEGORICAL, LGBM_NUMERIC, add_store_features, fit_categories, to_lgbm_frame)


def test_validation_rows_are_encoded_from_train_only():
    days = pd.date_range("2015-01-21", periods=10, freq="D").repeat(2)
    train = pd.DataFrame({"created_at": days,
                          "store_id": ["a", "b"] * 10,
                          "duration": np.arange(20, dtype=float) * 60})
    val = pd.DataFrame({"created_at": [pd.Timestamp("2015-02-12 12:00")] * 2,
                        "store_id": ["a", "zzz"],
                        "duration": [1.0, 2.0]})
    enc_train, (enc_val,) = add_store_features(train, [val], smoothing=1)
    assert enc_train[["store_te", "store_share"]].notna().all().all()
    assert enc_val["store_te"].iloc[1] == pytest.approx(train["duration"].mean())   # unseen store
    assert enc_val["store_share"].iloc[1] == 0.0
    
    
def test_lgbm_frame_nulls_categories_unseen_in_train():
    cols = LGBM_NUMERIC + LGBM_CATEGORICAL
    train = pd.DataFrame({c: [0.0, 0.0] for c in cols})
    for c in LGBM_CATEGORICAL:
        train[c] = ["a", "b"]
    new = train.copy()
    first = LGBM_CATEGORICAL[0]
    new[first] = ["a", "zzz"]
    X = to_lgbm_frame(new, fit_categories(train))
    assert list(X[first].cat.categories) == ["a", "b"]
    assert X[first].isna().tolist() == [False, True]