import numpy as np
import pandas as pd

from src.serving import prepare_orders

STORE_MAP = {1: "mexican", 2: "japanese"}


def prep(**overrides):
    row = dict(created_at="2015-02-10 02:00:00", store_id=1, total_items=3, subtotal=2500.0,
               num_distinct_items=2, estimated_order_place_duration=300)
    row.update(overrides)
    return prepare_orders(pd.DataFrame([row]), STORE_MAP).iloc[0]


def test_minimal_order_derives_category_and_leaves_missing_fields_nan():
    r = prep()
    assert r["store_primary_category"] == "mexican"
    assert r["market_id"] == "unknown" and r["order_protocol"] == "unknown"
    assert np.isnan(r["total_onshift_dashers"]) and r["is_dasher_data_missing"] == 1
    assert np.isnan(r["estimated_total"])


def test_unseen_store_falls_back_to_unknown_category():
    assert prep(store_id=999999)["store_primary_category"] == "unknown"


def test_explicit_category_is_kept():
    assert prep(store_primary_category="pizza")["store_primary_category"] == "pizza"


def test_bad_values_are_cleaned_like_in_training():
    r = prep(subtotal=0.0, total_onshift_dashers=-3, min_item_price=900.0, max_item_price=100.0)
    assert np.isnan(r["subtotal"]) and np.isnan(r["total_onshift_dashers"])
    assert np.isnan(r["min_item_price"]) and np.isnan(r["max_item_price"])


def test_time_features_use_pacific_time():
    r = prep()                      # 02:00 UTC on Feb 10 = 18:00 Pacific on Monday Feb 9
    assert r["hour"] == 18 and r["day_of_week"] == 0