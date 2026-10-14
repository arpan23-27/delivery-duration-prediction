import numpy as np
import pandas as pd

from src.features import build_features


def make_row(**overrides):
    row = dict(
        created_at=pd.Timestamp("2015-02-01 10:00:00"),   # 02:00 Pacific, a Sunday
        market_id=1.0, order_protocol=3.0, store_primary_category="italian",
        total_items=4, subtotal=2000.0, num_distinct_items=2,
        min_item_price=200.0, max_item_price=1200.0,
        total_onshift_dashers=10.0, total_busy_dashers=5.0, total_outstanding_orders=8.0,
        estimated_order_place_duration=300, estimated_store_to_consumer_driving_duration=600.0,
    )
    row.update(overrides)
    return row


def build(**overrides):
    return build_features(pd.DataFrame([make_row(**overrides)])).iloc[0]


def test_time_features_use_pacific_time():
    r = build()
    assert r["hour"] == 2
    assert r["day_of_week"] == 6 and r["is_weekend"] == 1


def test_marketplace_features_normal_case():
    r = build()
    assert r["available_dashers"] == 5
    assert r["busy_ratio"] == 0.5
    assert r["outstanding_per_dasher"] == 0.8
    assert r["is_dasher_data_missing"] == 0


def test_marketplace_features_busy_exceeds_onshift_and_zero_onshift():
    r = build(total_busy_dashers=30.0)
    assert r["available_dashers"] == 0          # clipped
    assert r["busy_ratio"] == 2.0               # capped
    r0 = build(total_onshift_dashers=0.0)
    assert np.isnan(r0["busy_ratio"])           # no division by zero


def test_missing_dashers_flagged():
    r = build(total_onshift_dashers=np.nan, total_busy_dashers=np.nan,
              total_outstanding_orders=np.nan)
    assert r["is_dasher_data_missing"] == 1


def test_order_and_estimate_features():
    r = build()
    assert r["avg_item_price"] == 500.0
    assert r["price_range"] == 1000.0
    assert r["items_per_distinct"] == 2.0
    assert r["estimated_total"] == 900.0
    assert np.isnan(build(estimated_store_to_consumer_driving_duration=np.nan)["estimated_total"])


def test_categoricals_become_labels():
    r = build(market_id=np.nan)
    assert r["market_id"] == "unknown"
    assert r["order_protocol"] == "3"