import numpy as np
import pandas as pd

from src.cleaning import (
    build_store_category_map,
    drop_invalid_rows,
    fill_category,
    fix_values,
)


def make_row(**overrides):
    row = dict(
        created_at=pd.Timestamp("2015-02-01 10:00:00"),
        actual_delivery_time=pd.Timestamp("2015-02-01 10:45:00"),
        store_id=1,
        store_primary_category="italian",
        total_items=2,
        subtotal=2000,
        min_item_price=500,
        max_item_price=1500,
        total_onshift_dashers=10.0,
        total_busy_dashers=5.0,
        total_outstanding_orders=8.0,
        duration=2700.0,
    )
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_valid_row_is_unchanged():
    out = fix_values(frame(make_row()))
    assert out.loc[0, "min_item_price"] == 500
    assert out.loc[0, "max_item_price"] == 1500
    assert out.loc[0, "subtotal"] == 2000
    assert out.loc[0, "total_busy_dashers"] == 5.0


def test_negative_counts_become_nan():
    out = fix_values(frame(make_row(total_busy_dashers=-1.0,
                                    total_outstanding_orders=-6.0)))
    assert np.isnan(out.loc[0, "total_busy_dashers"])
    assert np.isnan(out.loc[0, "total_outstanding_orders"])
    assert out.loc[0, "total_onshift_dashers"] == 10.0


def test_negative_min_price_becomes_nan_and_max_is_kept():
    out = fix_values(frame(make_row(min_item_price=-86)))
    assert np.isnan(out.loc[0, "min_item_price"])
    assert out.loc[0, "max_item_price"] == 1500


def test_reversed_prices_both_become_nan():
    out = fix_values(frame(make_row(min_item_price=900, max_item_price=800)))
    assert np.isnan(out.loc[0, "min_item_price"])
    assert np.isnan(out.loc[0, "max_item_price"])


def test_zero_subtotal_becomes_nan():
    out = fix_values(frame(make_row(subtotal=0)))
    assert np.isnan(out.loc[0, "subtotal"])


def test_drop_invalid_rows_and_log():
    df = frame(
        make_row(),                        # valid
        make_row(duration=np.nan),         # missing target
        make_row(duration=20000.0),        # > 3 h
        make_row(duration=100.0),          # < 5 min
        make_row(total_items=99),          # too many items
    )
    out, log = drop_invalid_rows(df)
    assert len(out) == 1
    assert log["rows_removed"].tolist() == [1, 1, 1, 1]


def test_category_fill_uses_store_mode_then_unknown():
    df = frame(
        make_row(store_id=1, store_primary_category="italian"),
        make_row(store_id=1, store_primary_category="italian"),
        make_row(store_id=1, store_primary_category=np.nan),
        make_row(store_id=2, store_primary_category=np.nan),
    )
    store_map = build_store_category_map(df)
    assert store_map == {1: "italian"}
    out = fill_category(df, store_map)
    assert out["store_primary_category"].tolist() == [
        "italian", "italian", "italian", "unknown"
    ]