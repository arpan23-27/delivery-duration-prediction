""" Cleaning rules for the delivery-duration dataset.

Every rule below comes from the Phase 1 audit (notebooks/01_eda.ipynb).
"""

import numpy as np 
import pandas as pd

# ---- Thresholds (named so they are visible and easy to change) ----
MAX_DURATION_S = 3 * 60 * 60   # drop deliveries longer than 3 hours
MIN_DURATION_S = 5 * 60  # drop deliveries shorter than 5 minutes
MAX_ITEMS = 50   # drop orders with more than 50 items

NON_NEGATIVE_COLS = [
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "min_item_price",
]

UNKNOWN_CATEGORY = "unknown"

def add_target(df):
    """Add `duration` (seconds from order creation to delivery)."""
    df = df.copy()
    df["duration"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds()
    return df



def fix_values(df):
    """Blank out impossible values. Safe to apply at training AND serving time."""
    df = df.copy()

    # Integer columns cannot hold NaN in pandas 3, so convert them first.
    float_cols = ["subtotal", "min_item_price", "max_item_price"]
    df[float_cols] = df[float_cols].astype("float64")

    # Negative counts / prices are invalid -> NaN
    for col in NON_NEGATIVE_COLS:
        df.loc[df[col] < 0, col] = np.nan

    # min > max cannot be repaired from other columns -> blank both
    reversed_mask = df["min_item_price"] > df["max_item_price"]
    df.loc[reversed_mask, ["min_item_price", "max_item_price"]] = np.nan

    # A zero subtotal with items and positive prices is invalid -> NaN
    df.loc[df["subtotal"] == 0, "subtotal"] = np.nan

    return df





def drop_invalid_rows(df):
    """Training-only row drops. Returns (cleaned_df, log_df)."""
    rules = [
        ("missing target", lambda d: d["duration"].isna()),
        (f"duration > {MAX_DURATION_S} s", lambda d: d["duration"] > MAX_DURATION_S),
        (f"duration < {MIN_DURATION_S} s", lambda d: d["duration"] < MIN_DURATION_S),
        (f"total_items > {MAX_ITEMS}", lambda d: d["total_items"] > MAX_ITEMS),
    ]

    log = []
    for name, rule in rules:
        mask = rule(df)               # evaluated on rows that survived so far
        log.append({"rule": name, "rows_removed": int(mask.sum())})
        df = df.loc[~mask]

    return df.reset_index(drop=True), pd.DataFrame(log)




def build_store_category_map(df):
    """Map store_id -> its most common known category."""
    known = df.dropna(subset=["store_primary_category"])
    return (
        known.groupby("store_id")["store_primary_category"]
        .agg(lambda s: s.value_counts().sort_index().idxmax())
        .to_dict()
    )



def fill_category(df, store_category_map):
    """Fill missing categories from the store map, else 'unknown'."""
    df = df.copy()
    from_store = df["store_id"].map(store_category_map)
    df["store_primary_category"] = (
        df["store_primary_category"].fillna(from_store).fillna(UNKNOWN_CATEGORY)
    )
    return df



def clean_training_data(raw):
    """Full training-time pipeline. Returns (clean_df, drop_log, store_map)."""
    df = add_target(raw)
    df = fix_values(df)
    df, drop_log = drop_invalid_rows(df)
    store_map = build_store_category_map(df)
    df = fill_category(df, store_map)
    return df, drop_log, store_map




