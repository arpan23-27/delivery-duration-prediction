"""Stateless feature engineering: no statistics are learned from the data here."""
import numpy as np
import pandas as pd


LOCAL_TZ = "US/Pacific"          # created_at is UTC; the stores operate in US/Pacific
BUSY_RATIO_CAP = 2.0             # busy > on-shift happens in ~20% of rows; cap extremes
DASHER_COLS = ["total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]



ENGINEERED_NUMERIC = [
    "hour", "day_of_week", "is_weekend",
    "is_dasher_data_missing", "available_dashers", "busy_ratio", "outstanding_per_dasher",
    "avg_item_price", "price_range", "items_per_distinct",
    "estimated_total",
]
RAW_NUMERIC = [
    "total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price",
    "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders",
    "estimated_order_place_duration", "estimated_store_to_consumer_driving_duration",
]


NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = ["market_id", "order_protocol", "store_primary_category"]


def _safe_divide(numerator, denominator):
    """Divide, returning NaN where the denominator is missing or not positive."""
    return numerator / denominator.where(denominator > 0)


def add_time_features(df):
    df = df.copy()
    local = df["created_at"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    df["hour"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek          # Monday = 0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    return df


def add_marketplace_features(df):
    df = df.copy()
    onshift = df["total_onshift_dashers"]
    busy = df["total_busy_dashers"]
    outstanding = df["total_outstanding_orders"]

    df["is_dasher_data_missing"] = df[DASHER_COLS].isna().any(axis=1).astype("int8")
    df["available_dashers"] = (onshift - busy).clip(lower=0)
    df["busy_ratio"] = _safe_divide(busy, onshift).clip(upper=BUSY_RATIO_CAP)
    df["outstanding_per_dasher"] = _safe_divide(outstanding, onshift)
    return df


def add_order_features(df):
    df = df.copy()
    df["avg_item_price"] = _safe_divide(df["subtotal"], df["total_items"])
    df["price_range"] = df["max_item_price"] - df["min_item_price"]
    df["items_per_distinct"] = _safe_divide(df["total_items"], df["num_distinct_items"])
    return df


def add_estimate_features(df):
    df = df.copy()
    df["estimated_total"] = (
        df["estimated_order_place_duration"]
        + df["estimated_store_to_consumer_driving_duration"]
    )
    return df


def prepare_categoricals(df):
    """Turn id columns into readable string labels; missing -> 'unknown'."""
    df = df.copy()
    for col in ["market_id", "order_protocol"]:
        df[col] = df[col].astype("Int64").astype("string").fillna("unknown")
    df["store_primary_category"] = df["store_primary_category"].astype("string").fillna("unknown")
    return df


def build_features(df):
    df = add_time_features(df)
    df = add_marketplace_features(df)
    df = add_order_features(df)
    df = add_estimate_features(df)
    df = prepare_categoricals(df)
    return df