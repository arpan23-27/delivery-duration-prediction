"""Serving core: raw orders -> predicted delivery duration, reusing the training code paths."""
import json
from pathlib import Path

import pandas as pd

import src.cleaning as C
import src.features as F
import src.pipeline as PL

RAW_COLUMNS = [
    "created_at", "market_id", "store_id", "store_primary_category", "order_protocol",
    "total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price",
    "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders",
    "estimated_order_place_duration", "estimated_store_to_consumer_driving_duration",
]
FLOAT_COLUMNS = [
    "market_id", "order_protocol", "total_items", "subtotal", "num_distinct_items",
    "min_item_price", "max_item_price", "total_onshift_dashers", "total_busy_dashers",
    "total_outstanding_orders", "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
]


def load_bundle(artifacts_dir="artifacts"):
    d = Path(artifacts_dir)
    artifacts = PL.load_artifacts(d / "final.joblib")
    raw_map = json.loads((d / "store_category_map.json").read_text())
    store_map = {int(k): v for k, v in raw_map.items()}      # JSON keys are strings
    return {"artifacts": artifacts, "store_map": store_map}


def prepare_orders(raw, store_map):
    """raw: DataFrame holding any subset of RAW_COLUMNS. Returns engineered features."""
    df = raw.reindex(columns=RAW_COLUMNS).copy()
    df["created_at"] = pd.to_datetime(df["created_at"])            # naive UTC
    for c in FLOAT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["store_id"] = df["store_id"].astype("int64")
    df["store_primary_category"] = df["store_primary_category"].astype("str")
    df = C.fix_values(df)                                           # same value fixes as training
    df = C.fill_category(df, store_map)                             # category from store_id
    return F.build_features(df)


def predict_orders(bundle, raw):
    """Predicted delivery duration in seconds, one value per row of `raw`."""
    feats = prepare_orders(raw, bundle["store_map"])
    return PL.predict(bundle["artifacts"], feats)