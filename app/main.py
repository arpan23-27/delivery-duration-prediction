"""FastAPI service for delivery-duration prediction."""
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi.responses import RedirectResponse

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

import src.serving as SV

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
MAX_BATCH = 1000

EXAMPLE_ORDER = {
    "store_id": 1845, "total_items": 3, "num_distinct_items": 3, "subtotal": 2500,
    "estimated_order_place_duration": 446, "created_at": "2015-02-14T01:30:00Z",
    "market_id": 1, "order_protocol": 1, "min_item_price": 557, "max_item_price": 1239,
    "total_onshift_dashers": 33, "total_busy_dashers": 14, "total_outstanding_orders": 21,
    "estimated_store_to_consumer_driving_duration": 861,
}


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_ORDER]})

    # required
    store_id: int = Field(ge=0)
    total_items: int = Field(ge=1)
    num_distinct_items: int = Field(ge=1)
    subtotal: float = Field(ge=0, description="order subtotal in cents")
    estimated_order_place_duration: float = Field(ge=0, description="seconds")
    # optional: missing values are handled as in training
    created_at: Optional[datetime] = Field(
        None, description="ISO 8601; naive timestamps are read as UTC; default is now")
    market_id: Optional[int] = Field(None, ge=1)
    order_protocol: Optional[int] = Field(None, ge=1)
    store_primary_category: Optional[str] = Field(
        None, description="overrides the category looked up from store_id")
    min_item_price: Optional[float] = None
    max_item_price: Optional[float] = None
    total_onshift_dashers: Optional[float] = None
    total_busy_dashers: Optional[float] = None
    total_outstanding_orders: Optional[float] = None
    estimated_store_to_consumer_driving_duration: Optional[float] = Field(
        None, description="seconds")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orders: list[Order] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    store_known: bool = Field(description="False if the store was never seen in training data")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = SV.load_bundle(ARTIFACTS_DIR)
    app.state.metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    yield


app = FastAPI(title="Delivery duration prediction", version="0.6.0", lifespan=lifespan)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


def _utc_naive(ts):
    if ts is None:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def known_stores(bundle) -> pd.Index:
    """Store ids the target encoder saw during fitting."""
    fitted = bundle.get("artifacts", bundle)          # tolerate wrapped or flat bundles
    return fitted["encoder"].stats_.index


def _predict(request: Request, orders: list[Order]) -> list[Prediction]:
    bundle = request.app.state.bundle
    df = pd.DataFrame([o.model_dump() for o in orders])
    df["created_at"] = [_utc_naive(o.created_at) for o in orders]
    seconds = SV.predict_orders(bundle, df)
    known = df["store_id"].isin(known_stores(bundle))
    version = request.app.state.metadata["model_version"]
    return [Prediction(predicted_duration_seconds=round(float(s), 1),
                       predicted_duration_minutes=round(float(s) / 60, 2),
                       store_known=bool(k), model_version=version)
            for s, k in zip(seconds, known)]


@app.get("/health")
def health(request: Request):
    meta = request.app.state.metadata
    return {"status": "ok", "model_version": meta["model_version"],
            "test_mae_min": meta["test_metrics"]["mae_min"]}


@app.post("/predict", response_model=Prediction)
def predict(order: Order, request: Request):
    return _predict(request, [order])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchRequest, request: Request):
    return _predict(request, batch.orders)