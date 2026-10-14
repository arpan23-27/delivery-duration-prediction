import pytest
from fastapi.testclient import TestClient

from app.main import MAX_BATCH, app, known_stores

ORDER = {"store_id": 1, "total_items": 3, "num_distinct_items": 2, "subtotal": 2500,
         "estimated_order_place_duration": 300}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:                       # runs the lifespan (loads artifacts)
        yield c


@pytest.fixture(scope="module")
def known_store(client):
    return int(known_stores(client.app.state.bundle)[0])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_version"]


def test_predict_known_store_returns_sane_duration(client, known_store):
    r = client.post("/predict", json={**ORDER, "store_id": known_store})
    assert r.status_code == 200
    body = r.json()
    assert 10 < body["predicted_duration_minutes"] < 150
    assert body["predicted_duration_seconds"] == pytest.approx(
        body["predicted_duration_minutes"] * 60, abs=1)
    assert body["store_known"] is True


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"

def test_unseen_store_still_predicts_and_is_flagged(client):
    r = client.post("/predict", json={**ORDER, "store_id": 999_999_999})
    assert r.status_code == 200 and r.json()["store_known"] is False


@pytest.mark.parametrize("bad", [
    {"subtotal": -5},
    {"total_items": 0},
    {"store_id": "abc"},
    {"unexpected_field": 1},
])
def test_invalid_payloads_are_rejected(client, bad):
    assert client.post("/predict", json={**ORDER, **bad}).status_code == 422


def test_missing_required_field_is_rejected(client):
    incomplete = {k: v for k, v in ORDER.items() if k != "subtotal"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_timezone_aware_and_naive_timestamps_agree(client, known_store):
    base = {**ORDER, "store_id": known_store}
    zulu = client.post("/predict", json={**base, "created_at": "2015-02-10T02:00:00Z"}).json()
    naive = client.post("/predict", json={**base, "created_at": "2015-02-10T02:00:00"}).json()
    pacific = client.post("/predict", json={**base, "created_at": "2015-02-09T18:00:00-08:00"}).json()
    assert zulu["predicted_duration_seconds"] == naive["predicted_duration_seconds"] \
        == pacific["predicted_duration_seconds"]


def test_questionable_values_are_cleaned_not_rejected(client, known_store):
    r = client.post("/predict", json={**ORDER, "store_id": known_store, "subtotal": 0,
                                      "total_onshift_dashers": -3,
                                      "min_item_price": 900, "max_item_price": 100})
    assert r.status_code == 200


def test_batch_matches_single_predictions(client, known_store):
    a = {**ORDER, "store_id": known_store, "created_at": "2015-02-10T02:00:00Z"}
    b = {**ORDER, "store_id": known_store, "total_items": 8, "created_at": "2015-02-10T03:00:00Z"}
    batch = client.post("/predict/batch", json={"orders": [a, b]}).json()
    single = [client.post("/predict", json=o).json() for o in (a, b)]
    assert [p["predicted_duration_seconds"] for p in batch] == \
           [p["predicted_duration_seconds"] for p in single]


def test_batch_size_limits(client):
    assert client.post("/predict/batch", json={"orders": []}).status_code == 422
    too_many = {"orders": [ORDER] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422