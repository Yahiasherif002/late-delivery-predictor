import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from ldp.api import create_app
from ldp.features import FEATURES, derive, haversine_km, history, orders_in_previous_days
from ldp.train import make_pipeline


def test_haversine_matches_the_sao_paulo_to_rio_distance():
    assert 350 < haversine_km(-23.5505, -46.6333, -22.9068, -43.1729) < 365


def test_history_only_counts_orders_delivered_before_the_purchase():
    t = pd.Timestamp
    orders = pd.DataFrame({
        "seller_id": ["a", "a", "a", "b"],
        "purchased_at": [t("2018-01-01"), t("2018-01-03"), t("2018-01-10"), t("2018-01-02")],
        "delivered_at": [t("2018-01-05"), t("2018-01-08"), t("2018-01-20"), t("2018-01-04")],
        "late": [1, 0, 0, 0],
    })

    seller = history(orders, "seller_id", "delivered_at", "late", prior_weight=1.0)

    assert seller["n"].tolist() == [0, 0, 2, 0]
    # The third order sees seller a's two deliveries (one late) and a market rate of 1 late in 3.
    assert np.isclose(seller.loc[2, "mean"], (1 + 1 / 3) / (2 + 1))


def test_orders_in_previous_days_counts_a_rolling_window():
    times = pd.Series(pd.to_datetime(["2018-01-10", "2018-01-01", "2018-01-05", "2018-01-09"]))
    assert orders_in_previous_days(times, days=7).tolist() == [2, 0, 1, 1]


def synthetic_orders(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "promised_days": rng.uniform(5, 40, n), "approval_hours": rng.uniform(0, 48, n),
        "purchase_weekday": rng.integers(0, 7, n),
        "n_items": rng.integers(1, 4, n), "n_sellers": 1, "price": rng.uniform(10, 500, n),
        "freight": rng.uniform(5, 60, n), "weight_kg": rng.uniform(0.1, 10, n), "volume_l": rng.uniform(0.5, 40, n),
        "distance_km": rng.uniform(1, 3000, n), "seller_prior_orders": rng.integers(0, 300, n),
        "seller_prior_late_rate": rng.uniform(0, 0.4, n), "seller_handling_days": rng.uniform(0, 6, n),
        "route_prior_late_rate": rng.uniform(0, 0.3, n), "orders_last_7d": rng.integers(500, 2500, n),
        "customer_state": rng.choice(["SP", "RJ", "MG"], n), "seller_state": rng.choice(["SP", "PR"], n),
        "category": rng.choice(["bed_bath_table", "health_beauty"], n),
    })
    late = (df["distance_km"] / df["promised_days"] > 80).astype(int)
    return derive(df), late


def test_the_api_scores_an_order_and_rejects_invalid_input(tmp_path):
    x, y = synthetic_orders()
    pipeline = make_pipeline(LogisticRegression(max_iter=1000)).fit(x[FEATURES], y)
    joblib.dump({"pipeline": pipeline, "threshold": 0.5}, tmp_path / "model.joblib")
    client = TestClient(create_app(tmp_path))
    order = {"promised_days": 6, "approval_hours": 2, "purchase_weekday": 0, "n_items": 1,
             "n_sellers": 1, "price": 120, "freight": 40, "weight_kg": 2, "volume_l": 10, "distance_km": 2500,
             "seller_prior_orders": 50, "seller_prior_late_rate": 0.2, "seller_handling_days": 2.5,
             "route_prior_late_rate": 0.1, "orders_last_7d": 1500, "customer_state": "RJ",
             "seller_state": "SP", "category": "health_beauty"}

    far = client.post("/predict", json=order)
    near = client.post("/predict", json={**order, "distance_km": 20, "promised_days": 30})

    assert far.status_code == 200
    assert far.json()["late_probability"] > near.json()["late_probability"]
    assert far.json()["high_risk"] is True
    assert client.post("/predict", json={**order, "purchase_weekday": 7}).status_code == 422
