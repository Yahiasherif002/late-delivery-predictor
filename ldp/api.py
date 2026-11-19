"""Score an order's risk of arriving late.

    uvicorn ldp.api:create_app --factory
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ldp.features import FEATURES, derive

MODELS = Path(__file__).resolve().parent.parent / "models"


class Order(BaseModel):
    promised_days: float = Field(gt=0, description="Days between purchase and the promised delivery date")
    approval_hours: float = Field(ge=0)
    purchase_weekday: int = Field(ge=0, le=6, description="0 = Monday")
    n_items: int = Field(ge=1)
    n_sellers: int = Field(ge=1)
    price: float = Field(ge=0)
    freight: float = Field(ge=0)
    weight_kg: float = Field(ge=0)
    volume_l: float = Field(ge=0)
    distance_km: float = Field(ge=0)
    seller_prior_orders: int = Field(ge=0)
    seller_prior_late_rate: float = Field(ge=0, le=1)
    seller_handling_days: float = Field(ge=0, description="Seller's average days from approval to carrier")
    route_prior_late_rate: float = Field(ge=0, le=1, description="Late rate of the seller-to-customer route")
    orders_last_7d: int = Field(ge=0, description="Marketplace orders placed in the previous 7 days")
    customer_state: str = Field(min_length=2, max_length=2)
    seller_state: str = Field(min_length=2, max_length=2)
    category: str


class Risk(BaseModel):
    late_probability: float
    high_risk: bool


def create_app(models: Path = MODELS) -> FastAPI:
    saved = joblib.load(models / "model.joblib")
    pipeline, threshold = saved["pipeline"], saved["threshold"]
    app = FastAPI(title="Late Delivery Predictor")

    @app.post("/predict", response_model=Risk)
    def predict(order: Order) -> Risk:
        row = derive(pd.DataFrame([order.model_dump()]))[FEATURES]
        probability = float(pipeline.predict_proba(row)[0, 1])
        return Risk(late_probability=probability, high_risk=probability >= threshold)

    return app
