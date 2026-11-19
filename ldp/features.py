"""Turn the Olist tables into one row per delivered order, using only what is known when the order is approved."""

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "promised_days", "approval_hours", "purchase_weekday", "n_items", "n_sellers", "price",
    "freight", "freight_ratio", "weight_kg", "volume_l", "distance_km", "same_state", "seller_prior_orders",
    "seller_prior_late_rate", "seller_handling_days", "route_prior_late_rate", "orders_last_7d",
]
CATEGORICAL = ["customer_state", "seller_state", "category"]
FEATURES = NUMERIC + CATEGORICAL

BRAZIL_LAT, BRAZIL_LNG = (-34.0, 6.0), (-74.0, -34.0)


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.radians(v) for v in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from other features; shared by training and the API."""
    df = df.copy()
    df["freight_ratio"] = df["freight"] / df["price"].where(df["price"] > 0)
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    return df


def history(orders: pd.DataFrame, key: str, done_at: str, value: str, prior_weight: float = 20.0) -> pd.DataFrame:
    """Count and mean of `value` over earlier orders with the same `key`, for every order.

    Only orders whose `done_at` falls before this order was placed are counted, so nothing from the future
    leaks in. The mean is smoothed towards the marketplace mean at that time, so a seller with two orders
    is not scored 0% or 100%.
    """
    done = orders[[key, done_at, value]].dropna().sort_values(done_at)
    groups = done.groupby(key)
    done = done.assign(n=groups.cumcount() + 1, total=groups[value].cumsum(),
                       all_n=np.arange(1, len(done) + 1), all_total=done[value].cumsum())

    left = orders[[key, "purchased_at"]].reset_index().sort_values("purchased_at")
    own = pd.merge_asof(left, done[[key, done_at, "n", "total"]], left_on="purchased_at", right_on=done_at,
                        by=key, allow_exact_matches=False)
    market = pd.merge_asof(left, done[[done_at, "all_n", "all_total"]], left_on="purchased_at",
                           right_on=done_at, allow_exact_matches=False)

    n = own["n"].fillna(0).to_numpy()
    total = own["total"].fillna(0).to_numpy()
    base = (market["all_total"] / market["all_n"]).to_numpy()
    mean = (total + prior_weight * base) / (n + prior_weight)
    return pd.DataFrame({"n": n, "mean": mean}, index=left["index"].to_numpy()).sort_index()


def orders_in_previous_days(purchased_at: pd.Series, days: int = 7) -> np.ndarray:
    times = purchased_at.sort_values()
    start = np.searchsorted(times.to_numpy(), (times - pd.Timedelta(days=days)).to_numpy())
    counts = pd.Series(np.arange(len(times)) - start, index=times.index)
    return counts.reindex(purchased_at.index).to_numpy()


def build_dataset(raw: Path | str) -> pd.DataFrame:
    raw = Path(raw)

    def read(name, **kwargs):
        return pd.read_csv(raw / f"olist_{name}_dataset.csv", **kwargs)

    orders = read("orders", parse_dates=["order_purchase_timestamp", "order_approved_at",
                                         "order_delivered_carrier_date", "order_delivered_customer_date",
                                         "order_estimated_delivery_date"])
    orders = orders[(orders["order_status"] == "delivered")
                    & orders["order_delivered_customer_date"].notna()
                    & orders["order_approved_at"].notna()]
    orders = orders.rename(columns={"order_purchase_timestamp": "purchased_at", "order_approved_at": "approved_at",
                                    "order_delivered_carrier_date": "carrier_at",
                                    "order_delivered_customer_date": "delivered_at",
                                    "order_estimated_delivery_date": "estimated_at"})

    geo = read("geolocation")
    geo = geo[geo["geolocation_lat"].between(*BRAZIL_LAT) & geo["geolocation_lng"].between(*BRAZIL_LNG)]
    geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    geo.columns = ["lat", "lng"]

    customers = read("customers").join(geo, on="customer_zip_code_prefix")
    sellers = read("sellers").join(geo, on="seller_zip_code_prefix")
    products = read("products")
    products["volume_l"] = products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"] / 1000
    names = pd.read_csv(raw / "product_category_name_translation.csv")
    products = products.merge(names, on="product_category_name", how="left")

    items = (read("order_items")
             .merge(products[["product_id", "product_category_name_english", "product_weight_g", "volume_l"]],
                    on="product_id", how="left")
             .merge(sellers[["seller_id", "seller_state", "lat", "lng"]], on="seller_id", how="left")
             .merge(orders[["order_id", "customer_id"]], on="order_id")
             .merge(customers[["customer_id", "lat", "lng"]], on="customer_id", suffixes=("_seller", "_customer")))
    items["distance_km"] = haversine_km(items["lat_seller"], items["lng_seller"],
                                        items["lat_customer"], items["lng_customer"])

    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_sellers=("seller_id", "nunique"), price=("price", "sum"),
        freight=("freight_value", "sum"), weight_kg=("product_weight_g", "sum"), volume_l=("volume_l", "sum"),
        distance_km=("distance_km", "max"))
    per_order["weight_kg"] /= 1000
    main_item = (items.sort_values("price", ascending=False).drop_duplicates("order_id")
                 .set_index("order_id")[["seller_id", "seller_state", "product_category_name_english"]]
                 .rename(columns={"product_category_name_english": "category"}))

    df = (orders.merge(customers[["customer_id", "customer_state"]], on="customer_id")
          .join(per_order, on="order_id", how="inner")
          .join(main_item, on="order_id"))
    df["category"] = df["category"].fillna("unknown")
    df["late"] = (df["delivered_at"].dt.normalize() > df["estimated_at"].dt.normalize()).astype(int)
    df["promised_days"] = (df["estimated_at"] - df["purchased_at"]).dt.total_seconds() / 86400
    df["approval_hours"] = (df["approved_at"] - df["purchased_at"]).dt.total_seconds() / 3600
    df["purchase_weekday"] = df["purchased_at"].dt.weekday
    df = derive(df.reset_index(drop=True))
    df["handling_days"] = ((df["carrier_at"] - df["approved_at"]).dt.total_seconds() / 86400).clip(lower=0)
    df["route"] = df["seller_state"] + ">" + df["customer_state"]

    seller = history(df, "seller_id", "delivered_at", "late")
    df["seller_prior_orders"] = seller["n"]
    df["seller_prior_late_rate"] = seller["mean"]
    df["seller_handling_days"] = history(df, "seller_id", "carrier_at", "handling_days")["mean"]
    df["route_prior_late_rate"] = history(df, "route", "delivered_at", "late")["mean"]
    df["orders_last_7d"] = orders_in_previous_days(df["purchased_at"])
    return df[["order_id", "purchased_at", "delivered_at", "seller_id"] + FEATURES + ["late"]]
