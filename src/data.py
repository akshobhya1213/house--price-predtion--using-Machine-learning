"""Data loading. Priority: custom CSV -> California Housing (online) -> synthetic fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET = "price"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

def make_synthetic(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(32.6, 41.8, n)
    lon = rng.uniform(-124.3, -114.4, n)
    med_inc = np.clip(rng.lognormal(1.25, 0.45, n), 0.5, 15)
    age = rng.integers(1, 53, n).astype(float)
    rooms = np.clip(rng.normal(5.4, 1.2, n), 2, 12)
    bedrms = np.clip(rooms * rng.normal(0.21, 0.03, n), 0.8, 4)
    pop = np.clip(rng.lognormal(7.1, 0.6, n), 50, 8000)
    occup = np.clip(rng.normal(3.0, 0.7, n), 1.2, 8)
    def km(la, lo):
        return np.hypot((lat - la) * 111, (lon - lo) * 92)
    coastal = np.exp(-np.minimum(km(34.05, -118.25), km(37.77, -122.42)) / 80)
    price = 0.35 + 0.42 * med_inc + 1.1 * coastal - 0.004 * age + 0.06 * (rooms - 5.4) - 0.12 * (occup - 3) - 0.00002 * pop + rng.normal(0, 0.28, n)
    price = np.clip(price, 0.15, 5.0)
    df = pd.DataFrame({"MedInc": med_inc, "HouseAge": age, "AveRooms": rooms, "AveBedrms": bedrms, "Population": pop, "AveOccup": occup, "Latitude": lat, "Longitude": lon, TARGET: price})
    for col in ["AveRooms", "HouseAge"]:
        df.loc[rng.random(n) < 0.01, col] = np.nan
    return df

def load_data(source: str = "auto") -> tuple[pd.DataFrame, str]:
    if source.endswith(".csv"):
        df = pd.read_csv(source)
        if TARGET not in df.columns:
            raise ValueError(f"CSV must contain a '{TARGET}' column")
        return df, f"csv:{source}"
    if source in ("auto", "california"):
        try:
            from sklearn.datasets import fetch_california_housing
            d = fetch_california_housing(as_frame=True)
            df = d.frame.rename(columns={"MedHouseVal": TARGET})
            return df[FEATURES + [TARGET]], "california_housing"
        except Exception as e:
            if source == "california":
                raise
            print(f"[data] California Housing unavailable ({type(e).__name__}); using synthetic data.")
    return make_synthetic(), "synthetic"
