"""Predict from the command line."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.data import FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
DEFAULTS = {"MedInc": 4.0, "HouseAge": 25, "AveRooms": 5.4, "AveBedrms": 1.1, "Population": 1200, "AveOccup": 3.0, "Latitude": 34.05, "Longitude": -118.25}

def predict(features: dict) -> float:
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{k: float(features[k]) for k in FEATURES}])
    return float(model.predict(row)[0]) * 100_000

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    print(f"Predicted price: ${predict(vars(ap.parse_args())):,.0f}")
