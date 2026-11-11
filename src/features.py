"""Feature engineering shared by training and inference."""
import numpy as np
import pandas as pd

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
    X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"]
    X["LogPopulation"] = np.log1p(X["Population"])
    X["DistLA"] = np.hypot(X["Latitude"] - 34.05, X["Longitude"] + 118.25)
    X["DistSF"] = np.hypot(X["Latitude"] - 37.77, X["Longitude"] + 122.42)
    X["DistMetro"] = X[["DistLA", "DistSF"]].min(axis=1)
    return X.replace([np.inf, -np.inf], np.nan)
