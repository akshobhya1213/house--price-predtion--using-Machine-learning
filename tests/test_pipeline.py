import pandas as pd
from src.data import FEATURES, TARGET, make_synthetic
from src.features import add_features
from src.train import build_pipeline, metrics
from sklearn.linear_model import Ridge

def test_synthetic_shape():
    df = make_synthetic(500)
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert df[TARGET].between(0.15, 5.0).all()

def test_feature_engineering_adds_columns():
    out = add_features(make_synthetic(50)[FEATURES])
    assert {"BedroomRatio", "RoomsPerPerson", "DistMetro"} <= set(out.columns)

def test_pipeline_fits_and_handles_nan():
    df = make_synthetic(600)
    pipe = build_pipeline(Ridge()).fit(df[FEATURES], df[TARGET])
    X = df[FEATURES].head(5).copy()
    X.iloc[0, 0] = float("nan")
    assert len(pipe.predict(X)) == 5
    assert metrics(df[TARGET], pipe.predict(df[FEATURES]))["R2"] > 0.5
