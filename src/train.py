"""Train, compare, tune and evaluate house-price models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from src.data import FEATURES, TARGET, load_data
from src.features import add_features

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])

def metrics(y_true, y_pred) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}

def main(source: str = "auto") -> dict:
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    df, data_name = load_data(source)
    print(f"[data] {data_name}: {df.shape[0]} rows")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED),
    }
    cv_results = {}
    for name, model in candidates.items():
        scores = -cross_val_score(build_pipeline(model), X_train, y_train, cv=5,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_results[name] = {"cv_rmse_mean": float(scores.mean()), "cv_rmse_std": float(scores.std())}
        print(f"[cv] {name:22s} RMSE {scores.mean():.4f} +/- {scores.std():.4f}")
    search = RandomizedSearchCV(
        build_pipeline(HistGradientBoostingRegressor(random_state=SEED)),
        {"model__learning_rate": [0.03, 0.05, 0.1, 0.15],
         "model__max_iter": [200, 400, 600],
         "model__max_leaf_nodes": [15, 31, 63],
         "model__min_samples_leaf": [10, 20, 40],
         "model__l2_regularization": [0.0, 0.1, 1.0]},
        n_iter=12, cv=3, scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    tuned_cv = -search.best_score_
    cv_results["HGB_tuned"] = {"cv_rmse_mean": float(tuned_cv), "best_params": search.best_params_}
    best_name = min(cv_results, key=lambda k: cv_results[k]["cv_rmse_mean"])
    best = search.best_estimator_ if best_name == "HGB_tuned" else build_pipeline(candidates[best_name]).fit(X_train, y_train)
    pred = best.predict(X_test)
    test_metrics = metrics(y_test, pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_test, pred, s=6, alpha=0.4)
    lim = [0, max(y_test.max(), pred.max())]
    ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="Actual ($100k)", ylabel="Predicted ($100k)", title="Predicted vs Actual")
    ax[1].hist(y_test - pred, bins=50)
    ax[1].set(title="Residuals", xlabel="Actual - Predicted")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "evaluation.png", dpi=120)
    plt.close(fig)
    imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(FEATURES)[order], imp.importances_mean[order])
    ax.set(title="Permutation importance", xlabel="Drop in R2")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "feature_importance.png", dpi=120)
    plt.close(fig)
    joblib.dump(best, ROOT / "models" / "model.joblib")
    report = {"data_source": data_name, "n_rows": int(len(df)), "best_model": best_name,
              "cross_validation": cv_results, "test_metrics": test_metrics, "target_unit": "$100,000"}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    print("[save] models/model.joblib, reports/metrics.json, reports/*.png")
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="auto", help="auto | california | synthetic | path/to.csv")
    main(p.parse_args().data)
