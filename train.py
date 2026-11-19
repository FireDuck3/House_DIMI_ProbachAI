"""Train and save the XGBoost model for the Kaggle House Prices task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

SEED = 42
CANDIDATES = {
    "XGBoost_200": dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                        subsample=0.85, colsample_bytree=0.85),
    "XGBoost_600": dict(n_estimators=600, learning_rate=0.03, max_depth=3,
                        subsample=0.85, colsample_bytree=0.85),
}

def build_pipeline(X: pd.DataFrame, params: dict) -> Pipeline:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(objective="reg:squarederror",
                               random_state=SEED, n_jobs=2, **params)),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    train_file = args.data_dir / "train.csv"
    test_file = args.data_dir / "test.csv"
    if not train_file.is_file() or not test_file.is_file():
        raise FileNotFoundError("Place Kaggle train.csv and test.csv in data/")
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    if not {"Id", "SalePrice"}.issubset(train.columns) or "Id" not in test:
        raise ValueError("Expected the Kaggle House Prices train/test CSVs.")
    X = train.drop(columns=["Id", "SalePrice"])
    y = np.log(train["SalePrice"])
    X_test = test.reindex(columns=X.columns)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_rmse = float(np.sqrt(mean_squared_error(
        y_valid, baseline.predict(np.zeros((len(y_valid), 1)))
    )))
    scores = {}
    for name, params in CANDIDATES.items():
        model = build_pipeline(X_train, params)
        model.fit(X_train, y_train)
        scores[name] = float(np.sqrt(mean_squared_error(y_valid, model.predict(X_valid))))
        print(f"{name}: validation RMSE(log SalePrice) = {scores[name]:.5f}")

    best_name = min(scores, key=scores.get)
    best = build_pipeline(X, CANDIDATES[best_name])
    best.fit(X, y)  # Retrain the selected model on all 1,460 labeled samples.

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    model_path = Path("models/house_prices_xgboost.joblib")
    joblib.dump(best, model_path)

    prices = np.exp(best.predict(X_test))
    pd.DataFrame({"Id": test["Id"], "SalePrice": prices}).to_csv(
        "results/submission.csv", index=False
    )
    metrics = {
        "dataset": "Kaggle House Prices - Advanced Regression Techniques",
        "seed": SEED,
        "train_rows": len(train),
        "validation_rows": len(y_valid),
        "final_training_rows": len(X),
        "baseline_rmse_log": baseline_rmse,
        "candidate_rmse_log": scores,
        "selected_model": best_name,
        "validation_rmse_log": scores[best_name],
        "target_transform": "natural logarithm",
        "validation_split": 0.2,
        "validation_note": "80/20 holdout; this is not a Kaggle leaderboard score"
    }
    Path("results/metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reloaded = joblib.load(model_path)
    assert len(reloaded.predict(X_test.head(3))) == 3
    print(f"Saved trained model: {model_path}")
    print(f"Best model: {best_name}; RMSE(log)={scores[best_name]:.5f}")
    print("Saved: results/metrics.json and results/submission.csv")

if __name__ == "__main__":
    main()
