"""
train_web.py — Train a streamlined 6-feature model for the CarDekho Indian Web App.

Dataset: Vehicle Dataset from CarDekho (Car details v3)
Source:  Kaggle (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
License: Open Data Commons Public Domain Dedication and License (PDDL) / CC0
Country: India (Prices in INR / ₹)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from data_preprocessing import (
    load_and_clean,
    WEB_NUMERIC_FEATURES,
    WEB_CATEGORICAL_FEATURES,
    WEB_FEATURES,
    TARGET,
)

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_details_v3.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def build_web_pipeline() -> Pipeline:
    """Build scikit-learn pipeline for 6 key features with Gradient Boosting."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), WEB_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                WEB_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            random_state=RANDOM_STATE,
        )),
    ])


def main() -> None:
    print("🌐 Training Web App Model (6 Essential Parameters — Indian Dataset)")
    print("=" * 65)

    # 1. Load & clean
    df = load_and_clean(DATA_PATH)

    # 2. Features and Target
    X = df[WEB_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # 3. Train & Evaluate
    pipeline = build_web_pipeline()

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    print(f"5-Fold CV MAE: ₹{cv_mae:,.0f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    print(f"Test MAE : ₹{test_mae:,.0f} (₹{test_mae / 100_000:.2f} Lakh)")
    print(f"Test RMSE: ₹{test_rmse:,.0f} (₹{test_rmse / 100_000:.2f} Lakh)")
    print(f"Test R²  : {test_r2:.4f}")

    # 4. Extract categories for web form
    brands = sorted(df["brand"].unique().tolist())
    fuels = sorted(df["fuel"].unique().tolist())
    transmissions = sorted(df["transmission"].unique().tolist())

    print(f"\nDiscovered {len(brands)} Indian car brands: {', '.join(brands[:8])}...")

    # 5. Save model bundle
    MODEL_DIR.mkdir(exist_ok=True)
    web_model_path = MODEL_DIR / "web_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "brands": brands,
            "fuels": fuels,
            "transmissions": transmissions,
            "features": WEB_FEATURES,
            "r2": round(test_r2, 4),
            "mae": round(test_mae, 2),
        },
        web_model_path,
    )
    print(f"\n💾 Web model saved successfully to: {web_model_path}")


if __name__ == "__main__":
    main()
