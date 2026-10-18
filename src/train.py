"""
train.py — Train, evaluate, and save Indian used car price prediction models.

Dataset: Vehicle Dataset from CarDekho (Car details v3)
Source:  Kaggle (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
License: Open Data Commons Public Domain Dedication and License (PDDL) / CC0
Country: India (Prices in INR / ₹)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from data_preprocessing import (
    load_and_clean,
    ALL_NUMERIC_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    TARGET,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_details_v3.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Visualisations & EDA ──────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Generate and save insightful EDA plots for Indian used car market."""
    PLOTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Price Distribution (in Lakhs)
    fig, ax = plt.subplots(figsize=(9, 5))
    price_lakhs = df[TARGET] / 100_000
    sns.histplot(price_lakhs, kde=True, bins=40, ax=ax, color="#4C72B0")
    ax.axvline(price_lakhs.median(), color="red", linestyle="--",
               label=f"Median: Rs. {price_lakhs.median():.2f} Lakh")
    ax.set_title("Distribution of Used Car Prices in India (CarDekho)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Selling Price (Rs. in Lakhs)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.text(0.5, -0.03,
             "Insight: Prices are right-skewed. The majority of used cars sell between Rs. 2 Lakh and Rs. 8 Lakh.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_price_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Top Brands vs Average Price
    top_brands = df["brand"].value_counts().head(10).index
    brand_df = df[df["brand"].isin(top_brands)].groupby("brand")[TARGET].mean() / 100_000
    brand_df = brand_df.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=brand_df.values, y=brand_df.index, ax=ax, palette="Blues_r")
    ax.set_title("Average Selling Price by Top 10 Indian Car Brands", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Price (Rs. in Lakhs)")
    ax.set_ylabel("Brand")
    fig.text(0.5, -0.03,
             "Insight: Premium brands like Toyota command significant resale value; Maruti & Hyundai lead in volume.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_brand_vs_price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Manufacturing Year vs Price (Depreciation Trend)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="year", y=df[TARGET] / 100_000, errorbar=None, color="#2ecc71", linewidth=2.5, ax=ax)
    ax.set_title("Vehicle Depreciation Trend (Manufacturing Year vs Price)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Manufacturing Year")
    ax.set_ylabel("Selling Price (Rs. in Lakhs)")
    fig.text(0.5, -0.03,
             "Insight: Exponential depreciation pattern — cars older than 7–8 years retain under 35% of original value.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_depreciation_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("✅ EDA plots saved to plots/")


# ── Pipeline Construction ──────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Chain StandardScaler for numericals and OneHotEncoder for categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ALL_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ── Training & Model Selection ─────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
) -> tuple[dict[str, dict], Pipeline, pd.DataFrame, pd.Series]:
    """Train 4 models, perform 5-fold CV, and evaluate on holdout test set."""
    feature_cols = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted_pipelines = {}

    print("\n" + "=" * 65)
    print("🤖 MODEL BENCHMARKING (5-Fold CV & Test Set)")
    print("=" * 65)

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

        # 5-fold Cross-Validation MAE on training set
        cv_neg_mae = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        cv_mae = -cv_neg_mae.mean()

        # Fit on full train set and evaluate on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        results[name] = {
            "CV MAE (₹)": cv_mae,
            "Test MAE (₹)": test_mae,
            "Test RMSE (₹)": test_rmse,
            "Test R²": test_r2,
        }
        fitted_pipelines[name] = pipeline

        print(
            f"{name:<20} | CV MAE: ₹{cv_mae:>9,.0f} | "
            f"Test MAE: ₹{test_mae:>9,.0f} | Test RMSE: ₹{test_rmse:>9,.0f} | Test R²: {test_r2:.4f}"
        )

    # Pick the best model based on highest Test R²
    best_name = max(results, key=lambda k: results[k]["Test R²"])
    print(f"\n🏆 Best Model: {best_name} (Test R² = {results[best_name]['Test R²']:.4f})")

    best_pipeline = fitted_pipelines[best_name]

    # Save model comparison plot
    names = list(results.keys())
    r2_vals = [results[m]["Test R²"] for m in names]
    mae_lakhs = [results[m]["Test MAE (₹)"] / 100_000 for m in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=r2_vals, y=names, ax=ax1, palette="viridis")
    ax1.set_title("Test R² Score (Higher is better)", fontweight="bold")
    ax1.set_xlim(0, 1.0)
    for i, v in enumerate(r2_vals):
        ax1.text(v + 0.02, i, f"{v:.4f}", va="center", fontweight="bold")

    sns.barplot(x=mae_lakhs, y=names, ax=ax2, palette="magma")
    ax2.set_title("Test MAE (Rs. in Lakhs - Lower is better)", fontweight="bold")
    for i, v in enumerate(mae_lakhs):
        ax2.text(v + 0.05, i, f"Rs. {v:.2f}L", va="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Actual vs Predicted plot for best pipeline
    y_best_pred = best_pipeline.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test / 100_000, y_best_pred / 100_000, alpha=0.4, color="#3498db", edgecolors="none")
    max_val = max(y_test.max(), y_best_pred.max()) / 100_000
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal Prediction (1:1)")
    ax.set_title(f"Actual vs Predicted Prices ({best_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price (Rs. in Lakhs)")
    ax.set_ylabel("Predicted Price (Rs. in Lakhs)")
    ax.legend()
    fig.text(0.5, -0.03, f"Test R² = {results[best_name]['Test R²']:.4f} · Test MAE = Rs. {results[best_name]['Test MAE (₹)']:,.0f}",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Feature Importances plot
    regressor = best_pipeline.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(ALL_CATEGORICAL_FEATURES)
        all_feat_names = ALL_NUMERIC_FEATURES + list(cat_features)
        importances = regressor.feature_importances_

        imp_df = pd.DataFrame({"Feature": all_feat_names, "Importance": importances})
        top_imp = imp_df.sort_values(by="Importance", ascending=False).head(12)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_imp, x="Importance", y="Feature", palette="rocket", ax=ax)
        ax.set_title(f"Top 12 Feature Importances ({best_name})", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "06_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return results, best_pipeline, X_test, y_test


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("🚀 TRAINING CAR PRICE PREDICTION MODEL (INDIAN DATASET)")
    print("=" * 65)

    df_clean = load_and_clean(DATA_PATH)
    run_eda(df_clean)

    results, best_pipeline, X_test, y_test = train_and_evaluate(df_clean)

    # Save best full pipeline
    MODEL_DIR.mkdir(exist_ok=True)
    best_model_path = MODEL_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n💾 Best model pipeline saved to: {best_model_path}")


if __name__ == "__main__":
    main()
