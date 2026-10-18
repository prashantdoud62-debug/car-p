"""
data_preprocessing.py — Data loading, cleaning, and feature engineering for Indian Used Cars.

Dataset: Vehicle Dataset from CarDekho (Car details v3)
Source:  Kaggle (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
File:    data/car_details_v3.csv
License: Open Data Commons Public Domain Dedication and License (PDDL) / CC0
Country: India (Prices in INR / ₹)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Feature Definitions ────────────────────────────────────────────────────────
# Target variable: selling_price (in Indian Rupees, ₹)
TARGET = "selling_price"

# Reference year for vehicle age calculation (2020 is dataset capture year)
REFERENCE_YEAR = 2020

# Full feature set for comprehensive modeling
ALL_NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "car_age",
    "engine_cc",
    "max_power_bhp",
    "mileage_kmpl",
    "seats",
]

ALL_CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

# 6 Essential Features for easy, intuitive user input (Web App & Streamlined ML)
WEB_NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "max_power_bhp",
]

WEB_CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "transmission",
]

WEB_FEATURES = WEB_NUMERIC_FEATURES + WEB_CATEGORICAL_FEATURES


# ── Loading ────────────────────────────────────────────────────────────────────

def load_data(filepath: Path) -> pd.DataFrame:
    """Load the raw CarDekho CSV file and return a pandas DataFrame."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Source: Kaggle (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)\n"
            "Please ensure data/car_details_v3.csv is present."
        )
    df = pd.read_csv(filepath)
    return df


# ── Inspection helpers ─────────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """Print comprehensive dataset information for transparency and viva verification."""
    print("=" * 65)
    print("🚗 INDIAN USED CARS DATASET INSPECTION (CarDekho)")
    print("=" * 65)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    missing = df.isnull().sum()
    print(f"\nMissing values per column:\n{missing[missing > 0]}")
    print(f"Total missing cells: {missing.sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nNumerical summary:")
    print(df.describe())


# ── Cleaning & Feature Engineering ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the Indian CarDekho dataset:
      1. Drop rows missing the target (selling_price).
      2. Extract numeric values from strings:
         - engine: '1248 CC' -> 1248.0
         - max_power: '74 bhp' -> 74.0
         - mileage: '23.4 kmpl' -> 23.4
      3. Extract brand from vehicle name (e.g., 'Maruti Swift' -> 'Maruti').
      4. Engineer 'car_age' = 2020 - year.
      5. Impute missing values with column medians / modes.
      6. Remove duplicate rows.
      7. Remove price outliers using IQR for robust linear modeling.
    """
    df = df.copy()

    # 1. Drop rows missing target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped_target = before - len(df)
    if dropped_target > 0:
        print(f"Dropped {dropped_target} rows missing {TARGET}.")

    # 2. Parse string numeric columns
    df["engine_cc"] = pd.to_numeric(
        df["engine"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce"
    )
    df["max_power_bhp"] = pd.to_numeric(
        df["max_power"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce"
    )
    df["mileage_kmpl"] = pd.to_numeric(
        df["mileage"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce"
    )

    # 3. Extract manufacturer brand
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0].title())

    # Standardize brands
    for cat_col in ["brand", "fuel", "seller_type", "transmission", "owner"]:
        df[cat_col] = df[cat_col].astype(str).str.strip()

    # 4. Feature engineering: car_age
    df["car_age"] = REFERENCE_YEAR - df["year"]

    # 5. Impute missing values with medians
    for num_col in ["engine_cc", "max_power_bhp", "mileage_kmpl", "seats"]:
        if num_col in df.columns:
            median_val = df[num_col].median()
            df[num_col] = df[num_col].fillna(median_val)
            print(f"  Filled {num_col} missing values with median = {median_val:.2f}")

    # 6. Remove duplicate entries
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {dup_count} duplicate rows.")

    # 7. Price outlier handling (clean extreme spurious anomalies)
    q1 = df[TARGET].quantile(0.01)
    q99 = df[TARGET].quantile(0.99)
    outliers = ((df[TARGET] < q1) | (df[TARGET] > q99)).sum()
    df = df[(df[TARGET] >= q1) & (df[TARGET] <= q99)].reset_index(drop=True)
    print(f"Trimmed {outliers} extreme 1% price outliers (price range: ₹{df[TARGET].min():,} to ₹{df[TARGET].max():,}).")

    print(f"\nFinal Cleaned Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Convenience: load + clean in one call ──────────────────────────────────────

def load_and_clean(filepath: Path) -> pd.DataFrame:
    """Load raw Indian CarDekho data, inspect it, clean it, and return the clean DataFrame."""
    df = load_data(filepath)
    inspect_data(df)
    print("\n" + "=" * 65)
    print("🧹 CLEANING & FEATURE EXTRACTION")
    print("=" * 65)
    df_clean = clean_data(df)
    return df_clean


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "data" / "car_details_v3.csv"
    df_clean = load_and_clean(data_file)
    print("\nSample processed records:")
    print(df_clean[["brand", "year", "km_driven", "fuel", "transmission", "max_power_bhp", "selling_price"]].head(5))
