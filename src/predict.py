"""
predict.py — Load the saved Indian car price pipeline and predict car prices.

Usage:
    python src/predict.py                  # uses the built-in sample Indian car
    python src/predict.py --interactive    # prompts you for each feature via terminal

Dataset: Vehicle Dataset from CarDekho (Car details v3)
Source:  Kaggle (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
License: Open Data Commons Public Domain Dedication and License (PDDL) / CC0
Country: India (Currency: INR / ₹)
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path

from data_preprocessing import ALL_NUMERIC_FEATURES, ALL_CATEGORICAL_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_pipeline.joblib"

# ── Realistic Sample Indian Car (e.g. Maruti Swift VDI 2017) ───────────────────
SAMPLE_CAR = {
    "brand": "Maruti",
    "year": 2017,
    "km_driven": 48000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "engine_cc": 1248.0,
    "max_power_bhp": 74.0,
    "mileage_kmpl": 24.3,
    "seats": 5.0,
    "car_age": 3,
}


def load_pipeline(path: Path):
    """Load the saved scikit-learn pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}.\n"
            "Run `python src/train.py` first to train and save the model."
        )
    return joblib.load(path)


def format_inr(price: float) -> str:
    """Format an amount into Indian numbering system (Lakhs / Rupees)."""
    lakhs = price / 100_000
    return f"₹{lakhs:.2f} Lakh (₹{price:,.0f})"


def get_car_from_user() -> dict:
    """Prompt the user to enter car features interactively via CLI."""
    print("\n🚗 Enter Indian car details (press Enter to accept default):\n")
    car = {}
    defaults = SAMPLE_CAR

    # Categorical prompts
    for feat in ALL_CATEGORICAL_FEATURES:
        default = defaults.get(feat, "")
        raw = input(f"  {feat.replace('_', ' ').title()} [{default}]: ").strip()
        car[feat] = raw if raw else default

    # Numeric prompts
    for feat in ALL_NUMERIC_FEATURES:
        if feat == "car_age":
            continue  # auto-computed from year
        default = defaults.get(feat, 0)
        raw = input(f"  {feat.replace('_', ' ').title()} [{default}]: ").strip()
        car[feat] = float(raw) if raw else float(default)

    # Compute car_age automatically
    car["car_age"] = 2020 - int(car["year"])

    return car


def predict_price(pipeline, car_dict: dict) -> float:
    """Predict the price for a single car given as a dict."""
    df = pd.DataFrame([car_dict])
    df = df[ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES]
    prediction = pipeline.predict(df)[0]
    return max(0.0, float(prediction))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Indian used car price using trained ML model.")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter car features interactively via CLI.",
    )
    args = parser.parse_args()

    pipeline = load_pipeline(MODEL_PATH)
    print("✅ Model loaded successfully from models/best_pipeline.joblib\n")

    if args.interactive:
        car = get_car_from_user()
    else:
        car = SAMPLE_CAR
        print("Using sample car details (run with --interactive to enter your own):")
        for k, v in car.items():
            print(f"  {k.replace('_', ' ').title():<18}: {v}")

    predicted_price = predict_price(pipeline, car)
    
    # Calculate benchmark showroom price and depreciation
    age = max(1, 2020 - int(car["year"])) if int(car["year"]) <= 2020 else 1
    depreciation_table = {1: 0.15, 2: 0.25, 3: 0.32, 4: 0.40, 5: 0.48, 6: 0.54, 7: 0.60, 8: 0.65}
    dep_rate = depreciation_table.get(age, min(0.85, 0.50 + (age - 5) * 0.05))
    total_dep = min(0.85, dep_rate + min(0.08, (float(car["km_driven"]) / 100_000) * 0.03))
    retention = max(0.15, 1.0 - total_dep)
    showroom_price = round((predicted_price / retention) / 5000.0) * 5000.0
    depreciation_amt = max(0.0, showroom_price - predicted_price)
    retained_pct = round((predicted_price / showroom_price) * 100, 1)

    print("\n" + "=" * 55)
    print("📊 PRICE & VALUATION COMPARISON (INR)")
    print("=" * 55)
    print(f"  🏷️  Original Showroom Price (New) : {format_inr(showroom_price)}")
    print(f"  🚗  Estimated Resale Value (Used) : {format_inr(predicted_price)}")
    print(f"  📉  Depreciation Loss / Savings   : -{format_inr(depreciation_amt)} (-{100 - retained_pct:.1f}%)")
    print(f"  📈  Value Retained                : {retained_pct}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
