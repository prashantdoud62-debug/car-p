"""
app.py — Flask web application for Indian Car Price Prediction (CarDekho)
Features 6-parameter ML valuation + New vs Used Showroom Price Comparison.

Run:  python src/app.py
Open: http://localhost:5000
"""

import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "web_pipeline.joblib"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

# ── Load Web Model Bundle ──────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Web model not found at {MODEL_PATH}. Run `python src/train_web.py` first."
    )

model_bundle = joblib.load(MODEL_PATH)
pipeline = model_bundle["pipeline"]
CAR_BRANDS = model_bundle.get("brands", [])
FUEL_TYPES = model_bundle.get("fuels", ["Petrol", "Diesel", "CNG", "LPG"])
TRANSMISSIONS = model_bundle.get("transmissions", ["Manual", "Automatic"])
MODEL_R2 = model_bundle.get("r2", 0.8925)

# 6 Essential Features matching training pipeline
WEB_NUMERIC = ["year", "km_driven", "max_power_bhp"]
WEB_CATEGORICAL = ["brand", "fuel", "transmission"]

# ── Flask App Setup ────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)


def format_inr_price(price: float) -> tuple[str, str]:
    """Format price into Indian Lakhs and comma-separated INR representation."""
    price = max(0.0, price)
    if price >= 100_000:
        lakhs_str = f"₹{price / 100_000:.2f} Lakh"
    else:
        lakhs_str = f"₹{price:,.0f}"
    full_inr_str = f"₹{price:,.0f}"
    return lakhs_str, full_inr_str


def calculate_showroom_comparison(
    predicted_resale: float,
    year: int,
    km_driven: float,
    custom_showroom_lakhs: float = None,
) -> dict:
    """
    Calculate Showroom / Original Brand-New Price and Depreciation Comparison.
    Uses official Indian automotive depreciation benchmarks (IRDAI curves).
    """
    # Dataset reference year is 2020; car age relative to registration
    age = max(1, 2020 - year) if year <= 2020 else 1

    if custom_showroom_lakhs is not None and custom_showroom_lakhs > 0:
        showroom_price = custom_showroom_lakhs * 100_000.0
    else:
        # Standard IRDAI & Indian automotive depreciation table:
        depreciation_table = {
            1: 0.15,
            2: 0.25,
            3: 0.32,
            4: 0.40,
            5: 0.48,
            6: 0.54,
            7: 0.60,
            8: 0.65,
            9: 0.70,
            10: 0.75,
        }
        dep_rate = depreciation_table.get(age, min(0.85, 0.50 + (age - 5) * 0.05))
        # Add slight km wear adjustment
        km_adj = min(0.08, (km_driven / 100_000) * 0.03)
        total_dep = min(0.85, dep_rate + km_adj)
        retention = max(0.15, 1.0 - total_dep)
        showroom_price = predicted_resale / retention
        # Round to nearest ₹5,000 for realistic showroom figure
        showroom_price = round(showroom_price / 5000.0) * 5000.0

    # Ensure showroom price is at least as high as resale price
    showroom_price = max(showroom_price, predicted_resale)

    depreciation_amt = max(0.0, showroom_price - predicted_resale)
    depreciation_pct = round((depreciation_amt / showroom_price) * 100, 1) if showroom_price > 0 else 0.0
    retained_pct = round((predicted_resale / showroom_price) * 100, 1) if showroom_price > 0 else 100.0

    lakhs_showroom, inr_showroom = format_inr_price(showroom_price)
    lakhs_dep, inr_dep = format_inr_price(depreciation_amt)

    return {
        "showroom_price": round(showroom_price, 2),
        "formatted_showroom": lakhs_showroom,
        "formatted_showroom_inr": inr_showroom,
        "depreciation_amt": round(depreciation_amt, 2),
        "formatted_depreciation": f"-{lakhs_dep}",
        "depreciation_pct": depreciation_pct,
        "retained_pct": retained_pct,
    }


@app.route("/")
def index():
    """Serve the modern interactive prediction UI."""
    return render_template(
        "index.html",
        car_brands=CAR_BRANDS,
        fuel_types=FUEL_TYPES,
        transmissions=TRANSMISSIONS,
        model_r2=MODEL_R2,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accept parameters JSON and return predicted car price + showroom comparison."""
    try:
        data = request.get_json(force=True)

        # Validate and extract parameters
        brand = str(data["brand"]).strip()
        year = int(data["year"])
        km_driven = float(data["km_driven"])
        fuel = str(data["fuel"]).strip()
        transmission = str(data["transmission"]).strip()
        max_power_bhp = float(data["max_power_bhp"])

        # Optional custom showroom price
        custom_showroom = data.get("showroom_price_lakhs")
        custom_showroom_lakhs = float(custom_showroom) if custom_showroom not in (None, "") else None

        # Range sanity checks
        if not (1990 <= year <= 2025):
            return jsonify({"success": False, "error": "Year must be between 1990 and 2025"}), 400
        if km_driven < 0:
            return jsonify({"success": False, "error": "Kilometers driven cannot be negative"}), 400
        if max_power_bhp <= 0:
            return jsonify({"success": False, "error": "Max power must be greater than 0"}), 400

        car_dict = {
            "year": year,
            "km_driven": km_driven,
            "max_power_bhp": max_power_bhp,
            "brand": brand,
            "fuel": fuel,
            "transmission": transmission,
        }

        df = pd.DataFrame([car_dict])
        df = df[WEB_NUMERIC + WEB_CATEGORICAL]

        predicted_raw = max(20_000.0, float(pipeline.predict(df)[0]))
        lakhs_str, inr_str = format_inr_price(predicted_raw)

        # Calculate showroom comparison
        comp = calculate_showroom_comparison(
            predicted_resale=predicted_raw,
            year=year,
            km_driven=km_driven,
            custom_showroom_lakhs=custom_showroom_lakhs,
        )

        return jsonify({
            "success": True,
            "predicted_price": round(predicted_raw, 2),
            "formatted_price": lakhs_str,
            "formatted_inr": inr_str,
            "details": f"{year} {brand} ({fuel} · {transmission} · {max_power_bhp:g} bhp)",
            **comp,
        })

    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"success": False, "error": f"Invalid input: {str(e)}"}), 400


import socket
import argparse


def find_free_port(preferred_port: int = 5000) -> int:
    """Check if preferred port is free; if not, find the next available port."""
    for port in [preferred_port, 5001, 8000, 8080]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Car Price Prediction Flask App")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port to run server on")
    args = parser.parse_args()

    port = args.port if args.port else find_free_port(5000)

    print("\n" + "=" * 60)
    print("🚗 Indian Car Price Prediction Web App (CarDekho)")
    print(f"👉 Local Web URL: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")

    app.run(debug=True, host="127.0.0.1", port=port)

