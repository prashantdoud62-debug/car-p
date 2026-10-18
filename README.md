# 🚗 CarDekho Indian Used Car Price Prediction & Valuation Web App

An end-to-end Machine Learning project and interactive Flask web application to predict realistic market resale valuations for used cars in India, trained on the CarDekho dataset.

---

## 📌 Project Overview

- **Dataset**: Vehicle Dataset from CarDekho (`Car details v3`) containing thousands of real-world Indian used car transactions.
- **Machine Learning Models**: Benchmarked multiple regressors including Linear Regression, Ridge, Random Forest, and Gradient Boosting (`R² ~ 0.89+`).
- **Web Application**: Interactive Flask UI providing instant valuation estimates, depreciation insights, and comparisons with original showroom prices.

---

## 🗂️ Project Structure

```text
├── data/
│   ├── car_details_v3.csv          # CarDekho used cars dataset
│   └── imports-85.data             # Benchmark auto dataset
├── models/
│   ├── web_pipeline.joblib         # Streamlined 6-feature pipeline for web inference
│   └── best_pipeline.joblib        # Full-feature ML pipeline
├── notebooks/
│   └── car_price_prediction.ipynb  # Interactive data exploration & experiment notebook
├── plots/                          # EDA & Model evaluation visualizations
│   ├── 01_price_distribution.png
│   ├── 02_brand_vs_price.png
│   ├── 03_depreciation_trend.png
│   ├── 04_model_comparison.png
│   ├── 05_actual_vs_predicted.png
│   └── 06_feature_importance.png
├── src/
│   ├── app.py                      # Flask web application server
│   ├── data_preprocessing.py       # Data cleaning, unit parsing & feature engineering
│   ├── predict.py                  # CLI prediction interface
│   ├── train.py                    # Comprehensive training & benchmarking pipeline
│   └── train_web.py                # Web pipeline trainer
├── static/
│   └── style.css                   # Web UI styles
├── templates/
│   └── index.html                  # Valuation dashboard template
├── requirements.txt                # Python package dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prashantdoud62-debug/car-p.git
   cd car-p
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Web Application

Start the Flask development server:
```bash
python src/app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🧠 Retraining Models

To retrain the complete benchmarking suite and generate EDA visual plots:
```bash
python src/train.py
```

To train the streamlined 6-parameter pipeline used by the web interface:
```bash
python src/train_web.py
```

---

## 📊 Key Features & Tech Stack

- **Tech Stack**: Python 3, Flask, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
- **Preprocessing**: Cleans numerical text fields (`mileage`, `engine`, `max_power`), handles missing values, and encodes categorical attributes (`brand`, `fuel`, `transmission`, etc.).
- **Evaluation Metrics**: Evaluated via Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score.
