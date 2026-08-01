# 🏠 House Price Prediction using Machine Learning

🔗 **[Live Demo](https://house-price-prediction-by-salman.streamlit.app/)**

## 📌 Project Overview

This project predicts house prices from property characteristics — living area, bedrooms, bathrooms, location, condition, and more — using historical King County, WA housing sales data.

It demonstrates a complete, iterative Machine Learning workflow:

- Data Cleaning & Feature Engineering
- Exploratory Data Analysis (EDA)
- Model Training across four algorithms (Linear Regression, Random Forest, XGBoost, CatBoost)
- Hyperparameter Tuning with `RandomizedSearchCV`
- Model Evaluation & Residual/Error Diagnostics
- Diagnosis-Driven Feature Engineering
- Model Ensembling
- Feature Importance Analysis
- Deployment as an interactive Streamlit app

---

## 🎯 Problem Statement

Accurately estimating house prices matters for buyers, sellers, and real estate professionals. This project builds a model that predicts house prices from historical sales data and property attributes, with an emphasis on understanding *where* the model succeeds or struggles, not just reporting a single accuracy number.

---

## 📂 Dataset Information

| Feature | Description |
|----------|-------------|
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft_living | Living area in square feet |
| sqft_lot | Lot size in square feet |
| floors | Number of floors |
| waterfront | Waterfront property indicator |
| view | Quality of view |
| condition | Overall house condition |
| sqft_above | Square footage above ground |
| sqft_basement | Basement square footage |
| city | Property city |
| statezip | ZIP code |
| month_sold | Month property was sold |
| house_age | Age of house at time of sale |
| is_renovated | Renovation indicator |
| price_per_sqft_zipcode_avg | Engineered: average price/sqft for the property's ZIP code, based on training data |

**Target variable:** `price` (modeled as `log_price` to reduce the impact of right-skew from high-value homes)

---

## 🧠 Models & Results

Four models were trained and tuned, then the two strongest were combined into a final ensemble:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $131,190 | $660,682 | 72.4% |
| Random Forest (tuned) | $95,900 | $202,018 | 76.8% |
| XGBoost (tuned) | $92,898 | $198,527 | 77.3% |
| CatBoost (tuned) | $92,583 | $198,331 | 76.7% |
| **Ensemble (XGBoost + CatBoost)** | **$91,591** | **$196,849** | **77.3%** |

**Final model:** a simple average of XGBoost and CatBoost predictions (in log-space). The two models make partially different errors — a price-bracket analysis showed XGBoost performs marginally better in the $415K–$723K range, while CatBoost is notably stronger above $723K — so averaging them reduces error further than either model alone.

### Key findings during development

- Diagnosed and fixed a bug where XGBoost's evaluation was silently reusing Random Forest's predictions, which had made the two models appear identical.
- Found that each model's hyperparameter search (`RandomizedSearchCV`) wasn't actually connected to the final trained model — searches were run, but final models used separate hardcoded parameters. Fixed so tuning results are used directly.
- Residual analysis by price bracket showed all models had consistent ~12–14% error in the $300K–$723K range, but roughly 3x higher dollar error for homes above $723K.
- Engineered `price_per_sqft_zipcode_avg` in response to that finding — it improved MAE, RMSE, and R² across every model, confirming the earlier performance ceiling was a missing-feature problem rather than a model-choice one.

### Known limitations

- High-value homes (>$723K) remain the hardest segment to price accurately, even after feature engineering.
- Target encoding and `price_per_sqft_zipcode_avg` carry a mild same-row leakage risk; a more rigorous version would use K-fold (out-of-fold) encoding.
- Results are based on a single train/test split rather than cross-validated estimates.
- The model shows non-monotonic behavior for bedroom count and floors in isolation (confirmed via partial dependence analysis) — price peaks around 4 bedrooms / 1.5 floors and declines beyond that. This reflects real patterns in the training data (larger bedroom/floor counts correlate with a different, typically lower-value housing segment) rather than a modeling error.

---

## 🔑 Important Features

Based on feature importance analysis (Random Forest) and correlation with price:

- Living area (`sqft_living`)
- Number of bathrooms
- Waterfront status
- House age
- View quality
- Location (`city` / `statezip`, plus the engineered `price_per_sqft_zipcode_avg`)

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🖥️ App Demo

A Streamlit app (`app.py`) provides an interactive interface: enter property details and get an instant price estimate from the final ensemble model, along with the model's MAE/R² for context.

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── notebooks/
│   └── housePricePrediction.ipynb
├── data/
│   └── data.csv
├── models/
│   ├── house_price_xgb_model.pkl
│   ├── house_price_cb_model.pkl
│   ├── target_encoder.pkl
│   ├── zipcode_avg_price_per_sqft.pkl
│   └── global_avg_price_per_sqft.pkl
├── screenshots/
│   └── (EDA plots, feature importance, etc.)
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn, XGBoost, CatBoost, category_encoders
- **Deployment:** Streamlit

---

## 🚀 Possible Future Improvements

- K-fold (out-of-fold) target encoding to remove leakage risk
- A segmented or two-stage model specifically for high-value homes
- LightGBM as an additional ensemble member
- Cross-validated performance estimates
