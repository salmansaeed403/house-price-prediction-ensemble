import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ------------------------------------------------------------------
# Load models and preprocessing artifacts (cached so this only runs
# once, not on every widget interaction / rerun)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb_model = pickle.load(open('models/house_price_xgb_model.pkl', 'rb'))
    cb_model = pickle.load(open('models/house_price_cb_model.pkl', 'rb'))
    encoder = pickle.load(open('models/target_encoder.pkl', 'rb'))
    zipcode_avg_price_per_sqft = pickle.load(
        open('models/zipcode_avg_price_per_sqft.pkl', 'rb')
    )
    global_avg_price_per_sqft = pickle.load(
        open('models/global_avg_price_per_sqft.pkl', 'rb')
    )
    return xgb_model, cb_model, encoder, zipcode_avg_price_per_sqft, global_avg_price_per_sqft


try:
    (
        xgb_model,
        cb_model,
        encoder,
        zipcode_avg_price_per_sqft,
        global_avg_price_per_sqft,
    ) = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure the `models/` folder contains "
        "house_price_xgb_model.pkl, house_price_cb_model.pkl, "
        "target_encoder.pkl, zipcode_avg_price_per_sqft.pkl, and "
        "global_avg_price_per_sqft.pkl, placed next to this app."
    )
    st.stop()


CITY_OPTIONS = [
    'Algona', 'Auburn', 'Beaux Arts Village', 'Bellevue',
    'Black Diamond', 'Bothell', 'Burien', 'Carnation',
    'Clyde Hill', 'Covington', 'Des Moines', 'Duvall',
    'Enumclaw', 'Fall City', 'Federal Way',
    'Inglewood-Finn Hill', 'Issaquah', 'Kenmore',
    'Kent', 'Kirkland', 'Lake Forest Park',
    'Maple Valley', 'Medina', 'Mercer Island',
    'Milton', 'Newcastle', 'Normandy Park',
    'North Bend', 'Pacific', 'Preston',
    'Ravensdale', 'Redmond', 'Renton',
    'Sammamish', 'SeaTac', 'Seattle',
    'Shoreline', 'Skykomish', 'Snoqualmie',
    'Snoqualmie Pass', 'Tukwila', 'Vashon',
    'Woodinville', 'Yarrow Point'
]

STATEZIP_OPTIONS = [
    'WA 98001', 'WA 98002', 'WA 98003', 'WA 98004',
    'WA 98005', 'WA 98006', 'WA 98007', 'WA 98008',
    'WA 98010', 'WA 98011', 'WA 98014', 'WA 98019',
    'WA 98022', 'WA 98023', 'WA 98024', 'WA 98027',
    'WA 98028', 'WA 98029', 'WA 98030', 'WA 98031',
    'WA 98032', 'WA 98033', 'WA 98034', 'WA 98038',
    'WA 98039', 'WA 98040', 'WA 98042', 'WA 98045',
    'WA 98047', 'WA 98050', 'WA 98051', 'WA 98052',
    'WA 98053', 'WA 98055', 'WA 98056', 'WA 98057',
    'WA 98058', 'WA 98059', 'WA 98065', 'WA 98068',
    'WA 98070', 'WA 98072', 'WA 98074', 'WA 98075',
    'WA 98077', 'WA 98092', 'WA 98102', 'WA 98103',
    'WA 98105', 'WA 98106', 'WA 98107', 'WA 98108',
    'WA 98109', 'WA 98112', 'WA 98115', 'WA 98116',
    'WA 98117', 'WA 98118', 'WA 98119', 'WA 98122',
    'WA 98125', 'WA 98126', 'WA 98133', 'WA 98136',
    'WA 98144', 'WA 98146', 'WA 98148', 'WA 98155',
    'WA 98166', 'WA 98168', 'WA 98177', 'WA 98178',
    'WA 98188', 'WA 98198', 'WA 98199', 'WA 98288',
    'WA 98354'
]

# Approximate performance of the final ensemble on held-out test data,
# shown to the user for context on how much to trust an estimate.
MODEL_MAE = 91_591
MODEL_R2 = 77.3


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption(
    "Estimate a King County, WA home's market value from an "
    "XGBoost + CatBoost ensemble model."
)

st.divider()

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Living Area (sqft)", min_value=100, value=1500)
    sqft_above = st.number_input("Sqft Above Ground", min_value=100, value=1200)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, value=0)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=100, value=5000)
    floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)

with col2:
    city = st.selectbox("City", CITY_OPTIONS, index=CITY_OPTIONS.index("Seattle"))
    statezip = st.selectbox(
        "ZIP Code", STATEZIP_OPTIONS, index=STATEZIP_OPTIONS.index("WA 98119")
    )
    condition = st.slider("Condition (1-5)", 1, 5, 3)
    view = st.slider("View Rating (0-4)", 0, 4, 0)
    waterfront = st.checkbox("Waterfront property")
    is_renovated = st.checkbox("Has been renovated")
    house_age = st.number_input("House Age (years)", min_value=0, value=20)
    month_sold = st.slider("Month Sold", 1, 12, 6)

st.divider()

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [int(waterfront)],
        'view': [view],
        'condition': [condition],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'city': [city],
        'statezip': [statezip],
        'month_sold': [month_sold],
        'house_age': [house_age],
        'is_renovated': [int(is_renovated)],
    })

    # Same location-price feature engineered during training
    input_df['price_per_sqft_zipcode_avg'] = (
        input_df['statezip']
        .map(zipcode_avg_price_per_sqft)
        .fillna(global_avg_price_per_sqft)
    )

    # XGBoost needs target-encoded categoricals
    input_encoded = encoder.transform(input_df)
    xgb_pred_log = xgb_model.predict(input_encoded)

    # CatBoost uses raw categoricals natively
    cb_pred_log = cb_model.predict(input_df)

    # Final prediction: average of both models, in log-space
    ensemble_log = (xgb_pred_log + cb_pred_log) / 2
    prediction = np.expm1(ensemble_log)[0]

    st.metric("Estimated Price", f"${prediction:,.0f}")
    st.caption(
        f"On held-out test data, this model has an average error "
        f"(MAE) of about ${MODEL_MAE:,.0f} and explains {MODEL_R2:.1f}% "
        f"of price variance (R²). Treat this as an estimate, not a "
        f"formal appraisal."
    )

with st.expander("About this model"):
    st.markdown(
        """
        Trained on historical King County, WA home sales.

        **Final model:** an ensemble that averages predictions from a
        tuned **XGBoost** model and a tuned **CatBoost** model, which
        outperformed either model alone, as well as Random Forest and
        Linear Regression baselines, on mean absolute error and RMSE.

        **Key features used:** living area, bathrooms, waterfront
        status, house age, view rating, and location — including an
        engineered feature capturing each ZIP code's average
        price-per-square-foot.
        """
    )