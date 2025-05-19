import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("ev_sales_model.pkl")

st.title("Electric Vehicle Sales Prediction in India")

# --- User Inputs ---
year = st.selectbox("Select Year", list(range(2014, 2025)))
month = st.slider("Month (1-12)", 1, 12, 1)
day = st.slider("Day", 1, 31, 1)

state = st.selectbox("State", ['Maharashtra', 'Karnataka', 'Uttar Pradesh', 'goa','Andra pradesh', 'TamilNadu'])  # Add more as needed
vehicle_class = st.selectbox("Vehicle Class", ['MOTOR CAR', 'BUS', 'MOTOR CAB'])  # Add more
vehicle_category = st.selectbox("Vehicle Category", ['2-Wheelers', '4-Wheelers', 'Bus', 'Others'])
vehicle_type = st.selectbox("Vehicle Type", ['2W_Personal', '4W_Shared', 'Bus', 'Others'])

# --- Data Preparation ---
# One-hot encoding manually since we can't use pd.get_dummies with one sample
features = {
    'Year': year,
    'Month': month,
    'Day': day
}

# Prepare all dummy column names as used during model training
dummy_columns = model.feature_names_in_

# Initialize all to 0
for col in dummy_columns:
    if col not in features:
        features[col] = 0

# Activate selected one-hot values
features[f'State_{state}'] = 1
features[f'Vehicle_Class_{vehicle_class}'] = 1
features[f'Vehicle_Category_{vehicle_category}'] = 1
features[f'Vehicle_Type_{vehicle_type}'] = 1

# --- Convert to DataFrame ---
df_input = pd.DataFrame([features])

# --- Prediction ---
if st.button("Predict EV Sales"):
    prediction = model.predict(df_input)
    st.success(f"Predicted EV Sales Quantity: {int(prediction[0])}")
    