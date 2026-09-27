import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Food Delivery ETA Prediction",
    layout="wide"
)

st.title("Food Delivery ETA Prediction")

st.write(
    "Enter the delivery information below to predict "
    "the estimated delivery time."
)

with open(f"{path}/models/model.json", "r") as file:
    model_info = json.load(file)

st.header("Model Information")

st.write(
    f"**Model :** {model_info['model']}"
)

st.subheader("Hyperparameters")

hyperparameters = model_info["hyperparameters"]

st.write(f"**Number of Estimators :** {hyperparameters['n_estimators']}")
st.write(f"**Maximum Depth :** {hyperparameters['max_depth']}")
st.write(f"**Minimum Samples Split :** {hyperparameters['min_samples_split']}")
st.write(f"**Minimum Samples Leaf :** {hyperparameters['min_samples_leaf']}")
st.write(f"**Maximum Features :** {hyperparameters['max_features']}")

st.header("Model Performance")

testing = model_info["testing"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MAE", f"{testing['mae']:.2f} min")

with col2:
    st.metric("MSE", f"{testing['mse']:.2f}")

with col3:
    st.metric("RMSE", f"{testing['rmse']:.2f} min")

with col4:
    st.metric("R² Score", f"{testing['r2_percent']:.2f}%")

st.subheader("Cross Validation")

cv = model_info["cross_validation"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Mean MAE",
        f"{cv['mean_mae']:.3f}",
        f"{cv['mae_std']:.3f}"
    )

with col2:
    st.metric(
        "Mean RMSE",
        f"{cv['mean_rmse']:.3f}",
        f"{cv['rmse_std']:.3f}"
    )

with col3:
    st.metric(
        "Mean R²",
        f"{cv['mean_r2']:.3f}"
    )

with col4:
    st.metric(
        "R² Std",
        f"{cv['r2_std']:.3f}"
    )

st.subheader("Training vs Testing")

training = model_info["training"]

performance_data = {
    "Metric": ["MAE", "MSE", "RMSE", "R²"],
    "Training": [
        training["mae"],
        training["mse"],
        training["rmse"],
        training["r2_percent"]
    ],
    "Testing": [
        testing["mae"],
        testing["mse"],
        testing["rmse"],
        testing["r2_percent"]
    ]
}

st.dataframe(
    performance_data,
    width="stretch"
)

st.subheader("Delivery Information")

delivery_person_age = st.number_input(
    "Delivery Person Age",
    min_value=18,
    max_value=70,
    value=30
)

delivery_person_ratings = st.number_input(
    "Delivery Person Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.5,
    step=0.1
)

vehicle_condition = st.number_input(
    "Vehicle Condition",
    min_value=0,
    max_value=2,
    value=2
)

multiple_deliveries = st.number_input(
    "Multiple Deliveries",
    min_value=0,
    max_value=3,
    value=0
)

distance_km = st.number_input(
    "Distance (km)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1
)

order_picked_hour = st.number_input(
    "Order Picked Hour",
    min_value=0,
    max_value=23,
    value=12
)

pickup_delay_minute = st.number_input(
    "Pickup Delay (minutes)",
    min_value=0.0,
    max_value=120.0,
    value=10.0,
    step=1.0
)

weatherconditions = st.selectbox(
    "Weather Conditions",
    [
        "Sunny",
        "Stormy",
        "Sandstorms",
        "Cloudy",
        "Windy",
        "Fog"
    ]
)

road_traffic_density = st.selectbox(
    "Road Traffic Density",
    [
        "Low",
        "Medium",
        "High",
        "Jam"
    ]
)

type_of_vehicle = st.selectbox(
    "Vehicle Type",
    [
        "Motorcycle",
        "Scooter",
        "Electric_scooter"
    ]
)

festival = st.selectbox(
    "Festival",
    [
        "No",
        "Yes"
    ]
)

city = st.selectbox(
    "City",
    [
        "Metropolitian",
        "Urban",
        "Semi-Urban"
    ]
)

if st.button("Predict Delivery Time"):

    input_data = pd.DataFrame({
        "Delivery_person_Age": [delivery_person_age],
        "Delivery_person_Ratings": [delivery_person_ratings],
        "Vehicle_condition": [vehicle_condition],
        "multiple_deliveries": [multiple_deliveries],
        "Distance_km": [distance_km],
        "Order_picked_hour": [order_picked_hour],
        "Pickup_delay_minute": [pickup_delay_minute],
        "Weatherconditions": [weatherconditions],
        "Road_traffic_density": [road_traffic_density],
        "Type_of_vehicle": [type_of_vehicle],
        "Festival": [festival],
        "City": [city]
    })

    model = joblib.load(f"{path}/models/model.joblib")

    prediction = model.predict(input_data)

    predicted_time = prediction[0]

    st.success(
        f"Estimated Delivery Time : **{predicted_time:.0f} minutes**"
    )
