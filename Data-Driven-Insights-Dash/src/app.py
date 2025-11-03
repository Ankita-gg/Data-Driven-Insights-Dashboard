import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# Define paths relative to the current script
base_path = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(base_path, "../assets")
data_path = os.path.join(base_path, "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load the trained model and scaler
pipeline_path = os.path.join(assets_path, "pipeline.joblib")
pipeline = joblib.load(pipeline_path)
model = pipeline["model"]
scaler = pipeline["scaler"]

# Streamlit UI setup
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Data-Driven Insights Dashboard")
st.write("Predict Customer Churn using a trained ML model built with Random Forest")

# Sidebar input form
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 10.0, 150.0, 70.0)
total = st.sidebar.number_input("Total Charges", 10.0, 8000.0, 1000.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Predict churn
if st.sidebar.button("Predict"):
    # Convert categorical inputs to numeric
    input_data = np.array([[0 if gender == "Male" else 1, senior, tenure, monthly, total,
                            0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    result = "🔴 Churn Risk" if prediction == 1 else "🟢 Retained Customer"

    st.subheader("Prediction Result")
    st.success(f"The model predicts: {result}")

st.markdown("---")

# Show model performance image
conf_matrix_path = os.path.join(assets_path, "confusion_matrix.png")
if os.path.exists(conf_matrix_path):
    st.image(conf_matrix_path, caption="Model Performance (Confusion Matrix)")
else:
    st.warning("Confusion matrix image not found.")

