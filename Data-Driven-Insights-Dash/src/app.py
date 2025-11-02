import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load trained model
pipeline = joblib.load("assets/pipeline.joblib")
model = pipeline["model"]
scaler = pipeline["scaler"]

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

if st.sidebar.button("Predict"):
    # Convert input to numeric array
    input_data = np.array([[0 if gender == "Male" else 1, senior, tenure, monthly, total, 
                            0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    result = "🔴 Churn Risk" if prediction == 1 else "🟢 Retained Customer"
    st.subheader("Prediction Result")
    st.success(f"The model predicts: {result}")

st.markdown("---")
st.image("assets/confusion_matrix.png", caption="Model Performance (Confusion Matrix)")
