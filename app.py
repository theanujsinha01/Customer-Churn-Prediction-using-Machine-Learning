import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load saved model and encoders
model_data = joblib.load('random_forest_model.pkl')  # This should be a dict with 'model' and 'feature_names'
loaded_encoder = joblib.load('label_encoders.pkl')   # This should be a dict with encoders for each column

# Extract model and feature names
loaded_model = model_data['model']
feature_names = model_data['feature_names']

# Page setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Prediction App")
st.info("⚠️ Make sure your model was trained using the same preprocessing logic.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Convert inputs into a DataFrame
input_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# Encode categorical features using saved LabelEncoders
for col, le in loaded_encoder.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Add any missing features (columns used in training but not in form)
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # Default fill (use 'No' or 0 as per your training data)

# Reorder columns to match the training feature order
input_df = input_df[feature_names]

# Prediction
if st.button("Predict Churn"):
    prediction = loaded_model.predict(input_df)

    if prediction[0] == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
