import streamlit as st
import pandas as pd
import numpy as np
from fraud_detection import load_model, predict_transaction

st.set_page_config(page_title="Fraud Detection App", layout="centered")

# Load trained model and scaler
model, scaler = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# Input fields (you can modify based on your dataset features)
# For this demo, let’s assume the dataset uses 30 anonymized features (V1 to V28, Time, Amount)

# We'll generate dummy inputs to match 30 columns for now
input_data = []

st.subheader("Enter Transaction Features")

for i in range(30):  # The dataset has 30 features
    val = st.number_input(f"Feature {i+1}", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Check for Fraud"):
    result = predict_transaction(model, scaler, input_array)
    if result == 1:
        st.error("⚠️ This transaction is predicted to be **Fraudulent**.")
    else:
        st.success("✅ This transaction is predicted to be **Legitimate**.")
