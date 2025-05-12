import streamlit as st
import numpy as np
from fraud_detection import load_model, predict_transaction

# Set Streamlit page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This app uses a machine learning model to predict the probability of a credit card transaction being fraudulent.
Please enter the transaction details below:
""")

# Load the model
model = load_model()

# Input fields for features
feature_names = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

input_data = []
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", value=0.0, format="%.6f")
        input_data.append(val)

# Prediction
if st.button("🔍 Predict"):
    prediction = predict_transaction(np.array(input_data), model)
    if prediction == 1:
        st.error("⚠️ This transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ This transaction is predicted to be LEGITIMATE.")
