import streamlit as st
import pandas as pd
from fraud_detection import FraudDetector  # Import the class

def main():
    st.title("Credit Card Fraud Detection")

    # Create an instance of the FraudDetector
    detector = FraudDetector()
    detector.load_model()  # Load the model

    # Input fields for user to enter transaction details
    st.subheader("Enter Transaction Details")
    
    # Create input fields for V1 to V28
    v1 = st.number_input("V1", value=0.0, step=0.01)
    v2 = st.number_input("V2", value=0.0, step=0.01)
    v3 = st.number_input("V3", value=0.0, step=0.01)
    v4 = st.number_input("V4", value=0.0, step=0.01)
    v5 = st.number_input("V5", value=0.0, step=0.01)
    v6 = st.number_input("V6", value=0.0, step=0.01)
    v7 = st.number_input("V7", value=0.0, step=0.01)
    v8 = st.number_input("V8", value=0.0, step=0.01)
    v9 = st.number_input("V9", value=0.0, step=0.01)
    v10 = st.number_input("V10", value=0.0, step=0.01)
    v11 = st.number_input("V11", value=0.0, step=0.01)
    v12 = st.number_input("V12", value=0.0, step=0.01)
    v13 = st.number_input("V13", value=0.0, step=0.01)
    v14 = st.number_input("V14", value=0.0, step=0.01)
    v15 = st.number_input("V15", value=0.0, step=0.01)
    v16 = st.number_input("V16", value=0.0, step=0.01)
    v17 = st.number_input("V17", value=0.0, step=0.01)
    v18 = st.number_input("V18", value=0.0, step=0.01)
    v19 = st.number_input("V19", value=0.0, step=0.01)
    v20 = st.number_input("V20", value=0.0, step=0.01)
    v21 = st.number_input("V21", value=0.0, step=0.01)
    v22 = st.number_input("V22", value=0.0, step=0.01)
    v23 = st.number_input("V23", value=0.0, step=0.01)
    v24 = st.number_input("V24", value=0.0, step=0.01)
    v25 = st.number_input("V25", value=0.0, step=0.01)
    v26 = st.number_input("V26", value=0.0, step=0.01)
    v27 = st.number_input("V27", value=0.0, step=0.01)
    v28 = st.number_input("V28", value=0.0, step=0.01)
    amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=1.0, step=0.01)

    # Create a button to make a prediction
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = {
            'V1': v1,
            'V2': v2,
            'V3': v3,
            'V4': v4,
            'V5': v5,
            'V6': v6,
            'V7': v7,
            'V8': v8,
            'V9': v9,
            'V10': v10,  }
