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
  
    amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=1.0, step=0.01)

    # Create a button to make a prediction
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = {
            'V1': v1,
            'V2': v2,
            'V3': v3,
           
        
        }
           

