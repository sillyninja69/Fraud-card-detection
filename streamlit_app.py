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
    amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=1.0, step=0.01)
    v1 = st.number_input("V1", value=0.0, step=0.01)
    v2 = st.number_input("V2", value=0.0, step=0.01)
    v3 = st.number_input("V3", value=0.0, step=0.01)
    # Add more input fields for V4 to V28 as needed

    # Create a button to make a prediction
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = {
            'V1': v1,
            'V2': v2,
            'V3': v3,
            'Amount': amount
            # Add more fields as necessary
        }
        
        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])  # Convert to DataFrame

        # Debugging: Print the input DataFrame
        st.write("Input DataFrame:", input_df)

        # Make prediction
        try:
            prediction = detector.predict_fraud(input_df)
            if prediction[0] == 1:
                st.error("Transaction is likely fraudulent.")
            else:
                st.success("Transaction is legitimate.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
