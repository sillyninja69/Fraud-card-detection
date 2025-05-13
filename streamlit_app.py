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

    # Input fields for features V1 to V28
    input_data = {"Amount": amount}
    for i in range(1, 29):
        input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

    # Create a button to make a prediction
    if st.button("Predict"):
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
