import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset safely
def load_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "creditcard.csv")
    df = pd.read_csv(dataset_path)

    if "Class" not in df.columns:
        raise ValueError("The dataset does not contain a 'Class' column.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y

# Train and save the model
def train_model():
    X, y = load_dataset()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler

# Load model and scaler
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Training a new one...")
        return train_model()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Predict transaction class
def predict_transaction(input_data, model, scaler):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return prediction[0]


