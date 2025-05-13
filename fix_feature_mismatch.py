import pandas as pd
import joblib

# Load the model
model = joblib.load("model.pkl")

# Load the dataset for prediction
df = pd.read_csv("creditcard.csv")

# Define the expected feature order
feature_order = [f'V{i}' for i in range(1, 29)] + ['Amount']

# Ensure input features align with the model
X = df[feature_order]

# Check feature count
print("Input features shape:", X.shape)
print("Model expects:", model.n_features_in_)

# Predict and print results
predictions = model.predict(X)
print("Predictions:", predictions)
