import pandas as pd
import joblib
import os
import gdown

# Load dataset
df = pd.read_csv("creditcard.csv")

# Preprocess the data
if "Class" in df.columns:
    X = df.drop("Class", axis=1)
    y = df["Class"]
else:
    raise ValueError("The dataset does not contain a 'Class' column.")

# Function to load the model
def load_model():
    model_filename = "model.pkl"
    model_url = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"  # Replace with your actual file ID

    if not os.path.exists(model_filename):
        gdown.download(model_url, model_filename, quiet=False)

    model = joblib.load(model_filename)
    return model

# Function to predict fraud
def predict_transaction(input_data, model):
    prediction = model.predict([input_data])
    return prediction[0]

