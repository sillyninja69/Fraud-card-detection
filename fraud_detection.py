import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Columns in DataFrame:", df.columns.tolist())  # Debug print to inspect columns
        return df
    except FileNotFoundError:
        print("Error: The file was not found.")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Process data
def process_data(df):
    if "Class" not in df.columns:
        raise KeyError("Column 'Class' not found in the dataset. Please check the file.")
    
    # Dropping the 'Class' column to separate features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    return X, y

# Train model
def train_model(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Save the model
def save_model(model, model_filename):
    try:
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Load the trained model
def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Make predictions
def predict_transaction(model, transaction_data):
    try:
        prediction = model.predict(transaction_data)
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise

# Main function to tie everything together
def main():
    # Load dataset
    df = load_data("creditcard.csv")
    
    # Process data
    X, y = process_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save the model
    save_model(model, "fraud_model.pkl")

if __name__ == "__main__":
    main()
