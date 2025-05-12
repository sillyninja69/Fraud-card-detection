import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

try:
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    raise ImportError("imblearn package is not installed. Please install it with 'pip install imblearn'.")

try:
    import gdown
except ImportError:
    raise ImportError("gdown package is not installed. Please install it with 'pip install gdown'.")

def main():
    # Download the dataset 
    if not os.path.exists("creditcard.csv"):
        # Use the file ID for direct download with gdown
        file_id = "1mD7t_kGWg9DThiEj5PJDxUuXlLFUYieC"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "creditcard.csv", quiet=False)

    if not os.path.exists("creditcard.csv"):
        raise FileNotFoundError("Failed to download creditcard.csv dataset.")

    df = pd.read_csv("creditcard.csv")

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Handle class imbalance using undersampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

if __name__ == "__main__":
    main()
