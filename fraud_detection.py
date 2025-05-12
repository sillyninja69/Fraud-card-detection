import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import joblib
import gdown
import os

# Download the dataset 
if not os.path.exists("creditcard.csv"):
    url = "https://drive.google.com/uc?id=1mD7t_kGWg9DThiEj5PJDxUuXlLFUYieC"
    gdown.download(url, "creditcard.csv", quiet=False)

df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.columns)
print(df.columns.tolist())

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
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

