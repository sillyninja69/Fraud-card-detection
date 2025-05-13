import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Ensure data has all required features
required_features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
if not all(feature in df.columns for feature in required_features):
    raise ValueError("Dataset does not have the required features.")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Enforce correct feature order
feature_order = [f'V{i}' for i in range(1, 29)] + ['Amount']
X = X[feature_order]

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

# Verify feature count
print("Shape of X_train_scaled:", X_train_scaled.shape)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
