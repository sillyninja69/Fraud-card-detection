import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_dataset():
    try:
        df = pd.read_csv("creditcard.csv")  # update path if needed
    except FileNotFoundError:
        raise FileNotFoundError("Dataset file 'creditcard.csv' not found. Please check the file path.")

    # Handle common casing issues
    colnames = [col.lower() for col in df.columns]
    if 'class' not in colnames:
        raise ValueError("The dataset does not contain a 'Class' or 'class' column.")

    # Normalize the column name
    df.columns = [col.lower() for col in df.columns]

    X = df.drop("class", axis=1)
    y = df["class"]
    return X, y

def train_model():
    X, y = load_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler

def load_model():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        return train_model()

    return model, scaler

def predict_transaction(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]



