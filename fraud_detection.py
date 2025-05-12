import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_dataset():
    df = pd.read_csv('your_dataset.csv')
    
    # Check for the column name, consider different possible names
    if 'Class' not in df.columns and 'class' not in df.columns and 'target' not in df.columns:
        raise ValueError("The dataset does not contain a 'Class', 'class', or 'target' column.")
    
    # Use the appropriate column as the target
    if 'Class' in df.columns:
        target_column = 'Class'
    elif 'class' in df.columns:
        target_column = 'class'
    else:
        target_column = 'target'  # or whichever name is in your dataset
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
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



