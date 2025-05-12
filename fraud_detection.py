import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import gdown

class FraudDetector:
    def __init__(self):
        self.model = None
        self.model_file = 'model.pkl'
        self.dataset_file = 'creditcard.csv'
        self.feature_order = [f'V{i}' for i in range(1, 29)] + ['Amount']

    def load_dataset(self):
        """Load the dataset from Google Drive."""
        # Google Drive file ID from the provided link
        file_id = '1mD7t_kGWg9DThiEj5PJDxUuXlLFUYieC'
        gdown.download(f'https://drive.google.com/uc?id={file_id}', self.dataset_file, quiet=False)

        try:
            df = pd.read_csv(self.dataset_file)
            print(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise Exception(f"Dataset file '{self.dataset_file}' not found. Please ensure it is in the correct directory.")

    def preprocess_data(self, data):
        """Preprocess the data for training."""
        X = data.drop('Class', axis=1)
        y = data['Class']
        # Log transform the Amount column to reduce skewness
        X['Amount'] = np.log1p(X['Amount'])
        return X, y

    def train_model(self, X, y):
        """Train the Logistic Regression model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(X_train, y_train)

        # Save the model to a file
        joblib.dump(model, self.model_file)
        print(f"Model trained and saved to '{self.model_file}'.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return model

    def load_model(self):
        """Load the trained model from a file."""
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            print(f"Model loaded from '{self.model_file}'.")
        else:
            # Load the dataset and train the model if the model file does not exist
            data = self.load_dataset()
            X, y = self.preprocess_data(data)
            self.model = self.train_model(X, y)

    def predict_fraud(self, input_data):
        """
        Predict fraud for given transactions.

        Parameters:
        - input_data: pandas DataFrame with columns matching features V1..V28, Amount

        Returns:
        - numpy array of predictions (0=legit, 1=fraud)
        """
        if self.model is None:
            raise Exception("Model not trained. Call load_model() first.")

        # Ensure all needed columns are present
        for feature in self.feature_order:
            if feature not in input_data.columns:
                raise ValueError(f"Missing required feature column: {feature}")

        # Apply log1p transform to Amount column to be consistent
        input_data = input_data.copy()
        input_data['Amount'] = np.log1p(input_data['Amount'])

        X_input = input_data[self.feature_order].values
        preds = self.model.predict(X_input)
        return preds

def main():
    detector = FraudDetector()
    detector.load_model()

    # Example usage: predict fraud for the first 5 samples
    sample_df = detector.load_dataset().head(5)
    print("\nPredictions for first 5 samples:")
    preds = detector.predict_fraud(sample_df)
    print(preds)

if __name__ == "__main__":
    main()
