import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

# Step 1: Load the dataset
df = pd.read_csv("creditcard.csv")

# Step 2: Check first few rows
print("🔹 First 5 rows of the dataset:")
print(df.head())

# Step 3: Dataset info
print("\n🔹 Dataset Info:")
print(df.info())

# Step 4: Check missing values
print("\n🔹 Missing values in each column:")
print(df.isnull().sum())

# Step 5: Class distribution
print("\n🔹 Class distribution (0 = legit, 1 = fraud):")
print(df['Class'].value_counts())

# Step 6: Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Step 7: Handle imbalance with undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print("\n✅ After undersampling:")
print(y_resampled.value_counts())

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Step 9: Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 10: Predictions and Evaluation
y_pred = model.predict(X_test)

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Save the model (future use in GUI)
import joblib
joblib.dump(model, "fraud_model.pkl")
