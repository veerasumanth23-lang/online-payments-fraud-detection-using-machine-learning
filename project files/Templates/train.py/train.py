# train.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/fraud.csv")

# Drop unnecessary columns
data = data.drop(["nameOrig", "nameDest"], axis=1)

# Convert categorical column to numeric
le = LabelEncoder()
data["type"] = le.fit_transform(data["type"])

# Features and Target
X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/model.pkl")

print("Model saved successfully!")
