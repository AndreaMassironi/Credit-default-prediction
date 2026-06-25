import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Reload data and rebuild the SAME test split (deterministic: random_state=42)
df = pd.read_csv("data/clean.csv")
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the already-trained model — NO retraining
model = joblib.load("model.joblib")
predictions = model.predict(X_test)