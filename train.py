import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv("data/clean.csv")

# Split features (X) from the target (y)
X = df.drop("loan_status", axis=1)   # everything except the target
y = df["loan_status"]                # 1 = default, 0 = repaid

# Train/test split: 80% , 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a first model: Logistic Regression
# Build the pipeline (scale -> classify)
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced")
)

# Fine-tuning: try several C values, keep the best by F1 (cross-validated)
param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best C:", grid.best_params_)
print("Best CV f1:", round(grid.best_score_, 3))

model = grid.best_estimator_

# Predict on the test set and measure accuracy
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions))
joblib.dump(model, "model.joblib")
print("Saved model.joblib")