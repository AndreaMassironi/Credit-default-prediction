import os
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("laotse/credit-risk-dataset")
df = pd.read_csv(os.path.join(path, "credit_risk_dataset.csv"))

print("Missing before:\n", df.isnull().sum())

df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

# 3) Drop rows with impossible values (outliers)
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 60]

# 4) Final check
print("\nMissing after:\n", df.isnull().sum())
print("Rows left:", df.shape[0])

# A model needs numbers, not text -> convert text columns into 0/1 columns (one-hot)
df = pd.get_dummies(df, drop_first=True, dtype=int)

print("\nShape after encoding:", df.shape)
print(df.head())

os.makedirs("data", exist_ok=True)
df.to_csv("data/clean.csv", index=False)
print("Saved data/clean.csv")