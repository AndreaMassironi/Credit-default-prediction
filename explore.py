import os
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("laotse/credit-risk-dataset")
df = pd.read_csv(os.path.join(path, "credit_risk_dataset.csv"))


print(df.head())
print(df.info())
print(df.describe())
