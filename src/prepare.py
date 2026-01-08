import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/data.csv")

# preprocessing
df["feature2"] = df["feature2"] * 3

df.to_csv("data/processed/processed.csv", index=False)
