import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/processed/processed.csv")

X = df[["feature1", "feature2"]]
y = df["target"]

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

os.makedirs("metrics", exist_ok=True)

with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mse}, f)
