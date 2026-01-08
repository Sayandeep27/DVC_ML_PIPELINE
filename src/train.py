import pandas as pd
import pickle
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed/processed.csv")

X = df[["feature1", "feature2"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"]
)

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
