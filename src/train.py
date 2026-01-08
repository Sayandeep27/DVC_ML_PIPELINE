import pandas as pd
import pickle
import yaml
import os
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

params = yaml.safe_load(open("params.yaml"))

mlflow.set_experiment("DVC_ML_Pipeline")

with mlflow.start_run():

    mlflow.log_param("test_size", params["train"]["test_size"])
    mlflow.log_param("random_state", params["train"]["random_state"])

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

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    mlflow.log_metric("mse_scaled", mse * 1e30)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.sklearn.log_model(model, "model")

    # Explicit artifacts
    mlflow.log_artifact("models/model.pkl", artifact_path="dvc_model")
    mlflow.log_artifact("params.yaml", artifact_path="config")
