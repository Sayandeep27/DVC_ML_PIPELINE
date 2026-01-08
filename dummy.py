import mlflow

client = mlflow.tracking.MlflowClient()
run = client.get_run("b22d447c17044e43b6caf1835be9d2cf")
print(run.data.metrics)