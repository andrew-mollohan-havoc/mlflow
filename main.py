import logging
import mlflow
import os

logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)

TOKEN = os.getenv("MLFLOW_QUERY_TOKEN")

# Connect to remote MLflow server
mlflow.set_tracking_uri("https://mlflow.havocai.net?token=" + TOKEN if TOKEN else "https://mlflow.havocai.net")
mlflow.set_experiment("Default")
logger.debug("Connected to MLflow tracking server at https://mlflow.havocai.net and set experiment to 'Default'")
