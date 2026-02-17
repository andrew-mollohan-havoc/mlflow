import logging
import mlflow
import os

# Set up logging for MLflow
logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)
# Connect to remote MLflow server
TOKEN = os.getenv("MLFLOW_QUERY_TOKEN")
mlflow.set_tracking_uri("https://mlflow.havocai.net?token=" + TOKEN if TOKEN else "https://mlflow.havocai.net")
mlflow.set_experiment("andrewm")
logger.debug("Connected to MLflow tracking server at https://mlflow.havocai.net and set experiment to 'andrewm'")
