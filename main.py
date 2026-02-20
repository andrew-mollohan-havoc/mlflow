import logging
import mlflow
import os

# Set up logging for MLflow
logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)
# Connect to local MLflow server by default (override with MLFLOW_TRACKING_URI)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("andrewm")
logger.debug("Connected to MLflow tracking server at %s and set experiment to 'andrewm'", tracking_uri)
