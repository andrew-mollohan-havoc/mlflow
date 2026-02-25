"""Example model training and MLflow logging."""

from __future__ import annotations

import mlflow
import sklearn.metrics
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .tracking import configure_mlflow


def run_training(run_name: str = "model_training_run") -> tuple[float, str]:
    """Train a simple model and log to MLflow.

    Returns:
        (rmse, run_id)
    """
    configure_mlflow()

    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    with mlflow.start_run(run_name=run_name) as run:
        params = {"n_estimators": 100, "max_depth": 6, "max_features": 3}
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)
        rmse = sklearn.metrics.root_mean_squared_error(y_test, predictions, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(rf, "model")

    return rmse, run.info.run_id
