import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics

# 1. Set the tracking URI (defaults to local MLflow server)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)

# 2. Set the experiment (creates a new one if name doesn't exist)
experiment_name = "Diabetes_RF_Experiment"
mlflow.set_experiment(experiment_name)

# Prepare data
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# 2. Start an MLflow run
with mlflow.start_run(run_name="model_training_run") as run:
    # Define and log parameters
    params = {"n_estimators": 100, "max_depth": 6, "max_features": 3}
    mlflow.log_params(params)

    # Create and train model
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    # Make predictions and log metrics
    predictions = rf.predict(X_test)
    rmse = sklearn.metrics.root_mean_squared_error(y_test, predictions, squared=False)
    mlflow.log_metric("rmse", rmse)

    # Log the model
    mlflow.sklearn.log_model(rf, "model")

    print(f"RMSE: {rmse}")
    print(f"MLflow Run ID: {run.info.run_id}")

# The run automatically ends when exiting the 'with' block
