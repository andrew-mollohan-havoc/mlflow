import mlflow

from mlflow_harness.training import run_training


def test_run_training_returns_rmse_and_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr("mlflow_harness.training.configure_mlflow", lambda: None)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-training")

    rmse, run_id = run_training(run_name="test-run")

    assert isinstance(rmse, float)
    assert rmse >= 0.0
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_run_training_logs_params_and_metric(tmp_path, monkeypatch):
    monkeypatch.setattr("mlflow_harness.training.configure_mlflow", lambda: None)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-training-params")

    _, run_id = run_training()

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)

    assert run.data.params["n_estimators"] == "100"
    assert run.data.params["max_depth"] == "6"
    assert run.data.params["max_features"] == "3"
    assert "rmse" in run.data.metrics


def test_run_training_default_run_name(tmp_path, monkeypatch):
    monkeypatch.setattr("mlflow_harness.training.configure_mlflow", lambda: None)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-training-name")

    _, run_id = run_training()

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)

    assert run.info.run_name == "model_training_run"
