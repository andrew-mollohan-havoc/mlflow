import mlflow
from mlflow.entities import ViewType

from mlflow_harness.experiments import list_experiments


def test_list_experiments_returns_result(monkeypatch):
    mock_experiments = [object(), object()]

    monkeypatch.setattr("mlflow_harness.experiments.configure_mlflow", lambda: None)
    monkeypatch.setattr(
        mlflow, "search_experiments", lambda view_type=ViewType.ALL: mock_experiments
    )

    result = list_experiments()

    assert result is mock_experiments


def test_list_experiments_default_view_type_is_all(monkeypatch):
    captured = {}

    monkeypatch.setattr("mlflow_harness.experiments.configure_mlflow", lambda: None)

    def fake_search(view_type=ViewType.ACTIVE_ONLY):
        captured["view_type"] = view_type
        return []

    monkeypatch.setattr(mlflow, "search_experiments", fake_search)

    list_experiments()

    assert captured["view_type"] == ViewType.ALL


def test_list_experiments_passes_custom_view_type(monkeypatch):
    captured = {}

    monkeypatch.setattr("mlflow_harness.experiments.configure_mlflow", lambda: None)

    def fake_search(view_type=ViewType.ACTIVE_ONLY):
        captured["view_type"] = view_type
        return []

    monkeypatch.setattr(mlflow, "search_experiments", fake_search)

    list_experiments(view_type=ViewType.DELETED_ONLY)

    assert captured["view_type"] == ViewType.DELETED_ONLY
