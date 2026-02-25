import mlflow

from mlflow_harness.config import Settings
from mlflow_harness.tracking import apply_query_token_patch, configure_mlflow


def test_configure_mlflow_sets_tracking(monkeypatch):
    called = {}

    def _set_tracking_uri(uri):
        called["uri"] = uri

    def _set_experiment(name):
        called["experiment"] = name

    monkeypatch.setattr(mlflow, "set_tracking_uri", _set_tracking_uri)
    monkeypatch.setattr(mlflow, "set_experiment", _set_experiment)

    settings = Settings(
        tracking_uri="http://example",
        experiment_name="example-exp",
        query_token=None,
    )
    configure_mlflow(settings, enable_token_patch=False)

    assert called["uri"] == "http://example"
    assert called["experiment"] == "example-exp"


def test_apply_query_token_patch_appends_token(monkeypatch):
    from mlflow.utils import rest_utils

    captured = {}

    def fake_http_request(host_creds, endpoint, method, *args, **kwargs):
        captured["endpoint"] = endpoint
        return "ok"

    monkeypatch.setattr(rest_utils, "http_request", fake_http_request)

    patched = apply_query_token_patch("abc123")
    assert patched is True

    rest_utils.http_request(None, "/api/2.0/mlflow/experiments/list", "GET")

    assert "token=abc123" in captured["endpoint"]
