import logging

import mlflow

from mlflow_harness.config import Settings
from mlflow_harness.tracking import _PATCH_ATTR, apply_query_token_patch, configure_mlflow

# ---------------------------------------------------------------------------
# apply_query_token_patch
# ---------------------------------------------------------------------------


def test_apply_query_token_patch_none_returns_false():
    assert apply_query_token_patch(None) is False


def test_apply_query_token_patch_empty_string_returns_false():
    assert apply_query_token_patch("") is False


def test_apply_query_token_patch_appends_token(monkeypatch):
    from mlflow.utils import rest_utils

    captured = {}

    def fake_http_request(host_creds, endpoint, method, *args, **kwargs):
        captured["endpoint"] = endpoint
        return "ok"

    monkeypatch.setattr(rest_utils, "http_request", fake_http_request)

    assert apply_query_token_patch("abc123") is True

    rest_utils.http_request(None, "/api/2.0/mlflow/experiments/list", "GET")

    assert "token=abc123" in captured["endpoint"]


def test_apply_query_token_patch_existing_query_string_uses_ampersand(monkeypatch):
    from mlflow.utils import rest_utils

    captured = {}

    def fake_http_request(host_creds, endpoint, method, *args, **kwargs):
        captured["endpoint"] = endpoint
        return "ok"

    monkeypatch.setattr(rest_utils, "http_request", fake_http_request)
    apply_query_token_patch("tok")

    rest_utils.http_request(None, "/api?foo=bar", "GET")

    assert "token=tok" in captured["endpoint"]
    assert captured["endpoint"].count("?") == 1


def test_apply_query_token_patch_idempotent(monkeypatch):
    """Calling the patch twice must not double-wrap the original function."""
    from mlflow.utils import rest_utils

    call_count = {"n": 0}

    def fake_http_request(host_creds, endpoint, method, *args, **kwargs):
        call_count["n"] += 1
        return "ok"

    monkeypatch.setattr(rest_utils, "http_request", fake_http_request)

    apply_query_token_patch("tok")
    apply_query_token_patch("tok")  # second call should be a no-op

    rest_utils.http_request(None, "/ep", "GET")

    assert call_count["n"] == 1


def test_apply_query_token_patch_rest_utils_unavailable(monkeypatch):
    """Returns False gracefully when rest_utils cannot be imported."""
    import sys

    import mlflow.utils as _mlflow_utils

    monkeypatch.setitem(sys.modules, "mlflow.utils.rest_utils", None)
    monkeypatch.delattr(_mlflow_utils, "rest_utils")

    result = apply_query_token_patch("token")
    assert result is False


def test_apply_query_token_patch_artifact_repo_unavailable(monkeypatch):
    """Returns True even when the artifact repo module cannot be imported."""
    import sys

    import mlflow.store.artifact as _artifact_pkg
    from mlflow.utils import rest_utils

    monkeypatch.setattr(rest_utils, "http_request", lambda *a, **kw: "ok")
    monkeypatch.setitem(sys.modules, "mlflow.store.artifact.http_artifact_repo", None)
    monkeypatch.delattr(_artifact_pkg, "http_artifact_repo", raising=False)

    result = apply_query_token_patch("tok-artifact")
    assert result is True


def test_apply_query_token_patch_sets_patch_attr(monkeypatch):
    from mlflow.utils import rest_utils

    monkeypatch.setattr(rest_utils, "http_request", lambda *a, **kw: "ok")

    apply_query_token_patch("tok")

    assert getattr(rest_utils.http_request, _PATCH_ATTR, False) is True


# ---------------------------------------------------------------------------
# configure_mlflow
# ---------------------------------------------------------------------------


def test_configure_mlflow_sets_tracking_uri_and_experiment(monkeypatch):
    called = {}

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: called.__setitem__("uri", uri))
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: called.__setitem__("exp", name))

    settings = Settings(
        tracking_uri="http://example",
        experiment_name="example-exp",
        query_token=None,
    )
    configure_mlflow(settings, enable_token_patch=False)

    assert called["uri"] == "http://example"
    assert called["exp"] == "example-exp"


def test_configure_mlflow_returns_settings(monkeypatch):
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: None)

    settings = Settings(tracking_uri="http://x", experiment_name="exp", query_token=None)
    result = configure_mlflow(settings, enable_token_patch=False)

    assert result is settings


def test_configure_mlflow_uses_get_settings_when_none(monkeypatch):
    from mlflow_harness.config import Settings as S

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: None)

    default = S(tracking_uri="http://default", experiment_name="default-exp", query_token=None)
    monkeypatch.setattr("mlflow_harness.tracking.get_settings", lambda: default)

    result = configure_mlflow(enable_token_patch=False)

    assert result is default


def test_configure_mlflow_with_token_calls_patch(monkeypatch):
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: None)

    patch_calls = []
    monkeypatch.setattr(
        "mlflow_harness.tracking.apply_query_token_patch",
        lambda token: patch_calls.append(token),
    )

    settings = Settings(tracking_uri="http://x", experiment_name="exp", query_token="secret")
    configure_mlflow(settings, enable_token_patch=True)

    assert patch_calls == ["secret"]


def test_configure_mlflow_logger_sets_debug_level(monkeypatch):
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: None)

    settings = Settings(tracking_uri="http://x", experiment_name="exp", query_token=None)
    configure_mlflow(settings, enable_token_patch=False, logger_name="test_harness_logger")

    logger = logging.getLogger("test_harness_logger")
    assert logger.level == logging.DEBUG


def test_configure_mlflow_logger_name_none_does_not_raise(monkeypatch):
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: None)

    settings = Settings(tracking_uri="http://x", experiment_name="exp", query_token=None)
    configure_mlflow(settings, enable_token_patch=False, logger_name=None)
