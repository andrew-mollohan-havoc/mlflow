import os
from contextlib import contextmanager
from unittest.mock import MagicMock

import mlflow
import pytest

from mlflow_harness.validation import _VALIDATION_EXPERIMENT, validate_s3_artifacts


def _make_run(
    run_id="test-run-123", artifact_uri="s3://my-bucket/experiments/0/test-run-123/artifacts"
):
    run = MagicMock()
    run.info.run_id = run_id
    run.info.artifact_uri = artifact_uri
    return run


def _mock_start_run(run):
    @contextmanager
    def _ctx(run_name=None):
        yield run

    return _ctx


def _patch_mlflow(monkeypatch, run):
    monkeypatch.setattr("mlflow_harness.validation.configure_mlflow", lambda: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda key, value: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "start_run", _mock_start_run(run))


def test_validate_s3_artifacts_returns_run_id_and_uri(monkeypatch):
    run = _make_run()
    _patch_mlflow(monkeypatch, run)

    run_id, artifact_uri = validate_s3_artifacts()

    assert run_id == "test-run-123"
    assert artifact_uri == "s3://my-bucket/experiments/0/test-run-123/artifacts"


def test_validate_s3_artifacts_raises_for_local_path(monkeypatch):
    run = _make_run(artifact_uri="/local/path/artifacts")
    _patch_mlflow(monkeypatch, run)

    with pytest.raises(ValueError, match="local storage"):
        validate_s3_artifacts()


def test_validate_s3_artifacts_raises_when_mlruns_in_uri(monkeypatch):
    run = _make_run(artifact_uri="http://server/mlruns/0/abc/artifacts")
    _patch_mlflow(monkeypatch, run)

    with pytest.raises(ValueError, match="local storage"):
        validate_s3_artifacts()


def test_validate_s3_artifacts_uses_correct_experiment(monkeypatch):
    run = _make_run()
    captured = {}

    monkeypatch.setattr("mlflow_harness.validation.configure_mlflow", lambda: None)
    monkeypatch.setattr(
        mlflow, "set_experiment", lambda name: captured.__setitem__("experiment", name)
    )
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "start_run", _mock_start_run(run))

    validate_s3_artifacts()

    assert captured["experiment"] == _VALIDATION_EXPERIMENT


def test_validate_s3_artifacts_cleans_up_temp_file_on_error(monkeypatch):
    """Temp file must be deleted even when log_artifact raises."""
    run = _make_run()
    created_files = []

    def fake_log_artifact(local_path, *args, **kwargs):
        created_files.append(local_path)
        raise RuntimeError("S3 upload failed")

    monkeypatch.setattr("mlflow_harness.validation.configure_mlflow", lambda: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "log_artifact", fake_log_artifact)
    monkeypatch.setattr(mlflow, "start_run", _mock_start_run(run))

    with pytest.raises(RuntimeError, match="S3 upload failed"):
        validate_s3_artifacts()

    assert len(created_files) == 1
    assert not os.path.exists(created_files[0])
