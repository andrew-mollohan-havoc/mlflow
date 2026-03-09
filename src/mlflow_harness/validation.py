"""End-to-end S3 artifact storage validation."""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import mlflow

from .tracking import configure_mlflow

_VALIDATION_RUN_NAME = "s3-artifact-validation"
_VALIDATION_EXPERIMENT = "s3-artifact-validation"


def validate_s3_artifacts() -> Tuple[str, str]:
    """Create a test run, log a metric and an artifact, and return (run_id, artifact_uri).

    Validates the full path: MLflow client → server proxy → S3.
    Set MLFLOW_QUERY_TOKEN in your environment to authenticate.

    Raises:
        ValueError: if the artifact URI does not point to S3 after the run.
    """
    configure_mlflow()
    mlflow.set_experiment(_VALIDATION_EXPERIMENT)

    with mlflow.start_run(run_name=_VALIDATION_RUN_NAME) as run:
        mlflow.log_metric("validation", 1.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("s3-artifact-validation-ok")
            tmp = f.name

        try:
            mlflow.log_artifact(tmp, artifact_path="validation")
        finally:
            os.unlink(tmp)

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    if artifact_uri.startswith("/") or "mlruns" in artifact_uri:
        raise ValueError(
            f"Artifact URI points to local storage, not S3: {artifact_uri!r}\n"
            "Check that --serve-artifacts and --artifacts-destination are set on the server."
        )

    return run_id, artifact_uri
