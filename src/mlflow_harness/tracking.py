"""MLflow client setup helpers."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlencode

import mlflow

from .config import Settings, get_settings

_PATCH_ATTR = "_mlflow_harness_patched"


def apply_query_token_patch(query_token: Optional[str]) -> bool:
    """Patch MLflow REST calls to append a query token, if provided."""
    if not query_token:
        return False

    try:
        from mlflow.utils import rest_utils
    except Exception:
        return False

    if getattr(rest_utils.http_request, _PATCH_ATTR, False):
        return True

    original_http_request = rest_utils.http_request

    def _http_request_with_token(host_creds, endpoint, method, *args, **kwargs):
        sep = "&" if "?" in endpoint else "?"
        endpoint = f"{endpoint}{sep}{urlencode({'token': query_token})}"
        return original_http_request(host_creds, endpoint, method, *args, **kwargs)

    setattr(_http_request_with_token, _PATCH_ATTR, True)
    rest_utils.http_request = _http_request_with_token

    # MLflow's artifact repo (HttpArtifactRepository) imports http_request directly
    # at module load time, so the rest_utils patch above does not reach it.
    # Patch the bound reference in that module explicitly.
    try:
        import mlflow.store.artifact.http_artifact_repo as _http_artifact_repo

        _http_artifact_repo.http_request = _http_request_with_token
    except Exception:
        pass

    return True


def configure_mlflow(
    settings: Optional[Settings] = None,
    *,
    enable_token_patch: bool = True,
    logger_name: Optional[str] = "mlflow",
) -> Settings:
    """Configure MLflow tracking URI, experiment, and optional token patch."""
    if settings is None:
        settings = get_settings()

    if enable_token_patch:
        apply_query_token_patch(settings.query_token)

    if logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    return settings
