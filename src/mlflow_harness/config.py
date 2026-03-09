"""Configuration loading for MLflow harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import find_dotenv, load_dotenv

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "Default"


def _load_dotenv(dotenv_path: Optional[str] = None) -> Optional[str]:
    """Load .env file if present and return the path used."""
    if dotenv_path is None:
        dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    return dotenv_path


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    experiment_name: str
    query_token: Optional[str]


def get_settings(dotenv_path: Optional[str] = None) -> Settings:
    _load_dotenv(dotenv_path=dotenv_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    query_token = os.getenv("MLFLOW_QUERY_TOKEN")

    return Settings(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        query_token=query_token,
    )
