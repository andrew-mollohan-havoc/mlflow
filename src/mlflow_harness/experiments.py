"""Experiment listing helpers."""

from __future__ import annotations

import mlflow
from mlflow.entities import ViewType

from .tracking import configure_mlflow


def list_experiments(view_type: ViewType = ViewType.ALL):
    configure_mlflow()
    return mlflow.search_experiments(view_type=view_type)
