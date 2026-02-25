import mlflow
from mlflow.entities import ViewType

from mlflow_harness.config import get_settings
from mlflow_harness.tracking import apply_query_token_patch, configure_mlflow


def main() -> None:
    settings = get_settings()
    # Preserve the token-based query patching behavior from the original script.
    apply_query_token_patch(settings.query_token)
    configure_mlflow(settings, enable_token_patch=False)

    experiments = mlflow.search_experiments(view_type=ViewType.ALL)
    for exp in experiments:
        print(
            f"Experiment ID: {exp.experiment_id}, Name: {exp.name}, "
            f"Lifecycle Stage: {exp.lifecycle_stage}"
        )


if __name__ == "__main__":
    main()
