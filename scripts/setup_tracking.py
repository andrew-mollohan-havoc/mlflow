from mlflow_harness.tracking import configure_mlflow


def main() -> None:
    settings = configure_mlflow()
    print(
        "Connected to MLflow tracking server at "
        f"{settings.tracking_uri} and set experiment to "
        f"'{settings.experiment_name}'"
    )


if __name__ == "__main__":
    main()
