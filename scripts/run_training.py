from mlflow_harness.training import run_training


def main() -> None:
    rmse, run_id = run_training()
    print(f"RMSE: {rmse}")
    print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()
