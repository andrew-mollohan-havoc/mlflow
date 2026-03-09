from mlflow_harness.validation import validate_s3_artifacts


def main() -> None:
    run_id, artifact_uri = validate_s3_artifacts()
    print(f"Run ID:       {run_id}")
    print(f"Artifact URI: {artifact_uri}")


if __name__ == "__main__":
    main()
