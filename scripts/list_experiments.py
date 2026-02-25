from mlflow_harness.experiments import list_experiments


def main() -> None:
    experiments = list_experiments()
    for exp in experiments:
        print(
            "Experiment ID: "
            f"{exp.experiment_id}, Name: {exp.name}, "
            f"Lifecycle Stage: {exp.lifecycle_stage}"
        )


if __name__ == "__main__":
    main()
