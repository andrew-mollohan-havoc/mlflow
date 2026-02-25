# MLflow Local Test Harness

Minimal setup for running MLflow client calls against a configured tracking server.

Repo structure:

- `src/mlflow_harness/` — reusable MLflow helpers and training example
- `scripts/` — runnable entrypoints
- `tests/` — basic unit tests

## Prerequisites

- Python 3

## Setup

```bash
make venv
make install
make activate
```

All commands in this repo use the project virtual environment. If you run scripts directly, use the venv Python:

```bash
venv/bin/python scripts/setup_tracking.py
```

## Docker Dependency

Running MLflow locally uses the official Docker image. Make sure Docker Desktop (or equivalent) is installed and running before starting the tracking server.

## Run MLflow Locally (Docker)

```bash
# Pull the MLflow image (optional; up will also pull)
make mlflow-docker-pull

# Start the tracking server on http://127.0.0.1:5000
make mlflow-docker-up

# View logs
make mlflow-docker-logs

# Stop and remove the container
make mlflow-docker-down
```

## Environment Variables

Environment variables are loaded automatically from `.env` if present.
Copy `.env.example` to `.env` and set values as needed:

```bash
MLFLOW_TRACKING_URI="https://your-tracking-server"
MLFLOW_EXPERIMENT_NAME="your-experiment"
MLFLOW_QUERY_TOKEN="optional-token"
```

## Git Hygiene

The repository ignores local-only files like virtual environments, caches, MLflow local data (`mlruns/`, `mlartifacts/`, `mlflow.db`), and `.env` secrets. Keep `.env.example` tracked as the template.

## Example run commands

```bash
# Connect to the tracking server and set the experiment
make setup-tracking

# List all experiments (uses token patching if configured)
make list-experiments

# Train a model and log to MLflow
make run-training
```

## Run Everything (venv + install + tests + runs)

```bash
make verify
```

You can also run the scripts directly:

```bash
python scripts/setup_tracking.py
python scripts/list_experiments.py
python scripts/run_training.py
```

## Tests

```bash
make test
```

## Local MLflow Data

Local runs and artifacts are stored in `mlruns/` and `mlartifacts/` when you use the Docker-backed tracking server. These directories are intentionally gitignored because they are environment-specific and can be large and frequently changing.

## Cleanup

```bash
make clean
```

## Docs

MLflow quickstart: <https://mlflow.org/docs/latest/ml/getting-started/running-notebooks/>
