# MLflow Local Test Harness

Minimal setup for running MLflow client calls against a configured tracking server.

## Prerequisites

- Python 3

## Setup

```bash
make venv
make install
make activate
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

If you are using an API token for query access, export it before running scripts:

```bash
export MLFLOW_QUERY_TOKEN="..."
```

You can also copy `.env.example` to `.env` and load it with your preferred tool.

## Example run commands

```bash
# Connect to the tracking server and set the Default experiment
python main.py

# Patch MLflow REST calls to append the token query param (if set)
python jank.py
```

## Local MLflow Data

Local runs and artifacts are stored in `mlruns/` and `mlartifacts/` when you use the Docker-backed tracking server. These directories are intentionally gitignored because they are environment-specific and can be large and frequently changing.

## Cleanup

```bash
make clean
```

## Docs

MLflow quickstart: <https://mlflow.org/docs/latest/ml/getting-started/running-notebooks/>
