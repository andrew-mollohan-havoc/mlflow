# MLflow Local Test Harness

Minimal setup for running MLflow client calls against the configured tracking server.

## Prerequisites

- Python 3

## Setup

```bash
make venv
make install
make activate
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

## Cleanup

```bash
make clean
```

## Docs

MLflow quickstart: <https://mlflow.org/docs/latest/ml/getting-started/running-notebooks/>
