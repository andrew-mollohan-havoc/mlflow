# MLflow Harness

A test harness for validating [HavocAI](https://havocai.com) MLflow tracking instances. It verifies that the tracking server is reachable, experiments can be created and queried, model training runs log correctly, and artifacts are stored in S3 (not local storage).

## What it tests

| Script | What it validates |
| --- | --- |
| `make setup-tracking` | Server is reachable; experiment is created or found |
| `make list-experiments` | Experiments can be listed (with token auth if configured) |
| `make run-training` | Full training run: params, metrics, and model artifact logged |
| `make validate-s3-artifacts` | Artifacts land in S3, not local storage |

## Prerequisites

- Python 3.9+
- Docker (for running a local MLflow server)

## Quick start

```bash
# 1. Create venv and install dependencies
make install

# 2. Copy env template and fill in your server details
cp .env.example .env

# 3. Start a local MLflow server (or point .env at an existing one)
make mlflow-docker-up

# 4. Run the full harness
make verify
```

## Environment variables

Copy `.env.example` to `.env` and set:

```bash
MLFLOW_TRACKING_URI="https://your-tracking-server"   # default: http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME="your-experiment"             # default: Default
MLFLOW_QUERY_TOKEN="optional-auth-token"             # required for HavocAI instances
```

### Authentication: query token workaround

> **This is a temporary workaround. See the long-term plan below.**

The production MLflow instance sits behind a reverse proxy that enforces OIDC authentication. Because MLflow's Python client has no built-in support for OIDC flows, we currently bypass the check by appending a shared token as a query parameter to every request:

```text
https://mlflow.example.com/api/2.0/mlflow/experiments/list?token=<MLFLOW_QUERY_TOKEN>
```

The harness implements this via a monkey-patch in [src/mlflow_harness/tracking.py](src/mlflow_harness/tracking.py) (`apply_query_token_patch`). It wraps MLflow's internal `http_request` function at runtime so the token is injected transparently without changing any MLflow source.

**Why this is a problem:** a shared bypass token is not scoped to a user or service, cannot be rotated per caller, and leaks full MLflow access to anyone who holds it. It is purely a stopgap to unblock development.

**Long-term solution:** provision a [Zitadel](https://zitadel.com) service account for each automated workload (CI, training jobs, this harness). Configure MLflow's reverse proxy to accept Zitadel-issued JWT bearer tokens, and have each client obtain a short-lived token via the Zitadel client-credentials flow before making MLflow API calls. This removes the shared secret entirely and gives per-service auditability and revocation.

## Running the harness

```bash
# Connect to the server and set the active experiment
make setup-tracking

# List all experiments
make list-experiments

# Train a RandomForest on the diabetes dataset and log to MLflow
make run-training

# Verify artifacts are stored in S3 (requires a server with --serve-artifacts)
make validate-s3-artifacts

# Run all of the above in sequence
make verify
```

## Development

### Tests

Tests run without a live server by mocking MLflow calls. The three training tests write to a local temp directory — no server needed.

```bash
# Run tests
make test

# Run tests with line-by-line coverage report
make coverage
```

### Linting and formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
# Check for lint errors (no changes made)
make lint

# Auto-fix lint errors where possible
make lint-fix

# Format source files in place
make format
```

`make lint` and `make format` target `src/` and `tests/`. Rules are configured in [pyproject.toml](pyproject.toml) under `[tool.ruff]`.

## Local MLflow server (Docker)

```bash
# Start the server on http://127.0.0.1:5000 with persistent local storage
make mlflow-docker-up

# View recent logs
make mlflow-docker-logs

# Stop and remove the container
make mlflow-docker-down
```

To override the port or image:

```bash
make mlflow-docker-up MLFLOW_PORT=5001
```

## Project layout

```text
src/mlflow_harness/
  config.py        # Settings dataclass; loads .env via python-dotenv
  tracking.py      # configure_mlflow() + query-token monkey-patch
  experiments.py   # list_experiments() wrapper
  training.py      # RandomForest training run example
  validation.py    # S3 artifact storage validation

scripts/           # CLI entry points (called by Makefile targets)
tests/             # Unit tests (pytest)
```

## Cleanup

```bash
# Remove the venv
make clean
```

## Reference

- [MLflow docs](https://mlflow.org/docs/latest/)
