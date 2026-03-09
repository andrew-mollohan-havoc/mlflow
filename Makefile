.PHONY: activate all clean coverage format install lint lint-fix list-experiments \
	mlflow-docker-down mlflow-docker-logs mlflow-docker-login mlflow-docker-pull \
	mlflow-docker-up mlflow-docker-up-defaults mlflow-local \
	run-training setup-tracking test validate-s3-artifacts venv verify

VENV_DIR := venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
RUFF := $(VENV_BIN)/ruff
MLFLOW_PORT ?= 5000
MLFLOW_DOCKER_IMAGE ?= ghcr.io/mlflow/mlflow
MLFLOW_DOCKER_CONTAINER ?= mlflow-tracking
MLFLOW_DOCKER_MLRUNS ?= ./mlruns
MLFLOW_DOCKER_MLARTIFACTS ?= ./mlartifacts
GHCR_USER ?= USERNAME

install: venv
	@. $(VENV_DIR)/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt && \
		pip install -e .

venv:
	python3 -m venv $(VENV_DIR)

verify:
	$(MAKE) install
	$(MAKE) test
	$(MAKE) setup-tracking
	$(MAKE) list-experiments
	$(MAKE) run-training

all: verify

activate: venv
	@echo "Spawning a shell with the venv activated. Exit to return."
	@. $(VENV_DIR)/bin/activate && exec $${SHELL:-/bin/bash} -i

list-experiments: install
	$(PYTHON) scripts/list_experiments.py

clean:
	rm -rf $(VENV_DIR)
	rm -rf __pycache__

run-training: install
	$(PYTHON) scripts/run_training.py

validate-s3-artifacts: install
	$(PYTHON) scripts/validate_s3_artifacts.py

setup-tracking: install
	$(PYTHON) scripts/setup_tracking.py

lint: install
	$(RUFF) check src tests

lint-fix: install
	$(RUFF) check src tests --fix

format: install
	$(RUFF) format src tests

test: install
	$(PYTEST) -q

coverage: install
	$(PYTEST) --cov=src/mlflow_harness --cov-report=term-missing -q

mlflow-local: mlflow-docker-up

mlflow-docker-down:
	docker rm -f $(MLFLOW_DOCKER_CONTAINER)

mlflow-docker-logs:
	docker logs --tail 50 $(MLFLOW_DOCKER_CONTAINER)

mlflow-docker-login:
	@echo "Using CR_PAT env var for GitHub Container Registry access"
	echo $$CR_PAT | docker login ghcr.io -u $(GHCR_USER) --password-stdin

mlflow-docker-pull:
	docker pull $(MLFLOW_DOCKER_IMAGE)

mlflow-docker-up-defaults: mlflow-docker-pull
	docker run -d --name $(MLFLOW_DOCKER_CONTAINER) \
		-p $(MLFLOW_PORT):5000 \
		$(MLFLOW_DOCKER_IMAGE) \
		mlflow server --host 0.0.0.0 --port 5000

mlflow-docker-up: mlflow-docker-pull
	mkdir -p $(MLFLOW_DOCKER_MLRUNS) $(MLFLOW_DOCKER_MLARTIFACTS)
	docker run -d --name $(MLFLOW_DOCKER_CONTAINER) \
		-p $(MLFLOW_PORT):5000 \
		-v $(abspath $(MLFLOW_DOCKER_MLRUNS)):/mlflow/mlruns \
		-v $(abspath $(MLFLOW_DOCKER_MLARTIFACTS)):/mlflow/mlartifacts \
		$(MLFLOW_DOCKER_IMAGE) \
		mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri /mlflow/mlruns \
		--default-artifact-root /mlflow/mlartifacts
