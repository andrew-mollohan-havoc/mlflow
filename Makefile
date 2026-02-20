.PHONY: venv activate install clean \
	mlflow-docker-login mlflow-docker-pull mlflow-docker-up mlflow-docker-down mlflow-docker-logs

VENV_DIR := venv
MLFLOW_PORT ?= 5000
MLFLOW_DOCKER_IMAGE ?= ghcr.io/mlflow/mlflow
MLFLOW_DOCKER_CONTAINER ?= mlflow-tracking
MLFLOW_DOCKER_MLRUNS ?= ./mlruns
MLFLOW_DOCKER_MLARTIFACTS ?= ./mlartifacts
GHCR_USER ?= USERNAME

venv:
	python3 -m venv $(VENV_DIR)

activate:
	@echo "Run: source $(VENV_DIR)/bin/activate"

install: venv
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

clean:
	rm -rf $(VENV_DIR)
	rm -rf __pycache__

mlflow-docker-login:
	@echo "Using CR_PAT env var for GitHub Container Registry access"
	echo $$CR_PAT | docker login ghcr.io -u $(GHCR_USER) --password-stdin

mlflow-docker-pull:
	docker pull $(MLFLOW_DOCKER_IMAGE)

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

mlflow-docker-down:
	docker rm -f $(MLFLOW_DOCKER_CONTAINER)

mlflow-docker-logs:
	docker logs --tail 50 $(MLFLOW_DOCKER_CONTAINER)
