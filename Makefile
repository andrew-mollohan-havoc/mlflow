.PHONY: venv activate install clean

VENV_DIR := venv

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
