#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = team2_mlops2026
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER ?= .venv/bin/python
MLFLOW_PORT ?= 5001

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff format --check
	$(PYTHON_INTERPRETER) -m ruff check

## Format source code with ruff
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff check --fix
	$(PYTHON_INTERPRETER) -m ruff format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	python$(PYTHON_VERSION) -m venv .venv
	@echo "Activate with: source .venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Start the local MLflow server (copy .env.example to .env first)
.PHONY: mlflow
mlflow:
	$(PYTHON_INTERPRETER) -m mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT)


## Build raw, interim, and cleaned data (downloads the full MiBici history)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.data.raw_data_extraction
	$(PYTHON_INTERPRETER) -m team2_mlops2026.data.raw_data_normalization
	$(PYTHON_INTERPRETER) -m team2_mlops2026.data.interim_data_cleaning


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python$(PYTHON_VERSION) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
