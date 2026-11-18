#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = team2_mlops2026
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER ?= .venv/bin/python
MLFLOW_PORT ?= 5001
SERVE_HOST ?= 127.0.0.1
SERVE_PORT ?= 8000

# Docker image of the prediction service. The version is __version__ of the package, the
# commit the one checked out, and the model URIs those of models/model_card.json.
DOCKER_IMAGE ?= ml-service
DOCKERHUB_REPO ?= racj90120hplmnv09/mibici-ml-service
DOCKER_PLATFORMS ?= linux/amd64,linux/arm64
DOCKER_CONTAINER ?= mibici-ml-service
SMOKE_IMAGE ?= $(DOCKER_IMAGE):latest
VERSION = $(shell $(PYTHON_INTERPRETER) -c "import re; print(re.search(r'__version__ = .(.+).', open('team2_mlops2026/__init__.py').read()).group(1))")
GIT_COMMIT = $(shell git rev-parse HEAD)
MODEL_URIS = $(shell $(PYTHON_INTERPRETER) -c "import json; c = json.load(open('models/model_card.json')); print(','.join(m['uri'] for m in c['models'].values()))")
DOCKER_LABELS = --build-arg VERSION=$(VERSION) --build-arg GIT_COMMIT=$(GIT_COMMIT) --build-arg MODELS=$(MODEL_URIS)

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
	$(PYTHON_INTERPRETER) -m pytest -q


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


## Build the station-hour features and the PCA station profiles from data/processed
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.features
	$(PYTHON_INTERPRETER) -m team2_mlops2026.station_profiles


## Train and score the baselines and candidates with their default params
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.train fit


## Tune the finalists' hyperparameters inside the training split
.PHONY: tune
tune:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.train tune


## Rolling-origin backtest and PCA ablation of the baselines and tuned finalists
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.train backtest
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.train ablation


## Register the champion and challenger of each target in the MLflow Model Registry
.PHONY: register
register:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.registry


## Forecast the next 24 h of every active station with the registered champions
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.predict


## Export the registered champions to models/ (with model_card.json) for the prediction service
.PHONY: export
export:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.modeling.export


## Serve the exported champions with FastAPI on http://127.0.0.1:8000 (Swagger UI at /docs)
.PHONY: serve
serve:
	$(PYTHON_INTERPRETER) -m uvicorn team2_mlops2026.serving.app:app --host $(SERVE_HOST) --port $(SERVE_PORT)


## Write the OpenAPI schema and an example request and response to docs/api/
.PHONY: api-docs
api-docs:
	$(PYTHON_INTERPRETER) -m team2_mlops2026.serving.app api-docs


## Build the image of the service with the models in models/: ml-service:latest and :<version>
.PHONY: docker-build
docker-build:
	docker build $(DOCKER_LABELS) -t $(DOCKER_IMAGE):latest -t $(DOCKER_IMAGE):$(VERSION) .


## Run the image on http://127.0.0.1:8000 (Ctrl+C stops it)
.PHONY: docker-run
docker-run:
	docker run --rm --name $(DOCKER_CONTAINER) -p $(SERVE_PORT):8000 $(DOCKER_IMAGE):latest


## Start an image in the background and check it against docs/api/ (SMOKE_IMAGE=repo:tag)
.PHONY: docker-smoke
docker-smoke:
	docker run -d --rm --name $(DOCKER_CONTAINER)-smoke $(SMOKE_IMAGE)
	docker exec $(DOCKER_CONTAINER)-smoke python -m team2_mlops2026.serving.app check; \
	status=$$?; docker stop $(DOCKER_CONTAINER)-smoke > /dev/null; exit $$status


## Build for amd64 and arm64 and push :<version> and :latest to Docker Hub (commit first)
.PHONY: docker-push
docker-push:
	@git diff --quiet HEAD -- || { echo "Commit first: the image is labelled with the commit"; exit 1; }
	@! docker buildx imagetools inspect $(DOCKERHUB_REPO):$(VERSION) > /dev/null 2>&1 \
		|| { echo "$(DOCKERHUB_REPO):$(VERSION) is already published: bump __version__"; exit 1; }
	docker buildx build --platform $(DOCKER_PLATFORMS) $(DOCKER_LABELS) \
		-t $(DOCKERHUB_REPO):$(VERSION) -t $(DOCKERHUB_REPO):latest --push .


## Copy the models of the published image into models/, to rebuild it without the MLflow server
.PHONY: docker-models
docker-models:
	docker create --name $(DOCKER_CONTAINER)-models $(DOCKERHUB_REPO):$(VERSION) > /dev/null
	docker cp $(DOCKER_CONTAINER)-models:/app/models/. models/
	docker rm $(DOCKER_CONTAINER)-models > /dev/null


## Every step after the data, in order: features, train, tune, evaluate, register, predict
.PHONY: pipeline
pipeline:
	$(MAKE) features
	$(MAKE) train
	$(MAKE) tune
	$(MAKE) evaluate
	$(MAKE) register
	$(MAKE) predict


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
