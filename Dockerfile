# syntax=docker/dockerfile:1
#
# Prediction service with the exported champions: FastAPI on port 8000, no MLflow server needed.
#
#   make export                                   # models:/mibici-<target>@champion -> models/
#   docker build -t ml-service:latest .
#   docker run -p 8000:8000 ml-service:latest     # http://127.0.0.1:8000/docs
#
# The base is the Python of the training environment (3.12.12), pinned by digest so a rebuild
# starts from the same image on amd64 and arm64.
ARG PYTHON_IMAGE=python:3.12.12-slim-trixie@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c

# --- build: the virtual environment with the pinned serving dependencies and the package ---
FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt
COPY pyproject.toml README.md LICENSE ./
COPY team2_mlops2026 team2_mlops2026
RUN pip install --no-deps .

# --- runtime: the environment, the models and the API examples, run as a non-root user ---
FROM ${PYTHON_IMAGE}
# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY models models
RUN test -f models/model_card.json \
    || { echo "models/ is empty: run 'make export' (or 'make docker-models') before building"; exit 1; }
COPY docs/api/example_request.json docs/api/example_response.json docs/api/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_SOURCE=local \
    MODEL_DIR=/app/models \
    API_DOCS_DIR=/app/docs/api \
    MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
CMD ["uvicorn", "team2_mlops2026.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Set by `make docker-build` / `make docker-push` (see the README); "unknown" otherwise
ARG VERSION=unknown
ARG GIT_COMMIT=unknown
ARG MODELS=unknown
LABEL org.opencontainers.image.title="mibici-ml-service" \
      org.opencontainers.image.description="Hourly departures and arrivals of every MiBici station (Guadalajara): FastAPI POST /predict with the registered champions" \
      org.opencontainers.image.source="https://github.com/CarlosHeinze/MLOps_Team2" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      team2_mlops2026.models="${MODELS}"
