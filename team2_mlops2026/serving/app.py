"""
Online prediction service: the champions' forecasts over HTTP with FastAPI.

- ``POST /predict`` takes feature rows (``serving.schemas.FeatureRow``) and returns the expected
  departures, arrivals and net flow of each station-hour, plus the model versions that forecast
  them. It calls ``modeling.predict.forecast``, the function of the batch forecast, so both give
  the same numbers for the same rows.
- ``GET /health`` says whether the models are loaded (200) or not (503), for the container's
  health check. ``GET /model`` returns the model card: versions, URIs, metrics and digests.
- The models load once, at start-up, from ``MODEL_SOURCE``:
  - ``local`` (default, used by the Docker image): the folder that ``modeling.export`` writes,
    ``models/`` or ``MODEL_DIR``.
  - ``registry``: ``models:/mibici-<target>@<MODEL_ALIAS>`` (default ``champion``) from the
    MLflow server at ``MLFLOW_TRACKING_URI``.
  If they cannot load, the service still starts and answers 503, and the log says why.
- Errors: 422 with the field and the reason for invalid rows, 400 for an empty or oversized
  batch, 503 without models, and 500 with no traceback if the forecast itself fails.
- Swagger UI at ``/docs``; the OpenAPI schema and an example request and response are kept in
  ``docs/api/`` (``api-docs`` command).

Usage:
    make serve   # uvicorn team2_mlops2026.serving.app:app --port 8000
    MODEL_SOURCE=registry make serve
    python -m team2_mlops2026.serving.app api-docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator
import typer

from team2_mlops2026.config import MODELS_DIR, PROJ_ROOT
from team2_mlops2026.features import FEATURES_DIR, STATION, TARGETS, TIME
from team2_mlops2026.modeling import export, predict
from team2_mlops2026.modeling.registry import CHAMPION
from team2_mlops2026.serving.schemas import (
    MAX_ROWS,
    HealthResponse,
    ModelCard,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    to_frame,
    to_payload,
)

LOCAL, REGISTRY = "local", "registry"
API_DOCS_DIR = PROJ_ROOT / "docs" / "api"
OPENAPI_FILE = "openapi.json"
EXAMPLE_REQUEST, EXAMPLE_RESPONSE = "example_request.json", "example_response.json"
# The example rows: the morning peak of the night replayed in notebook 2.05
EXAMPLE_START = "2026-09-09 08:00"
EXAMPLE_STATIONS = 3

try:
    VERSION = version("team2_mlops2026")
except PackageNotFoundError:  # running from a source tree that was not installed
    VERSION = "0+unknown"


@dataclass(frozen=True)
class LoadedModels:
    models: dict[str, BaseEstimator]
    card: dict  # the model card, as served by GET /model
    source: str

    def info(self) -> dict[str, ModelInfo]:
        return {
            t: ModelInfo(**{k: e[k] for k in ModelInfo.model_fields})
            for t, e in self.card["models"].items()
        }


def load_models(
    source: str = LOCAL, models_dir: str | Path = MODELS_DIR, alias: str = CHAMPION
) -> LoadedModels:
    """Each target's model and the model card, from ``models_dir`` or from the Registry."""
    if source == LOCAL:
        models, card = export.load_exported(models_dir)
    elif source == REGISTRY:
        from team2_mlops2026.modeling.registry import load_model

        loaded = {target: load_model(target, alias) for target in TARGETS}
        models = {target: model for target, (model, _) in loaded.items()}
        card = {"models": {t: export.card_entry(v, alias) for t, (_, v) in loaded.items()}}
    else:
        raise ValueError(f"MODEL_SOURCE must be {LOCAL!r} or {REGISTRY!r}, not {source!r}")
    return LoadedModels(models, {**card, "source": source}, source)


def loaded_models(request: Request) -> LoadedModels:
    loaded = request.app.state.loaded
    if loaded is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The models are not loaded; see the service log (run `make export` first?)",
        )
    return loaded


Models = Annotated[LoadedModels, Depends(loaded_models)]
NOT_LOADED = {503: {"description": "The models are not loaded"}}
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Forecast the departures and arrivals of station-hours",
    responses={
        400: {"description": f"Empty batch or more than {MAX_ROWS:,} rows"},
        500: {"description": "The forecast failed (details in the service log)"},
        **NOT_LOADED,
    },
)
def predict_rows(body: PredictRequest, loaded: Models) -> dict:
    """Expected departures and arrivals of each row (clipped at 0) and the net flow
    (arrivals - departures), with the champion of each target, as the batch forecast."""
    rows = to_frame(body.rows)
    try:
        out = predict.forecast(loaded.models, rows)
    except Exception:  # noqa: BLE001  (any model error becomes a 500 without traceback)
        logger.exception("The forecast of {} rows failed", len(rows))
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "The forecast failed; see the service log"
        ) from None
    return {"predictions": out.to_dict("records"), "models": loaded.info()}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Whether the models are loaded",
    responses={503: {"model": HealthResponse, "description": "The models are not loaded"}},
)
def health(request: Request):
    loaded = request.app.state.loaded
    body = HealthResponse(
        status="ok" if loaded else "unavailable",
        models_loaded=loaded is not None,
        source=request.app.state.source,
    )
    code = status.HTTP_200_OK if loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(jsonable_encoder(body), status_code=code)


@router.get(
    "/model",
    response_model=ModelCard,
    summary="The model card of the served models",
    responses=NOT_LOADED,
)
def model_card(loaded: Models) -> dict:
    """Name, version, alias, URI, source run and Registry tags of each target's model."""
    return loaded.card


async def _batch_size_errors(request: Request, exc: RequestValidationError):
    """An empty or oversized batch is a bad request (400), not an invalid row (422)."""
    errors = exc.errors()
    if errors and all(
        e["type"] in ("too_short", "too_long") and tuple(e["loc"]) == ("body", "rows")
        for e in errors
    ):
        return JSONResponse(
            {"detail": f"Send between 1 and {MAX_ROWS:,} rows"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return await request_validation_exception_handler(request, exc)


def create_app(
    source: str | None = None, models_dir: str | Path | None = None, alias: str | None = None
) -> FastAPI:
    """The FastAPI app. Unset arguments come from ``MODEL_SOURCE``, ``MODEL_DIR`` and
    ``MODEL_ALIAS``."""
    source = source or os.getenv("MODEL_SOURCE", LOCAL)
    models_dir = Path(models_dir or os.getenv("MODEL_DIR", MODELS_DIR))
    alias = alias or os.getenv("MODEL_ALIAS", CHAMPION)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.source = source
        try:
            app.state.loaded = load_models(source, models_dir, alias)
            logger.success(
                "Serving {}",
                ", ".join(i.uri for i in app.state.loaded.info().values()),
            )
        except Exception:  # noqa: BLE001  (the service starts and answers 503)
            app.state.loaded = None
            logger.exception("The models could not be loaded from {}", source)
        yield

    app = FastAPI(
        title="MiBici station demand forecast",
        version=VERSION,
        summary="Hourly departures and arrivals of every MiBici station (Guadalajara)",
        description=(
            "Serves the registered champions `models:/mibici-departures@champion` and "
            "`models:/mibici-arrivals@champion`. Send the feature rows of the station-hours "
            "to forecast; the batch forecast (`modeling.predict`) gives the same numbers."
        ),
        lifespan=lifespan,
    )
    app.state.source, app.state.loaded = source, None
    app.add_exception_handler(RequestValidationError, _batch_size_errors)
    app.include_router(router)
    return app


app = create_app()


# --- docs/api ------------------------------------------------------------------


def example_rows(
    src: str | Path = FEATURES_DIR, start: str = EXAMPLE_START, stations: int = EXAMPLE_STATIONS
) -> pd.DataFrame:
    """The rows of the hour ``start`` at the ``stations`` busiest stations of the last week."""
    from team2_mlops2026.features import forecast_rows
    from team2_mlops2026.modeling.train import load_profiles

    start = pd.Timestamp(start)
    scores, _ = load_profiles(src)
    rows = forecast_rows(predict.load_history(src, start), start, hours=1)
    rows = predict.join_profiles(rows, scores)
    return rows.nlargest(stations, "departures_mean_7d").sort_values([STATION, TIME])


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


cli = typer.Typer()


@cli.command("api-docs")
def api_docs(
    dst: Annotated[Path, typer.Option(help="Destination folder")] = API_DOCS_DIR,
    src: Annotated[Path, typer.Option(help="Folder with the features and profiles")] = (
        FEATURES_DIR
    ),
    models_dir: Annotated[Path, typer.Option(help="Exported models")] = MODELS_DIR,
):
    """Write the OpenAPI schema, and an example request and its response from the exported
    models, to docs/api/."""
    from fastapi.testclient import TestClient

    dst.mkdir(parents=True, exist_ok=True)
    service = create_app(LOCAL, models_dir)
    _write_json(dst / OPENAPI_FILE, service.openapi())
    request = to_payload(example_rows(src))
    with TestClient(service) as client:
        response = client.post("/predict", json=request)
    response.raise_for_status()
    _write_json(dst / EXAMPLE_REQUEST, request)
    _write_json(dst / EXAMPLE_RESPONSE, response.json())
    logger.success(
        "Wrote {}, {} and {} to {}", OPENAPI_FILE, EXAMPLE_REQUEST, EXAMPLE_RESPONSE, dst
    )


@cli.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
):
    """Start the service with uvicorn."""
    import uvicorn

    uvicorn.run("team2_mlops2026.serving.app:app", host=host, port=port)


if __name__ == "__main__":
    cli()
