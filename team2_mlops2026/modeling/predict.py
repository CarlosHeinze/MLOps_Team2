"""
Batch forecast of the next day with the registered champions:
data/processed/mibici/features/ + models:/mibici-<target>@champion -> data/processed/mibici/predictions/

- The forecast for day D+1 is made on the night of day D (ML Canvas). By default it covers the
  24 hours after the last hour of the feature table, for every station with a trip in the last
  28 days.
- The rows come from ``features.forecast_rows``, which reads only the history before
  ``start``. A past ``start`` therefore replays the forecast made then, which can be scored
  against what happened (notebook 2.05).
- The station's profile scores of the week are joined as in training. A week after the last
  scored one takes the scores of that last week, since the weekly scores are only computed when
  the profiles table is rebuilt.
- Each target's model is loaded from the Model Registry by alias (``modeling.registry``).
- Output: one Parquet file per forecast, with the expected departures and arrivals of every
  station-hour and their net flow (arrivals - departures). A negative net flow empties the
  station and a positive one fills it.
- ``predict_tracked`` logs a ``predict-<start>`` run with the model versions, the input datasets
  with their digests, and the forecast file as a dataset.

Usage (CLI):
    python -m team2_mlops2026.modeling.predict
    python -m team2_mlops2026.modeling.predict --start "2026-09-15 00:00" --no-track
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import typer

from team2_mlops2026.data.trip_data import PROCESSED_MIBICI_DIR
from team2_mlops2026.features import (
    FEATURES_DIR,
    FEATURES_FILE,
    FORECAST_HOURS,
    STATION,
    TARGETS,
    TIME,
    WARMUP_HOURS,
    forecast_rows,
)
from team2_mlops2026.modeling.registry import CHAMPION, load_model
from team2_mlops2026.modeling.train import load_profiles
from team2_mlops2026.station_profiles import WEEK, week_start

PREDICTIONS_DIR = PROCESSED_MIBICI_DIR / "predictions"
RUN_PREFIX = "predict"
NET_FLOW = "net_flow"
DATASET = "mibici_predictions"

app = typer.Typer()


@dataclass(frozen=True)
class Forecast:
    start: pd.Timestamp
    predictions: pd.DataFrame  # station_id, hour_start, departures, arrivals, net_flow
    versions: dict[str, object]  # target -> the ModelVersion that forecast it
    path: Path | None = None
    run_id: str | None = None


def last_hour(src: str | Path = FEATURES_DIR) -> pd.Timestamp:
    """The last hour of the feature table."""
    return pd.read_parquet(Path(src) / FEATURES_FILE, columns=[TIME])[TIME].max()


def load_history(src: str | Path, start: pd.Timestamp) -> pd.DataFrame:
    """The rows of the feature table in the ``WARMUP_HOURS`` before ``start``."""
    window = [
        (TIME, ">=", start - pd.Timedelta(hours=WARMUP_HOURS)),
        (TIME, "<", start),
    ]
    return pd.read_parquet(Path(src) / FEATURES_FILE, filters=window)


def join_profiles(rows: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """The station's profile scores of the week, as in ``train.load_features``. A week after
    the last scored one takes the scores of that last week."""
    week = week_start(rows[TIME]).clip(upper=scores[WEEK].max()).astype(scores[WEEK].dtype)
    joined = rows.assign(**{WEEK: week}).merge(scores, on=[STATION, WEEK], how="left")
    return joined.drop(columns=WEEK)


def forecast(models: Mapping[str, BaseEstimator], rows: pd.DataFrame) -> pd.DataFrame:
    """Each target's forecast for ``rows`` (clipped at 0), and the net flow."""
    out = rows[[STATION, TIME]].reset_index(drop=True)
    for target in TARGETS:
        out[target] = np.clip(models[target].predict(rows), 0, None)
    out[NET_FLOW] = out["arrivals"] - out["departures"]
    return out


def output_path(dst: str | Path, start: pd.Timestamp) -> Path:
    return Path(dst) / f"forecast_{start:%Y%m%d_%H}h.parquet"


def predict(
    start: str | pd.Timestamp | None = None,
    hours: int = FORECAST_HOURS,
    src: str | Path = FEATURES_DIR,
    dst: str | Path | None = PREDICTIONS_DIR,
    alias: str = CHAMPION,
) -> Forecast:
    """Forecasts the ``hours`` hours from ``start`` (default: the hour after the last data) with
    the models that hold ``alias``, and writes them to ``dst`` unless it is None."""
    start = last_hour(src) + pd.Timedelta(hours=1) if start is None else pd.Timestamp(start)
    scores, _ = load_profiles(src)
    rows = join_profiles(forecast_rows(load_history(src, start), start, hours), scores)
    loaded = {target: load_model(target, alias) for target in TARGETS}
    predictions = forecast({t: model for t, (model, _) in loaded.items()}, rows)
    versions = {t: version for t, (_, version) in loaded.items()}

    path = None
    if dst is not None:
        path = output_path(dst, start)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.part")
        try:
            predictions.to_parquet(tmp, index=False)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    logger.success(
        "Forecast of {:,} station-hours ({} stations) from {} with {}",
        len(predictions),
        predictions[STATION].nunique(),
        start,
        ", ".join(f"{v.name} v{v.version}" for v in versions.values()),
    )
    return Forecast(start, predictions, versions, path)


def forecast_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": len(predictions),
        "stations": predictions[STATION].nunique(),
        **{f"total_{t}": float(predictions[t].sum()) for t in TARGETS},
        "max_station_net_flow": float(predictions.groupby(STATION)[NET_FLOW].sum().abs().max()),
    }


def predict_tracked(
    start: str | pd.Timestamp | None = None,
    hours: int = FORECAST_HOURS,
    src: str | Path = FEATURES_DIR,
    dst: str | Path = PREDICTIONS_DIR,
    alias: str = CHAMPION,
) -> Forecast:
    """``predict()`` inside a ``predict-<start>`` MLflow run.

    The run logs the feature and profile tables as inputs and the forecast file as a dataset
    (each with a digest), the start, horizon and alias as params, the name, version and source
    run of each model as params, and the forecast's size and totals as metrics.
    """
    import mlflow

    from team2_mlops2026 import tracking
    from team2_mlops2026.modeling.train import log_inputs

    src, dst = Path(src), Path(dst)
    start = last_hour(src) + pd.Timedelta(hours=1) if start is None else pd.Timestamp(start)
    with tracking.start_run(f"{RUN_PREFIX}-{start:%Y-%m-%d-%H}h") as run:
        log_inputs(src)
        result = predict(start, hours, src, dst, alias)
        mlflow.set_tag("git_commit", tracking.git_commit())
        mlflow.log_params(
            {
                "start": str(result.start),
                "hours": hours,
                "alias": alias,
                **{f"{t}_model": f"{v.name}/{v.version}" for t, v in result.versions.items()},
                **{f"{t}_model_run_id": v.run_id for t, v in result.versions.items()},
            }
        )
        mlflow.log_metrics(forecast_metrics(result.predictions))
        tracking.log_files_dataset([result.path], dst, DATASET, "predictions")
    return replace(result, run_id=run.info.run_id)


@app.command()
def main(
    start: Annotated[
        str | None,
        typer.Option(help="First hour to forecast (default: the hour after the last data)"),
    ] = None,
    hours: Annotated[int, typer.Option(help="Hours to forecast")] = FORECAST_HOURS,
    src: Annotated[
        Path, typer.Option(help="Folder with the station-hour features and profiles")
    ] = FEATURES_DIR,
    dst: Annotated[Path, typer.Option(help="Destination folder")] = PREDICTIONS_DIR,
    alias: Annotated[str, typer.Option(help="Registry alias of the models")] = CHAMPION,
    track: Annotated[bool, typer.Option(help="Log the run to the MLflow server")] = True,
):
    """Forecast the next hours of every active station with the registered models."""
    (predict_tracked if track else predict)(start, hours, src, dst, alias)


if __name__ == "__main__":
    app()
