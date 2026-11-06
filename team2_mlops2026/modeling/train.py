"""
Training and tracking of the demand models:
data/processed/mibici/features/ -> one MLflow run per model and target

- Each model is fitted on the hours before ``TEST_START`` and scored on the later hours with
  ``evaluate.segment_report``. There is one model per target (departures, arrivals).
- The inputs are the station-hour features plus each station's PCA profile scores
  (``station_profiles``). A station without a profile gets NaN.
- Every run logs the feature and profile tables as input datasets, with the digests of the steps
  that wrote them, plus the model's params, the segment metrics and the fitted model.

Usage (CLI):
    python -m team2_mlops2026.modeling.train
    python -m team2_mlops2026.modeling.train --model seasonal_naive --target departures --no-track

Usage (notebook):
    from team2_mlops2026.modeling.train import load_training_data, train_tracked
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Annotated
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import typer

from team2_mlops2026.features import FEATURES_DIR, FEATURES_FILE, STATION, TARGETS, time_split
from team2_mlops2026.modeling.evaluate import TEST_START, segment_report
from team2_mlops2026.modeling.models import BASELINES, FAMILY, TRUSTED_TYPES
from team2_mlops2026.station_profiles import PROFILES_FILE

RUN_PREFIX = "train"
MODEL_ARTIFACT = "model"
INPUT_EXAMPLE_ROWS = 5

app = typer.Typer()


@dataclass(frozen=True)
class TrainedModel:
    name: str
    target: str
    estimator: BaseEstimator
    report: pd.DataFrame  # segment_report on the test split
    run_id: str | None = None


def load_training_data(
    src: str | Path = FEATURES_DIR, test_start: str = TEST_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Station-hour features with each station's profile scores, split at ``test_start``.

    The profiles are fitted on the hours before ``station_profiles.DEFAULT_FIT_END``, which is
    the default ``test_start``, so the scores carry no test information.
    """
    src = Path(src)
    scores = pd.read_parquet(src / PROFILES_FILE)
    scores = scores.astype({c: "float32" for c in scores.columns if c != STATION})
    features = pd.read_parquet(src / FEATURES_FILE).merge(scores, on=STATION, how="left")
    return time_split(features, test_start)


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns a model may read: both targets are dropped, since the same hour's arrivals
    are not known when its departures are forecast."""
    return frame.drop(columns=list(TARGETS))


def report_metrics(report: pd.DataFrame) -> dict[str, float]:
    """``segment_report`` as flat metrics (``overall_wape``, ``peak_rows``...), without the
    undefined ones."""
    return {
        f"{segment}_{'rows' if metric == 'n' else metric}": float(value)
        for (segment, metric), value in report.stack().items()
        if np.isfinite(value)
    }


def _check_names(models: Sequence[str], targets: Sequence[str]) -> None:
    unknown = [m for m in models if m not in BASELINES] + [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(
            f"Unknown models or targets: {', '.join(unknown)}. "
            f"Models: {', '.join(BASELINES)}; targets: {', '.join(TARGETS)}"
        )


def train_model(name: str, target: str, train: pd.DataFrame, test: pd.DataFrame) -> TrainedModel:
    """Fits the model ``name`` for ``target`` on ``train`` and scores it on ``test``."""
    estimator = BASELINES[name](target).fit(model_inputs(train), train[target])
    report = segment_report(test, estimator.predict(model_inputs(test)), target, train)
    logger.success(
        "{} for {}: WAPE {:.3f} overall, {:.3f} at peak hours",
        name,
        target,
        report.loc["overall", "wape"],
        report.loc["peak", "wape"],
    )
    return TrainedModel(name, target, estimator, report)


def train(
    models: Sequence[str] = tuple(BASELINES),
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
) -> list[TrainedModel]:
    """Trains every model for every target on the same split."""
    _check_names(models, targets)
    train_rows, test_rows = load_training_data(src, test_start)
    return [
        train_model(name, target, train_rows, test_rows)
        for name, target in product(models, targets)
    ]


def train_tracked(
    models: Sequence[str] = tuple(BASELINES),
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
) -> list[TrainedModel]:
    """``train()`` with one MLflow run per model and target in the project's experiment.

    Each run logs the feature and profile tables as inputs (each with a digest), the model's
    params, the segment metrics, the tags ``model_family``, ``target`` and ``git_commit``, and
    the fitted model with a signature and an input example.
    """
    import mlflow
    from mlflow.models import infer_signature
    import mlflow.sklearn

    from team2_mlops2026 import tracking

    _check_names(models, targets)
    src = Path(src)
    train_rows, test_rows = load_training_data(src, test_start)
    example = model_inputs(test_rows.head(INPUT_EXAMPLE_ROWS))
    results = []
    for name, target in product(models, targets):
        with tracking.start_run(f"{RUN_PREFIX}-{name}-{target}") as run:
            tracking.log_files_dataset(
                [src / FEATURES_FILE], src, "mibici_station_hour_features", "features"
            )
            tracking.log_files_dataset(
                [src / PROFILES_FILE], src, "mibici_station_profiles", "profiles"
            )
            result = train_model(name, target, train_rows, test_rows)
            mlflow.set_tags(
                {"model_family": FAMILY, "target": target, "git_commit": tracking.git_commit()}
            )
            mlflow.log_params(
                {
                    "model": name,
                    "test_start": test_start,
                    "train_rows": len(train_rows),
                    **result.estimator.get_params(),
                    "target": target,
                }
            )
            mlflow.log_metrics(report_metrics(result.report))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # hint that integer columns cannot hold NaN
                signature = infer_signature(example, result.estimator.predict(example))
            mlflow.sklearn.log_model(
                result.estimator,
                name=MODEL_ARTIFACT,
                signature=signature,
                input_example=example,
                skops_trusted_types=TRUSTED_TYPES,
            )
        results.append(replace(result, run_id=run.info.run_id))
    return results


@app.command()
def main(
    model: Annotated[
        list[str] | None, typer.Option(help="Model to train, repeatable (default: all baselines)")
    ] = None,
    target: Annotated[
        list[str] | None, typer.Option(help="Target, repeatable (default: both)")
    ] = None,
    src: Annotated[
        Path, typer.Option(help="Folder with the station-hour features and profiles")
    ] = FEATURES_DIR,
    test_start: Annotated[
        str, typer.Option(help="YYYY-MM: first month of the test split")
    ] = TEST_START,
    track: Annotated[bool, typer.Option(help="Log the runs to the MLflow server")] = True,
):
    (train_tracked if track else train)(
        model or tuple(BASELINES), target or TARGETS, src, test_start
    )


if __name__ == "__main__":
    app()
