"""
Training and tracking of the demand models:
data/processed/mibici/features/ -> one MLflow run per model and target

- Each model is fitted on the hours before ``TEST_START`` and scored on the later hours with
  ``evaluate.segment_report``. There is one model per target (departures, arrivals).
- The models are the baselines and the ML candidates of ``models.MODELS``.
- The inputs are the station-hour features plus the station's PCA profile scores of the week
  (``station_profiles``). A station-week without scores gets NaN. A test split that starts before
  the fit end the profiles table records is rejected, since the components would carry
  test-period demand.
- Every run logs the feature and profile tables as input datasets, with the digests of the steps
  that wrote them, plus the model's params, the segment metrics, the fit time, the model size
  and the fitted model.
- ``tune`` searches the hyperparameters of the finalists inside the training split
  (``modeling.tuning``).

Usage (CLI):
    python -m team2_mlops2026.modeling.train fit
    python -m team2_mlops2026.modeling.train fit --model lgbm_poisson --target departures --no-track
    python -m team2_mlops2026.modeling.train tune --model lgbm_tweedie --n-trials 20

Usage (notebook):
    from team2_mlops2026.modeling.train import load_training_data, train_tracked
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import time
from typing import Annotated
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import typer

from team2_mlops2026.features import (
    FEATURES_DIR,
    FEATURES_FILE,
    STATION,
    TARGETS,
    TIME,
    time_split,
)
from team2_mlops2026.modeling.evaluate import TEST_START, segment_report
from team2_mlops2026.modeling.models import MODELS, TRUSTED_TYPES
from team2_mlops2026.station_profiles import FIT_END_KEY, PROFILES_FILE, WEEK, week_start

RUN_PREFIX = "train"
MODEL_ARTIFACT = "model"
INPUT_EXAMPLE_ROWS = 5
# 1.05: each later component adds 3% of the variance or less, and only PC1-PC5 are stable
# from a station's first four weeks
PROFILE_COMPONENTS = 5

app = typer.Typer()


@dataclass(frozen=True)
class TrainedModel:
    name: str
    target: str
    estimator: BaseEstimator
    report: pd.DataFrame  # segment_report on the test split
    fit_seconds: float
    run_id: str | None = None


def load_profiles(src: str | Path = FEATURES_DIR) -> tuple[pd.DataFrame, str]:
    """The first ``PROFILE_COMPONENTS`` weekly profile scores of each station, and the
    ``fit_end`` that their table records."""
    scores = pd.read_parquet(Path(src) / PROFILES_FILE)
    if FIT_END_KEY not in scores.attrs:
        raise ValueError(
            f"{PROFILES_FILE} does not record its fit end: rebuild it with "
            "python -m team2_mlops2026.station_profiles"
        )
    fit_end = scores.attrs[FIT_END_KEY]
    components = [f"pc{i}" for i in range(1, PROFILE_COMPONENTS + 1) if f"pc{i}" in scores]
    scores = scores[[STATION, WEEK, *components]].astype(dict.fromkeys(components, "float32"))
    return scores, fit_end


def load_features(
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Station-hour features with the station's profile scores of the week, from the hours in
    [``start``, ``end``) only, if given.

    A ``test_start`` before the profiles' fit end is rejected, since the components would carry
    test-period demand.
    """
    src = Path(src)
    scores, fit_end = load_profiles(src)
    if pd.Timestamp(test_start) < pd.Timestamp(fit_end):
        raise ValueError(
            f"The test split starts at {test_start}, before the profiles' fit end "
            f"{fit_end}: refit the profiles with --fit-end {test_start} or earlier"
        )
    window = [(TIME, ">=", pd.Timestamp(start))] if start else []
    window += [(TIME, "<", pd.Timestamp(end))] if end else []
    features = pd.read_parquet(src / FEATURES_FILE, filters=window or None)
    features[WEEK] = week_start(features[TIME]).astype(scores[WEEK].dtype)
    return features.merge(scores, on=[STATION, WEEK], how="left").drop(columns=WEEK)


def load_training_data(
    src: str | Path = FEATURES_DIR, test_start: str = TEST_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``load_features``, split at ``test_start``."""
    return time_split(load_features(src, test_start), test_start)


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns a model may read: both targets are dropped, since the same hour's arrivals
    are not known when its departures are forecast."""
    return frame.drop(columns=list(TARGETS))


def model_params(estimator: BaseEstimator) -> dict:
    """The params of the model itself: for a ``Pipeline``, those of its last step, since every
    candidate shares the same preprocessing."""
    return (estimator[-1] if isinstance(estimator, Pipeline) else estimator).get_params()


def report_metrics(report: pd.DataFrame) -> dict[str, float]:
    """``segment_report`` as flat metrics (``overall_wape``, ``peak_rows``...), without the
    undefined ones."""
    return {
        f"{segment}_{'rows' if metric == 'n' else metric}": float(value)
        for (segment, metric), value in report.stack().items()
        if np.isfinite(value)
    }


def check_names(models: Sequence[str], targets: Sequence[str]) -> None:
    unknown = [m for m in models if m not in MODELS] + [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(
            f"Unknown models or targets: {', '.join(unknown)}. "
            f"Models: {', '.join(MODELS)}; targets: {', '.join(TARGETS)}"
        )


def train_model(name: str, target: str, train: pd.DataFrame, test: pd.DataFrame) -> TrainedModel:
    """Fits the model ``name`` for ``target`` on ``train`` and scores it on ``test``."""
    start = time.perf_counter()
    estimator = MODELS[name].build(target).fit(model_inputs(train), train[target])
    fit_seconds = time.perf_counter() - start
    report = segment_report(test, estimator.predict(model_inputs(test)), target, train)
    logger.success(
        "{} for {}: WAPE {:.3f} overall, {:.3f} at peak hours (fitted in {:.0f} s)",
        name,
        target,
        report.loc["overall", "wape"],
        report.loc["peak", "wape"],
        fit_seconds,
    )
    return TrainedModel(name, target, estimator, report, fit_seconds)


def train(
    models: Sequence[str] = tuple(MODELS),
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
) -> list[TrainedModel]:
    """Trains every model for every target on the same split."""
    check_names(models, targets)
    train_rows, test_rows = load_training_data(src, test_start)
    return [
        train_model(name, target, train_rows, test_rows)
        for name, target in product(models, targets)
    ]


def log_inputs(src: Path) -> None:
    """Logs the feature and profile tables as inputs of the active MLflow run."""
    from team2_mlops2026 import tracking

    tracking.log_files_dataset(
        [src / FEATURES_FILE], src, "mibici_station_hour_features", "features"
    )
    tracking.log_files_dataset([src / PROFILES_FILE], src, "mibici_station_profiles", "profiles")


def log_model(estimator: BaseEstimator, example: pd.DataFrame) -> None:
    """Logs ``estimator`` as the active run's model, with a signature and an input example, and
    the size of its skops file as ``model_size_mb``."""
    import mlflow
    from mlflow.models import infer_signature
    import mlflow.sklearn
    import skops.io

    mlflow.log_metric("model_size_mb", len(skops.io.dumps(estimator)) / 1e6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # hint that integer columns cannot hold NaN
        signature = infer_signature(example, estimator.predict(example))
    mlflow.sklearn.log_model(
        estimator,
        name=MODEL_ARTIFACT,
        signature=signature,
        input_example=example,
        skops_trusted_types=TRUSTED_TYPES,
    )


def train_tracked(
    models: Sequence[str] = tuple(MODELS),
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
) -> list[TrainedModel]:
    """``train()`` with one MLflow run per model and target in the project's experiment.

    Each run logs:
    - the feature and profile tables as inputs, each with a digest
    - the model's params and the profiles' fit end
    - the segment metrics, plus ``fit_seconds`` and ``model_size_mb``
    - the tags ``model_family``, ``target`` and ``git_commit``
    - the fitted model, with a signature and an input example
    """
    import mlflow

    from team2_mlops2026 import tracking

    check_names(models, targets)
    src = Path(src)
    _, profiles_fit_end = load_profiles(src)
    train_rows, test_rows = load_training_data(src, test_start)
    example = model_inputs(test_rows.head(INPUT_EXAMPLE_ROWS))
    results = []
    for name, target in product(models, targets):
        with tracking.start_run(f"{RUN_PREFIX}-{name}-{target}") as run:
            log_inputs(src)
            result = train_model(name, target, train_rows, test_rows)
            mlflow.set_tags(
                {
                    "model_family": MODELS[name].family,
                    "target": target,
                    "git_commit": tracking.git_commit(),
                }
            )
            mlflow.log_params(
                {
                    "model": name,
                    "test_start": test_start,
                    "profiles_fit_end": profiles_fit_end,
                    "train_rows": len(train_rows),
                    **model_params(result.estimator),
                    "target": target,
                }
            )
            mlflow.log_metrics(
                {**report_metrics(result.report), "fit_seconds": result.fit_seconds}
            )
            log_model(result.estimator, example)
        results.append(replace(result, run_id=run.info.run_id))
    return results


Targets = Annotated[list[str] | None, typer.Option(help="Target, repeatable (default: both)")]
Source = Annotated[Path, typer.Option(help="Folder with the station-hour features and profiles")]
TestStart = Annotated[str, typer.Option(help="YYYY-MM: first month of the test split")]
Track = Annotated[bool, typer.Option(help="Log the runs to the MLflow server")]


@app.command()
def fit(
    model: Annotated[
        list[str] | None, typer.Option(help="Model to train, repeatable (default: all models)")
    ] = None,
    target: Targets = None,
    src: Source = FEATURES_DIR,
    test_start: TestStart = TEST_START,
    track: Track = True,
):
    """Train and score the models with their default params."""
    (train_tracked if track else train)(model or tuple(MODELS), target or TARGETS, src, test_start)


@app.command()
def tune(
    model: Annotated[
        list[str] | None, typer.Option(help="Model to tune, repeatable (default: the finalists)")
    ] = None,
    target: Targets = None,
    src: Source = FEATURES_DIR,
    test_start: TestStart = TEST_START,
    n_trials: Annotated[
        int | None, typer.Option(help="Random trials besides the defaults (default: 20)")
    ] = None,
    track: Track = True,
):
    """Search the hyperparameters inside the training split and refit the best ones."""
    from team2_mlops2026.modeling import tuning  # tuning builds on this module

    (tuning.tune_tracked if track else tuning.tune)(
        model or tuning.FINALISTS, target or TARGETS, src, test_start, n_trials or tuning.N_TRIALS
    )


if __name__ == "__main__":
    app()
