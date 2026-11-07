"""
MLflow Model Registry: the champion and the challenger of each target.

- One registered model per target, ``mibici-departures`` and ``mibici-arrivals``. The batch
  forecast (``modeling.predict``) loads ``models:/mibici-<target>@champion``.
- Each version is the model a run logged, so the Registry links it to that run's params,
  metrics, input datasets (with their digests) and artifacts. A finalist's version is the
  model of its latest ``tune-<model>-<target>`` run: the tuned params, refitted on the training
  split. A baseline's version is the model of its latest ``train-<model>-<target>`` run.
- The ranking is the one of notebook 2.04 (``backtest.rank_backtests``): the success criterion on
  the mean of the monthly backtest folds, then the peak-hour WAPE. ``champion`` goes to the first
  candidate that passed. ``challenger`` goes to the runner-up if it passed too, otherwise to the
  best baseline, so the comparison stays in the Registry.
- Every version gets a description and tags: algorithm, target, training window, test-split
  metrics, backtest summary, input digests and git commit. The test-split metrics of a finalist
  come from its ``ablation-<model>-<target>-pca`` run, which scored that same model.
- A run already registered keeps its version: registering again refreshes its tags and moves
  the aliases.

Usage (CLI):
    python -m team2_mlops2026.modeling.registry
    python -m team2_mlops2026.modeling.registry --target departures --champion hgb_poisson

Usage (notebook):
    from team2_mlops2026.modeling.registry import load_model, register_champions
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import typer

from team2_mlops2026.features import RUN_NAME as FEATURES_RUN
from team2_mlops2026.features import TARGETS
from team2_mlops2026.modeling import backtest as bt
from team2_mlops2026.modeling.models import BASELINES, MODELS
from team2_mlops2026.modeling.train import MODEL_ARTIFACT
from team2_mlops2026.modeling.train import RUN_PREFIX as TRAIN_PREFIX
from team2_mlops2026.modeling.tuning import RUN_PREFIX as TUNE_PREFIX

MODEL_NAME = "mibici-{}"
CHAMPION, CHALLENGER = "champion", "challenger"
TEST_METRICS = ("overall_wape", "overall_mae", "overall_poisson_deviance", "peak_wape", "peak_mae")
BACKTEST_METRICS = ("peak_wape", "overall_wape", "overall_poisson_deviance", "group_1_daily_wape")
DIGESTS = {
    "mibici_station_hour_features": "features_digest",
    "mibici_station_profiles": "profiles_digest",
}

app = typer.Typer()


def model_name(target: str) -> str:
    return MODEL_NAME.format(target)


def model_uri(target: str, alias: str = CHAMPION) -> str:
    return f"models:/{model_name(target)}@{alias}"


def _client():
    import mlflow

    from team2_mlops2026 import tracking

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    return mlflow.MlflowClient()


def load_model(target: str, alias: str = CHAMPION) -> tuple[BaseEstimator, object]:
    """The model of ``target`` that holds ``alias``, and its ``ModelVersion``."""
    import mlflow.sklearn

    version = _client().get_model_version_by_alias(model_name(target), alias)
    return mlflow.sklearn.load_model(model_uri(target, alias)), version


def pick(ranking: pd.DataFrame, champion: str | None = None) -> tuple[str, str]:
    """(champion, challenger) from a ``rank_candidates`` table.

    The champion is ``champion`` if given, otherwise the first candidate that passed the
    success criterion. The challenger is the next candidate that passed, otherwise the best
    baseline.
    """
    passed = list(ranking.index[ranking["passed"].astype(bool)])
    if champion is None:
        if not passed:
            raise ValueError(
                "No candidate passed the success criterion; name one with --champion to "
                "register it anyway"
            )
        champion = passed[0]
    elif champion not in ranking.index:
        raise ValueError(f"{champion!r} is not a ranked candidate: {', '.join(ranking.index)}")
    others = [name for name in passed if name != champion]
    return champion, others[0] if others else ranking.loc[champion, "baseline"]


def source_runs(name: str, target: str) -> tuple[object, object]:
    """The run whose logged model is registered, and the run with its test-split metrics.

    For a baseline both are its ``train`` run. For a finalist they are its ``tune`` run and the
    ``ablation ... pca`` run, which must have scored that same model.
    """
    from team2_mlops2026 import tracking

    if name in BASELINES:
        run = tracking.latest_run(f"{TRAIN_PREFIX}-{name}-{target}")
        return run, run
    model_run = tracking.latest_run(f"{TUNE_PREFIX}-{name}-{target}")
    test_run = tracking.latest_run(f"{bt.ABLATION_PREFIX}-{name}-{target}-pca")
    if test_run.data.tags.get("params_from") != model_run.info.run_id:
        raise ValueError(
            f"The latest ablation run of {name} for {target} scored another model than the "
            f"latest tune run {model_run.info.run_id}: run the ablation again"
        )
    return model_run, test_run


def algorithm(name: str, target: str) -> str:
    """The regressor's class, such as ``LGBMRegressor``."""
    estimator = MODELS[name].build(target)
    return type(estimator[-1] if isinstance(estimator, Pipeline) else estimator).__name__


def _training_window(model_run) -> tuple[str, str]:
    from team2_mlops2026 import tracking

    try:
        start = tracking.latest_run(FEATURES_RUN).data.params.get("start", "table start")
    except LookupError:
        start = "table start"
    return start, model_run.data.params["test_start"]


@dataclass(frozen=True)
class Entry:
    """A model to register for a target, with what its description and tags report."""

    name: str
    target: str
    alias: str
    model_run: object
    test_run: object
    backtest_run: object
    ranking: pd.Series | None  # its row of the ranking; None for a baseline

    def tags(self) -> dict[str, str]:
        from team2_mlops2026 import tracking

        start, end = _training_window(self.model_run)
        test = self.test_run.data.metrics
        backtest = self.backtest_run.data.metrics
        n_folds = self.backtest_run.data.params.get("n_folds", "?")
        tags = {
            "model": self.name,
            "model_family": MODELS[self.name].family,
            "algorithm": algorithm(self.name, self.target),
            "target": self.target,
            "training_window": f"{start} to {end} (exclusive)",
            "train_rows": self.model_run.data.params.get("train_rows", ""),
            "source_run": self.model_run.info.run_name,
            "test_metrics_run": self.test_run.info.run_id,
            "backtest_run": self.backtest_run.info.run_id,
            "backtest_folds": n_folds,
            "git_commit": self.model_run.data.tags.get("git_commit", ""),
            **{f"test_{m}": f"{test[m]:.4f}" for m in TEST_METRICS if m in test},
            **{
                f"backtest_{m}": (
                    f"{backtest[f'{self.target}_{m}']:.4f} "
                    f"± {backtest.get(f'{self.target}_{m}_std', float('nan')):.4f}"
                )
                for m in BACKTEST_METRICS
                if f"{self.target}_{m}" in backtest
            },
            **{
                tag: digest
                for name, digest in tracking.input_digests(self.model_run.info.run_id).items()
                if (tag := DIGESTS.get(name))
            },
        }
        if self.ranking is not None:
            tags |= {
                "success_criterion": "passed" if self.ranking["passed"] else "failed",
                "best_baseline": str(self.ranking["baseline"]),
                "peak_wape_improvement": f"{self.ranking['peak_wape_improvement']:.4f}",
            }
        return tags

    def description(self, tags: dict[str, str]) -> str:
        role = {
            CHAMPION: "Champion: the forecast that modeling.predict serves.",
            CHALLENGER: "Challenger: kept for comparison with the champion.",
        }[self.alias]
        test = {m: tags.get(f"test_{m}", "?") for m in TEST_METRICS}
        summary = (
            f"{self.name} ({tags['algorithm']}) forecast of the hourly {self.target} of every "
            f"MiBici station. {role}"
        )
        trained = f"- Trained on the station-hours from {tags['training_window']}, run "
        tested = (
            f"- Test split: WAPE {test['overall_wape']} overall and {test['peak_wape']} at peak "
            f"hours, MAE {test['overall_mae']}, Poisson deviance "
            f"{test['overall_poisson_deviance']}."
        )
        backtest = (
            f"- Backtest over {tags['backtest_folds']} monthly folds: peak WAPE "
            f"{tags.get('backtest_peak_wape', '?')}."
        )
        lines = [summary, "", f"{trained}{tags['source_run']}.", tested, backtest]
        if self.ranking is not None:
            lines.append(
                f"- Success criterion {tags['success_criterion']}: peak WAPE "
                f"{float(tags['peak_wape_improvement']):.1%} below {tags['best_baseline']}."
            )
        return "\n".join(lines)


def register(entry: Entry):
    """Registers the model of ``entry.model_run`` (or reuses its version), refreshes its
    description and tags and points ``entry.alias`` at it. Returns the ``ModelVersion``."""
    import mlflow
    from mlflow.exceptions import MlflowException

    client = _client()
    name = model_name(entry.target)
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(
            name,
            tags={"target": entry.target},
            description=(
                f"Hourly {entry.target} of every MiBici station (Guadalajara), one global model "
                f"for all stations. Aliases: {CHAMPION} (served by modeling.predict) and "
                f"{CHALLENGER}."
            ),
        )
    run_id = entry.model_run.info.run_id
    existing = client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    version = (
        existing[0]
        if existing
        else mlflow.register_model(f"runs:/{run_id}/{MODEL_ARTIFACT}", name)
    )
    tags = entry.tags()
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)
    client.update_model_version(name, version.version, description=entry.description(tags))
    client.set_registered_model_alias(name, entry.alias, version.version)
    logger.success("{} version {} ({}) is the {}", name, version.version, entry.name, entry.alias)
    return client.get_model_version(name, version.version)


def register_champions(
    targets: Sequence[str] = TARGETS,
    champion: str | None = None,
    models: Sequence[str] = bt.BACKTEST_MODELS,
) -> dict[str, dict[str, object]]:
    """Registers the champion and the challenger of each target from the latest backtest runs
    of ``models``. Returns {target: {alias: ModelVersion}}."""
    loaded = bt.load_backtests(models)
    rankings = bt.rank_backtests({name: table for name, (_, table) in loaded.items()}, targets)
    registered = {}
    for target in targets:
        ranking = rankings[target]
        chosen = dict(zip((CHAMPION, CHALLENGER), pick(ranking, champion)))
        registered[target] = {}
        for alias, name in chosen.items():
            model_run, test_run = source_runs(name, target)
            entry = Entry(
                name,
                target,
                alias,
                model_run,
                test_run,
                loaded[name][0],
                ranking.loc[name] if name in ranking.index else None,
            )
            registered[target][alias] = register(entry)
    return registered


@app.command()
def main(
    target: Annotated[
        list[str] | None, typer.Option(help="Target, repeatable (default: both)")
    ] = None,
    champion: Annotated[
        str | None,
        typer.Option(help="Candidate to promote instead of the backtest's first (every target)"),
    ] = None,
):
    """Register the champion and the challenger of each target in the MLflow Model Registry."""
    register_champions(target or TARGETS, champion)


if __name__ == "__main__":
    app()
