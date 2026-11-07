"""
Hyperparameter tuning of the ML candidates, validated inside the training split:
data/processed/mibici/features/ -> one MLflow parent run per model and target, with one nested
run per trial

- Search data: the ``SEARCH_MONTHS`` months before the test split (2024-10 to 2025-09). The test
  months are never read: they stay untouched for the final backtest.
- Folds (``tuning_folds``): each of the last ``VALIDATION_MONTHS`` months of that window
  validates one fold. A fold fits on a random ``FIT_SAMPLE`` of the window's earlier hours and
  stops early on the ``STOPPING_DAYS`` just before its month, so it never sees a later hour.
- Search: random search over the model's ``ModelSpec.search_space``. Trial 0 holds the defaults,
  fitted as ``train`` fits them (no early stopping), so they are scored on the same folds.
- Objective: the mean Poisson deviance over the folds. The overall and peak-hour WAPE and the
  daily WAPE of demand group 1 are logged too.
- Refit: the best trial's params on the whole training split, with the boosting iterations its
  early stopping kept (the mean over the folds).

The reasons for the search method, the window and the sample are in ``references/ml-canvas.md``
§8.

Usage (CLI):
    python -m team2_mlops2026.modeling.train tune
    python -m team2_mlops2026.modeling.train tune --model hgb_poisson --n-trials 5 --no-track

Usage (notebook):
    from team2_mlops2026.modeling.tuning import FINALISTS, tune_tracked, tuning_folds
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import product
from pathlib import Path
import time

import lightgbm
from lightgbm import LGBMRegressor
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterSampler

from team2_mlops2026.features import FEATURES_DIR, SEED, STATION, TARGETS, TIME, WEEKDAY
from team2_mlops2026.modeling.evaluate import TEST_START, segment_report
from team2_mlops2026.modeling.models import MODELS, ModelSpec
from team2_mlops2026.modeling.train import (
    INPUT_EXAMPLE_ROWS,
    check_names,
    load_features,
    load_profiles,
    log_inputs,
    log_model,
    model_inputs,
    model_params,
    report_metrics,
)

FINALISTS = ("lgbm_tweedie", "hgb_poisson")  # notebook 2.02
RUN_PREFIX = "tune"
DEFAULTS_RUN = "defaults"
SEARCH_MONTHS = 12
VALIDATION_MONTHS = 3
STOPPING_DAYS = 14
FIT_SAMPLE = 1 / 3  # of each fold's fit hours
PATIENCE = 20  # boosting iterations without improvement before early stopping
N_TRIALS = 20
VALIDATION_METRICS = (
    "overall_poisson_deviance",
    "overall_wape",
    "peak_wape",
    "group_1_daily_wape",
)
OBJECTIVE = "val_overall_poisson_deviance"
# The param that sets the number of boosting iterations, per regressor class
ITERATIONS = {HistGradientBoostingRegressor: "max_iter", LGBMRegressor: "n_estimators"}


@dataclass(frozen=True)
class TuningFold:
    """Fit on [``fit_start``, ``stop_start``), stop early on [``stop_start``, ``month``) and
    validate on ``month``."""

    month: pd.Period
    fit_start: pd.Timestamp
    stop_start: pd.Timestamp

    @property
    def val_start(self) -> pd.Timestamp:
        return self.month.start_time

    @property
    def val_end(self) -> pd.Timestamp:
        return (self.month + 1).start_time

    def split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        times = frame[TIME]
        return (
            frame[times.between(self.fit_start, self.stop_start, inclusive="left")],
            frame[times.between(self.stop_start, self.val_start, inclusive="left")],
            frame[times.between(self.val_start, self.val_end, inclusive="left")],
        )


def tuning_folds(
    test_start: str = TEST_START,
    search_months: int = SEARCH_MONTHS,
    validation_months: int = VALIDATION_MONTHS,
    stopping_days: int = STOPPING_DAYS,
) -> list[TuningFold]:
    """One fold per month of the last ``validation_months`` before ``test_start`` (YYYY-MM),
    each fitted on the hours of the ``search_months`` window before it."""
    test = pd.Period(test_start, freq="M")
    fit_start = (test - search_months).start_time
    folds = [
        TuningFold(month, fit_start, month.start_time - pd.Timedelta(days=stopping_days))
        for month in pd.period_range(test - validation_months, periods=validation_months)
    ]
    if not 0 < validation_months < search_months or folds[0].stop_start <= fit_start:
        raise ValueError(
            f"{validation_months} validation months and {stopping_days} stopping days leave no "
            f"fit hours in a {search_months}-month window"
        )
    return folds


@dataclass(frozen=True, eq=False)
class PreparedFold:
    """A fold's rows, already through the model's preprocessing."""

    month: str
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_stop: np.ndarray
    y_stop: np.ndarray
    X_val: np.ndarray
    val: pd.DataFrame  # station, hour, weekday and target of the validation rows
    fit: pd.DataFrame  # station and target of the fit rows, to rank the stations


def prepare_folds(
    frame: pd.DataFrame,
    folds: Sequence[TuningFold],
    preprocess: BaseEstimator,
    target: str,
    fit_sample: float = FIT_SAMPLE,
    seed: int = SEED,
) -> list[PreparedFold]:
    """Splits ``frame`` into the folds and fits a clone of ``preprocess`` on each fit sample.

    The preprocessing has no tuned params, so every trial of a model shares it.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for fold in folds:
        fit, stop, val = fold.split(frame)
        fit = fit[rng.random(len(fit)) < fit_sample]
        step = clone(preprocess)
        prepared.append(
            PreparedFold(
                month=str(fold.month),
                X_fit=step.fit_transform(model_inputs(fit), fit[target]),
                y_fit=fit[target].to_numpy(),
                X_stop=step.transform(model_inputs(stop)),
                y_stop=stop[target].to_numpy(),
                X_val=step.transform(model_inputs(val)),
                val=val[[STATION, TIME, WEEKDAY, target]],
                fit=fit[[STATION, target]],
            )
        )
    return prepared


def fit_regressor(
    regressor: BaseEstimator, fold: PreparedFold, early_stopping: bool
) -> int | None:
    """Fits ``regressor`` on the fold's fit rows, if ``early_stopping`` stopping on the Poisson
    deviance of its stop rows. Returns the boosting iterations the fitted model uses (None for a
    model without them)."""
    if isinstance(regressor, LGBMRegressor) and early_stopping:
        regressor.set_params(metric="poisson")
        regressor.fit(
            fold.X_fit,
            fold.y_fit,
            eval_X=(fold.X_stop,),
            eval_y=(fold.y_stop,),
            callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False)],
        )
        return regressor.best_iteration_
    if isinstance(regressor, HistGradientBoostingRegressor) and early_stopping:
        regressor.set_params(early_stopping=True, n_iter_no_change=PATIENCE, scoring="loss")
        regressor.fit(fold.X_fit, fold.y_fit, X_val=fold.X_stop, y_val=fold.y_stop)
        return regressor.n_iter_
    regressor.fit(fold.X_fit, fold.y_fit)
    if isinstance(regressor, LGBMRegressor):
        return regressor.n_estimators_
    return regressor.n_iter_ if isinstance(regressor, HistGradientBoostingRegressor) else None


@dataclass(frozen=True)
class Trial:
    number: int  # 0 holds the defaults
    params: dict  # the search-space params of the regressor
    iterations: int | None  # boosting iterations, mean over the folds (None without them)
    metrics: dict[str, float]  # mean over the folds: val_overall_poisson_deviance...
    fold_metrics: list[dict[str, float]]
    fit_seconds: float
    refit_params: dict = field(default_factory=dict)  # params plus the iterations

    @property
    def run_name(self) -> str:
        return DEFAULTS_RUN if self.number == 0 else f"trial-{self.number:02d}"


def _builtin(value):
    return value.item() if isinstance(value, np.generic) else value


def space_values(regressor: BaseEstimator, keys) -> dict:
    """The values of the search-space ``keys`` in ``regressor``, also when they belong to the
    regressor it wraps (``NonNegative``)."""
    params = regressor.get_params()
    return {k: _builtin(params[k] if k in params else params[f"regressor__{k}"]) for k in keys}


def search_space_table(spec: ModelSpec, target: str = TARGETS[0]) -> pd.DataFrame:
    """Per tuned param: its distribution, the range it is sampled from and the default."""
    defaults = space_values(spec.build(target)[-1], spec.search_space)
    return pd.DataFrame(
        {
            param: {
                "distribution": dist.dist.name,
                "low": _builtin(dist.support()[0]),
                "high": _builtin(dist.support()[1]),
                "default": defaults[param],
            }
            for param, dist in spec.search_space.items()
        }
    ).T.rename_axis("param")


def run_trial(
    spec: ModelSpec,
    target: str,
    folds: Sequence[PreparedFold],
    number: int,
    params: dict,
    early_stopping: bool = True,
) -> Trial:
    """Fits ``spec`` with ``params`` on every fold and scores it on the validation month."""
    start = time.perf_counter()
    fold_metrics, kept = [], []
    for fold in folds:
        regressor = spec.build(target, **params)[-1]
        kept.append(fit_regressor(regressor, fold, early_stopping))
        report = segment_report(fold.val, regressor.predict(fold.X_val), target, fold.fit)
        metrics = report_metrics(report)
        fold_metrics.append({f"val_{m}": metrics.get(m, np.nan) for m in VALIDATION_METRICS})
    iterations = None if None in kept else round(float(np.mean(kept)))
    param = ITERATIONS.get(type(regressor))
    return Trial(
        number=number,
        params=params,
        iterations=iterations,
        metrics={k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]},
        fold_metrics=fold_metrics,
        fit_seconds=time.perf_counter() - start,
        refit_params={**params, param: iterations} if param and iterations else dict(params),
    )


def search(
    spec: ModelSpec,
    target: str,
    folds: Sequence[PreparedFold],
    n_trials: int = N_TRIALS,
    seed: int = SEED,
    on_trial: Callable[[Trial], None] | None = None,
) -> list[Trial]:
    """The defaults (trial 0), then ``n_trials`` random samples of the search space with early
    stopping. ``on_trial`` is called with every finished trial."""
    defaults = space_values(spec.build(target)[-1], spec.search_space)
    samples = ParameterSampler(spec.search_space, n_iter=n_trials, random_state=seed)
    trials = []
    for number, params in enumerate([defaults, *samples]):
        params = {k: _builtin(v) for k, v in params.items()}
        trial = run_trial(spec, target, folds, number, params, early_stopping=number > 0)
        logger.info(
            "{} for {}, {}: Poisson deviance {:.4f}, peak WAPE {:.3f} ({} iterations, {:.0f} s)",
            spec.name,
            target,
            trial.run_name,
            trial.metrics[OBJECTIVE],
            trial.metrics["val_peak_wape"],
            trial.iterations,
            trial.fit_seconds,
        )
        if on_trial:
            on_trial(trial)
        trials.append(trial)
    return trials


@dataclass(frozen=True)
class TunedModel:
    name: str
    target: str
    trials: list[Trial]
    best: Trial
    estimator: BaseEstimator  # the best params, refitted on the whole training split
    fit_seconds: float
    run_id: str | None = None

    @property
    def defaults(self) -> Trial:
        return self.trials[0]


def tune_model(
    name: str,
    target: str,
    search_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    folds: Sequence[TuningFold],
    n_trials: int = N_TRIALS,
    seed: int = SEED,
    on_trial: Callable[[Trial], None] | None = None,
) -> TunedModel:
    """Searches the params of model ``name`` for ``target`` on the ``folds`` of
    ``search_rows``, then refits the best trial on ``train_rows``."""
    spec = MODELS[name]
    prepared = prepare_folds(search_rows, folds, spec.build(target)[0], target, seed=seed)
    trials = search(spec, target, prepared, n_trials, seed, on_trial)
    del prepared
    best = min(trials, key=lambda t: t.metrics[OBJECTIVE])
    start = time.perf_counter()
    estimator = spec.build(target, **best.refit_params)
    estimator.fit(model_inputs(train_rows), train_rows[target])
    fit_seconds = time.perf_counter() - start
    logger.success(
        "{} for {}: best {}, validation Poisson deviance {:.4f} against {:.4f} with the defaults "
        "(refitted in {:.0f} s)",
        name,
        target,
        best.run_name,
        best.metrics[OBJECTIVE],
        trials[0].metrics[OBJECTIVE],
        fit_seconds,
    )
    return TunedModel(name, target, trials, best, estimator, fit_seconds)


def _search_data(
    models: Sequence[str],
    targets: Sequence[str],
    src: str | Path,
    test_start: str,
    folds: Sequence[TuningFold] | None,
) -> tuple[list[TuningFold], pd.DataFrame, pd.DataFrame]:
    """The folds, the hours of the search window and the training split (no test hours)."""
    check_names(models, targets)
    untunable = [m for m in models if not MODELS[m].search_space]
    if untunable:
        raise ValueError(f"Models without a search space: {', '.join(untunable)}")
    folds = list(folds or tuning_folds(test_start))
    if folds[-1].val_end > pd.Timestamp(test_start):
        raise ValueError(f"The tuning folds validate on hours of the test split ({test_start})")
    train_rows = load_features(src, test_start, end=test_start)
    return folds, train_rows[train_rows[TIME] >= folds[0].fit_start], train_rows


def tune(
    models: Sequence[str] = FINALISTS,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
    n_trials: int = N_TRIALS,
    folds: Sequence[TuningFold] | None = None,
    seed: int = SEED,
) -> list[TunedModel]:
    """Tunes every model for every target on the same folds (default: ``tuning_folds``)."""
    folds, search_rows, train_rows = _search_data(models, targets, src, test_start, folds)
    return [
        tune_model(name, target, search_rows, train_rows, folds, n_trials, seed)
        for name, target in product(models, targets)
    ]


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if np.isfinite(v)}


def log_trial(trial: Trial, name: str, target: str) -> None:
    """Logs ``trial`` as a run nested in the active one."""
    import mlflow

    with mlflow.start_run(run_name=trial.run_name, nested=True):
        mlflow.set_tags(
            {"model_family": MODELS[name].family, "target": target, "trial": trial.number}
        )
        mlflow.log_params(trial.params)
        mlflow.log_metrics(
            _finite(
                {
                    **trial.metrics,
                    "fit_seconds": trial.fit_seconds,
                    **({"iterations": trial.iterations} if trial.iterations else {}),
                }
            )
        )
        for step, fold in enumerate(trial.fold_metrics):
            mlflow.log_metrics(_finite({f"fold_{k}": v for k, v in fold.items()}), step=step)


def tune_tracked(
    models: Sequence[str] = FINALISTS,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
    n_trials: int = N_TRIALS,
    folds: Sequence[TuningFold] | None = None,
    seed: int = SEED,
) -> list[TunedModel]:
    """``tune()`` with one MLflow parent run per model and target (``tune-<model>-<target>``)
    and one nested run per trial.

    Each trial run logs its search-space params, the boosting iterations, the mean validation
    metrics and the ones of every fold (``fold_...``, one step per fold), and its fit time.
    The parent run logs:
    - the feature and profile tables as inputs, each with a digest
    - the search settings, the refitted model's params and the best trial's number
    - the best trial's validation metrics, and the defaults' as ``default_val_...``
    - ``fit_seconds`` and ``model_size_mb`` of the refit
    - the tags ``model_family``, ``target`` and ``git_commit``
    - the refitted model, with a signature and an input example
    """
    import mlflow

    from team2_mlops2026 import tracking

    src = Path(src)
    _, profiles_fit_end = load_profiles(src)
    folds, search_rows, train_rows = _search_data(models, targets, src, test_start, folds)
    example = model_inputs(train_rows.tail(INPUT_EXAMPLE_ROWS))
    settings = {
        "test_start": test_start,
        "profiles_fit_end": profiles_fit_end,
        "train_rows": len(train_rows),
        "search_start": folds[0].fit_start.strftime("%Y-%m"),
        "validation_months": ",".join(str(f.month) for f in folds),
        "stopping_days": (folds[0].val_start - folds[0].stop_start).days,
        "fit_sample": round(FIT_SAMPLE, 3),
        "patience": PATIENCE,
        "n_trials": n_trials,
        "seed": seed,
        "search": "random",
    }
    results = []
    for name, target in product(models, targets):
        with tracking.start_run(f"{RUN_PREFIX}-{name}-{target}") as run:
            log_inputs(src)
            mlflow.set_tags(
                {
                    "model_family": MODELS[name].family,
                    "target": target,
                    "git_commit": tracking.git_commit(),
                }
            )
            on_trial = partial(log_trial, name=name, target=target)
            result = tune_model(
                name, target, search_rows, train_rows, folds, n_trials, seed, on_trial
            )
            mlflow.log_params(
                {
                    "model": name,
                    **settings,
                    "best_trial": result.best.number,
                    **model_params(result.estimator),
                    "target": target,
                }
            )
            mlflow.log_metrics(
                _finite(
                    {
                        **result.best.metrics,
                        **{f"default_{k}": v for k, v in result.defaults.metrics.items()},
                        "fit_seconds": result.fit_seconds,
                    }
                )
            )
            log_model(result.estimator, example)
        results.append(replace(result, run_id=run.info.run_id))
    return results
