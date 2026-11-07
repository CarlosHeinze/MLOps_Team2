"""
Rolling-origin backtest of the finalists against the baselines, and the PCA ablation:
data/processed/mibici/features/ -> one MLflow parent run per model (``backtest-<model>``) with one
nested run per fold

The ML Canvas (``references/ml-canvas.md`` §4, Impact simulation) fixes the protocol:

- Folds: ``evaluate.rolling_origin_splits``, 12 monthly folds from 2025-10 to 2026-09. Each fold
  refits the model on every hour before its month and forecasts that month, as the monthly
  retraining of the canvas would.
- Models: the two baselines and the tuned finalists. A finalist takes the params of the model
  its latest ``tune-<model>-<target>`` run logged; that model is cloned, so every fold refits it
  from scratch. ``tuned=False`` uses the catalog defaults instead.
- Each fold scores both targets with ``segment_report``, plus a segment for the stations opened
  from ``NEW_SINCE`` on (about 100 stations opened in May-August 2026, a drift check), and the
  decision metric (``imbalance_scores``), which needs the forecasts of both targets.
- The parent run logs the mean and standard deviation of every fold metric, the key metrics of
  each fold as steps (MLflow's chart view then draws one line per model across the folds), the
  fold table and the error by hour as CSV, the out-of-sample forecasts as Parquet, and figures:
  error per month and per hour, residuals, forecasts against actual values for sample stations,
  and the permutation importance of the last fold's model.
- The success criterion and the champion of each target are decided on the mean report over the
  folds (``evaluate.rank_candidates``), in notebook 2.04 and by the registry step.

PCA ablation (the Phase 1 review asked whether PCA helps the model): each finalist with and
without the station profile scores (``pc1``-``pc5`` and their missing flag), trained and scored
on the split of ``train``. With ``tuned=True`` the variant with the scores is the model the tune
run already refitted on that split, so only the variant without them is fitted.

Usage (CLI):
    python -m team2_mlops2026.modeling.train backtest
    python -m team2_mlops2026.modeling.train backtest --model seasonal_naive --n-folds 2 --no-track
    python -m team2_mlops2026.modeling.train ablation

Usage (notebook):
    from team2_mlops2026.modeling.backtest import load_backtests, rank_backtests
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from pathlib import Path
import tempfile
import time

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_poisson_deviance
from sklearn.pipeline import Pipeline

from team2_mlops2026.features import FEATURES_DIR, SEED, STATION, TARGETS, TIME
from team2_mlops2026.modeling.evaluate import (
    MIN_PREDICTION,
    N_FOLDS,
    TEST_START,
    Fold,
    aggregate_folds,
    grouped_metrics,
    imbalance_scores,
    rank_candidates,
    rolling_origin_splits,
    segment_report,
)
from team2_mlops2026.modeling.models import BASELINES, MODELS
from team2_mlops2026.modeling.train import (
    MODEL_ARTIFACT,
    check_names,
    load_features,
    load_profiles,
    load_training_data,
    log_inputs,
    model_inputs,
    model_params,
    report_metrics,
)
from team2_mlops2026.modeling.tuning import FINALISTS
from team2_mlops2026.modeling.tuning import RUN_PREFIX as TUNE_PREFIX

RUN_PREFIX = "backtest"
ABLATION_PREFIX = "ablation"
DEFAULTS = "defaults"  # where the params of a model that was not tuned come from
BACKTEST_MODELS = (*BASELINES, *FINALISTS)
NEW_SINCE = "2026-05"  # canvas §4: about 100 stations opened in May-August 2026
NEW_STATIONS = "new_stations"
# Logged per fold on the parent run as steps, for MLflow's chart view
STEP_METRICS = ("overall_wape", "peak_wape", "overall_poisson_deviance", "group_1_daily_wape")
IMBALANCE_METRICS = ("imbalance_precision", "imbalance_recall")
IMPORTANCE_ROWS = 50_000
IMPORTANCE_REPEATS = 5
PLOT_DAYS = 7
ARTIFACT_DIR = "backtest"
FOLDS_FILE = "folds.csv"
IMBALANCE_FILE = "imbalance.csv"
BY_HOUR_FILE = "error_by_hour.csv"
FORECASTS_FILE = "forecasts.parquet"
FORECAST = "{}_forecast"  # column of a target's forecast in the forecasts table
FOLD = "fold"


# --- models ----------------------------------------------------------------------


def load_tuned(name: str, target: str) -> tuple[BaseEstimator, object]:
    """The fitted model of the latest ``tune-<name>-<target>`` run, and that run."""
    import mlflow.sklearn

    from team2_mlops2026 import tracking

    run = tracking.latest_run(f"{TUNE_PREFIX}-{name}-{target}")
    model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/{MODEL_ARTIFACT}")
    return model, run


def model_template(name: str, target: str, tuned: bool = True) -> tuple[BaseEstimator, str]:
    """The unfitted estimator of model ``name`` for ``target``, and where its params come from:
    the id of the latest ``tune-<name>-<target>`` run for a tuned finalist, else "defaults"."""
    spec = MODELS[name]
    if not (tuned and spec.search_space):
        return spec.build(target), DEFAULTS
    model, run = load_tuned(name, target)
    return clone(model), run.info.run_id


def new_stations(features: pd.DataFrame, since: str) -> pd.Index:
    """The stations whose first hour in ``features`` is in or after the month ``since``."""
    opened = features.groupby(STATION)[TIME].min()
    return opened.index[opened >= pd.Timestamp(since)]


def _poisson_deviance(y_true, y_pred) -> float:
    return mean_poisson_deviance(y_true, np.maximum(y_pred, MIN_PREDICTION))


POISSON_SCORER = make_scorer(_poisson_deviance, greater_is_better=False)


def permutation_importances(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y,
    n_rows: int = IMPORTANCE_ROWS,
    n_repeats: int = IMPORTANCE_REPEATS,
    seed: int = SEED,
) -> pd.DataFrame:
    """How much the Poisson deviance of ``estimator`` rises when each column of ``X`` is
    shuffled, on a random sample of ``n_rows``: the mean and std over ``n_repeats`` shuffles,
    largest first. The columns are the raw inputs, so ``station_id`` covers its target
    encoding and ``pc1``... the station profile."""
    rows = np.random.default_rng(seed).choice(len(X), min(n_rows, len(X)), replace=False)
    result = permutation_importance(
        estimator,
        X.iloc[rows],
        np.asarray(y)[rows],
        scoring=POISSON_SCORER,
        n_repeats=n_repeats,
        random_state=seed,
    )
    return pd.DataFrame(
        {"mean": result.importances_mean, "std": result.importances_std},
        index=pd.Index(X.columns, name="feature"),
    ).sort_values("mean", ascending=False)


# --- backtest --------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    month: str
    reports: dict[str, pd.DataFrame]  # segment_report per target
    imbalance: dict[str, float]  # imbalance_scores, when both targets are forecast
    fit_seconds: dict[str, float]  # per target
    train_rows: int

    def metrics(self) -> dict[str, float]:
        """Flat metrics: ``<target>_<segment>_<metric>``, the imbalance scores and
        ``<target>_fit_seconds``, without the undefined ones."""
        return {
            **{
                f"{target}_{key}": value
                for target, report in self.reports.items()
                for key, value in report_metrics(report).items()
            },
            **{k: float(v) for k, v in self.imbalance.items() if np.isfinite(v)},
            **{f"{target}_fit_seconds": s for target, s in self.fit_seconds.items()},
        }


@dataclass(frozen=True, eq=False)
class BacktestResult:
    name: str
    sources: dict[str, str]  # per target: the tune run its params come from, or "defaults"
    folds: list[FoldResult]
    forecasts: pd.DataFrame  # out-of-sample forecasts of every fold, with the actual values
    importance: dict[str, pd.DataFrame]  # per target, of the last fold's model (ML models)
    run_id: str | None = None

    @property
    def targets(self) -> list[str]:
        return list(self.sources)

    def summary(self, target: str) -> pd.DataFrame:
        """``aggregate_folds`` of the ``target`` reports: (metric, mean|std) per segment."""
        return aggregate_folds({f.month: f.reports[target] for f in self.folds})

    def mean_report(self, target: str) -> pd.DataFrame:
        """The mean over the folds of each segment metric, shaped as a ``segment_report``."""
        return self.summary(target).xs("mean", axis=1, level=1)

    def fold_table(self) -> pd.DataFrame:
        """One row per fold, target and segment, with the segment metrics."""
        return pd.concat(
            {(f.month, t): r for f in self.folds for t, r in f.reports.items()},
            names=[FOLD, "target"],
        ).reset_index()

    def imbalance_table(self) -> pd.DataFrame:
        """One row per fold with the decision metric (empty without both targets)."""
        rows = {f.month: f.imbalance for f in self.folds if f.imbalance}
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis(FOLD)

    def error_by(self, key: str) -> pd.DataFrame:
        """``count_metrics`` of the out-of-sample forecasts per ``key`` ("hour" of the day or
        ``FOLD``), one row per target and key."""
        keys = (
            self.forecasts[TIME].dt.hour.rename("hour") if key == "hour" else self.forecasts[key]
        )
        return pd.concat(
            {
                target: grouped_metrics(
                    self.forecasts,
                    self.forecasts[target],
                    self.forecasts[FORECAST.format(target)],
                    keys,
                )
                for target in self.targets
            },
            names=["target"],
        )


def _forecast_rows(test: pd.DataFrame, month: str, predictions: Mapping[str, np.ndarray]):
    frame = test[[STATION, TIME, *predictions]].reset_index(drop=True)
    frame.insert(0, FOLD, month)
    for target, y_pred in predictions.items():
        frame[FORECAST.format(target)] = np.asarray(y_pred, dtype="float32")
    return frame


def run_fold(
    fold: Fold,
    features: pd.DataFrame,
    templates: Mapping[str, BaseEstimator],
    stations: Mapping[str, pd.Index] | None = None,
) -> tuple[FoldResult, pd.DataFrame, dict[str, BaseEstimator], pd.DataFrame]:
    """Refits a clone of each target's template on the hours before ``fold.month`` and scores
    its forecasts of the month. Returns the result, the forecasts, the fitted models and the
    test rows."""
    train, test = fold.split(features)
    X_train, X_test = model_inputs(train), model_inputs(test)
    reports, fit_seconds, predictions, fitted = {}, {}, {}, {}
    for target, template in templates.items():
        start = time.perf_counter()
        fitted[target] = clone(template).fit(X_train, train[target])
        fit_seconds[target] = time.perf_counter() - start
        predictions[target] = fitted[target].predict(X_test)
        reports[target] = segment_report(test, predictions[target], target, train, stations)
    imbalance = (
        imbalance_scores(test, predictions["departures"], predictions["arrivals"])
        if set(TARGETS) <= set(templates)
        else {}
    )
    result = FoldResult(str(fold.month), reports, imbalance, fit_seconds, len(train))
    return result, _forecast_rows(test, str(fold.month), predictions), fitted, test


def backtest_model(
    name: str,
    features: pd.DataFrame,
    folds: Sequence[Fold],
    templates: Mapping[str, tuple[BaseEstimator, str]],
    stations: Mapping[str, pd.Index] | None = None,
    on_fold: Callable[[FoldResult, int], None] | None = None,
) -> BacktestResult:
    """Runs every fold for model ``name`` with the ``templates`` of each target (estimator and
    the source of its params). ``on_fold`` is called with every finished fold and its index."""
    results, forecasts, importance = [], [], {}
    for step, fold in enumerate(folds):
        result, rows, fitted, test = run_fold(
            fold, features, {t: est for t, (est, _) in templates.items()}, stations
        )
        logger.success(
            "{} {}: {} (fitted in {:.0f} s)",
            name,
            result.month,
            ", ".join(
                f"{t} peak WAPE {r.loc['peak', 'wape']:.3f}" for t, r in result.reports.items()
            ),
            sum(result.fit_seconds.values()),
        )
        if on_fold:
            on_fold(result, step)
        results.append(result)
        forecasts.append(rows)
        if step == len(folds) - 1:
            X_test = model_inputs(test)
            importance = {
                target: permutation_importances(model, X_test, test[target])
                for target, model in fitted.items()
                if isinstance(model, Pipeline)
            }
        del fitted, test
    return BacktestResult(
        name=name,
        sources={t: source for t, (_, source) in templates.items()},
        folds=results,
        forecasts=pd.concat(forecasts, ignore_index=True),
        importance=importance,
    )


def _backtest_data(
    models: Sequence[str],
    targets: Sequence[str],
    src: str | Path,
    first_test_month: str,
    n_folds: int,
) -> tuple[pd.DataFrame, list[Fold], dict[str, pd.Index]]:
    check_names(models, targets)
    features = load_features(src, first_test_month)
    folds = rolling_origin_splits(features, first_test_month, n_folds)
    return features, folds, {NEW_STATIONS: new_stations(features, NEW_SINCE)}


def backtest(
    models: Sequence[str] = BACKTEST_MODELS,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    first_test_month: str = TEST_START,
    n_folds: int = N_FOLDS,
    tuned: bool = True,
) -> list[BacktestResult]:
    """Backtests every model on the same ``n_folds`` monthly folds from ``first_test_month``."""
    features, folds, stations = _backtest_data(models, targets, src, first_test_month, n_folds)
    return [
        backtest_model(
            name, features, folds, {t: model_template(name, t, tuned) for t in targets}, stations
        )
        for name in models
    ]


# --- MLflow ----------------------------------------------------------------------


def _finite(metrics: Mapping[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if np.isfinite(v)}


def log_fold(result: FoldResult, step: int, name: str) -> None:
    """Logs ``result`` as a run nested in the active one, and its key metrics on the active run
    as step ``step``."""
    import mlflow

    with mlflow.start_run(run_name=f"fold-{result.month}", nested=True):
        mlflow.set_tags({"model_family": MODELS[name].family, FOLD: result.month})
        mlflow.log_params({"test_month": result.month, "train_rows": result.train_rows})
        mlflow.log_metrics(result.metrics())
    metrics = result.metrics()
    keys = [f"{t}_{m}" for t in result.reports for m in STEP_METRICS] + list(IMBALANCE_METRICS)
    mlflow.log_metrics(_finite({f"fold_{k}": metrics.get(k, np.nan) for k in keys}), step=step)


def summary_metrics(result: BacktestResult) -> dict[str, float]:
    """The parent run's metrics: the mean over the folds of every fold metric as
    ``<target>_<segment>_<metric>`` and its standard deviation with a ``_std`` suffix, the same
    for the imbalance scores, and the total fit time per target."""
    metrics = {}
    for target in result.targets:
        summary = result.summary(target)
        mean = report_metrics(summary.xs("mean", axis=1, level=1))
        std = report_metrics(summary.xs("std", axis=1, level=1))
        metrics |= {f"{target}_{k}": v for k, v in mean.items()}
        metrics |= {f"{target}_{k}_std": v for k, v in std.items() if not k.endswith("_rows")}
        metrics[f"{target}_fit_seconds"] = sum(f.fit_seconds[target] for f in result.folds)
    imbalance = result.imbalance_table()
    for col in imbalance:
        metrics[col] = imbalance[col].mean()
        metrics[f"{col}_std"] = imbalance[col].std()
    return _finite(metrics)


def sample_stations(forecasts: pd.DataFrame, target: str, new: pd.Index) -> dict[str, int]:
    """Three stations to plot, from the last fold: the busiest, a median one and the busiest
    of the ``new`` stations (when one is active then)."""
    last = forecasts[forecasts[FOLD] == forecasts[FOLD].max()]
    demand = last.groupby(STATION)[target].sum().sort_values()
    picks = {"busiest": demand.index[-1], "median": demand.index[len(demand) // 2]}
    opened = demand[demand.index.isin(new)]
    if len(opened):
        picks["newest busiest"] = opened.index[-1]
    return {label: int(station) for label, station in picks.items()}


def backtest_figures(result: BacktestResult, new: pd.Index) -> dict[str, object]:
    """The figures logged with a backtest run, by file name."""
    from team2_mlops2026 import plots

    plots.use_style()
    figures = {}
    by_fold = result.error_by(FOLD)["wape"].unstack("target")
    peaks = pd.DataFrame(
        {
            f"{t} peak": [f.reports[t].loc["peak", "wape"] for f in result.folds]
            for t in result.targets
        },
        index=[f.month for f in result.folds],
    )
    figures["error_by_month.png"] = plots.plot_lines(
        peaks.join(by_fold.add_suffix(" overall")),
        f"{result.name}: WAPE per test month (dashed: overall)",
        "WAPE",
        "test month",
        muted=[f"{t} overall" for t in result.targets],
    )
    figures["error_by_hour.png"] = plots.plot_lines(
        result.error_by("hour")["wape"].unstack("target"),
        f"{result.name}: WAPE by hour of the day over the {len(result.folds)} test months",
        "WAPE",
        "hour of the day",
    )
    forecasts = result.forecasts
    figures["residuals.png"] = plots.plot_residuals(
        pd.DataFrame({t: forecasts[t] - forecasts[FORECAST.format(t)] for t in result.targets}),
        f"{result.name}: residuals over the {len(result.folds)} test months",
    )
    target = result.targets[0]
    last = forecasts[forecasts[FOLD] == forecasts[FOLD].max()]
    first_day = last[TIME].min().normalize()
    monday = first_day + pd.Timedelta(days=(7 - first_day.weekday()) % 7)
    week = last[last[TIME].between(monday, monday + pd.Timedelta(days=PLOT_DAYS), "left")]
    panels = {
        f"station {station} ({label})": week[week[STATION] == station]
        .set_index(TIME)[[target, FORECAST.format(target)]]
        .set_axis(["actual", result.name], axis=1)
        for label, station in sample_stations(forecasts, target, new).items()
    }
    figures["forecasts.png"] = plots.plot_station_forecasts(
        panels, f"{result.name}: hourly {target} from {monday:%Y-%m-%d}", target
    )
    for t, table in result.importance.items():
        figures[f"importance_{t}.png"] = plots.plot_importance(
            table, f"{result.name}, {t}: permutation importance ({result.folds[-1].month})"
        )
    return figures


def log_backtest(result: BacktestResult, new: pd.Index) -> None:
    """Logs the summary metrics, the tables and the figures of ``result`` on the active run."""
    import matplotlib.pyplot as plt
    import mlflow

    mlflow.log_metrics(summary_metrics(result))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        result.fold_table().to_csv(tmp / FOLDS_FILE, index=False)
        result.error_by("hour").to_csv(tmp / BY_HOUR_FILE)
        if result.folds[0].imbalance:
            result.imbalance_table().to_csv(tmp / IMBALANCE_FILE)
        for target, table in result.importance.items():
            table.to_csv(tmp / f"importance_{target}.csv")
        result.forecasts.to_parquet(tmp / FORECASTS_FILE, index=False)
        mlflow.log_artifacts(str(tmp), ARTIFACT_DIR)
    for file_name, figure in backtest_figures(result, new).items():
        mlflow.log_figure(figure, f"{ARTIFACT_DIR}/{file_name}")
        plt.close(figure)


def backtest_tracked(
    models: Sequence[str] = BACKTEST_MODELS,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    first_test_month: str = TEST_START,
    n_folds: int = N_FOLDS,
    tuned: bool = True,
) -> list[BacktestResult]:
    """``backtest()`` with one MLflow parent run per model (``backtest-<model>``) and one
    nested run per fold (``fold-<YYYY-MM>``).

    Each fold run logs every segment metric of both targets, the decision metric and the fit
    times. The parent run logs:
    - the feature and profile tables as inputs, each with a digest
    - the folds, the profiles' fit end, each target's model params (``<target>.<param>``) and,
      as the tag ``params_from_<target>``, the tune run they come from (or "defaults")
    - the mean and the standard deviation of the fold metrics, and the key ones per fold as
      steps (``fold_<target>_peak_wape``...)
    - the tags ``model_family`` and ``git_commit``
    - the artifacts under ``backtest/``: the fold table, the decision metric per fold, the error
      by hour, the permutation importance, the forecasts and the figures
    """
    import mlflow

    from team2_mlops2026 import tracking

    src = Path(src)
    _, profiles_fit_end = load_profiles(src)
    features, folds, stations = _backtest_data(models, targets, src, first_test_month, n_folds)
    results = []
    for name in models:
        templates = {t: model_template(name, t, tuned) for t in targets}
        with tracking.start_run(f"{RUN_PREFIX}-{name}") as run:
            log_inputs(src)
            mlflow.set_tags(
                {
                    "model_family": MODELS[name].family,
                    "git_commit": tracking.git_commit(),
                    **{f"params_from_{t}": source for t, (_, source) in templates.items()},
                }
            )
            mlflow.log_params(
                {
                    "model": name,
                    "targets": ",".join(targets),
                    "first_test_month": str(folds[0].month),
                    "last_test_month": str(folds[-1].month),
                    "n_folds": len(folds),
                    "profiles_fit_end": profiles_fit_end,
                    "new_since": NEW_SINCE,
                    "tuned": tuned,
                    **{
                        f"{t}.{k}": v
                        for t, (est, _) in templates.items()
                        for k, v in model_params(est).items()
                    },
                }
            )
            result = backtest_model(
                name, features, folds, templates, stations, partial(log_fold, name=name)
            )
            log_backtest(result, stations[NEW_STATIONS])
        results.append(replace(result, run_id=run.info.run_id))
    return results


def load_backtests(
    models: Sequence[str] = BACKTEST_MODELS,
) -> dict[str, tuple[object, pd.DataFrame]]:
    """The latest finished ``backtest-<model>`` run of each model and its fold table."""
    import mlflow

    from team2_mlops2026 import tracking

    loaded = {}
    for name in models:
        run = tracking.latest_run(f"{RUN_PREFIX}-{name}")
        path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path=f"{ARTIFACT_DIR}/{FOLDS_FILE}"
        )
        loaded[name] = (run, pd.read_csv(path))
    return loaded


def mean_reports(fold_tables: Mapping[str, pd.DataFrame], target: str) -> dict[str, pd.DataFrame]:
    """Per model, the mean ``target`` report over the folds of its fold table."""
    reports = {}
    for name, table in fold_tables.items():
        rows = table[table["target"] == target].drop(columns="target")
        by_fold = {
            month: r.drop(columns=FOLD).set_index("segment") for month, r in rows.groupby(FOLD)
        }
        reports[name] = aggregate_folds(by_fold).xs("mean", axis=1, level=1)
    return reports


def rank_backtests(
    fold_tables: Mapping[str, pd.DataFrame], targets: Sequence[str] = TARGETS
) -> dict[str, pd.DataFrame]:
    """``rank_candidates`` of the backtested models for each target, on their mean reports:
    the first row is the champion if it passed."""
    return {
        target: rank_candidates(
            mean_reports(fold_tables, target), [n for n in fold_tables if n in BASELINES]
        )
        for target in targets
    }


# --- PCA ablation ----------------------------------------------------------------


@dataclass(frozen=True)
class AblationResult:
    name: str
    target: str
    profiles: bool  # whether the model reads the station profile scores
    report: pd.DataFrame  # segment_report on the test split
    fit_seconds: float  # NaN when the tune run's fitted model was reused
    source: str  # the tune run its params come from, or "defaults"
    inputs: tuple[str, ...]  # the columns the model read
    run_id: str | None = None

    @property
    def variant(self) -> str:
        return "pca" if self.profiles else "no-pca"


def profile_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("pc") and c[2:].isdigit()]


def ablate_model(
    name: str,
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_start: str,
    tuned: bool = True,
) -> list[AblationResult]:
    """Model ``name`` for ``target`` with and without the profile scores, on one split."""
    results = []
    if tuned and MODELS[name].search_space:
        fitted, run = load_tuned(name, target)
        if run.data.params.get("test_start") != test_start:
            raise ValueError(
                f"The tune run {run.info.run_id} was fitted before {run.data.params.get('test_start')}, "
                f"not before {test_start}"
            )
        template, source = clone(fitted), run.info.run_id
        pca_fit = (fitted, np.nan)
    else:
        template, source = model_template(name, target, tuned=False)
        pca_fit = None
    for profiles in (True, False):
        drop = [] if profiles else profile_columns(train)
        X_test = model_inputs(test).drop(columns=drop)
        if profiles and pca_fit:
            estimator, fit_seconds = pca_fit
        else:
            start = time.perf_counter()
            estimator = clone(template).fit(model_inputs(train).drop(columns=drop), train[target])
            fit_seconds = time.perf_counter() - start
        report = segment_report(test, estimator.predict(X_test), target, train)
        logger.success(
            "{} for {} {} the profile scores: WAPE {:.3f} overall, {:.3f} at peak hours",
            name,
            target,
            "with" if profiles else "without",
            report.loc["overall", "wape"],
            report.loc["peak", "wape"],
        )
        results.append(
            AblationResult(
                name, target, profiles, report, fit_seconds, source, tuple(X_test.columns)
            )
        )
    return results


def ablation(
    models: Sequence[str] = FINALISTS,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
    tuned: bool = True,
) -> list[AblationResult]:
    """``ablate_model`` for every model and target on the split at ``test_start``."""
    check_names(models, targets)
    train, test = load_training_data(src, test_start)
    return [
        result
        for name, target in product(models, targets)
        for result in ablate_model(name, target, train, test, test_start, tuned)
    ]


def ablation_tracked(
    models: Sequence[str] = FINALISTS,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    test_start: str = TEST_START,
    tuned: bool = True,
) -> list[AblationResult]:
    """``ablation()`` with one MLflow run per model, target and variant
    (``ablation-<model>-<target>-pca`` and ``...-no-pca``), tagged ``experiment=pca_ablation``,
    with the inputs, the params, the segment metrics and the fit time."""
    import mlflow

    from team2_mlops2026 import tracking

    check_names(models, targets)
    src = Path(src)
    train, test = load_training_data(src, test_start)
    results = []
    for name, target in product(models, targets):
        for result in ablate_model(name, target, train, test, test_start, tuned):
            with tracking.start_run(f"{ABLATION_PREFIX}-{name}-{target}-{result.variant}") as run:
                log_inputs(src)
                mlflow.set_tags(
                    {
                        "experiment": "pca_ablation",
                        "model_family": MODELS[name].family,
                        "target": target,
                        "git_commit": tracking.git_commit(),
                        "params_from": result.source,
                    }
                )
                mlflow.log_params(
                    {
                        "model": name,
                        "target": target,
                        "test_start": test_start,
                        "profiles": "pc1-pc5 + missing flag" if result.profiles else "none",
                        "train_rows": len(train),
                    }
                )
                mlflow.log_metrics(
                    _finite({**report_metrics(result.report), "fit_seconds": result.fit_seconds})
                )
            results.append(replace(result, run_id=run.info.run_id))
    return results
