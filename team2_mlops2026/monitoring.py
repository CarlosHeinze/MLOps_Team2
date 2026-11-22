"""Replay held-out MiBici station-hours with the served models and Phase 2 baselines.

Use `python -m team2_mlops2026.monitoring --help` or `make monitor`. The replay compares
each window with a held-out reference and each injected scenario with its unmodified source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, ClassVar, Protocol
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import typer

from team2_mlops2026 import plots
from team2_mlops2026.config import MODELS_DIR
from team2_mlops2026.features import (
    DEMAND_FEATURES,
    FEATURES_DIR,
    FEATURES_FILE,
    LAGS,
    MEAN_HOURS,
    PROFILE_SCORES,
    SEED,
    STATION,
    STATION_ATTRS,
    TARGETS,
    TIME,
    WEEKDAY,
)
from team2_mlops2026.modeling.backtest import NEW_SINCE, NEW_STATIONS
from team2_mlops2026.modeling.evaluate import (
    MIN_IMPROVEMENT,
    PEAK_HOURS,
    TEST_START,
    imbalance_scores,
    segment_report,
    success_criterion,
)
from team2_mlops2026.modeling.models import BASELINES
from team2_mlops2026.modeling.predict import forecast
from team2_mlops2026.modeling.registry import CHAMPION
from team2_mlops2026.modeling.train import load_features, model_inputs, report_metrics
from team2_mlops2026.serving.app import LOCAL, REGISTRY, LoadedModels, load_models

MONITORING_DIR = FEATURES_DIR.parent / "monitoring"
SIM_MONTH = "2025-11"
REAL_MONTH = "2026-09"
DEFAULT_SEVERITY = 0.25
PSI_BINS = 10
PSI_FLOOR = 1e-6
PSI_WARNING = 0.10
PSI_ALERT = 0.25
MISSING_INCREASE = 0.10
UNKNOWN_STATIONS = 0.10
BACKTEST_SIGMAS = 2
BACKTEST_PEAK_WAPE = {"departures": (0.520, 0.042), "arrivals": (0.483, 0.039)}
METRIC_ATOL = 1e-12

app = typer.Typer()


class Scenario(Protocol):
    name: str
    severity: float

    @property
    def params(self) -> dict: ...

    def apply(self, rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame: ...


@dataclass(frozen=True)
class Control:
    severity: float = 0.0
    name: ClassVar[str] = "control"

    def __post_init__(self):
        if not 0 <= self.severity <= 1:
            raise ValueError("Severity must be between 0 and 1")

    @property
    def params(self) -> dict:
        return asdict(self)

    def apply(self, rows, rng):
        return rows


class RealDrift(Control):
    name = "real"


def _scale(rows, columns, factor, *, counts=False):
    for column in columns:
        values = rows[column] * factor
        rows[column] = values.round().astype(rows[column].dtype) if counts else values


class MeanShift(Control):
    name = "mean-shift"

    def apply(self, rows, rng):
        if not self.severity:
            return rows
        changed = rows.copy()
        _scale(changed, DEMAND_FEATURES, 1 + self.severity)
        _scale(changed, TARGETS, 1 + self.severity, counts=True)
        return changed


class CovariateShift(Control):
    name = "covariate-only"

    def apply(self, rows, rng):
        if not self.severity:
            return rows
        changed = rows.copy()
        _scale(changed, DEMAND_FEATURES, 1 + self.severity)
        return changed


def _stations(rows, rng, severity):
    stations = np.sort(rows[STATION].unique())
    return rng.permutation(stations)[: int(np.ceil(severity * len(stations)))]


def _null_columns(rows, mask, columns):
    for column in columns:
        rows[column] = rows[column].mask(mask)


class MissingFeatures(Control):
    name = "missing-features"

    def apply(self, rows, rng):
        if not self.severity:
            return rows
        changed = rows.copy()
        groups = (list(rows.filter(regex=PROFILE_SCORES)), DEMAND_FEATURES, STATION_ATTRS)
        for columns in groups:
            mask = rows[STATION].isin(_stations(rows, rng, self.severity))
            _null_columns(changed, mask, columns)
        return changed


def _commute(times):
    return (times.dt.day_of_week < 5) & times.dt.hour.isin(PEAK_HOURS)


class SeasonalShift(Control):
    name = "seasonal"

    def apply(self, rows, rng):
        """Holiday-like commuter reduction, with the calendar and station identities intact."""
        if not self.severity:
            return rows
        changed = rows.copy()
        times = rows[TIME]
        _scale(changed, TARGETS, 1 - self.severity * _commute(times), counts=True)
        for lag, hours in LAGS.items():
            factor = 1 - self.severity * _commute(times - pd.Timedelta(hours=hours))
            _scale(changed, [f"{target}_{lag}" for target in TARGETS], factor)
        _scale(
            changed,
            [f"{target}_same_hour_4w" for target in TARGETS],
            1 - self.severity * _commute(times),
        )
        # Approximate the commuter share of a full week's rolling mean by its share of hours.
        factor = 1 - self.severity * (5 * len(PEAK_HOURS) / MEAN_HOURS)
        _scale(changed, [f"{target}_mean_7d" for target in TARGETS], factor)
        return changed


@dataclass(frozen=True)
class NewStations(Control):
    first_unused: int = 0
    name: ClassVar[str] = "new-stations"

    def apply(self, rows, rng):
        if not self.severity:
            return rows
        selected = _stations(rows, rng, self.severity)
        if self.first_unused <= int(rows[STATION].max()):
            raise ValueError("New station IDs must follow the existing station IDs")
        replacements = {station: self.first_unused + i for i, station in enumerate(selected)}
        changed = rows.copy()
        changed[STATION] = rows[STATION].replace(replacements)
        _null_columns(
            changed,
            rows[STATION].isin(selected),
            (*DEMAND_FEATURES, *rows.filter(regex=PROFILE_SCORES)),
        )
        return changed


SIMULATIONS = {
    s.name: s for s in (MeanShift, CovariateShift, MissingFeatures, SeasonalShift, NewStations)
}


@dataclass(frozen=True)
class Thresholds:
    psi_warning: float = PSI_WARNING
    psi_alert: float = PSI_ALERT
    missing_increase: float = MISSING_INCREASE
    unknown_share: float = UNKNOWN_STATIONS
    min_improvement: float = MIN_IMPROVEMENT
    peak_departures: float = (
        BACKTEST_PEAK_WAPE["departures"][0] + BACKTEST_SIGMAS * BACKTEST_PEAK_WAPE["departures"][1]
    )
    peak_arrivals: float = (
        BACKTEST_PEAK_WAPE["arrivals"][0] + BACKTEST_SIGMAS * BACKTEST_PEAK_WAPE["arrivals"][1]
    )

    def __post_init__(self):
        if not 0 < self.psi_warning <= self.psi_alert < np.inf:
            raise ValueError("PSI thresholds must satisfy 0 < warning <= alert")
        if not all(0 < value <= 1 for value in (self.missing_increase, self.unknown_share)):
            raise ValueError("Missing and unknown-station thresholds must be in (0, 1]")
        if not 0 <= self.min_improvement < 1 or not all(
            0 < self.peak_limit(target) < np.inf for target in TARGETS
        ):
            raise ValueError("Invalid performance thresholds")

    def peak_limit(self, target: str) -> float:
        return getattr(self, f"peak_{target}")


DEFAULT_THRESHOLDS = Thresholds()


def distribution(
    a: pd.Series, b: pd.Series, *, categorical: bool = False, compute_ks: bool = True
) -> dict:
    """PSI on reference quantile bins (or categories), plus KS on non-missing numbers."""
    observed_a, observed_b = a.dropna(), b.dropna()
    missing_a = (len(a) - len(observed_a)) / len(a)
    missing_b = (len(b) - len(observed_b)) / len(b)
    ks = pvalue = np.nan
    if categorical:
        counts = pd.concat([observed_a.value_counts(), observed_b.value_counts()], axis=1).fillna(
            0
        )
        mass_a, mass_b = (
            counts.iloc[:, 0].to_numpy() / len(a),
            counts.iloc[:, 1].to_numpy() / len(b),
        )
    elif observed_a.empty:
        mass_a, mass_b = [0], [1 - missing_b]
    else:
        left, right = observed_a.to_numpy(dtype=float), observed_b.to_numpy(dtype=float)
        quantiles = np.unique(np.quantile(left, np.linspace(0, 1, PSI_BINS + 1)))
        edges = np.r_[np.nextafter(quantiles[0], -np.inf), quantiles]
        mass_a = np.bincount(
            np.searchsorted(edges, left, side="left"), minlength=len(edges) + 1
        ) / len(a)
        mass_b = np.bincount(
            np.searchsorted(edges, right, side="left"), minlength=len(edges) + 1
        ) / len(b)
        if compute_ks and len(right):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="ks_2samp: Exact calculation unsuccessful"
                )
                ks, pvalue = ks_2samp(left, right)
    p, q = (
        np.maximum(np.r_[mass_a, missing_a], PSI_FLOOR),
        np.maximum(np.r_[mass_b, missing_b], PSI_FLOOR),
    )
    p, q = p / p.sum(), q / q.sum()
    return {
        "psi": float(np.sum((q - p) * np.log(q / p))),
        "ks": float(ks),
        "ks_pvalue": float(pvalue),
        "reference_missing": missing_a,
        "monitoring_missing": missing_b,
        "missing_increase": missing_b - missing_a,
    }


def drift_frame(rows, predictions):
    return (
        rows.drop(columns=TIME)
        .rename(columns={t: f"actual_{t}" for t in TARGETS})
        .assign(
            hour=rows[TIME].dt.hour,
            **{f"forecast_{t}": predictions[t].to_numpy() for t in TARGETS},
        )
    )


def drift_report(reference, current, control=None):
    records = {}
    for name in reference:
        categorical = name in (STATION, WEEKDAY, "hour")
        # Hours and weekdays have meaningful ordered CDFs as well as fixed categories.
        stats = distribution(reference[name], current[name], categorical=categorical)
        if categorical and name != STATION:
            ordered = distribution(reference[name], current[name])
            stats.update(ks=ordered["ks"], ks_pvalue=ordered["ks_pvalue"])
        if control is not None:
            stats["psi_vs_control"] = distribution(
                control[name], current[name], categorical=categorical, compute_ks=False
            )["psi"]
        records[name] = stats
    return pd.DataFrame.from_dict(records, orient="index").rename_axis("feature")


@dataclass(frozen=True)
class Scores:
    predictions: pd.DataFrame
    reports: dict[str, dict[str, pd.DataFrame]]
    decisions: dict[str, dict[str, float]]
    best: dict[str, dict]


def fit_baselines(train):
    return {
        name: {
            target: spec.build(target).fit(train[[STATION, TIME, WEEKDAY]], train[target])
            for target in TARGETS
        }
        for name, spec in BASELINES.items()
    }


def score(rows, models, baselines, train, *, decision_capacity=None, station_groups=None):
    inputs = model_inputs(rows)
    decision_rows = rows if decision_capacity is None else rows.assign(capacity=decision_capacity)
    eligible = (
        decision_rows.loc[decision_rows["capacity"].notna()]
        .groupby([STATION, decision_rows[TIME].dt.floor("D")])
        .ngroups
    )
    reports, decisions, best = {}, {}, {}
    for name, estimators in {CHAMPION: models, **baselines}.items():
        pred = forecast(estimators, inputs)
        reports[name] = {
            target: segment_report(rows, pred[target], target, train, stations=station_groups)
            for target in TARGETS
        }
        decisions[name] = imbalance_scores(decision_rows, pred["departures"], pred["arrivals"])
        decisions[name]["eligible_station_days"] = eligible
        if name == CHAMPION:
            champion = pred
    for target in TARGETS:
        candidates = {name: reports[name][target] for name in baselines}
        peak = {name: table.loc["peak", "wape"] for name, table in candidates.items()}
        valid = {name: value for name, value in peak.items() if np.isfinite(value)}
        best_name = min(valid, key=valid.get) if valid else None
        champion_wape = reports[CHAMPION][target].loc["peak", "wape"]
        result = {
            "baseline": best_name,
            "baseline_peak_wape": valid.get(best_name, np.nan),
            "peak_wape": float(champion_wape),
            "improvement": np.nan,
            "worse_groups": None,
        }
        if best_name is not None and valid[best_name] > 0 and np.isfinite(champion_wape):
            comparison = success_criterion(
                reports[CHAMPION][target], {name: candidates[name] for name in valid}
            )
            result.update(
                improvement=comparison.peak_wape_improvement, worse_groups=comparison.worse_groups
            )
        best[target] = result
    return Scores(champion, reports, decisions, best)


def assess(stats, best, comparator, unknown, thresholds):
    """Quality, distribution and measured performance checks are independent."""
    features = stats.loc[
        ~stats.index.str.startswith(("actual_", "forecast_"))
        & ~stats.index.isin(["month_sin", "month_cos"])
    ]
    predictions = stats.loc[stats.index.str.startswith("forecast_")]
    feature_drift = features.index[features["psi"] >= thresholds.psi_warning].tolist()
    prediction_drift = predictions.index[predictions["psi"] >= thresholds.psi_warning].tolist()
    missing = features.index[
        features["missing_increase"] >= thresholds.missing_increase - METRIC_ATOL
    ].tolist()
    quality = bool(missing) or unknown >= thresholds.unknown_share
    severe_psi = bool(
        (features["psi"] >= thresholds.psi_alert).any()
        or (predictions["psi"] >= thresholds.psi_alert).any()
    )
    performance = {}
    for target, observed in best.items():
        peak = observed["peak_wape"]
        margin = observed["improvement"]
        delta_peak = peak - comparator[target]["peak_wape"]
        delta_margin = margin - comparator[target]["improvement"]
        peak_alert = peak - thresholds.peak_limit(target) > METRIC_ATOL
        margin_alert = thresholds.min_improvement - margin > METRIC_ATOL
        performance[target] = {
            "peak_alert": bool(peak_alert),
            "improvement_alert": bool(margin_alert),
            "performance_degraded": bool(
                (peak_alert and delta_peak > METRIC_ATOL)
                or (margin_alert and delta_margin < -METRIC_ATOL)
            ),
            "peak_wape_delta": float(delta_peak),
            "improvement_delta": float(delta_margin),
        }
    flagged = any(p["peak_alert"] or p["improvement_alert"] for p in performance.values())
    degraded = any(p["performance_degraded"] for p in performance.values())
    detected = bool(feature_drift or prediction_drift)
    level = "alert" if quality or severe_psi or flagged else ("warning" if detected else "ok")
    if quality:
        action = "Review the feature pipeline and data contract, then replay the window."
    elif detected and degraded:
        action = (
            "Confirm the loss with observed labels, then consider make train tune evaluate register. "
            "Move champion only if the candidate passes the backtest criterion."
        )
    elif flagged:
        action = "Review the forecast errors, labels and baseline comparison."
    elif detected:
        action = "Keep watching the shifted distributions and forecast error."
    else:
        action = "Continue routine monitoring."
    return {
        "level": level,
        "action": action,
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "severe_psi": severe_psi,
        "missing_features": missing,
        "unknown_station_alert": bool(unknown >= thresholds.unknown_share),
        "quality_alert": quality,
        "performance": performance,
    }


def _month(value):
    return pd.Period(value, freq="M")


def _training_bounds(loaded):
    bounds = set()
    for target in TARGETS:
        window = loaded.card["models"][target].get("tags", {}).get("training_window", "")
        start, sep, end = window.removesuffix(" (exclusive)").rpartition(" to ")
        if not sep or pd.isna(pd.Timestamp(end)):
            raise ValueError(f"{target} lacks a valid training_window in its model card")
        start = pd.Timestamp(start) if start != "table start" else None
        bounds.add((start, pd.Timestamp(end)))
    if len(bounds) != 1:
        raise ValueError("Both served models must have the same training window")
    return bounds.pop()


def _window(src, month):
    start, end = month.start_time, (month + 1).start_time
    rows = load_features(
        src, test_start=start.isoformat(), start=start.isoformat(), end=end.isoformat()
    )
    if rows.empty or rows.duplicated([STATION, TIME]).any():
        raise ValueError(f"{month} needs nonempty, unique station-hours")
    if (
        rows[STATION].isna().any()
        or not np.isfinite(rows[list(TARGETS)].to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"{month} has missing station IDs or missing/non-finite actuals")
    if (rows[list(TARGETS)] < 0).any().any():
        raise ValueError(f"{month} has negative demand counts")
    return rows.sort_values([STATION, TIME], ignore_index=True)


def _digest(rows):
    """Hash the schema and every row, including changes after MLflow's pandas sample."""
    sha = hashlib.sha256(repr(list(zip(rows.columns, map(str, rows.dtypes)))).encode())
    for start in range(0, len(rows), 100_000):
        values = pd.util.hash_pandas_object(rows.iloc[start : start + 100_000], index=False)
        sha.update(values.to_numpy(dtype="<u8").tobytes())
    return sha.hexdigest()[:32]


def _describe(rows):
    return {
        "rows": len(rows),
        "stations": int(rows[STATION].nunique()),
        "first_hour": rows[TIME].min().isoformat(),
        "last_hour": rows[TIME].max().isoformat(),
        "digest": _digest(rows),
        "actual_totals": {target: float(rows[target].sum()) for target in TARGETS},
    }


def _json_ready(value):
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (int, float, np.generic)):
        return (
            None
            if not np.isfinite(value)
            else (value.item() if isinstance(value, np.generic) else value)
        )
    return value


def _save_json(path, value):
    path.write_text(
        json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _report_tables(scores):
    metrics, decisions = {}, {}
    for window, scored in scores.items():
        for model, targets in scored.reports.items():
            for target, table in targets.items():
                metrics[window, model, target] = table
        decisions[window] = pd.DataFrame.from_dict(scored.decisions, orient="index")
    return (
        pd.concat(metrics, names=["window", "model", "target"]),
        pd.concat(decisions, names=["window", "model"]),
    )


def _log_result(summary, stats, scores, source, rows, reference_rows, directory):
    import mlflow

    from team2_mlops2026 import tracking

    alerts = summary["alerts"]
    with tracking.start_run(summary["name"]) as run:
        tracking.log_dataset(
            reference_rows,
            source,
            "mibici_monitor_reference",
            "reference",
            digest=summary["reference"]["digest"],
        )
        tracking.log_dataset(
            rows,
            source,
            "mibici_monitoring",
            "monitoring",
            digest=summary["monitoring"]["digest"],
        )
        mlflow.log_params(
            {
                **summary["scenario_params"],
                **summary["thresholds"],
                **summary["windows"],
                "seed": summary["seed"],
            }
        )
        mlflow.set_tags(
            {
                "scenario": summary["scenario"],
                "data_kind": summary["data_kind"],
                "alert_level": alerts["level"],
                "proposed_action": alerts["action"],
                "git_commit": tracking.git_commit(),
                **{f"{target}_model_uri": uri for target, uri in summary["models"].items()},
            }
        )
        metrics = {
            "feature_drift_detected": bool(alerts["feature_drift"]),
            "prediction_drift_detected": bool(alerts["prediction_drift"]),
            "severe_psi": alerts["severe_psi"],
            "quality_alert": alerts["quality_alert"],
            "unknown_station_alert": alerts["unknown_station_alert"],
            "unknown_row_share": summary["unknown_row_share"],
            "unknown_station_share": summary["unknown_station_share"],
            "new_since_station_share": summary.get("new_since_station_share", np.nan),
            **{
                f"{target}_{key}": value
                for target, checks in alerts["performance"].items()
                for key, value in checks.items()
            },
            **{f"{feature}_{metric}": value for (feature, metric), value in stats.stack().items()},
        }
        for window, result in scores.items():
            for model, reports in result.reports.items():
                for target, table in reports.items():
                    metrics.update(
                        {
                            f"{window}_{model}_{target}_{key}": value
                            for key, value in report_metrics(table).items()
                        }
                    )
            for model, decisions in result.decisions.items():
                metrics.update(
                    {f"{window}_{model}_{key}": value for key, value in decisions.items()}
                )
            for target, values in result.best.items():
                metrics[f"{window}_{target}_improvement"] = values["improvement"]
                metrics[f"{window}_{target}_baseline_peak_wape"] = values["baseline_peak_wape"]
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if np.isfinite(value)}
        )
        summary["mlflow_run_id"] = run.info.run_id
        _save_json(directory / "summary.json", summary)
        for name in (
            "summary.json",
            "drift.csv",
            "performance.csv",
            "decisions.csv",
            "distribution.png",
            "psi.png",
        ):
            mlflow.log_artifact(str(directory / name))
    return run.info.run_id


def _save_case(directory, stats, frames, thresholds, scores):
    directory.mkdir(parents=True, exist_ok=True)
    stats.to_csv(directory / "drift.csv")
    performance, decisions = _report_tables(scores)
    performance.to_csv(directory / "performance.csv")
    decisions.to_csv(directory / "decisions.csv")
    numeric = stats.drop(index=[STATION, "month_sin", "month_cos"])
    feature = numeric["psi"].idxmax()
    figures = {
        "distribution.png": plots.plot_drift_distribution(
            *(frame[feature] for frame in frames), feature
        ),
        "psi.png": plots.plot_feature_psi(
            stats.drop(index=["month_sin", "month_cos"])["psi"],
            thresholds.psi_warning,
            thresholds.psi_alert,
        ),
    }
    for name, fig in figures.items():
        fig.savefig(directory / name, dpi=120)
        plt.close(fig)


def _save_curves(summaries, dst, thresholds, tracked):
    for name in dict.fromkeys(summary["scenario"] for summary in summaries):
        group = [summary for summary in summaries if summary["scenario"] == name]
        points = []
        if "control" in group[0]["scores"]:
            points.append((0.0, group[0]["scores"]["control"]))
        points += [(s["scenario_params"]["severity"], s["scores"]["monitoring"]) for s in group]
        if name == "real":
            continue
        fig = plots.plot_drift_performance(
            name,
            dict(points),
            group[0]["scores"]["reference"],
            {target: thresholds.peak_limit(target) for target in TARGETS},
        )
        path = dst / f"{name}-performance.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        if tracked:
            from mlflow import MlflowClient

            client = MlflowClient()
            for summary in group:
                client.log_artifact(summary["mlflow_run_id"], str(path))


def monitor(
    src: str | Path = FEATURES_DIR,
    dst: str | Path = MONITORING_DIR,
    *,
    loaded: LoadedModels | None = None,
    reference_month: str = TEST_START,
    sim_month: str = SIM_MONTH,
    real_month: str = REAL_MONTH,
    scenarios: Sequence[str] = (*SIMULATIONS, "real"),
    severities: Sequence[float] = (DEFAULT_SEVERITY,),
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    seed: int = SEED,
    data_kind: str = "historical",
    track: bool = False,
) -> list[dict]:
    """Run the frozen models and both baselines on disjoint monthly windows."""
    src, dst = Path(src), Path(dst)
    names = tuple(dict.fromkeys(scenarios))
    if not names or set(names) - {*SIMULATIONS, "real"}:
        raise ValueError("Select at least one supported scenario")
    if not severities or not all(0 <= value <= 1 for value in severities):
        raise ValueError("Select severity values between 0 and 1")
    if data_kind not in ("historical", "synthetic"):
        raise ValueError("Data kind must be historical or synthetic")
    plots.use_style()
    if track:
        import mlflow

        from team2_mlops2026 import tracking

        if mlflow.active_run() is not None or os.getenv("MLFLOW_RUN_ID"):
            raise ValueError("Monitoring needs a separate MLflow run per scenario and severity")
        tracking.check_tracking_server()

    if loaded is None:
        loaded = load_models()
    start, cutoff = _training_bounds(loaded)
    ref_month = _month(reference_month)
    sim = _month(sim_month) if set(names) & SIMULATIONS.keys() else None
    real = _month(real_month) if "real" in names else None
    months = [month for month in (ref_month, sim, real) if month is not None]
    if cutoff > ref_month.start_time or months != sorted(set(months)):
        raise ValueError(
            "Expected disjoint months in order: after training, reference, simulation, real"
        )

    filters = [(TIME, "<", cutoff)]
    if start is not None:
        filters.append((TIME, ">=", start))
    feature_file = src / FEATURES_FILE
    train = pd.read_parquet(
        feature_file, columns=[STATION, TIME, WEEKDAY, *TARGETS], filters=filters
    )
    if train.empty:
        raise ValueError("No feature rows in the models' training window for the baselines")
    baselines = fit_baselines(train)
    known = pd.Index(train[STATION].unique())
    # segment_report ranks the stations using only their mean demand in training.
    station_means = train.groupby(STATION, as_index=False)[list(TARGETS)].mean()
    del train
    reference_rows = _window(src, ref_month)
    reference = score(reference_rows, loaded.models, baselines, station_means)
    reference_drift = drift_frame(reference_rows, reference.predictions)
    reference_meta = _describe(reference_rows)
    control_rows = _window(src, sim) if sim is not None else None
    control = control_drift = control_meta = None
    cases: list[Scenario] = []
    if control_rows is not None:
        control = score(control_rows, loaded.models, baselines, station_means)
        control_drift = drift_frame(control_rows, control.predictions)
        control_meta = _describe(control_rows)
        first_unused = (
            int(max(known.max(), reference_rows[STATION].max(), control_rows[STATION].max())) + 1
        )
        cases.append(Control())
        for name in names:
            if name == "real":
                continue
            for severity in sorted(set(severities)):
                extra = {"first_unused": first_unused} if name == NewStations.name else {}
                cases.append(SIMULATIONS[name](severity, **extra))
    if real is not None:
        cases.append(RealDrift())

    summaries = []
    for scenario in cases:
        is_real = scenario.name == "real"
        month = real if is_real else sim
        window = _window(src, month) if is_real else control_rows
        rows = scenario.apply(window, np.random.default_rng(seed))
        station_groups = {}
        if is_real and month.start_time >= pd.Timestamp(NEW_SINCE):
            prior_ids = pd.read_parquet(
                feature_file, columns=[STATION], filters=[(TIME, "<", pd.Timestamp(NEW_SINCE))]
            )[STATION].unique()
            station_groups[NEW_STATIONS] = rows.loc[
                ~rows[STATION].isin(prior_ids), STATION
            ].unique()
        current = (
            control
            if not is_real and scenario.severity == 0
            else score(
                rows,
                loaded.models,
                baselines,
                station_means,
                decision_capacity=window["capacity"],
                station_groups=station_groups,
            )
        )
        current_drift = drift_frame(rows, current.predictions)
        stats = drift_report(reference_drift, current_drift, None if is_real else control_drift)
        unknown = ~rows[STATION].isin(known)
        scored = {"reference": reference, "monitoring": current}
        if not is_real:
            scored["control"] = control
        severity = np.format_float_positional(scenario.severity, trim="-")
        name = f"drift-{scenario.name}-{severity}"
        summary = {
            "name": name,
            "scenario": scenario.name,
            "scenario_params": scenario.params,
            "windows": {
                "reference_month": str(ref_month),
                "monitoring_month": str(month),
                "train_start": str(start or "table start"),
                "train_end": cutoff.isoformat(),
            },
            "seed": seed,
            "data_kind": data_kind,
            "thresholds": asdict(thresholds),
            "models": {target: loaded.card["models"][target]["uri"] for target in TARGETS},
            "reference": reference_meta,
            "control": None if is_real else control_meta,
            "monitoring": _describe(rows),
            "unknown_row_share": float(unknown.mean()),
            "unknown_station_share": rows.loc[unknown, STATION].nunique()
            / rows[STATION].nunique(),
            "scores": {key: value.best for key, value in scored.items()},
            "alerts": assess(
                stats,
                current.best,
                (reference if is_real else control).best,
                unknown.mean(),
                thresholds,
            ),
        }
        if NEW_STATIONS in station_groups:
            summary["new_since_station_share"] = (
                len(station_groups[NEW_STATIONS]) / rows[STATION].nunique()
            )
        directory = dst / name
        _save_case(directory, stats, (reference_drift, current_drift), thresholds, scored)
        if track:
            _log_result(summary, stats, scored, feature_file, rows, reference_rows, directory)
        else:
            _save_json(directory / "summary.json", summary)
        summaries.append(summary)
        logger.info("{}: {} — {}", name, summary["alerts"]["level"], summary["alerts"]["action"])
    _save_curves(summaries, dst, thresholds, track)
    _save_json(dst / "summaries.json", summaries)
    return summaries


def monitor_tracked(*args, **kwargs):
    """Write one MLflow run per scenario and severity."""
    return monitor(*args, track=True, **kwargs)


@app.command()
def main(
    src: Annotated[
        Path, typer.Option(help="Feature and profile Parquet directory")
    ] = FEATURES_DIR,
    dst: Annotated[Path, typer.Option(help="Reports and figures directory")] = MONITORING_DIR,
    models_dir: Annotated[
        Path, typer.Option(help="Trusted exported champions directory")
    ] = MODELS_DIR,
    model_source: Annotated[
        str, typer.Option(help="Same sources as the service: local or registry")
    ] = LOCAL,
    reference_month: Annotated[
        str, typer.Option(help="Held-out reference month (YYYY-MM)")
    ] = TEST_START,
    sim_month: Annotated[str, typer.Option(help="Month to perturb (YYYY-MM)")] = SIM_MONTH,
    real_month: Annotated[
        str, typer.Option(help="Unmodified replay month (YYYY-MM)")
    ] = REAL_MONTH,
    scenario: Annotated[
        list[str] | None,
        typer.Option(help="Repeat to select from: " + ", ".join((*SIMULATIONS, "real"))),
    ] = None,
    severity: Annotated[
        list[float] | None, typer.Option(help="Repeat for a sweep, in [0, 1]")
    ] = None,
    seed: Annotated[int, typer.Option(min=0)] = SEED,
    psi_warning: float = PSI_WARNING,
    psi_alert: float = PSI_ALERT,
    missing_increase: float = MISSING_INCREASE,
    unknown_share: float = UNKNOWN_STATIONS,
    min_improvement: float = MIN_IMPROVEMENT,
    peak_departures: float = DEFAULT_THRESHOLDS.peak_departures,
    peak_arrivals: float = DEFAULT_THRESHOLDS.peak_arrivals,
    data_kind: Annotated[
        str, typer.Option(help="Label input provenance: historical or synthetic")
    ] = "historical",
    no_track: Annotated[
        bool, typer.Option("--no-track", help="Write outputs locally only")
    ] = False,
):
    """Compare the served champions on held-out station-hours with and without drift."""
    try:
        if model_source not in (LOCAL, REGISTRY):
            raise ValueError("Use --model-source local or registry")
        thresholds = Thresholds(
            psi_warning,
            psi_alert,
            missing_increase,
            unknown_share,
            min_improvement,
            peak_departures,
            peak_arrivals,
        )
        run = monitor if no_track else monitor_tracked
        run(
            src,
            dst,
            loaded=load_models(model_source, models_dir),
            reference_month=reference_month,
            sim_month=sim_month,
            real_month=real_month,
            scenarios=scenario or (*SIMULATIONS, "real"),
            severities=severity or (DEFAULT_SEVERITY,),
            thresholds=thresholds,
            seed=seed,
            data_kind=data_kind,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
