"""
Evaluation of the station-hour demand forecasts, as fixed by the ML Canvas
(``references/ml-canvas.md`` §4, Impact simulation):

- Count metrics: MAE, RMSE, WAPE and mean Poisson deviance. WAPE divides the total error by the
  total demand, so the many zero hours do not blow it up as a per-row percentage error would.
  Each segment also gets the daily WAPE, the WAPE of the station-day totals.
- Segments: overall, weekday peak hours, the top-50 stations, demand groups and the stations
  unseen in training. Stations are ranked on the training split, so the test rows never choose
  their own segment.
- Decision metric: how well the forecasts flag the high-imbalance station-days, the days whose
  net flow exceeds half of the station's capacity.
- Rolling-origin folds: train on every hour before a month, test on that month, move one month on.
  ``aggregate_folds`` summarizes the folds' reports as their mean and standard deviation.
- Success criterion: what a model must achieve against the best baseline. The demand groups
  are compared on the daily WAPE: in the quietest group most hours have no trip, and there the
  hourly WAPE is lowest for a forecast of zero (notebook 2.02). ``rank_candidates`` applies it
  to every candidate and ranks them, so the first one that passes is the champion.

Usage (notebook):
    from team2_mlops2026.modeling.evaluate import rolling_origin_splits, segment_report
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from team2_mlops2026.features import STATION, TIME, WEEKDAY, time_split

METRICS = ("mae", "rmse", "wape", "poisson_deviance")
DAILY_WAPE = "daily_wape"
# Monday-Friday, one hour around the commuter peaks at 8:00 and 18:00 (notebook 1.03)
PEAK_HOURS = (7, 8, 9, 17, 18, 19)
TOP_STATIONS = 50
DEMAND_GROUPS = 5
MIN_PREDICTION = 1e-6  # Poisson deviance is undefined for a zero prediction
IMBALANCE_SHARE = 0.5  # of the capacity

TEST_START = "2025-10"  # the test period of notebooks 1.04 and 1.05
N_FOLDS = 12
MIN_IMPROVEMENT = 0.10  # in peak-hour WAPE over the best baseline


# --- metrics -------------------------------------------------------------------


def count_metrics(y_true, y_pred) -> dict[str, float]:
    """Row count plus MAE, RMSE, WAPE and mean Poisson deviance of count forecasts.

    Undefined values are NaN, not a division by zero: WAPE when there is no demand at all, and
    every metric for an empty segment.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{len(y_pred)} predictions for {len(y_true)} targets")
    if np.isnan(y_pred).any() or (y_pred < 0).any():
        raise ValueError("Predictions must be non-negative numbers")
    if len(y_true) == 0:
        return {"n": 0, **dict.fromkeys(METRICS, np.nan)}

    error = y_true - y_pred
    demand = y_true.sum()
    return {
        "n": len(y_true),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / demand) if demand > 0 else np.nan,
        "poisson_deviance": float(
            mean_poisson_deviance(y_true, np.maximum(y_pred, MIN_PREDICTION))
        ),
    }


def daily_wape(y_true, y_pred, station_day) -> float:
    """WAPE of the station-day totals, where ``station_day`` labels each row's station and day
    with an integer code. NaN when there is no demand."""
    y_true = np.asarray(y_true, dtype="float64")
    demand = y_true.sum()
    if demand <= 0:
        return np.nan
    error = np.bincount(station_day, y_true - np.asarray(y_pred, dtype="float64"))
    return float(np.abs(error).sum() / demand)


# --- segments ------------------------------------------------------------------


def is_peak(frame: pd.DataFrame) -> np.ndarray:
    return ((frame[WEEKDAY] < 5) & frame[TIME].dt.hour.isin(PEAK_HOURS)).to_numpy()


def station_segments(
    train: pd.DataFrame, target: str, top: int = TOP_STATIONS, groups: int = DEMAND_GROUPS
) -> pd.DataFrame:
    """Per station of ``train``: whether it is a top station and its demand group (1 = lowest).

    Both come from the station's mean hourly ``target``, so a busy station that opened late
    ranks by how busy it is, not by how long it has existed.
    """
    demand = train.groupby(STATION)[target].mean()
    ranks = demand.rank(method="first")  # equal demand must not merge quantile edges
    return pd.DataFrame(
        {
            "top": demand.index.isin(demand.nlargest(top).index),
            "group": pd.qcut(ranks, min(groups, len(demand)), labels=False) + 1,
        },
        index=demand.index,
    )


def segment_report(
    test: pd.DataFrame,
    y_pred,
    target: str,
    train: pd.DataFrame,
    stations: Mapping[str, Collection] | None = None,
) -> pd.DataFrame:
    """``count_metrics`` and ``daily_wape`` of the ``target`` forecasts per segment, one row
    each. ``stations`` adds one segment per named set of station ids, such as the stations
    opened in a given period."""
    segments = station_segments(train, target)
    station = test[STATION]
    group = station.map(segments["group"]).to_numpy()
    masks = {
        "overall": np.ones(len(test), dtype=bool),
        "peak": is_peak(test),
        f"top_{TOP_STATIONS}": station.isin(segments.index[segments["top"]]).to_numpy(),
        **{f"group_{g}": group == g for g in range(1, segments["group"].max() + 1)},
        "unseen_stations": np.isnan(group),
        **{name: station.isin(list(ids)).to_numpy() for name, ids in (stations or {}).items()},
    }
    y_true, y_pred = test[target].to_numpy(), np.asarray(y_pred)
    station_day = test.groupby([STATION, test[TIME].dt.floor("D")]).ngroup().to_numpy()
    report = pd.DataFrame.from_dict(
        {
            name: {
                **count_metrics(y_true[mask], y_pred[mask]),
                DAILY_WAPE: daily_wape(y_true[mask], y_pred[mask], station_day[mask]),
            }
            for name, mask in masks.items()
        },
        orient="index",
    )
    report.index.name = "segment"
    return report


def grouped_metrics(frame: pd.DataFrame, y_true, y_pred, by) -> pd.DataFrame:
    """``count_metrics`` of the forecasts per value of ``by`` (one key per row of ``frame``),
    such as the hour of the day or the month, one row each."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    keys = pd.Series(np.asarray(by), index=frame.index)
    table = pd.DataFrame.from_dict(
        {
            key: count_metrics(y_true[rows], y_pred[rows])
            for key, rows in keys.groupby(keys, sort=True).indices.items()
        },
        orient="index",
    )
    table.index.name = getattr(by, "name", None)
    return table


# --- decision metric -----------------------------------------------------------


def imbalance_days(frame: pd.DataFrame, departures, arrivals) -> pd.Series:
    """Per station-day: whether the absolute net flow exceeds ``IMBALANCE_SHARE`` of capacity.

    Stations without a known capacity are left out, since their threshold is unknown.
    """
    days = (
        pd.DataFrame(
            {
                STATION: frame[STATION].to_numpy(),
                "day": frame[TIME].dt.floor("D").to_numpy(),
                "net_flow": np.asarray(arrivals, "float64") - np.asarray(departures, "float64"),
                "capacity": frame["capacity"].to_numpy(),
            }
        )
        .groupby([STATION, "day"])
        .agg(net_flow=("net_flow", "sum"), capacity=("capacity", "first"))
        .dropna(subset="capacity")
    )
    return days["net_flow"].abs() > IMBALANCE_SHARE * days["capacity"]


def imbalance_scores(test: pd.DataFrame, departures, arrivals) -> dict[str, float]:
    """Precision and recall of the forecasts in flagging the actual high-imbalance days."""
    actual = imbalance_days(test, test["departures"], test["arrivals"])
    flagged = imbalance_days(test, departures, arrivals)
    hits = int((actual & flagged).sum())
    return {
        "imbalance_days": int(actual.sum()),
        "flagged_days": int(flagged.sum()),
        "imbalance_precision": hits / flagged.sum() if flagged.any() else np.nan,
        "imbalance_recall": hits / actual.sum() if actual.any() else np.nan,
    }


# --- rolling-origin folds ------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """Train on every hour before ``month``, test on the hours of ``month``."""

    month: pd.Period

    @property
    def test_start(self) -> pd.Timestamp:
        return self.month.start_time

    @property
    def test_end(self) -> pd.Timestamp:
        return (self.month + 1).start_time

    def split(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, later = time_split(features, self.test_start)
        return train, later[later[TIME] < self.test_end]


def rolling_origin_splits(
    features: pd.DataFrame, first_test_month: str = TEST_START, n_folds: int = N_FOLDS
) -> list[Fold]:
    """The ``n_folds`` monthly folds from ``first_test_month`` (YYYY-MM).

    Fails if the data has no hours before the first fold or none in any of the test months.
    """
    first = pd.Period(first_test_month, freq="M")
    folds = [Fold(first + k) for k in range(n_folds)]
    times = features[TIME]
    if not (times < folds[0].test_start).any():
        raise ValueError(f"No training hours before the first test month {first}")
    empty = [
        str(f.month)
        for f in folds
        if not times.between(f.test_start, f.test_end, inclusive="left").any()
    ]
    if empty:
        raise ValueError(f"No hours in the test months {', '.join(empty)}")
    return folds


def aggregate_folds(reports: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard deviation over the folds of every segment metric.

    ``reports`` maps each fold to its ``segment_report``. The result has one row per segment, in
    the order of the first fold, and the columns (metric, "mean") and (metric, "std"). A value
    that is undefined in a fold (NaN) is left out of that segment's statistics, and the standard
    deviation of a single fold is NaN. ``.xs("mean", axis=1, level=1)`` has the shape of a
    ``segment_report``, so the mean report can go through ``success_criterion``.
    """
    if not reports:
        raise ValueError("No fold reports to aggregate")
    order = next(iter(reports.values())).index
    stacked = pd.concat(reports, names=["fold"])
    summary = stacked.groupby(level="segment").agg(["mean", "std"])
    return summary.reindex([*order, *summary.index.difference(order)])


# --- success criterion ---------------------------------------------------------


@dataclass(frozen=True)
class SuccessCheck:
    baseline: str
    peak_wape_improvement: float  # relative reduction of the baseline's peak-hour WAPE
    worse_groups: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.peak_wape_improvement >= MIN_IMPROVEMENT and not self.worse_groups


def success_criterion(report: pd.DataFrame, baselines: Mapping[str, pd.DataFrame]) -> SuccessCheck:
    """Checks a model's ``segment_report`` against the baseline with the lowest peak-hour WAPE.

    The model passes if its peak-hour WAPE is at least ``MIN_IMPROVEMENT`` lower and no demand
    group has a higher daily WAPE than under that baseline. A group whose daily WAPE is
    undefined is not compared.
    """
    peak_wapes = {name: b.loc["peak", "wape"] for name, b in baselines.items()}
    if not np.isfinite([report.loc["peak", "wape"], *peak_wapes.values()]).all():
        raise ValueError("Peak-hour WAPE is undefined: the test rows have no peak-hour demand")
    name = min(peak_wapes, key=peak_wapes.get)
    baseline = baselines[name]
    groups = [s for s in report.index if s.startswith("group_")]
    return SuccessCheck(
        baseline=name,
        peak_wape_improvement=float(1 - report.loc["peak", "wape"] / baseline.loc["peak", "wape"]),
        worse_groups=tuple(
            g for g in groups if report.loc[g, DAILY_WAPE] > baseline.loc[g, DAILY_WAPE]
        ),
    )


def rank_candidates(reports: Mapping[str, pd.DataFrame], baselines: Iterable[str]) -> pd.DataFrame:
    """``success_criterion`` of every model of ``reports`` that is not a baseline, ranked.

    ``reports`` maps each model to its segment report for one target, such as the mean report
    of the backtest, and must include the ``baselines``. The models that pass come first, each
    group ordered by peak-hour WAPE, so the first row is the champion if it passed. Columns:
    peak_wape, baseline, peak_wape_improvement, worse_groups and passed.
    """
    baselines = {name: reports[name] for name in baselines}
    rows = {}
    for name, report in reports.items():
        if name in baselines:
            continue
        check = success_criterion(report, baselines)
        rows[name] = {
            "peak_wape": float(report.loc["peak", "wape"]),
            "baseline": check.baseline,
            "peak_wape_improvement": check.peak_wape_improvement,
            "worse_groups": ", ".join(check.worse_groups),
            "passed": check.passed,
        }
    if not rows:
        raise ValueError("No candidates to rank besides the baselines")
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")
    return table.sort_values(["passed", "peak_wape"], ascending=[False, True], kind="stable")
