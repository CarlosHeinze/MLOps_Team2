"""
Forecasting models of the station-hour demand, as scikit-learn estimators.

The baselines are the simple rules of the ML Canvas (``references/ml-canvas.md`` §4) that every
ML model must beat:

- ``SeasonalNaive``: the demand of the same station and hour one week earlier.
- ``HistoricalAverage``: the mean demand per station, weekday and hour of the training split.

Every model takes the station-hour features as a DataFrame and predicts one target, so training,
evaluation and MLflow logging treat all of them alike.

Usage (notebook):
    from team2_mlops2026.modeling.models import BASELINES, HistoricalAverage, SeasonalNaive
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from team2_mlops2026.features import STATION, TIME, WEEKDAY
from team2_mlops2026.station_profiles import HOURS_PER_WEEK

FAMILY = "baseline"
SEASON = "lag_7d"  # the feature with the demand one week earlier
# Classes skops may load from a logged model, besides the scikit-learn and NumPy ones
TRUSTED_TYPES = [
    "team2_mlops2026.modeling.models.HistoricalAverage",
    "team2_mlops2026.modeling.models.SeasonalNaive",
]


def _hour_of_week(X: pd.DataFrame) -> np.ndarray:
    return X[WEEKDAY].to_numpy("int64") * 24 + X[TIME].dt.hour.to_numpy("int64")


# Mixins go before BaseEstimator so that scikit-learn picks up their tags


class HistoricalAverage(RegressorMixin, BaseEstimator):
    """Mean demand per station, weekday and hour of the training rows.

    A station without training history at that weekday and hour gets the mean of all stations.
    """

    def fit(self, X: pd.DataFrame, y) -> HistoricalAverage:
        self.stations_, rows = np.unique(X[STATION].to_numpy("int64"), return_inverse=True)
        cells = rows * HOURS_PER_WEEK + _hour_of_week(X)
        size = len(self.stations_) * HOURS_PER_WEEK
        sums = np.bincount(cells, np.asarray(y, "float64"), size).reshape(-1, HOURS_PER_WEEK)
        counts = np.bincount(cells, minlength=size).reshape(-1, HOURS_PER_WEEK)
        with np.errstate(invalid="ignore"):  # 0 / 0 -> NaN: a station-hour never seen
            self.station_means_ = sums / counts
            self.fallback_means_ = sums.sum(axis=0) / counts.sum(axis=0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        rows = pd.Index(self.stations_).get_indexer(X[STATION].to_numpy("int64"))
        hour_of_week = _hour_of_week(X)
        y_pred = self.fallback_means_[hour_of_week]
        seen = rows >= 0
        station = self.station_means_[rows[seen], hour_of_week[seen]]
        y_pred[seen] = np.where(np.isnan(station), y_pred[seen], station)
        return y_pred


class SeasonalNaive(RegressorMixin, BaseEstimator):
    """Demand of the same station and hour one week earlier (``<target>_lag_7d``).

    In a station's first week that hour is unknown, so those rows get a ``HistoricalAverage``
    fitted on the training rows.
    """

    def __init__(self, target: str = "departures"):
        self.target = target

    def fit(self, X: pd.DataFrame, y) -> SeasonalNaive:
        self.fallback_ = HistoricalAverage().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        y_pred = X[f"{self.target}_{SEASON}"].to_numpy("float64", copy=True)
        unknown = np.isnan(y_pred)
        if unknown.any():
            y_pred[unknown] = self.fallback_.predict(X[unknown])
        return y_pred


# Unfitted model for a target, by the name the CLI and MLflow use
BASELINES: dict[str, Callable[[str], BaseEstimator]] = {
    "seasonal_naive": SeasonalNaive,
    "historical_average": lambda target: HistoricalAverage(),
}
