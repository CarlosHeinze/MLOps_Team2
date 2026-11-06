"""
Forecasting models of the station-hour demand, as scikit-learn estimators.

The baselines are the simple rules of the ML Canvas (``references/ml-canvas.md`` §4) that every
ML model must beat:

- ``SeasonalNaive``: the demand of the same station and hour one week earlier.
- ``HistoricalAverage``: the mean demand per station, weekday and hour of the training split.

The candidates are global ML models with a count loss (canvas §8), each one a ``Pipeline`` with
``features.make_preprocessor()``, so the encodings and scalings are fitted on the training rows:

- ``poisson_glm``: a Poisson GLM with a log link, the interpretable linear reference.
- ``hgb_poisson``: scikit-learn's histogram gradient boosting with a Poisson loss.
- ``lgbm_poisson`` and ``lgbm_tweedie``: LightGBM with a Poisson and a Tweedie objective.
- ``hgb_squared_error``: the same boosting with squared error, clipped at 0, as the contrast
  that shows what the count loss adds.

Every model takes the station-hour features as a DataFrame and predicts one target, so training,
evaluation and MLflow logging treat all of them alike. ``MODELS`` holds them all as ``ModelSpec``.

Usage (notebook):
    from team2_mlops2026.modeling.models import BASELINES, CANDIDATES, MODELS
    estimator = MODELS["lgbm_poisson"].build("departures", learning_rate=0.05)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from team2_mlops2026.features import SEED, STATION, TIME, WEEKDAY, make_preprocessor
from team2_mlops2026.station_profiles import HOURS_PER_WEEK

SEASON = "lag_7d"  # the feature with the demand one week earlier
# Classes skops may load from a logged model, besides the ones it trusts by default
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "numpy.dtype",
    "sklearn.compose._column_transformer.make_column_selector",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.model_selection._split.KFold",
    "team2_mlops2026.modeling.models.HistoricalAverage",
    "team2_mlops2026.modeling.models.NonNegative",
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


class NonNegative(RegressorMixin, BaseEstimator):
    """``regressor`` with its negative forecasts raised to 0, since demand is a count."""

    def __init__(self, regressor: BaseEstimator):
        self.regressor = regressor

    def fit(self, X, y) -> NonNegative:
        self.regressor_ = clone(self.regressor).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        return np.maximum(self.regressor_.predict(X), 0)


@dataclass(frozen=True)
class ModelSpec:
    """A model of the catalog, by the name the CLI and MLflow use.

    ``factory(target, **params)`` returns the unfitted estimator. ``params`` are its defaults,
    and ``search_space`` holds the distributions the tuning step samples overrides from.
    """

    name: str
    family: str  # the ``model_family`` tag that groups the runs in MLflow
    factory: Callable[..., BaseEstimator]
    params: Mapping[str, Any] = field(default_factory=dict)
    search_space: Mapping[str, Any] = field(default_factory=dict)

    def build(self, target: str, **overrides) -> BaseEstimator:
        return self.factory(target, **{**self.params, **overrides})


def _pipeline(regressor: BaseEstimator) -> Pipeline:
    # NumPy output: on a DataFrame, gradient boosting makes two extra copies of its input
    # (3.8 GB on the training split). The column names stay in get_feature_names_out().
    preprocess = make_preprocessor().set_output(transform="default")
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


# Early stopping is off: its "auto" mode validates on a random 10% of the rows, which ignores
# time order. Tuning validates on later hours instead.
HGB_DEFAULTS = {"early_stopping": False, "random_state": SEED}
HGB_SPACE = {
    "learning_rate": loguniform(0.02, 0.3),
    "max_iter": randint(100, 600),
    "max_leaf_nodes": randint(15, 128),
    "min_samples_leaf": randint(20, 500),
    "l2_regularization": loguniform(1e-4, 10),
}
# subsample only applies when bagging is on (subsample_freq > 0)
LGBM_DEFAULTS = {"subsample_freq": 1, "random_state": SEED, "n_jobs": -1, "verbose": -1}
LGBM_SPACE = {
    "n_estimators": randint(100, 800),
    "learning_rate": loguniform(0.02, 0.3),
    "num_leaves": randint(15, 255),
    "min_child_samples": randint(20, 500),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-4, 10),
}


def _catalog(*specs: ModelSpec) -> dict[str, ModelSpec]:
    return {spec.name: spec for spec in specs}


BASELINES = _catalog(
    ModelSpec("seasonal_naive", "baseline", SeasonalNaive),
    ModelSpec("historical_average", "baseline", lambda target: HistoricalAverage()),
)
CANDIDATES = _catalog(
    ModelSpec(
        "poisson_glm",
        "glm",
        lambda target, **params: _pipeline(PoissonRegressor(**params)),
        {"alpha": 1e-4, "max_iter": 300},
        {"alpha": loguniform(1e-6, 1e-1)},
    ),
    ModelSpec(
        "hgb_poisson",
        "boosting",
        lambda target, **params: _pipeline(
            HistGradientBoostingRegressor(loss="poisson", **params)
        ),
        HGB_DEFAULTS,
        HGB_SPACE,
    ),
    ModelSpec(
        "hgb_squared_error",
        "boosting",
        lambda target, **params: _pipeline(
            NonNegative(HistGradientBoostingRegressor(loss="squared_error", **params))
        ),
        HGB_DEFAULTS,
        HGB_SPACE,
    ),
    ModelSpec(
        "lgbm_poisson",
        "boosting",
        lambda target, **params: _pipeline(LGBMRegressor(objective="poisson", **params)),
        LGBM_DEFAULTS,
        LGBM_SPACE,
    ),
    ModelSpec(
        "lgbm_tweedie",
        "boosting",
        lambda target, **params: _pipeline(LGBMRegressor(objective="tweedie", **params)),
        {**LGBM_DEFAULTS, "tweedie_variance_power": 1.5},
        {**LGBM_SPACE, "tweedie_variance_power": uniform(1.1, 0.8)},
    ),
)
MODELS = BASELINES | CANDIDATES
