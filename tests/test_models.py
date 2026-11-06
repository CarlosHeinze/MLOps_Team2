import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler
import skops.io
from trip_factory import STATIONS, make_trips

from team2_mlops2026.features import SEED, STATION, TARGETS, TIME, WEEKDAY, build_features
from team2_mlops2026.modeling.models import (
    BASELINES,
    CANDIDATES,
    MODELS,
    TRUSTED_TYPES,
    HistoricalAverage,
    NonNegative,
    SeasonalNaive,
)

MONDAY = pd.Timestamp("2025-09-01")
DAY0 = pd.Timestamp("2024-01-01")  # a Monday


def rows(station, times, **columns) -> pd.DataFrame:
    times = pd.DatetimeIndex(times)
    return pd.DataFrame({STATION: station, TIME: times, WEEKDAY: times.day_of_week, **columns})


@pytest.fixture
def train():
    """Two Mondays at 08:00 and 09:00 for station 1 (demand 2 and 4 at 08:00) and station 2."""
    times = [MONDAY + pd.Timedelta(days=d, hours=h) for d in (0, 7) for h in (8, 9)]
    return pd.concat(
        [
            rows(1, times, departures=[2, 1, 4, 1]),
            rows(2, times, departures=[0, 0, 0, 0]),
        ],
        ignore_index=True,
    )


def test_historical_average_is_the_station_weekday_hour_mean(train):
    model = HistoricalAverage().fit(train.drop(columns="departures"), train["departures"])
    later = MONDAY + pd.Timedelta(weeks=2, hours=8)
    test = pd.concat([rows(1, [later]), rows(2, [later]), rows(3, [later])], ignore_index=True)
    # Station 3 is unseen: mean of all stations on Mondays at 08:00, (2 + 4 + 0 + 0) / 4
    assert model.predict(test).tolist() == [3, 0, 1.5]


def test_seasonal_naive_repeats_its_target_one_week_earlier(train):
    later = MONDAY + pd.Timedelta(weeks=2, hours=8)
    test = rows(
        [1, 1], [later, later], departures_lag_7d=[5.0, np.nan], arrivals_lag_7d=[9.0, 9.0]
    )
    model = SeasonalNaive("departures").fit(train, train["departures"])
    # Without the week-earlier hour it falls back to the station's historical average
    assert model.predict(test).tolist() == [5, 3]
    arrivals = SeasonalNaive("arrivals").fit(train, train["departures"])
    assert arrivals.predict(test).tolist() == [9, 9]


@pytest.mark.parametrize("model", [HistoricalAverage(), SeasonalNaive("arrivals")])
def test_baselines_behave_as_scikit_learn_estimators(model, train):
    with pytest.raises(NotFittedError):
        model.predict(train)
    assert clone(model).get_params() == model.get_params()


def test_catalog_builds_an_unfitted_model_per_target():
    assert BASELINES["seasonal_naive"].build("arrivals").target == "arrivals"
    assert isinstance(BASELINES["historical_average"].build("arrivals"), HistoricalAverage)
    assert MODELS == BASELINES | CANDIDATES and len(CANDIDATES) >= 3
    assert {spec.family for spec in MODELS.values()} == {"baseline", "glm", "boosting"}


def test_non_negative_clips_the_forecasts_at_zero():
    X, y = pd.DataFrame({"x": [0.0, 1.0, 2.0]}), np.array([-2.0, 0.0, 2.0])
    model = NonNegative(LinearRegression())
    with pytest.raises(NotFittedError):
        model.predict(X)
    assert model.fit(X, y).predict(X).tolist() == pytest.approx([0, 0, 2])
    assert clone(model).get_params()["regressor"] is not model.regressor


@pytest.fixture(scope="module")
def station_hours():
    """Six weeks of random commutes between stations 1 and 2, plus station 3 (not in the
    nomenclature and without a profile score) from week 5 on."""
    rng = np.random.default_rng(SEED)
    hours = DAY0 + pd.to_timedelta(rng.integers(0, 6 * 168, 3000), unit="h")
    origins = rng.choice([1, 2, 3], len(hours), p=[0.5, 0.4, 0.1])
    trips = make_trips(
        *(
            {"start": start, "Origen_Id": int(a), "Destino_Id": 1 if a != 1 else 2}
            for start, a in zip(hours, origins)
            if a != 3 or start >= DAY0 + pd.Timedelta(weeks=4)
        )
    )
    table = build_features(trips, STATIONS, DAY0 + pd.Timedelta(weeks=4))
    scores = pd.DataFrame({STATION: [1, 2], "pc1": [0.4, -0.4], "pc2": [0.1, -0.2]})
    return table.merge(scores, on=STATION, how="left")


@pytest.mark.parametrize("name", CANDIDATES)
@pytest.mark.parametrize("target", TARGETS)
def test_every_candidate_fits_and_forecasts_counts(name, target, station_hours):
    X = station_hours.drop(columns=list(TARGETS))
    model = CANDIDATES[name].build(target).fit(X, station_hours[target])
    y_pred = model.predict(X)
    assert y_pred.shape == (len(X),) and np.isfinite(y_pred).all() and (y_pred >= 0).all()
    # A logged model loads with the trusted types only
    assert set(skops.io.get_untrusted_types(data=skops.io.dumps(model))) <= set(TRUSTED_TYPES)


@pytest.mark.parametrize("name", CANDIDATES)
def test_search_space_samples_are_valid_overrides(name):
    spec = CANDIDATES[name]
    (sample,) = ParameterSampler(spec.search_space, n_iter=1, random_state=SEED)
    model = spec.build("departures", **sample)
    params = model[-1].get_params()
    assert all(params.get(k, params.get(f"regressor__{k}")) == v for k, v in sample.items())
    assert spec.build("departures")[-1].get_params() != params  # defaults left untouched
