import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from team2_mlops2026.features import STATION, TIME, WEEKDAY
from team2_mlops2026.modeling.models import BASELINES, HistoricalAverage, SeasonalNaive

MONDAY = pd.Timestamp("2025-09-01")


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
    assert BASELINES["seasonal_naive"]("arrivals").target == "arrivals"
    assert isinstance(BASELINES["historical_average"]("arrivals"), HistoricalAverage)
