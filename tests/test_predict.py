"""The inference steps of the batch forecast without MLflow. ``test_registry`` covers the
forecast with the registered champions, and ``test_pipeline_integration`` the whole chain."""

import numpy as np
import pandas as pd
from trip_factory import STATIONS, make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026.modeling import predict as pr
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.modeling.models import MODELS
from team2_mlops2026.station_profiles import WEEK

DAY0 = pd.Timestamp("2024-01-01")  # a Monday


class Fixed:
    """A fitted model stand-in that returns the given forecasts."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype="float64")

    def predict(self, X):
        assert len(X) == len(self.values)
        return self.values


def test_forecast_clips_negative_demand_and_nets_arrivals_minus_departures():
    rows = pd.DataFrame(
        {ft.STATION: [1, 1, 2], ft.TIME: DAY0 + pd.to_timedelta([0, 1, 0], unit="h")},
        index=[7, 8, 9],  # rows filtered out of a larger frame
    )
    models = {"departures": Fixed([1.0, -2.0, 3.0]), "arrivals": Fixed([4.0, 0.5, -1.0])}
    out = pr.forecast(models, rows)
    assert list(out.columns) == [ft.STATION, ft.TIME, "departures", "arrivals", pr.NET_FLOW]
    assert out.index.equals(pd.RangeIndex(3))
    assert out["departures"].tolist() == [1, 0, 3]
    assert out["arrivals"].tolist() == [4, 0.5, 0]
    assert out[pr.NET_FLOW].tolist() == [3, 0.5, -3]


def test_join_profiles_takes_the_last_scored_week_and_leaves_unscored_stations_empty():
    weeks = DAY0 + pd.to_timedelta([0, 7], unit="D")
    scores = pd.DataFrame(
        {ft.STATION: [1, 1, 2], WEEK: weeks[[0, 1, 0]], "pc1": [0.1, 0.2, 0.3], "pc2": 0.0}
    )
    times = DAY0 + pd.to_timedelta([1, 8, 15, 30, 8, 1], unit="D")
    rows = pd.DataFrame({ft.STATION: [1, 1, 1, 1, 2, 3], ft.TIME: times, "x": range(6)})
    joined = pr.join_profiles(rows, scores)
    assert list(joined.columns) == [ft.STATION, ft.TIME, "x", "pc1", "pc2"]
    pd.testing.assert_frame_equal(joined[rows.columns], rows)  # same rows, same order
    # Weeks 3 and 5 come after the last scored week, so they take its scores. Station 2 has no
    # score in that week and station 3 none at all.
    expected = [0.1, 0.2, 0.2, 0.2, np.nan, np.nan]
    np.testing.assert_array_equal(joined["pc1"], expected)


def test_a_candidate_forecasts_a_station_unseen_in_training_a_week_after_the_last_profile():
    """Six weeks of a daily 08:00 trip 1 -> 2; station 3, not in the nomenclature, opens after
    the training rows end. The profile scores stop four weeks before the forecast."""
    opens = DAY0 + pd.Timedelta(days=36, hours=12)
    trips = make_trips(
        *({"start": DAY0 + pd.Timedelta(days=d, hours=8)} for d in range(42)),
        *(
            {"start": opens + pd.Timedelta(days=d), "Origen_Id": 3, "Destino_Id": 1}
            for d in range(5)
        ),
    )
    table = ft.build_features(trips, STATIONS, DAY0)
    weeks = pd.date_range(DAY0, periods=2, freq="W-MON")
    scores = pd.DataFrame(
        {ft.STATION: [1, 2, 1, 2], WEEK: weeks.repeat(2), "pc1": [0.5, -0.5, 0.4, -0.4]}
    )
    train = pr.join_profiles(table[table[ft.TIME] < opens], scores)
    models = {
        target: MODELS["lgbm_tweedie"]
        .build(target, n_estimators=10)
        .fit(tr.model_inputs(train), train[target])
        for target in ft.TARGETS
    }

    rows = pr.join_profiles(ft.forecast_rows(table), scores)
    new = rows[ft.STATION] == 3
    assert new.any() and rows.loc[new, ["pc1", "latitude", "departures_lag_7d"]].isna().all().all()
    assert (rows.loc[~new, "pc1"].abs() == 0.4).all()  # the last scored week, weeks earlier
    out = pr.forecast(models, rows)
    demand = out[list(ft.TARGETS)].to_numpy()
    assert np.isfinite(demand).all() and (demand >= 0).all()
    assert set(out[ft.STATION]) == {1, 2, 3} and len(out) == 3 * ft.FORECAST_HOURS
