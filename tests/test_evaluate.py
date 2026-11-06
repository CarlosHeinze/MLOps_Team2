import numpy as np
import pandas as pd
import pytest

from team2_mlops2026.features import STATION, TIME, WEEKDAY
from team2_mlops2026.modeling import evaluate as ev

MONDAY = pd.Timestamp("2025-09-01")


def station_hours(stations, start, hours, **columns) -> pd.DataFrame:
    """Every hour from ``start`` for each station, with the columns the evaluation reads."""
    times = pd.date_range(start, periods=hours, freq="h")
    frame = pd.DataFrame(
        {
            STATION: np.repeat(stations, len(times)),
            TIME: np.tile(times, len(stations)),
        }
    )
    frame[WEEKDAY] = frame[TIME].dt.day_of_week
    return frame.assign(**{"departures": 0, "arrivals": 0, "capacity": 10.0, **columns})


def report(peak_wape, *group_wapes) -> pd.DataFrame:
    rows = {"peak": peak_wape, **{f"group_{g}": w for g, w in enumerate(group_wapes, 1)}}
    return pd.DataFrame({"wape": rows})


# --- metrics -------------------------------------------------------------------


def test_count_metrics_match_hand_computed_values():
    m = ev.count_metrics([0, 2, 4], [1, 2, 1])  # absolute errors 1, 0, 3
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert m["wape"] == pytest.approx(4 / 6)
    # 2 * mean(y log(y / p) - y + p) = 2/3 * (1 + 0 + 4 log 4 - 3)
    assert m["poisson_deviance"] == pytest.approx(2 / 3 * (4 * np.log(4) - 2))


def test_no_demand_leaves_wape_undefined_and_zero_predictions_scored():
    m = ev.count_metrics([0, 0], [0, 0])
    assert np.isnan(m["wape"])
    assert m["mae"] == 0 and m["poisson_deviance"] == pytest.approx(0, abs=1e-5)
    # Predicting no trips for an hour that had some is penalised, not a math error
    assert np.isfinite(ev.count_metrics([3], [0])["poisson_deviance"])


def test_empty_segment_has_undefined_metrics():
    m = ev.count_metrics([], [])
    assert m["n"] == 0
    assert all(np.isnan(m[k]) for k in ev.METRICS)


@pytest.mark.parametrize("y_pred", [[1, np.nan], [1, -0.5], [1]])
def test_invalid_predictions_are_rejected(y_pred):
    with pytest.raises(ValueError):
        ev.count_metrics([1, 2], y_pred)


# --- segments ------------------------------------------------------------------


def test_peak_hours_are_weekday_commute_hours():
    times = pd.Series(pd.to_datetime(["2025-09-01 08:00", "2025-09-01 12:00", "2025-09-06 08:00"]))
    frame = pd.DataFrame({TIME: times, WEEKDAY: times.dt.day_of_week})
    assert ev.is_peak(frame).tolist() == [True, False, False]  # Monday 8, Monday 12, Saturday 8


def test_station_segments_rank_by_mean_hourly_demand():
    # Station 3 has the most trips in total but the lowest hourly mean
    train = pd.DataFrame(
        {STATION: [1, 1, 2, 2, 3, 3, 3, 3, 3, 3], "departures": [4, 4, 2, 2, 1, 1, 1, 1, 1, 1]}
    )
    segments = ev.station_segments(train, "departures", top=1, groups=3)
    assert segments["top"].to_dict() == {1: True, 2: False, 3: False}
    assert segments["group"].to_dict() == {1: 3, 2: 2, 3: 1}


def test_segment_report_scores_each_segment_on_its_own_rows():
    train = station_hours([1, 2], MONDAY, 24).assign(departures=lambda f: f[STATION])
    test = station_hours([1, 2, 3], MONDAY + pd.Timedelta(days=1), 24, departures=2)
    y_pred = np.where(test[STATION] == 3, 0, 2)  # exact except at the unseen station 3

    result = ev.segment_report(test, y_pred, "departures", train)

    assert list(result.columns) == ["n", *ev.METRICS]
    assert result.loc["overall", "n"] == 72
    assert result.loc["peak", "n"] == 3 * len(ev.PEAK_HOURS)  # a Tuesday
    assert result.loc["top_50", "n"] == 48  # every station seen in training
    assert result.loc[["group_1", "group_2"], "mae"].eq(0).all()
    assert result.loc["unseen_stations", "n"] == 24
    assert result.loc["unseen_stations", "wape"] == 1
    assert result.loc["overall", "wape"] == pytest.approx(1 / 3)


# --- decision metric -----------------------------------------------------------


def test_imbalance_days_need_net_flow_above_half_the_capacity():
    frame = station_hours([1, 2, 3], MONDAY, 48)
    frame.loc[frame[STATION] == 3, "capacity"] = np.nan
    first_day = frame[TIME] < MONDAY + pd.Timedelta(days=1)
    noon = frame[TIME].dt.hour == 12
    # Day 1: station 1 nets +6 (above 5), station 2 nets -5 (not above); day 2: station 2 nets +6
    arrivals = np.select(
        [first_day & noon & (frame[STATION] == 1), ~first_day & noon & (frame[STATION] == 2)],
        [6, 6],
    )
    departures = np.where(first_day & noon & (frame[STATION] == 2), 5, 0)

    flags = ev.imbalance_days(frame, departures, arrivals)

    assert len(flags) == 4  # station 3 has no capacity, so its days are left out
    assert flags[flags].index.tolist() == [(1, MONDAY), (2, MONDAY + pd.Timedelta(days=1))]


def test_imbalance_scores_compare_flagged_with_actual_days():
    test = station_hours([1, 2], MONDAY, 24)
    test.loc[test[STATION] == 1, "arrivals"] = 1  # nets +24: an actual imbalance day
    flag_both = np.full(len(test), 1)

    scores = ev.imbalance_scores(test, departures=np.zeros(len(test)), arrivals=flag_both)
    assert scores == {
        "imbalance_days": 1,
        "flagged_days": 2,
        "imbalance_precision": 0.5,
        "imbalance_recall": 1.0,
    }
    quiet = ev.imbalance_scores(test, np.zeros(len(test)), np.zeros(len(test)))
    assert np.isnan(quiet["imbalance_precision"]) and quiet["imbalance_recall"] == 0


# --- rolling-origin folds ------------------------------------------------------


@pytest.fixture
def table():
    """Two stations, hourly from 2025-01-01 to the end of September 2026, as in the real table."""
    hours = int((pd.Timestamp("2026-10-01") - pd.Timestamp("2025-01-01")) / pd.Timedelta("1h"))
    return station_hours([1, 2], "2025-01-01", hours)


def test_default_folds_are_the_twelve_months_from_october_2025(table):
    folds = ev.rolling_origin_splits(table)
    assert [str(f.month) for f in folds] == [
        str(m) for m in pd.period_range("2025-10", "2026-09", freq="M")
    ]


def test_each_fold_trains_on_the_past_and_tests_on_its_month(table):
    for fold in ev.rolling_origin_splits(table):
        train, test = fold.split(table)
        assert train[TIME].max() < test[TIME].min()
        assert test[TIME].min() == fold.test_start
        assert test[TIME].max() == fold.test_end - pd.Timedelta(hours=1)
        assert len(train) + len(test) == (table[TIME] < fold.test_end).sum()


def test_folds_need_data_before_the_first_and_in_every_test_month(table):
    with pytest.raises(ValueError, match="No training hours"):
        ev.rolling_origin_splits(table, first_test_month="2025-01")
    with pytest.raises(ValueError, match="No training hours"):
        ev.rolling_origin_splits(table.iloc[:0])
    with pytest.raises(ValueError, match="test months 2026-10, 2026-11, 2026-12"):
        ev.rolling_origin_splits(table, first_test_month="2026-01", n_folds=12)

    gap = table[table[TIME].dt.to_period("M") != pd.Period("2026-02", freq="M")]
    with pytest.raises(ValueError, match="test months 2026-02$"):
        ev.rolling_origin_splits(gap)


# --- success criterion ---------------------------------------------------------


def test_success_needs_ten_percent_lower_peak_wape_than_the_best_baseline():
    baselines = {"seasonal_naive": report(0.40, 0.5), "historical_average": report(0.30, 0.6)}

    check = ev.success_criterion(report(0.26, 0.5), baselines)
    assert check.baseline == "historical_average"
    assert check.peak_wape_improvement == pytest.approx(1 - 0.26 / 0.30)
    assert check.passed

    assert not ev.success_criterion(report(0.28, 0.5), baselines).passed  # only 6.7% lower


def test_success_fails_if_any_demand_group_is_worse_than_the_baseline():
    baselines = {"historical_average": report(0.30, 0.6, 0.4)}
    check = ev.success_criterion(report(0.20, 0.5, 0.45), baselines)
    assert check.worse_groups == ("group_2",)
    assert not check.passed


def test_groups_with_undefined_wape_are_not_compared():
    baselines = {"historical_average": report(0.30, np.nan)}
    assert ev.success_criterion(report(0.20, np.nan), baselines).passed


def test_success_is_not_decided_without_peak_hour_demand():
    baselines = {"seasonal_naive": report(np.nan, 0.5), "historical_average": report(np.nan, 0.6)}
    with pytest.raises(ValueError, match="Peak-hour WAPE is undefined"):
        ev.success_criterion(report(np.nan, 0.5), baselines)
