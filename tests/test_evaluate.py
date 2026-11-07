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


def report(peak_wape, *group_wapes, hourly_group_wapes=None) -> pd.DataFrame:
    """Peak WAPE plus the groups' daily WAPE (also their hourly WAPE, unless given)."""
    groups = [f"group_{g}" for g in range(1, len(group_wapes) + 1)]
    hourly = group_wapes if hourly_group_wapes is None else hourly_group_wapes
    return pd.DataFrame(
        {
            "wape": {"peak": peak_wape, **dict(zip(groups, hourly))},
            ev.DAILY_WAPE: {"peak": peak_wape, **dict(zip(groups, group_wapes))},
        }
    )


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


def test_daily_wape_nets_the_hourly_errors_within_each_station_day():
    # Day 0: 1 trip forecast an hour late; day 1: 2 trips, none forecast
    y_true, y_pred, station_day = [1, 0, 2, 0], [0, 1, 0, 0], [0, 0, 1, 1]
    assert ev.daily_wape(y_true, y_pred, station_day) == pytest.approx(2 / 3)
    assert ev.count_metrics(y_true, y_pred)["wape"] == pytest.approx(4 / 3)
    assert np.isnan(ev.daily_wape([0, 0], [1, 1], [0, 1]))


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

    assert list(result.columns) == ["n", *ev.METRICS, ev.DAILY_WAPE]
    assert result.loc["overall", "n"] == 72
    assert result.loc["peak", "n"] == 3 * len(ev.PEAK_HOURS)  # a Tuesday
    assert result.loc["top_50", "n"] == 48  # every station seen in training
    assert result.loc[["group_1", "group_2"], "mae"].eq(0).all()
    assert result.loc["unseen_stations", "n"] == 24
    assert result.loc["unseen_stations", "wape"] == 1
    assert result.loc["overall", "wape"] == pytest.approx(1 / 3)
    assert result.loc["unseen_stations", ev.DAILY_WAPE] == 1


def test_segment_report_adds_a_segment_per_named_set_of_stations():
    train = station_hours([1, 2], MONDAY, 24, departures=1)
    test = station_hours([1, 2, 3], MONDAY + pd.Timedelta(days=1), 24, departures=2)
    y_pred = np.where(test[STATION] == 3, 1, 2)

    result = ev.segment_report(
        test, y_pred, "departures", train, stations={"new_stations": [3], "closed": [9]}
    )

    assert list(result.index[-2:]) == ["new_stations", "closed"]
    assert result.loc["new_stations", "n"] == 24
    assert result.loc["new_stations", "wape"] == pytest.approx(0.5)
    assert result.loc["closed", "n"] == 0 and np.isnan(result.loc["closed", "wape"])


def test_grouped_metrics_score_each_key_on_its_own_rows():
    frame = station_hours([1], MONDAY, 4)
    hours = frame[TIME].dt.hour.rename("hour")
    result = ev.grouped_metrics(frame, [2, 2, 2, 4], [2, 1, 3, 2], hours % 2)
    assert result.index.name == "hour"
    assert list(result.index) == [0, 1]
    assert result.loc[0, "n"] == 2 and result.loc[0, "wape"] == pytest.approx(1 / 4)  # rows 0, 2
    assert result.loc[1, "wape"] == pytest.approx(3 / 6)  # rows 1, 3


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


def test_aggregate_folds_takes_the_mean_and_std_of_each_segment_metric():
    folds = {
        "2025-10": pd.DataFrame(
            {"n": [10, 5], "wape": [0.5, 0.2]}, index=pd.Index(["overall", "peak"], name="segment")
        ),
        "2025-11": pd.DataFrame(
            {"n": [20, 0], "wape": [0.7, np.nan]},
            index=pd.Index(["overall", "peak"], name="segment"),
        ),
    }
    summary = ev.aggregate_folds(folds)
    assert list(summary.index) == ["overall", "peak"]
    assert summary.loc["overall", ("wape", "mean")] == pytest.approx(0.6)
    assert summary.loc["overall", ("wape", "std")] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert summary.loc["overall", ("n", "mean")] == 15
    # An undefined value is left out, so the peak mean is the first fold's alone
    assert summary.loc["peak", ("wape", "mean")] == pytest.approx(0.2)
    assert np.isnan(summary.loc["peak", ("wape", "std")])
    mean = summary.xs("mean", axis=1, level=1)
    assert list(mean.columns) == ["n", "wape"]
    with pytest.raises(ValueError, match="No fold reports"):
        ev.aggregate_folds({})


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


def test_demand_groups_are_compared_on_their_daily_wape():
    baselines = {"historical_average": report(0.30, 0.6, hourly_group_wapes=[1.2])}
    check = ev.success_criterion(report(0.20, 0.5, hourly_group_wapes=[1.3]), baselines)
    assert check.passed  # worse hour by hour, better per station-day


def test_groups_with_undefined_wape_are_not_compared():
    baselines = {"historical_average": report(0.30, np.nan)}
    assert ev.success_criterion(report(0.20, np.nan), baselines).passed


def test_success_is_not_decided_without_peak_hour_demand():
    baselines = {"seasonal_naive": report(np.nan, 0.5), "historical_average": report(np.nan, 0.6)}
    with pytest.raises(ValueError, match="Peak-hour WAPE is undefined"):
        ev.success_criterion(report(np.nan, 0.5), baselines)


def test_rank_candidates_puts_the_passing_models_first_by_peak_wape():
    reports = {
        "seasonal_naive": report(0.40, 0.5),
        "historical_average": report(0.30, 0.6),
        "fast_but_unfair": report(0.10, 0.7),  # worse than the baseline in group 1
        "close": report(0.26, 0.5),
        "best": report(0.20, 0.4),
        "not_enough": report(0.29, 0.4),
    }
    table = ev.rank_candidates(reports, ["seasonal_naive", "historical_average"])
    assert list(table.index) == ["best", "close", "fast_but_unfair", "not_enough"]
    assert table["passed"].tolist() == [True, True, False, False]
    assert table.loc["best", "baseline"] == "historical_average"
    assert table.loc["best", "peak_wape_improvement"] == pytest.approx(1 - 0.20 / 0.30)
    assert table.loc["fast_but_unfair", "worse_groups"] == "group_1"
    with pytest.raises(ValueError, match="No candidates"):
        ev.rank_candidates({"seasonal_naive": report(0.4, 0.5)}, ["seasonal_naive"])
