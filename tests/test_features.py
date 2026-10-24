import mlflow
import numpy as np
import pandas as pd
import pytest
from trip_factory import STATIONS, make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026 import tracking

DAY0 = pd.Timestamp("2024-01-01")  # a Monday
START = DAY0 + pd.Timedelta(weeks=4)  # 2024-01-29: first hour of the table
NEW_STATION_OPENS = START + pd.Timedelta(days=7, hours=12)


@pytest.fixture
def trips():
    """Every day for 6 weeks one 08:00 trip 1 -> 2; station 3 (not in the nomenclature)
    opens in week 6 with a daily 12:00 trip 3 -> 1."""
    rows = [{"start": DAY0 + pd.Timedelta(days=d, hours=8)} for d in range(42)]
    rows += [
        {"start": NEW_STATION_OPENS + pd.Timedelta(days=d), "Origen_Id": 3, "Destino_Id": 1}
        for d in range(7)
    ]
    return make_trips(*rows)


@pytest.fixture
def table(trips):
    return ft.build_features(trips, STATIONS, START)


def at(table, station, time):
    return table[(table[ft.STATION] == station) & (table[ft.TIME] == pd.Timestamp(time))].iloc[0]


# --- station-hour table --------------------------------------------------------


def test_table_has_every_hour_of_each_station_from_start(table):
    assert list(table.columns) == list(ft.FEATURE_COLUMNS)
    assert table[ft.TIME].min() == START
    station_1 = table[table[ft.STATION] == 1]
    assert station_1[ft.TIME].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert (station_1["departures"] == 0).mean() > 0.9  # zero hours are kept


def test_table_ends_at_the_last_departure_hour(trips):
    last_start = trips["Inicio_del_viaje"].max() + pd.Timedelta(hours=11, minutes=50)
    overnight = make_trips({"start": last_start, "seconds": 1800})  # ends after midnight
    table = ft.build_features(pd.concat([trips, overnight], ignore_index=True), STATIONS, START)
    assert table[ft.TIME].max() == last_start.floor("h")  # no hour with arrivals only


def test_departures_count_by_start_hour_and_arrivals_by_end_hour(table):
    eight = START + pd.Timedelta(hours=8)
    assert at(table, 1, eight)["departures"] == 1
    assert at(table, 2, eight)["arrivals"] == 1  # the 10 min trip ends in the same hour
    assert at(table, 1, eight + pd.Timedelta(hours=1))["departures"] == 0


def test_lags_look_back_two_days_one_week_and_four_weeks(table):
    eight = START + pd.Timedelta(days=2, hours=8)
    row = at(table, 1, eight)
    assert row["departures_lag_2d"] == 1
    assert row["departures_lag_7d"] == 1
    assert row["departures_same_hour_4w"] == 1
    assert at(table, 1, eight + pd.Timedelta(hours=1))["departures_lag_2d"] == 0


def test_hours_before_a_station_opened_are_dropped_and_unknown_in_lags(table):
    station_3 = table[table[ft.STATION] == 3]
    assert station_3[ft.TIME].min() == NEW_STATION_OPENS  # no fake zero demand before opening
    first = station_3.iloc[0]
    assert np.isnan(first["departures_lag_7d"]) and np.isnan(first["departures_same_hour_4w"])


def test_calendar_features_are_cyclical(table):
    assert np.allclose(table["hour_sin"] ** 2 + table["hour_cos"] ** 2, 1)
    midnight = at(table, 1, START)
    late = at(table, 1, START - pd.Timedelta(hours=1) + pd.Timedelta(days=1))  # 23:00
    distance = np.hypot(
        midnight["hour_sin"] - late["hour_sin"], midnight["hour_cos"] - late["hour_cos"]
    )
    assert distance < 0.3  # 23:00 sits next to 00:00
    assert midnight[ft.WEEKDAY] == 0  # Monday


def test_station_attributes_are_missing_for_unknown_or_retired_stations(trips):
    nomenclature = STATIONS.assign(dpcapacity=pd.array([0, 10], dtype="Int64"))
    table = ft.build_features(trips, nomenclature, START)
    capacity = table.groupby(ft.STATION)["capacity"].first()
    assert np.isnan(capacity[1])  # retired: capacity 0 is not a capacity
    assert capacity[2] == 10
    assert table.loc[table[ft.STATION] == 3, ["latitude", "longitude"]].isna().all().all()


def test_time_split_puts_every_test_row_after_every_train_row(table):
    train, test = ft.time_split(table, START + pd.Timedelta(days=7))
    assert len(train) + len(test) == len(table)
    assert train[ft.TIME].max() < test[ft.TIME].min()


# --- preprocessing -------------------------------------------------------------


def test_preprocessor_encodes_scales_and_handles_unseen_stations(table):
    train, test = ft.time_split(table, NEW_STATION_OPENS)  # station 3 only exists in test
    preprocessor = ft.make_preprocessor()
    x_train = preprocessor.fit_transform(train, train["departures"])
    x_test = preprocessor.transform(test)

    weekday = [c for c in x_train if c.startswith("weekday__")]
    assert len(weekday) == 7 and x_train[weekday].sum(axis=1).eq(1).all()
    assert x_train["demand__departures_lag_7d"].mean() == pytest.approx(0, abs=1e-6)
    assert list(x_test.columns) == list(x_train.columns)
    assert not x_test.isna().any().any()  # unseen station and its missing lags are handled


# --- tracked build -------------------------------------------------------------


def test_create_features_tracked_logs_input_and_output_datasets(trips, tmp_path, mlflow_store):
    src, stations_src = tmp_path / "processed", tmp_path / "interim"
    dest = src / "viajes" / "2024" / "viajes_2024_01.parquet"
    dest.parent.mkdir(parents=True)
    trips.to_parquet(dest, index=False)
    (stations_src / "estaciones").mkdir(parents=True)
    STATIONS.to_parquet(stations_src / "estaciones" / "nomenclatura_2024_01.parquet", index=False)

    dst = tmp_path / "features"
    summary = ft.create_features_tracked(src, stations_src, dst, start=str(START.date()))
    run = mlflow.get_run(summary["mlflow_run_id"])

    assert run.info.run_name == ft.RUN_NAME and run.info.status == "FINISHED"
    datasets = {i.dataset.name: i.dataset for i in run.inputs.dataset_inputs}
    assert set(datasets) == {
        "mibici_trips_processed",
        "mibici_stations",
        "mibici_station_hour_features",
    }
    output = dst / ft.FEATURES_FILE
    assert datasets["mibici_station_hour_features"].digest == tracking.files_digest([output], dst)
    assert run.data.params["start"] == str(START.date())
    assert run.data.metrics["rows"] == len(pd.read_parquet(output)) == summary["rows"]
    assert run.data.metrics["stations_without_attributes"] == 1
