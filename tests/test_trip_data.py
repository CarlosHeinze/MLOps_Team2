import pandas as pd
import pytest
from trip_factory import BY, STATIONS, make_trips

from team2_mlops2026.data import trip_data as td


@pytest.fixture
def interim_dir(tmp_path):
    """Two months, one of them with an all-null gender column (stored as Parquet null)."""
    src = tmp_path / "interim"
    for year_month, df in (
        ("2024_02", make_trips({"Genero": None, BY: None})),
        ("2024_03", make_trips({}, {"Viaje_Id": 2})),
    ):
        dest = src / "viajes" / "2024" / f"viajes_{year_month}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
    (src / "estaciones").mkdir()
    for name in ("nomenclatura_2024_01.parquet", "nomenclatura_2024_03.parquet"):
        STATIONS.to_parquet(src / "estaciones" / name, index=False)
    return src


# --- loading -----------------------------------------------------------------


def test_load_trips_reads_every_month_as_category(interim_dir):
    trips = td.load_trips(interim_dir, ["Usuario_Id", "Genero"])
    assert list(trips.columns) == ["Usuario_Id", "Genero"]
    assert len(trips) == 3
    assert trips["Genero"].dtype == "category"



def test_load_stations_reads_the_latest_nomenclature(interim_dir):
    assert td.latest_station_file(interim_dir).name == "nomenclatura_2024_03.parquet"
    assert list(td.load_stations(interim_dir)["id"]) == [1, 2]


def test_latest_station_file_requires_a_nomenclature(tmp_path):
    (tmp_path / "estaciones").mkdir()
    with pytest.raises(FileNotFoundError):
        td.latest_station_file(tmp_path)


# --- metrics -----------------------------------------------------------------


def test_haversine_one_hundredth_degree_of_latitude():
    assert td.haversine_km(20.0, -103.0, 20.01, -103.0) == pytest.approx(1.112, abs=1e-3)


def test_trip_metrics_values():
    metrics = td.trip_metrics(make_trips({"seconds": 360}), STATIONS)
    row = metrics.iloc[0]
    assert row["duration_s"] == 360
    assert row["distance_km"] == pytest.approx(1.0, abs=0.01)
    assert row["speed_kmh"] == pytest.approx(10.0, abs=0.1)
    assert row["age"] == 34


def test_trip_metrics_unknown_station_and_zero_duration_are_nan():
    metrics = td.trip_metrics(make_trips({"Destino_Id": 99}, {"seconds": 0}), STATIONS)
    assert pd.isna(metrics.loc[0, "distance_km"]) and pd.isna(metrics.loc[0, "speed_kmh"])
    assert pd.isna(metrics.loc[1, "speed_kmh"])


def test_trip_metrics_missing_birth_year_gives_nan_age():
    assert pd.isna(td.trip_metrics(make_trips({BY: None}), STATIONS).loc[0, "age"])

