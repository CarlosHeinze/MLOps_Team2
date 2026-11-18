"""The pipeline end to end on synthetic data: raw CSVs -> normalization -> cleaning -> features ->
station profiles -> training -> Model Registry -> batch forecast -> metrics.

Each step runs through its tracked entry point into the throwaway MLflow store of
``mlflow_store``, and every boundary checks the columns the next step reads. The feature and
forecast columns are written out literally: training, the logged model signature and the
forecast file rely on them, so renaming one must fail here even if every constant follows.
"""

from dataclasses import replace
from itertools import product

import mlflow
from mlflow.models import get_model_info
import numpy as np
import pandas as pd
import pytest
from trip_factory import make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026 import station_profiles as sp
from team2_mlops2026.data import interim_data_cleaning as idc
from team2_mlops2026.data import raw_data_normalization as rdn
from team2_mlops2026.data.trip_data import load_trips
from team2_mlops2026.modeling import evaluate as ev
from team2_mlops2026.modeling import predict as pr
from team2_mlops2026.modeling import registry as reg
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.modeling.models import MODELS
from team2_mlops2026.tracking import input_digests

pytestmark = pytest.mark.integration

DAY0 = pd.Timestamp("2024-01-01")  # a Monday
END = pd.Timestamp("2024-04-01")  # three months of trips
FEATURES_START = "2024-01-29"  # after the four weeks of warm-up the lags need
TEST_START = "2024-03"  # the held-out month, also the profiles' fit end
NEW_STATION_OPENS = pd.Timestamp("2024-03-11")

HOMES, OFFICES, NEW_STATION = (1, 2), (3, 4), 5  # station 5 is not in the nomenclature
NOMENCLATURE = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "name": ["Home A", "Home B", "Office A", "Office B"],
        "latitude": [20.0, 20.009, 20.0, 20.009],
        "longitude": [-103.0, -103.0, -103.009, -103.009],
        "dpcapacity": [12, 10, 20, 16],
    }
)
BASELINE, CANDIDATE = "seasonal_naive", "lgbm_tweedie"

# The contract of each table, as the next step reads it
FEATURE_COLUMNS = [
    "station_id",
    "hour_start",
    "departures",
    "arrivals",
    "departures_lag_2d",
    "departures_lag_7d",
    "departures_same_hour_4w",
    "departures_mean_7d",
    "arrivals_lag_2d",
    "arrivals_lag_7d",
    "arrivals_same_hour_4w",
    "arrivals_mean_7d",
    "weekday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "latitude",
    "longitude",
    "capacity",
]
MODEL_INPUTS = [c for c in FEATURE_COLUMNS if c not in ("departures", "arrivals")]
FORECAST_COLUMNS = ["station_id", "hour_start", "departures", "arrivals", "net_flow"]


def synthetic_trips(rng: np.random.Generator) -> list[dict]:
    """Weekday commutes from the homes to the offices at 07:00-09:00 and back at 17:00-19:00,
    plus random daytime trips between any two stations; station 5 joins from
    ``NEW_STATION_OPENS``, so it is unseen in training."""
    rows = []
    for day in pd.date_range(DAY0, END, freq="D", inclusive="left"):
        stations = [*HOMES, *OFFICES, *([NEW_STATION] if day >= NEW_STATION_OPENS else [])]
        workday = day.day_of_week < 5
        for hour in range(6, 23):
            if workday and hour in (7, 8, 9):
                pairs = [p for p in product(HOMES, OFFICES) for _ in range(rng.poisson(2))]
            elif workday and hour in (17, 18, 19):
                pairs = [p for p in product(OFFICES, HOMES) for _ in range(rng.poisson(2))]
            else:
                pairs = [rng.choice(stations, 2, replace=False) for _ in range(rng.poisson(1.5))]
            rows += [
                {
                    "start": day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 45))),
                    "Origen_Id": int(origin),
                    "Destino_Id": int(destination),
                }
                for origin, destination in pairs
            ]
    return rows


def write_raw_csvs(raw_dir, trips: pd.DataFrame) -> None:
    """One CSV per month and the station list, named and formatted as MiBici publishes them."""
    raw_dir.mkdir()
    months = trips["Inicio_del_viaje"].dt.to_period("M")
    for month, rows in trips.groupby(months):
        rows.to_csv(
            raw_dir / f"datos_abiertos_{month.year}_{month.month:02d}.csv",
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
    NOMENCLATURE.to_csv(raw_dir / "nomenclatura_2024_03.csv", index=False)


def test_pipeline_runs_from_raw_csvs_to_scored_forecasts(tmp_path, mlflow_store, monkeypatch):
    # A small candidate, so that it trains in a fraction of a second
    spec = MODELS[CANDIDATE]
    monkeypatch.setitem(
        MODELS, CANDIDATE, replace(spec, params={**spec.params, "n_estimators": 20})
    )
    raw, interim, processed = tmp_path / "raw", tmp_path / "interim", tmp_path / "processed"
    features_dir = tmp_path / "features"

    # --- load: raw CSVs with one duplicate row and one trip under a minute --------------
    valid = make_trips(*synthetic_trips(np.random.default_rng(ft.SEED)))
    too_short = make_trips({"start": DAY0 + pd.Timedelta(hours=12), "seconds": 30})
    write_raw_csvs(raw, pd.concat([valid, valid.head(1), too_short], ignore_index=True))

    normalized = rdn.normalize_tracked(raw, interim, workers=1)
    assert normalized["failed"] == [] and len(normalized["normalized"]) == 4
    assert tuple(load_trips(interim).columns) == rdn.TRIP_COLUMNS  # what cleaning validates

    # --- preprocess: cleaning -> features -> station profiles ---------------------------
    cleaned = idc.clean_tracked(interim, processed, workers=1)
    assert cleaned["failed"] == []
    trips = load_trips(processed, ft.TRIP_COLUMNS)  # the columns the features step reads
    assert len(trips) == len(valid)  # the duplicate and the short trip are gone

    built = ft.create_features_tracked(processed, interim, features_dir, start=FEATURES_START)
    table = pd.read_parquet(features_dir / ft.FEATURES_FILE)
    assert list(table.columns) == FEATURE_COLUMNS
    assert table.notna().any().all()  # a column built under another name would be all NaN
    assert not table.duplicated(["station_id", "hour_start"]).any()
    gaps = table.groupby("station_id")["hour_start"].diff().dropna()
    assert (gaps == pd.Timedelta(hours=1)).all()  # every hour of an active station
    assert set(table["station_id"]) == {*HOMES, *OFFICES, NEW_STATION}
    assert table["departures"].sum() == (trips["Inicio_del_viaje"] >= FEATURES_START).sum()

    profiled = sp.create_profiles_tracked(
        features_dir, features_dir, fit_end=TEST_START, min_trips=1, score_min_trips=1
    )
    scores = pd.read_parquet(features_dir / sp.PROFILES_FILE)
    components = [f"pc{i}" for i in range(1, profiled["components"] + 1)]
    assert list(scores.columns) == ["station_id", "week_start", *components]
    assert scores.attrs[sp.FIT_END_KEY] == TEST_START
    profile_inputs = components[: tr.PROFILE_COMPONENTS]

    # --- train: one baseline and one candidate per target -------------------------------
    results = tr.train_tracked([BASELINE, CANDIDATE], ft.TARGETS, features_dir, TEST_START)
    trained = {(r.name, r.target): r for r in results}
    assert set(trained) == set(product([BASELINE, CANDIDATE], ft.TARGETS))
    upstream = {
        **input_digests(built["mlflow_run_id"]),
        **input_digests(profiled["mlflow_run_id"]),
    }
    for result in results:
        assert input_digests(result.run_id).items() <= upstream.items()  # the tables just built
        signature = get_model_info(f"runs:/{result.run_id}/{tr.MODEL_ARTIFACT}").signature
        assert signature.inputs.input_names() == [*MODEL_INPUTS, *profile_inputs]

    # --- register: the candidate as champion of each target -----------------------------
    versions = {}
    for target in ft.TARGETS:
        run = mlflow.get_run(trained[CANDIDATE, target].run_id)
        versions[target] = reg.register(
            reg.Entry(CANDIDATE, target, reg.CHAMPION, run, run, run, None)
        )
        assert versions[target].run_id == run.info.run_id
        assert versions[target].tags["features_digest"] == upstream["mibici_station_hour_features"]

    # --- predict: the 24 h after the data, with the models loaded by alias --------------
    result = pr.predict_tracked(src=features_dir, dst=tmp_path / "predictions")
    forecast = result.predictions
    assert list(forecast.columns) == FORECAST_COLUMNS
    assert {t: v.version for t, v in result.versions.items()} == {
        t: v.version for t, v in versions.items()
    }
    hours = pd.date_range(table["hour_start"].max() + pd.Timedelta(hours=1), periods=24, freq="h")
    stations = {*HOMES, *OFFICES, NEW_STATION}  # each had a trip in the last 28 days
    expected = pd.MultiIndex.from_product([sorted(stations), hours])
    assert pd.MultiIndex.from_frame(forecast[["station_id", "hour_start"]]).equals(expected)
    demand = forecast[["departures", "arrivals"]].to_numpy()
    assert np.isfinite(demand).all() and (demand >= 0).all()
    assert np.allclose(forecast["net_flow"], forecast["arrivals"] - forecast["departures"])

    # --- metrics: the champions on the held-out month, against the baseline -------------
    train_rows, test_rows = tr.load_training_data(features_dir, TEST_START)
    assert test_rows["hour_start"].dt.to_period("M").astype(str).eq(TEST_START).all()
    inputs = tr.model_inputs(test_rows)
    for target in ft.TARGETS:
        champion, _ = reg.load_model(target)
        y_pred = champion.predict(inputs)
        assert np.allclose(y_pred, trained[CANDIDATE, target].estimator.predict(inputs))

        report = ev.segment_report(test_rows, y_pred, target, train_rows)
        scored = report.loc[["overall", "peak", "unseen_stations"], list(ev.METRICS)]
        assert np.isfinite(scored.to_numpy()).all()
        logged = mlflow.get_run(versions[target].run_id).data.metrics
        assert logged["overall_wape"] == pytest.approx(report.loc["overall", "wape"])

        check = ev.success_criterion(report, {BASELINE: trained[BASELINE, target].report})
        assert check.baseline == BASELINE and np.isfinite(check.peak_wape_improvement)
