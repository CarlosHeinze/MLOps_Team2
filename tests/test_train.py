import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from trip_factory import STATIONS, make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026 import station_profiles as sp
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.tracking import input_digests

DAY0 = pd.Timestamp("2024-01-01")  # a Monday
START = DAY0 + pd.Timedelta(weeks=4)  # first hour of the table
TEST_START = "2024-02-26"  # four weeks of training hours
NEW_STATION_OPENS = pd.Timestamp(TEST_START) + pd.Timedelta(days=2, hours=12)


@pytest.fixture
def lineage(tmp_path, mlflow_store):
    """Tracked features and profiles from 9 weeks of commutes 1 -> 2 at 08:00 and back at 18:00;
    station 3 opens in the test period with a daily 12:00 trip 3 -> 1."""
    commutes = [
        {"start": DAY0 + pd.Timedelta(days=d, hours=h), "Origen_Id": a, "Destino_Id": b}
        for d in range(63)
        for h, a, b in [(8, 1, 2), (18, 2, 1)]
    ]
    new = [
        {"start": NEW_STATION_OPENS + pd.Timedelta(days=d), "Origen_Id": 3, "Destino_Id": 1}
        for d in range(4)
    ]
    processed, interim, dst = tmp_path / "processed", tmp_path / "interim", tmp_path / "features"
    trips_file = processed / "viajes" / "2024" / "viajes_2024_01.parquet"
    trips_file.parent.mkdir(parents=True)
    make_trips(*commutes, *new).to_parquet(trips_file, index=False)
    (interim / "estaciones").mkdir(parents=True)
    STATIONS.to_parquet(interim / "estaciones" / "nomenclatura_2024_01.parquet", index=False)

    features = ft.create_features_tracked(processed, interim, dst, start=str(START.date()))
    profiles = sp.create_profiles_tracked(dst, dst, fit_end=TEST_START, min_trips=1)
    return {"dir": dst, "features": features, "profiles": profiles}


def test_training_data_joins_the_profiles_and_splits_in_time(lineage):
    train, test = tr.load_training_data(lineage["dir"], TEST_START, TEST_START)
    assert train[ft.TIME].max() < pd.Timestamp(TEST_START) == test[ft.TIME].min()
    assert "pc1" in train.columns and train["pc1"].notna().all()
    new = test[test[ft.STATION] == 3]
    assert len(new) and new["pc1"].isna().all()  # opened after the profiles were fitted
    assert not set(ft.TARGETS) & set(tr.model_inputs(test).columns)


def test_a_test_split_before_the_profiles_fit_end_is_rejected(lineage):
    with pytest.raises(ValueError, match="before the profiles' fit end 2024-02-26"):
        tr.load_training_data(lineage["dir"], "2024-02-19", TEST_START)


def test_report_metrics_are_flat_and_defined():
    report = pd.DataFrame(
        {"n": [10, 0], "wape": [0.5, np.nan]}, index=pd.Index(["overall", "unseen_stations"])
    )
    assert tr.report_metrics(report) == {
        "overall_rows": 10,
        "overall_wape": 0.5,
        "unseen_stations_rows": 0,
    }


def test_unknown_models_and_targets_are_rejected(lineage):
    with pytest.raises(ValueError, match="Unknown models or targets: prophet, bikes"):
        tr.train(["prophet"], ["bikes"], lineage["dir"], TEST_START, TEST_START)


def test_train_tracked_logs_each_baseline_and_target_with_its_lineage(lineage):
    results = tr.train_tracked(
        src=lineage["dir"], test_start=TEST_START, profiles_fit_end=TEST_START
    )
    assert {(r.name, r.target) for r in results} == {
        (m, t) for m in tr.BASELINES for t in ft.TARGETS
    }

    upstream = {
        **input_digests(lineage["features"]["mlflow_run_id"]),
        **input_digests(lineage["profiles"]["mlflow_run_id"]),
    }
    _, test = tr.load_training_data(lineage["dir"], TEST_START, TEST_START)
    for result in results:
        run = mlflow.get_run(result.run_id)
        assert run.info.run_name == f"train-{result.name}-{result.target}"
        assert run.info.status == "FINISHED"
        assert run.data.tags["model_family"] == "baseline"
        assert run.data.tags["target"] == result.target
        assert len(run.data.tags["git_commit"]) == 40
        assert run.data.params["test_start"] == TEST_START
        for metric in ("overall_wape", "overall_mae", "peak_wape", "peak_mae"):
            assert run.data.metrics[metric] == pytest.approx(
                result.report.loc[metric.split("_")[0], metric.split("_")[1]]
            )
        # The inputs are the exact tables the features and profiles steps wrote
        logged = input_digests(result.run_id)
        assert set(logged) == {"mibici_station_hour_features", "mibici_station_profiles"}
        assert logged == {name: upstream[name] for name in logged}

        model = mlflow.sklearn.load_model(f"runs:/{result.run_id}/{tr.MODEL_ARTIFACT}")
        X = tr.model_inputs(test)
        assert np.allclose(model.predict(X), result.estimator.predict(X))
