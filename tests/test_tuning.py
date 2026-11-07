import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from trip_factory import STATIONS, make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026 import station_profiles as sp
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.modeling import tuning as tu

DAY0 = pd.Timestamp("2024-01-01")
TEST_START = "2024-06"
# A 3-month window (2024-03 to 2024-05) that validates on 2024-04 and 2024-05
FOLDS = tu.tuning_folds(TEST_START, search_months=3, validation_months=2, stopping_days=7)


@pytest.mark.parametrize(
    "test_start, search_months, validation_months, stopping_days",
    [("2025-10", 12, 3, 14), ("2024-06", 3, 2, 7), ("2025-01", 6, 5, 20)],
)
def test_tuning_folds_validate_after_their_fit_and_stop_hours(
    test_start, search_months, validation_months, stopping_days
):
    folds = tu.tuning_folds(test_start, search_months, validation_months, stopping_days)
    test = pd.Period(test_start, freq="M")
    assert [f.month for f in folds] == list(
        pd.period_range(test - validation_months, test - 1, freq="M")
    )
    hours = pd.DataFrame(
        {ft.TIME: pd.date_range((test - search_months - 1).start_time, test.end_time, freq="h")}
    )
    for fold in folds:
        assert fold.fit_start == (test - search_months).start_time
        fit, stop, val = (part[ft.TIME] for part in fold.split(hours))
        assert len(fit) and len(stop) and len(val)
        assert fold.fit_start <= fit.min() and fit.max() < stop.min()
        assert stop.max() < val.min() == fold.val_start
        assert stop.max() - stop.min() == pd.Timedelta(days=stopping_days, hours=-1)
        assert val.max() < fold.val_end <= test.start_time  # never a test hour


@pytest.mark.parametrize(
    "search_months, validation_months, stopping_days", [(3, 3, 7), (3, 0, 7), (2, 1, 40)]
)
def test_tuning_folds_without_fit_hours_are_rejected(
    search_months, validation_months, stopping_days
):
    with pytest.raises(ValueError, match="leave no fit hours"):
        tu.tuning_folds(TEST_START, search_months, validation_months, stopping_days)


@pytest.fixture
def features_dir(tmp_path):
    """Five months of random trips between stations 1 and 2, as features and weekly profiles
    fitted before ``TEST_START``."""
    rng = np.random.default_rng(ft.SEED)
    hours = DAY0 + pd.to_timedelta(rng.integers(0, 152 * 24, 4000), unit="h")
    origins = rng.choice([1, 2], len(hours), p=[0.6, 0.4])
    trips = make_trips(
        *(
            {"start": h, "Origen_Id": int(a), "Destino_Id": 3 - int(a)}
            for h, a in zip(hours, origins)
        )
    )
    processed, interim, dst = tmp_path / "processed", tmp_path / "interim", tmp_path / "features"
    trips_file = processed / "viajes" / "2024" / "viajes_2024_01.parquet"
    trips_file.parent.mkdir(parents=True)
    trips.to_parquet(trips_file, index=False)
    (interim / "estaciones").mkdir(parents=True)
    STATIONS.to_parquet(interim / "estaciones" / "nomenclatura_2024_01.parquet", index=False)
    ft.create_features(processed, interim, dst, start="2024-02")
    sp.create_profiles(dst, dst, fit_end=TEST_START, min_trips=1, score_min_trips=1)
    return dst


def test_folds_that_reach_the_test_split_are_rejected(features_dir):
    late = tu.tuning_folds("2024-07", search_months=3, validation_months=2, stopping_days=7)
    with pytest.raises(ValueError, match="validate on hours of the test split"):
        tu.tune(["lgbm_tweedie"], ["departures"], features_dir, TEST_START, 1, late)


def test_models_without_a_search_space_are_rejected(features_dir):
    with pytest.raises(ValueError, match="without a search space: seasonal_naive"):
        tu.tune(["seasonal_naive"], ["departures"], features_dir, TEST_START, 1, FOLDS)


def test_a_tiny_search_is_tracked_as_nested_runs(features_dir, mlflow_store):
    n_trials = 2
    results = tu.tune_tracked(
        tu.FINALISTS, ["arrivals"], features_dir, TEST_START, n_trials, FOLDS
    )
    client = MlflowClient()
    for result in results:
        spec = tu.MODELS[result.name]
        parent = mlflow.get_run(result.run_id)
        assert parent.info.run_name == f"tune-{result.name}-arrivals"
        assert parent.info.status == "FINISHED"
        assert parent.data.params["validation_months"] == "2024-04,2024-05"
        assert parent.data.params["n_trials"] == str(n_trials)

        trials = mlflow.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{result.run_id}'",
            output_format="list",
        )
        assert sorted(t.info.run_name for t in trials) == ["defaults", "trial-01", "trial-02"]
        objectives = {}
        for trial in trials:
            assert set(spec.search_space) <= set(trial.data.params)
            assert trial.data.metrics["iterations"] > 0
            history = client.get_metric_history(trial.info.run_id, f"fold_{tu.OBJECTIVE}")
            assert [m.step for m in history] == [0, 1]  # one step per fold
            assert trial.data.metrics[tu.OBJECTIVE] == pytest.approx(
                np.mean([m.value for m in history])
            )
            objectives[trial.info.run_name] = trial.data.metrics[tu.OBJECTIVE]

        # The parent holds the best trial, the defaults and the refitted model
        assert parent.data.metrics[tu.OBJECTIVE] == pytest.approx(min(objectives.values()))
        assert parent.data.metrics[f"default_{tu.OBJECTIVE}"] == objectives["defaults"]
        assert result.best.run_name == min(objectives, key=objectives.get)
        iterations = tu.ITERATIONS[type(result.estimator[-1])]
        assert parent.data.params[iterations] == str(result.best.iterations)
        assert parent.data.metrics["model_size_mb"] > 0

        model = mlflow.sklearn.load_model(f"runs:/{result.run_id}/{tr.MODEL_ARTIFACT}")
        rows = tr.load_features(features_dir, TEST_START, end=TEST_START)
        X = tr.model_inputs(rows)
        assert np.allclose(model.predict(X), result.estimator.predict(X))


def test_search_space_table_lists_every_param_with_its_range_and_default():
    table = tu.search_space_table(tu.MODELS["lgbm_tweedie"])
    assert list(table.index) == list(tu.MODELS["lgbm_tweedie"].search_space)
    assert table.loc["learning_rate"].tolist() == ["loguniform", 0.02, 0.3, 0.1]
    assert table.loc["n_estimators"].tolist() == ["randint", 100, 799, 100]
    assert table.loc["tweedie_variance_power", "default"] == 1.5
