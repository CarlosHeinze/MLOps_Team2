import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from test_backtest import FIRST_TEST_MONTH, N_FOLDS, NEW_SINCE, write_features

from team2_mlops2026 import features as ft
from team2_mlops2026 import tracking
from team2_mlops2026.modeling import backtest as bt
from team2_mlops2026.modeling import predict as pr
from team2_mlops2026.modeling import registry as reg
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.modeling import tuning as tu
from team2_mlops2026.modeling.models import BASELINES
from team2_mlops2026.tracking import input_digests, latest_run

MODELS = ("seasonal_naive", "historical_average", "lgbm_tweedie")


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """lgbm_tweedie tuned, ablated and backtested next to the baselines for both targets on the
    synthetic features of ``test_backtest``, and registered as the champion of each target.

    Built once for the module, in a throwaway SQLite store (as ``mlflow_store``), since the
    chain of runs takes a while."""
    tmp_path = tmp_path_factory.mktemp("registry")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
        )
        mp.delenv("MLFLOW_RUN_ID", raising=False)
        mp.chdir(tmp_path)
        mp.setattr(bt, "NEW_SINCE", NEW_SINCE)
        src = write_features(tmp_path)
        folds = tu.tuning_folds(
            FIRST_TEST_MONTH, search_months=2, validation_months=1, stopping_days=7
        )
        tu.tune_tracked(["lgbm_tweedie"], ft.TARGETS, src, FIRST_TEST_MONTH, 1, folds)
        bt.ablation_tracked(["lgbm_tweedie"], ft.TARGETS, src, FIRST_TEST_MONTH)
        bt.backtest_tracked(MODELS, ft.TARGETS, src, FIRST_TEST_MONTH, N_FOLDS)
        tr.train_tracked(BASELINES, ft.TARGETS, src, FIRST_TEST_MONTH)
        versions = reg.register_champions(ft.TARGETS, champion="lgbm_tweedie", models=MODELS)
        yield {"src": src, "versions": versions, "tmp": tmp_path}
        mlflow.end_run()


# --- registry ------------------------------------------------------------------


def test_pick_promotes_the_first_candidate_that_passed():
    ranking = pd.DataFrame(
        {"passed": [True, True, False], "baseline": "seasonal_naive"}, index=["a", "b", "c"]
    )
    assert reg.pick(ranking) == ("a", "b")
    assert reg.pick(ranking, champion="b") == ("b", "a")
    assert reg.pick(ranking.iloc[[0, 2]]) == ("a", "seasonal_naive")  # no runner-up passed
    with pytest.raises(ValueError, match="No candidate passed"):
        reg.pick(ranking.iloc[[2]])
    with pytest.raises(ValueError, match="not a ranked candidate"):
        reg.pick(ranking, champion="z")


def test_champions_are_registered_from_their_runs_with_metadata_and_aliases(registered):
    client = MlflowClient()
    for target in ft.TARGETS:
        name = reg.model_name(target)
        champion = registered["versions"][target][reg.CHAMPION]
        tune = latest_run(f"tune-lgbm_tweedie-{target}")
        assert champion.run_id == tune.info.run_id
        assert client.get_model_version_by_alias(name, reg.CHAMPION).version == champion.version

        tags = champion.tags
        assert tags["model"] == "lgbm_tweedie" and tags["algorithm"] == "LGBMRegressor"
        assert tags["target"] == target
        assert tags["training_window"].endswith(f"to {FIRST_TEST_MONTH} (exclusive)")
        ablation = latest_run(f"ablation-lgbm_tweedie-{target}-pca")
        assert tags["test_metrics_run"] == ablation.info.run_id
        assert float(tags["test_overall_wape"]) == pytest.approx(
            ablation.data.metrics["overall_wape"], abs=1e-4
        )
        mean = latest_run("backtest-lgbm_tweedie").data.metrics[f"{target}_peak_wape"]
        assert tags["backtest_peak_wape"].startswith(f"{mean:.4f} ± ")
        assert tags["backtest_folds"] == str(N_FOLDS)
        digests = input_digests(tune.info.run_id)
        assert tags["features_digest"] == digests["mibici_station_hour_features"]
        assert tags["profiles_digest"] == digests["mibici_station_profiles"]
        assert len(tags["git_commit"]) == 40
        assert tags["success_criterion"] in {"passed", "failed"}
        assert champion.description.startswith("lgbm_tweedie (LGBMRegressor) forecast")

        challenger = registered["versions"][target][reg.CHALLENGER]
        baseline = challenger.tags["model"]
        assert baseline in BASELINES  # no other candidate was ranked
        assert challenger.run_id == latest_run(f"train-{baseline}-{target}").info.run_id
        assert "success_criterion" not in challenger.tags

    # Registering again keeps the versions and only moves the aliases
    again = reg.register_champions(ft.TARGETS, champion="lgbm_tweedie", models=MODELS)
    for target in ft.TARGETS:
        assert {a: v.version for a, v in again[target].items()} == {
            a: v.version for a, v in registered["versions"][target].items()
        }
        assert len(client.search_model_versions(f"name = '{reg.model_name(target)}'")) == 2


def test_the_champion_loads_by_alias_and_forecasts_like_its_run(registered):
    _, test = tr.load_training_data(registered["src"], FIRST_TEST_MONTH)
    sample = tr.model_inputs(test.head(500))
    model, version = reg.load_model("departures")
    assert version.version == registered["versions"]["departures"][reg.CHAMPION].version
    tuned = mlflow.sklearn.load_model(f"runs:/{version.run_id}/{tr.MODEL_ARTIFACT}")
    assert np.allclose(model.predict(sample), tuned.predict(sample))


# --- batch forecast --------------------------------------------------------------


def test_predict_forecasts_the_next_day_of_every_active_station(registered):
    src = registered["src"]
    result = pr.predict_tracked(src=src, dst=registered["tmp"] / "predictions")
    assert result.start == pr.last_hour(src) + pd.Timedelta(hours=1)

    predictions = result.predictions
    assert list(predictions.columns) == [ft.STATION, ft.TIME, *ft.TARGETS, pr.NET_FLOW]
    assert set(predictions[ft.STATION]) == {1, 2, 3}
    assert len(predictions) == 3 * ft.FORECAST_HOURS
    assert (predictions[list(ft.TARGETS)] >= 0).all().all()
    assert np.allclose(
        predictions[pr.NET_FLOW], predictions["arrivals"] - predictions["departures"]
    )
    pd.testing.assert_frame_equal(pd.read_parquet(result.path), predictions)

    run = mlflow.get_run(result.run_id)
    assert run.info.run_name == f"predict-{result.start:%Y-%m-%d-%H}h"
    for target in ft.TARGETS:
        champion = registered["versions"][target][reg.CHAMPION]
        assert run.data.params[f"{target}_model"] == f"{champion.name}/{champion.version}"
        assert run.data.params[f"{target}_model_run_id"] == champion.run_id
    digests = input_digests(result.run_id)
    assert set(digests) == {
        "mibici_station_hour_features",
        "mibici_station_profiles",
        pr.DATASET,
    }
    assert digests[pr.DATASET] == tracking.files_digest([result.path], result.path.parent)
    assert run.data.metrics["rows"] == len(predictions)


def test_a_past_start_replays_the_forecast_from_the_rows_of_the_table(registered):
    src = registered["src"]
    start = pd.Timestamp("2024-05-10")
    result = pr.predict(start, src=src, dst=None)
    assert result.path is None
    # The rows built from the history before start are the table's rows of that day
    day = tr.load_features(src, FIRST_TEST_MONTH, start=str(start), end="2024-05-11")
    model, _ = reg.load_model("arrivals")
    expected = day[[ft.STATION, ft.TIME]].assign(
        expected=np.clip(model.predict(tr.model_inputs(day)), 0, None)
    )
    replay = result.predictions.merge(expected, on=[ft.STATION, ft.TIME])
    assert len(replay) == len(day) == len(result.predictions)
    assert np.allclose(replay["arrivals"], replay["expected"])
