import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import PoissonRegressor
from sklearn.utils.validation import check_is_fitted
from trip_factory import STATIONS, make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026 import station_profiles as sp
from team2_mlops2026.modeling import backtest as bt
from team2_mlops2026.modeling import evaluate as ev
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.modeling import tuning as tu
from team2_mlops2026.tracking import input_digests

DAY0 = pd.Timestamp("2024-01-01")
FIRST_TEST_MONTH = "2024-04"  # two monthly folds: 2024-04 and 2024-05
N_FOLDS = 2
NEW_SINCE = "2024-05"  # station 3 opens then


@pytest.fixture
def features_dir(tmp_path, monkeypatch):
    """Five months of random trips between stations 1 and 2, and from May 2024 a new station 3,
    as features and weekly profiles fitted before ``FIRST_TEST_MONTH``."""
    monkeypatch.setattr(bt, "NEW_SINCE", NEW_SINCE)
    rng = np.random.default_rng(ft.SEED)
    hours = DAY0 + pd.to_timedelta(rng.integers(0, 152 * 24, 4000), unit="h")
    origins = rng.choice([1, 2], len(hours), p=[0.6, 0.4])
    trips = [
        {"start": h, "Origen_Id": int(a), "Destino_Id": 3 - int(a)} for h, a in zip(hours, origins)
    ]
    opened = pd.Timestamp(NEW_SINCE) + pd.Timedelta(days=3, hours=9)
    trips += [
        {"start": opened + pd.Timedelta(days=d, hours=h), "Origen_Id": 3, "Destino_Id": 1}
        for d in range(25)
        for h in (0, 9)
    ]
    processed, interim, dst = tmp_path / "processed", tmp_path / "interim", tmp_path / "features"
    trips_file = processed / "viajes" / "2024" / "viajes_2024_01.parquet"
    trips_file.parent.mkdir(parents=True)
    make_trips(*trips).to_parquet(trips_file, index=False)
    (interim / "estaciones").mkdir(parents=True)
    STATIONS.to_parquet(interim / "estaciones" / "nomenclatura_2024_01.parquet", index=False)
    ft.create_features(processed, interim, dst, start="2024-02")
    sp.create_profiles(dst, dst, fit_end=FIRST_TEST_MONTH, min_trips=1, score_min_trips=1)
    return dst


class LastHourSpy(RegressorMixin, BaseEstimator):
    """Forecasts the training mean and records the last hour it was trained on."""

    def fit(self, X, y):
        self.last_hour_ = X[ft.TIME].max()
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_each_fold_refits_on_the_hours_before_its_month(features_dir):
    features = tr.load_features(features_dir, FIRST_TEST_MONTH)
    for fold in ev.rolling_origin_splits(features, FIRST_TEST_MONTH, N_FOLDS):
        templates = {t: LastHourSpy() for t in ft.TARGETS}
        result, forecasts, fitted, _ = bt.run_fold(fold, features, templates)
        for target, model in fitted.items():
            assert model.last_hour_ == fold.test_start - pd.Timedelta(hours=1)
            assert templates[target] is not model  # the template itself stays unfitted
        assert forecasts[ft.TIME].min() == fold.test_start
        assert forecasts[ft.TIME].max() < fold.test_end
        assert result.month == str(fold.month)
        assert result.train_rows == (features[ft.TIME] < fold.test_start).sum()
        assert set(result.imbalance) == {
            "imbalance_days",
            "flagged_days",
            "imbalance_precision",
            "imbalance_recall",
        }


def test_backtest_scores_every_fold_and_the_new_stations(features_dir):
    (result,) = bt.backtest(
        ["seasonal_naive"], src=features_dir, first_test_month=FIRST_TEST_MONTH, n_folds=N_FOLDS
    )
    assert result.sources == {"departures": bt.DEFAULTS, "arrivals": bt.DEFAULTS}
    assert [f.month for f in result.folds] == ["2024-04", "2024-05"]
    april, may = (f.reports["departures"] for f in result.folds)
    assert april.loc[bt.NEW_STATIONS, "n"] == 0  # station 3 opens in May
    assert may.loc[bt.NEW_STATIONS, "n"] > 0
    assert result.importance == {}  # only the ML pipelines get a permutation importance

    mean = result.mean_report("departures")
    assert mean.loc["peak", "wape"] == pytest.approx(
        np.mean([april.loc["peak", "wape"], may.loc["peak", "wape"]])
    )
    assert mean.loc[bt.NEW_STATIONS, "wape"] == pytest.approx(may.loc[bt.NEW_STATIONS, "wape"])

    forecasts = result.forecasts
    assert len(forecasts) == sum(f.reports["departures"].loc["overall", "n"] for f in result.folds)
    assert set(forecasts[bt.FOLD]) == {"2024-04", "2024-05"}
    assert (forecasts["departures_forecast"] >= 0).all()
    by_hour = result.error_by("hour")
    assert by_hour.loc["departures", "n"].sum() == len(forecasts)
    by_fold = result.error_by(bt.FOLD)
    assert by_fold.loc[("departures", "2024-05"), "wape"] == pytest.approx(
        may.loc["overall", "wape"]
    )
    assert result.fold_table().groupby(["fold", "target"]).size().nunique() == 1


def test_backtest_tracked_logs_a_parent_run_with_one_nested_run_per_fold(
    features_dir, mlflow_store
):
    models = ["historical_average", "lgbm_tweedie"]
    results = bt.backtest_tracked(
        models, src=features_dir, first_test_month=FIRST_TEST_MONTH, n_folds=N_FOLDS, tuned=False
    )
    client = MlflowClient()
    for result in results:
        parent = mlflow.get_run(result.run_id)
        assert parent.info.run_name == f"backtest-{result.name}"
        assert parent.info.status == "FINISHED"
        assert parent.data.tags["params_from_departures"] == bt.DEFAULTS
        assert len(parent.data.tags["git_commit"]) == 40
        assert parent.data.params["n_folds"] == str(N_FOLDS)
        assert parent.data.params["first_test_month"] == FIRST_TEST_MONTH
        assert set(input_digests(result.run_id)) == {
            "mibici_station_hour_features",
            "mibici_station_profiles",
        }

        folds = mlflow.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{result.run_id}'",
            output_format="list",
        )
        assert sorted(f.info.run_name for f in folds) == ["fold-2024-04", "fold-2024-05"]
        peak = [f.data.metrics["departures_peak_wape"] for f in folds]
        assert parent.data.metrics["departures_peak_wape"] == pytest.approx(np.mean(peak))
        assert parent.data.metrics["departures_peak_wape_std"] == pytest.approx(
            np.std(peak, ddof=1)
        )
        assert "imbalance_precision" in parent.data.metrics
        history = client.get_metric_history(result.run_id, "fold_departures_peak_wape")
        assert [m.step for m in history] == [0, 1]

        artifacts = {a.path for a in client.list_artifacts(result.run_id, bt.ARTIFACT_DIR)}
        expected = {
            bt.FOLDS_FILE,
            bt.BY_HOUR_FILE,
            bt.IMBALANCE_FILE,
            bt.FORECASTS_FILE,
            "error_by_month.png",
            "error_by_hour.png",
            "residuals.png",
            "forecasts.png",
        }
        if result.name == "lgbm_tweedie":
            expected |= {f"importance_{t}.{ext}" for t in ft.TARGETS for ext in ("csv", "png")}
        assert {f"{bt.ARTIFACT_DIR}/{name}" for name in expected} == artifacts

    # The ranking reads the fold tables back from MLflow
    loaded = bt.load_backtests(models)
    assert {name: run.info.run_id for name, (run, _) in loaded.items()} == {
        r.name: r.run_id for r in results
    }
    ranking = bt.rank_backtests({name: table for name, (_, table) in loaded.items()})
    for target, table in ranking.items():
        assert list(table.index) == ["lgbm_tweedie"]
        assert table.loc["lgbm_tweedie", "baseline"] == "historical_average"
        mean = results[1].mean_report(target)
        assert table.loc["lgbm_tweedie", "peak_wape"] == pytest.approx(mean.loc["peak", "wape"])


@pytest.fixture
def tuned_departures(features_dir, mlflow_store):
    """A tiny tracked search of lgbm_tweedie for departures, refitted before ``FIRST_TEST_MONTH``."""
    folds = tu.tuning_folds(
        FIRST_TEST_MONTH, search_months=2, validation_months=1, stopping_days=7
    )
    (result,) = tu.tune_tracked(
        ["lgbm_tweedie"], ["departures"], features_dir, FIRST_TEST_MONTH, 1, folds
    )
    return result


def test_finalists_take_the_params_of_their_latest_tune_run(tuned_departures):
    template, source = bt.model_template("lgbm_tweedie", "departures")
    assert source == tuned_departures.run_id
    with pytest.raises(NotFittedError):
        check_is_fitted(template[-1])
    tuned = tr.model_params(tuned_departures.estimator)
    assert tr.model_params(template) == tuned
    # Without tuning, or without a tune run, the catalog defaults are not silently used
    _, source = bt.model_template("lgbm_tweedie", "departures", tuned=False)
    assert source == bt.DEFAULTS
    with pytest.raises(LookupError, match="tune-lgbm_tweedie-arrivals"):
        bt.model_template("lgbm_tweedie", "arrivals")


def test_ablation_scores_the_model_with_and_without_the_profile_scores(
    features_dir, tuned_departures
):
    with_pca, without = bt.ablation_tracked(
        ["lgbm_tweedie"], ["departures"], features_dir, FIRST_TEST_MONTH
    )
    assert bt.profile_columns(pd.DataFrame(columns=list(with_pca.inputs)))  # pc1...
    assert not bt.profile_columns(pd.DataFrame(columns=list(without.inputs)))
    assert set(with_pca.inputs) - set(without.inputs) == {
        c for c in with_pca.inputs if c.startswith("pc")
    }
    # The variant with the scores is the tune run's model, so only the other one is fitted
    assert np.isnan(with_pca.fit_seconds) and without.fit_seconds > 0
    assert with_pca.source == without.source == tuned_departures.run_id

    for result in (with_pca, without):
        run = mlflow.get_run(result.run_id)
        assert run.info.run_name == f"ablation-lgbm_tweedie-departures-{result.variant}"
        assert run.data.tags["experiment"] == "pca_ablation"
        assert run.data.metrics["peak_wape"] == pytest.approx(result.report.loc["peak", "wape"])
    assert mlflow.get_run(without.run_id).data.params["profiles"] == "none"

    with pytest.raises(ValueError, match="fitted before 2024-04"):
        bt.ablation(["lgbm_tweedie"], ["departures"], features_dir, "2024-05")


def test_permutation_importance_ranks_the_informative_column_first():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"noise": rng.normal(size=2000), "signal": rng.normal(size=2000)})
    y = rng.poisson(np.exp(0.8 * X["signal"]))
    model = PoissonRegressor().fit(X, y)
    importance = bt.permutation_importances(model, X, y, n_rows=1000, n_repeats=3)
    assert list(importance.index) == ["signal", "noise"]
    assert importance.index.name == "feature"
    assert importance.loc["signal", "mean"] > 0.1 > abs(importance.loc["noise", "mean"])
