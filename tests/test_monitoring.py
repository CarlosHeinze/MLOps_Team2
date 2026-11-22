import json

import mlflow
from mlflow import MlflowClient
from monitoring_factory import REFERENCE_MONTH, REPLAY_MONTH, SIM_MONTH, write_fixture
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from team2_mlops2026 import features as ft
from team2_mlops2026 import monitoring as mon
from team2_mlops2026.modeling.backtest import NEW_STATIONS
from team2_mlops2026.serving.app import load_models


@pytest.fixture
def rows():
    times = pd.Series(pd.to_datetime(["2026-04-01 08:00", "2026-04-01 14:00"]))
    return pd.DataFrame(
        {
            ft.STATION: [1, 1, 2, 2],
            ft.TIME: list(times) * 2,
            **{target: [8, 8, 12, 12] for target in ft.TARGETS},
            **{column: [8.0, 8.0, 12.0, 12.0] for column in ft.DEMAND_FEATURES},
            **{column: [1.0] * 4 for column in ft.STATION_ATTRS},
            "pc1": [1.0] * 4,
        }
    )


@pytest.mark.parametrize("scenario", [mon.Control, mon.RealDrift, *mon.SIMULATIONS.values()])
def test_zero_severity_preserves_values_index_and_types(rows, scenario):
    pd.testing.assert_frame_equal(scenario(0).apply(rows, np.random.default_rng(42)), rows)


def test_mean_and_covariate_shifts_perturb_only_intended_columns(rows):
    covariates = mon.CovariateShift(0.25).apply(rows, np.random.default_rng(42))
    mean = mon.MeanShift(0.25).apply(rows, np.random.default_rng(42))
    pd.testing.assert_frame_equal(
        covariates.drop(columns=list(ft.DEMAND_FEATURES)),
        rows.drop(columns=list(ft.DEMAND_FEATURES)),
    )
    pd.testing.assert_frame_equal(
        mean.drop(columns=list(ft.TARGETS)), covariates.drop(columns=list(ft.TARGETS))
    )
    assert covariates["departures_lag_7d"].tolist() == [10, 10, 15, 15]
    assert mean["departures"].tolist() == [10, 10, 15, 15]


def test_missing_features_are_reproducible_and_do_not_touch_actuals(rows):
    scenario = mon.MissingFeatures(0.5)
    expected = scenario.apply(rows, np.random.default_rng(42))
    pd.testing.assert_frame_equal(expected, scenario.apply(rows, np.random.default_rng(42)))
    assert expected["pc1"].isna().sum() == 2
    assert expected["capacity"].isna().sum() == 2
    assert expected["departures_lag_7d"].isna().sum() == 2
    pd.testing.assert_frame_equal(
        expected[[ft.STATION, ft.TIME, *ft.TARGETS]], rows[[ft.STATION, ft.TIME, *ft.TARGETS]]
    )
    assert rows.notna().all().all()


def test_holiday_pattern_reduces_only_commute_actuals_and_lag_matched_times(rows):
    shifted = mon.SeasonalShift(0.5).apply(rows, np.random.default_rng(42))
    assert shifted["departures"].tolist() == [4, 8, 6, 12]
    assert shifted["arrivals_lag_7d"].tolist() == [4, 8, 6, 12]
    assert shifted["arrivals_lag_2d"].tolist() == [4, 8, 6, 12]
    assert shifted["arrivals_mean_7d"].tolist() == pytest.approx(
        rows["arrivals_mean_7d"] * (1 - 0.5 * 30 / 168)
    )
    pd.testing.assert_frame_equal(
        shifted[[ft.STATION, ft.TIME, *ft.STATION_ATTRS]],
        rows[[ft.STATION, ft.TIME, *ft.STATION_ATTRS]],
    )


def test_new_stations_keep_actuals_and_attributes_and_clear_history(rows):
    shifted = mon.NewStations(0.5, first_unused=3).apply(rows, np.random.default_rng(42))
    cold = shifted[ft.STATION] == 3
    assert cold.sum() == 2
    assert shifted.loc[cold, [*ft.DEMAND_FEATURES, "pc1"]].isna().all().all()
    assert shifted.loc[~cold, list(ft.DEMAND_FEATURES)].notna().all().all()
    pd.testing.assert_frame_equal(
        shifted[[ft.TIME, *ft.TARGETS, *ft.STATION_ATTRS]],
        rows[[ft.TIME, *ft.TARGETS, *ft.STATION_ATTRS]],
    )


def test_psi_ks_missing_and_categorical_support():
    a = pd.Series(np.arange(100))
    same = mon.distribution(a, a)
    assert (same["psi"], same["ks"], same["ks_pvalue"]) == (0, 0, 1)
    shifted = mon.distribution(a, a + 100)
    assert shifted["psi"] >= mon.PSI_ALERT
    assert shifted["ks"] == 1
    assert shifted["ks_pvalue"] < 0.01
    for offset in (-1, 1):
        assert (
            mon.distribution(pd.Series([10] * 10), pd.Series([10 + offset] * 10))["psi"]
            > mon.PSI_ALERT
        )
    missing = mon.distribution(a, a.mask(a < 50))
    assert missing["missing_increase"] == 0.5
    assert mon.distribution(a, a.mask(a < 10))["missing_increase"] == mon.MISSING_INCREASE
    assert mon.distribution(a.mask(a >= 0), a.mask(a >= 0))["psi"] == 0
    categoric = mon.distribution(pd.Series([1, 2]), pd.Series([2, 3]), categorical=True)
    assert categoric["psi"] >= mon.PSI_ALERT and np.isnan(categoric["ks"])


def test_alert_boundaries_and_independent_loss():
    limits = mon.DEFAULT_THRESHOLDS
    stats = pd.DataFrame(
        {"psi": [limits.psi_warning], "missing_increase": [0.0]}, index=["capacity"]
    )
    best = {
        "departures": {"peak_wape": limits.peak_departures, "improvement": limits.min_improvement}
    }
    assert mon.assess(stats, best, best, 0, limits)["level"] == "warning"
    stats.loc["capacity", "psi"] = limits.psi_alert
    assessed = mon.assess(stats, best, best, 0, limits)
    assert assessed["level"] == "alert"
    assert not assessed["performance"]["departures"]["performance_degraded"]
    assert "Keep watching" in assessed["action"]
    stats.loc["capacity", "psi"] = 0.0
    stats.loc["capacity", "missing_increase"] = limits.missing_increase
    assessed = mon.assess(stats, best, best, limits.unknown_share, limits)
    assert assessed["quality_alert"] and assessed["unknown_station_alert"]
    assert assessed["missing_features"] == ["capacity"]
    stats.loc["capacity", "missing_increase"] = 0.0
    observed = {
        "departures": {
            "peak_wape": limits.peak_departures + 0.01,
            "improvement": limits.min_improvement - 0.01,
        }
    }
    assessed = mon.assess(stats, observed, best, 0, limits)
    assert not assessed["feature_drift"]
    assert assessed["performance"]["departures"]["performance_degraded"]
    stats.loc["capacity", "psi"] = limits.psi_warning
    assert (
        "make train tune evaluate register"
        in mon.assess(stats, observed, best, 0, limits)["action"]
    )


def test_digest_covers_tail_after_mlflow_pandas_sample():
    rows = pd.DataFrame({"value": range(10_010)})
    changed = rows.copy()
    changed.loc[10_001, "value"] = 0
    assert mon._digest(rows) != mon._digest(changed)


@pytest.mark.integration
def test_monitor_cli_no_track_and_complete_tracked_replay(tmp_path, mlflow_store):
    src, models = write_fixture(tmp_path / "fixture")
    loaded = load_models(models_dir=models)
    dst = tmp_path / "tracked"
    summary = mon.monitor_tracked(
        src,
        dst,
        loaded=loaded,
        reference_month=REFERENCE_MONTH,
        sim_month=SIM_MONTH,
        real_month=REPLAY_MONTH,
        data_kind="synthetic",
    )
    assert len(summary) == 7
    assert {row["alerts"]["level"] for row in summary} >= {"ok", "alert"}
    assert {row["scenario"] for row in summary} == {"control", "real", *mon.SIMULATIONS}
    for row in summary:
        run = mlflow.get_run(row["mlflow_run_id"])
        assert run.info.run_name == row["name"] and run.info.status == "FINISHED"
        assert run.data.tags["alert_level"] == row["alerts"]["level"]
        assert run.data.tags["data_kind"] == "synthetic"
        inputs = {item.dataset.name: item.dataset.digest for item in run.inputs.dataset_inputs}
        assert inputs["mibici_monitoring"] == row["monitoring"]["digest"]
        assert inputs["mibici_monitor_reference"] == row["reference"]["digest"]
        assert "monitoring_champion_departures_peak_wape" in run.data.metrics
        assert "monitoring_seasonal_naive_arrivals_overall_mae" in run.data.metrics
        names = {artifact.path for artifact in MlflowClient().list_artifacts(run.info.run_id)}
        assert names >= {
            "summary.json",
            "drift.csv",
            "performance.csv",
            "decisions.csv",
            "psi.png",
            "distribution.png",
        }
        if row["scenario"] != "real":
            assert f"{row['scenario']}-performance.png" in names
    by_name = {row["scenario"]: row for row in summary}
    replay = by_name["real"]
    assert replay["unknown_station_share"] == replay["new_since_station_share"] == 1 / 8
    real_metrics = pd.read_csv(dst / replay["name"] / "performance.csv")
    assert set(real_metrics["segment"]) >= {NEW_STATIONS, "unseen_stations", "group_1", "peak"}
    masked = pd.read_csv(dst / by_name["missing-features"]["name"] / "decisions.csv")
    assert (
        masked.groupby("window")["eligible_station_days"].first()["monitoring"]
        == masked.groupby("window")["eligible_station_days"].first()["control"]
    )
    assert (
        by_name["covariate-only"]["monitoring"]["actual_totals"]
        == by_name["control"]["monitoring"]["actual_totals"]
    )

    cli_out = tmp_path / "offline"
    args = [
        "--src",
        str(src),
        "--models-dir",
        str(models),
        "--dst",
        str(cli_out),
        "--no-track",
        "--scenario",
        "mean-shift",
        "--reference-month",
        REFERENCE_MONTH,
        "--sim-month",
        SIM_MONTH,
        "--severity",
        "0",
        "--severity",
        ".25",
        "--data-kind",
        "synthetic",
    ]
    output = CliRunner().invoke(mon.app, args)
    assert output.exit_code == 0, (output.output, output.exception)
    offline = json.loads((cli_out / "summaries.json").read_text())
    assert offline[1]["name"] == "drift-mean-shift-0"
    assert offline[1]["monitoring"]["digest"] == offline[0]["monitoring"]["digest"]
    assert offline[2]["monitoring"]["digest"] == by_name["mean-shift"]["monitoring"]["digest"]
    assert all("mlflow_run_id" not in case for case in offline)
    overlapping = CliRunner().invoke(mon.app, [*args, "--sim-month", REFERENCE_MONTH])
    assert overlapping.exit_code == 2
    assert "Expected disjoint months" in overlapping.output
