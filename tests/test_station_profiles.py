import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from team2_mlops2026 import station_profiles as sp
from team2_mlops2026 import tracking
from team2_mlops2026.features import FEATURES_FILE, STATION, TIME

WEEK0 = pd.Timestamp("2024-01-01")  # a Monday
HOURS = pd.date_range(WEEK0, periods=4 * 168, freq="h")


def station_rows(station: int, morning: str, evening: str, noise_seed: int | None = None):
    """Four weeks of one station: 10 trips at 08:00 and at 18:00 on weekdays, in the given
    directions, plus optional background noise."""
    df = pd.DataFrame({STATION: station, TIME: HOURS, "departures": 0, "arrivals": 0})
    weekday = df[TIME].dt.day_of_week < 5
    df.loc[weekday & (df[TIME].dt.hour == 8), morning] = 10
    df.loc[weekday & (df[TIME].dt.hour == 18), evening] = 10
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        for target in ("departures", "arrivals"):
            df[target] += rng.poisson(0.3, len(df))
    return df


@pytest.fixture
def features():
    """Stations 1-3 are residential (leave at 08:00, come back at 18:00), 4-6 the opposite."""
    residential = [station_rows(s, "departures", "arrivals", s) for s in (1, 2, 3)]
    work = [station_rows(s, "arrivals", "departures", s) for s in (4, 5, 6)]
    return pd.concat([*residential, *work], ignore_index=True)


def test_profile_is_the_share_of_trips_per_direction_and_hour_of_week():
    rows = station_rows(1, "departures", "departures")  # weekday departures only
    profile = sp.hour_of_week_profiles(rows, min_trips=1).loc[1]
    assert list(profile.index) == list(sp.PROFILE_COLUMNS) and len(profile) == 336
    assert profile.sum() == pytest.approx(1)
    assert profile["departures_mon_08"] == pytest.approx(0.1)  # 1 of 10 equal weekly peaks
    assert profile["arrivals_mon_08"] == 0 and profile["departures_sat_08"] == 0


def test_profiles_only_count_the_window_and_skip_quiet_stations(features):
    late = station_rows(7, "departures", "arrivals").assign(**{TIME: HOURS + pd.Timedelta(weeks=8)})
    quiet = station_rows(8, "departures", "arrivals").iloc[:24]  # 20 trips on Monday
    rows = pd.concat([features, late, quiet], ignore_index=True)
    profiles = sp.hour_of_week_profiles(rows, end=WEEK0 + pd.Timedelta(weeks=4), min_trips=100)
    assert list(profiles.index) == [1, 2, 3, 4, 5, 6]


def test_first_weeks_starts_at_each_station_first_hour(features):
    later = station_rows(7, "departures", "arrivals")
    later[TIME] += pd.Timedelta(weeks=2)  # opens two weeks after the others
    window = sp.first_weeks(pd.concat([features, later], ignore_index=True), weeks=1)
    span = window.groupby(STATION)[TIME].agg(["min", "max"])
    assert (span["max"] - span["min"] == pd.Timedelta(weeks=1) - pd.Timedelta(hours=1)).all()
    assert span.loc[7, "min"] == WEEK0 + pd.Timedelta(weeks=2)


def test_weekly_profiles_use_the_weeks_known_before_each_week_starts():
    rows = pd.DataFrame({STATION: 1, TIME: HOURS, "departures": 0, "arrivals": 0})
    week2 = WEEK0 + pd.Timedelta(weeks=2)
    rows.loc[rows[TIME] == week2 - pd.Timedelta(hours=49), "departures"] = 5  # Friday 23:00
    rows.loc[rows[TIME] == week2 - pd.Timedelta(hours=48), "arrivals"] = 5  # Saturday 00:00

    profiles = sp.weekly_profiles(rows, weeks=1, gap=48, min_trips=1)

    assert list(profiles.columns) == list(sp.PROFILE_COLUMNS)
    assert profiles.loc[(1, week2)].sum() == pytest.approx(1)
    assert profiles.loc[(1, week2), "departures_fri_23"] == 1  # the Saturday trip comes too late
    assert profiles.loc[(1, week2 + pd.Timedelta(weeks=1)), "arrivals_sat_00"] == 1
    assert profiles.index.get_level_values(sp.WEEK).tolist() == [week2, week2 + pd.Timedelta(weeks=1)]


def test_first_component_separates_residential_from_work_stations(features):
    profiles = sp.hour_of_week_profiles(features, min_trips=1)
    scores = sp.component_scores(sp.fit_pca(profiles, 2), profiles)
    assert list(scores.columns) == ["pc1", "pc2"]
    residential, work = scores.loc[[1, 2, 3], "pc1"], scores.loc[[4, 5, 6], "pc1"]
    assert residential.max() < work.min() or work.max() < residential.min()


def test_variance_target_keeps_just_enough_components(features):
    profiles = sp.hour_of_week_profiles(features, min_trips=1)
    pca = sp.fit_pca(profiles, 0.9)
    ratio = pca.explained_variance_ratio_
    assert ratio.sum() >= 0.9 and ratio[:-1].sum() < 0.9
    weights = sp.loadings(pca, n=1)
    assert list(weights.index) == list(sp.PROFILE_COLUMNS)
    # residential vs work: 08:00 departures and 08:00 arrivals weigh in opposite directions
    assert np.sign(weights.loc["departures_mon_08", "pc1"]) == -np.sign(
        weights.loc["arrivals_mon_08", "pc1"]
    )


def test_create_profiles_tracked_logs_input_output_and_variance(features, tmp_path, mlflow_store):
    src = tmp_path / "features"
    src.mkdir()
    features.to_parquet(src / FEATURES_FILE, index=False)

    summary = sp.create_profiles_tracked(src, src, fit_end="2024-02", min_trips=10)
    run = mlflow.get_run(summary["mlflow_run_id"])

    assert run.info.run_name == sp.RUN_NAME and run.info.status == "FINISHED"
    datasets = {i.dataset.name: i.dataset for i in run.inputs.dataset_inputs}
    assert set(datasets) == {"mibici_station_hour_features", "mibici_station_profiles"}
    output = src / sp.PROFILES_FILE
    assert datasets["mibici_station_profiles"].digest == tracking.files_digest([output], src)
    scores = pd.read_parquet(output)
    components = [f"pc{i}" for i in range(1, summary["components"] + 1)]
    assert list(scores.columns) == [STATION, sp.WEEK, *components]
    assert (scores[sp.WEEK].dt.day_of_week == 0).all() and len(scores) == summary["station_weeks"]
    assert run.data.params["fit_end"] == scores.attrs[sp.FIT_END_KEY] == "2024-02"
    assert run.data.metrics["stations"] == 6
    assert run.data.metrics["explained_variance"] >= sp.VARIANCE_TARGET
    artifacts = [a.path for a in MlflowClient().list_artifacts(run.info.run_id)]
    assert artifacts == ["explained_variance.json"]
