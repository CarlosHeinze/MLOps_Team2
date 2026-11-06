"""
Station usage profiles and their dimensionality reduction with PCA:
data/processed/mibici/features/station_hour_features.parquet -> .../features/station_profiles.parquet

A station's profile is the share of its trips that depart and that arrive in each of the 168
hours of the week: 336 values that sum to 1. Shares describe *when* and *in which direction* a
station is used, independently of how busy it is (the station's volume is already in the target
encoding of ``features``). Neighbouring hours move together, so PCA summarises the 336 shares in
a few components (notebook 1.05): for example, stations people leave in the morning and return
to in the evening (residential) against the opposite pattern (work and study areas).

- PCA is fitted only on hours before ``fit_end``, where the test period of notebook 1.04 starts,
  so no test information reaches the components. The table records its ``fit_end`` in the file
  metadata, so training can reject a test split that starts earlier.
- Shares are centred but not scaled: they are already on one scale, and scaling would blow up
  the noise of the almost empty night hours (some of them have no trips at all).
- Stations with fewer than ``min_trips`` trips in the window do not enter the fit: a few trips do
  not describe a pattern.
- The scores are time-aware: every week, each station's profile of its last four weeks, ending
  48 h before the week starts (the history known at its first forecast), is projected on the
  fitted components. Notebook 1.05 shows that four weeks place a station. So a station opened
  after ``fit_end`` gets scores from its fifth week on, and in training the stations opened
  then have no scores in their first weeks, just as new stations in the test period (notebook
  2.02: with one fixed score per station, "no profile" only meant a quiet station).

Usage (CLI):
    python -m team2_mlops2026.station_profiles
    python -m team2_mlops2026.station_profiles --fit-end 2025-10 --variance 0.9 --no-track

Usage (notebook):
    from team2_mlops2026.station_profiles import fit_pca, hour_of_week_profiles, weekly_profiles
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import typer

from team2_mlops2026.features import (
    FEATURES_DIR,
    FEATURES_FILE,
    FORECAST_GAP,
    STATION,
    TARGETS,
    TIME,
)

PROFILES_FILE = "station_profiles.parquet"
RUN_NAME = "fit-station-profiles"

DEFAULT_FIT_END = "2025-10"  # first month of the test period in notebook 1.04
FIT_END_KEY = "fit_end"  # key of the fit end in the table's metadata (DataFrame.attrs)
MIN_TRIPS = 500
VARIANCE_TARGET = 0.9  # keep the components that explain 90% of the variance
WEEK = "week_start"  # Monday 00:00 of the week the scores apply to
PROFILE_WEEKS = 4  # history behind each weekly score (1.05)
MIN_TRIPS_WEEKS = 100  # minimum trips in those weeks for a score (1.05)

HOURS_PER_WEEK = 168
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
# departures_mon_00 ... departures_sun_23, arrivals_mon_00 ... arrivals_sun_23
PROFILE_COLUMNS = tuple(
    f"{target}_{WEEKDAYS[h // 24]}_{h % 24:02d}"
    for target in TARGETS
    for h in range(HOURS_PER_WEEK)
)

app = typer.Typer()


def hour_of_week_profiles(
    features: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    min_trips: int = MIN_TRIPS,
) -> pd.DataFrame:
    """Share of each station's trips per direction and hour of the week (each row sums to 1).

    Only hours in [``start``, ``end``) count. ``features`` needs the station, hour and target
    columns of the station-hour table.
    """
    times = features[TIME]
    window = pd.Series(True, index=features.index)
    if start is not None:
        window &= times >= pd.Timestamp(start)
    if end is not None:
        window &= times < pd.Timestamp(end)
    rows = features[window]
    hour_of_week = rows[TIME].dt.day_of_week * 24 + rows[TIME].dt.hour
    counts = (
        rows.groupby([rows[STATION], hour_of_week])[list(TARGETS)]
        .sum()
        .unstack(fill_value=0)
        .reindex(
            columns=pd.MultiIndex.from_product([TARGETS, range(HOURS_PER_WEEK)]), fill_value=0
        )
    )
    total = counts.sum(axis=1)
    keep = total >= min_trips
    profiles = counts[keep].div(total[keep], axis=0)
    profiles.columns = list(PROFILE_COLUMNS)
    return profiles.rename_axis(STATION)


def first_weeks(features: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Rows within the first ``weeks`` weeks of each station, from its first hour in the table.

    For a station that opened after the table starts, that is its first month of service: the
    history a new station has when it first needs a profile.
    """
    first_hour = features.groupby(STATION)[TIME].transform("min")
    return features[features[TIME] < first_hour + pd.Timedelta(weeks=weeks)]


def week_start(times: pd.Series) -> pd.Series:
    """Monday 00:00 of each time's week."""
    return times.dt.normalize() - pd.to_timedelta(times.dt.day_of_week, unit="D")


def weekly_profiles(
    features: pd.DataFrame,
    weeks: int = PROFILE_WEEKS,
    gap: int = FORECAST_GAP,
    min_trips: int = MIN_TRIPS_WEEKS,
) -> pd.DataFrame:
    """Each station's profile for every week of ``features``, from its trips in the ``weeks``
    weeks that end ``gap`` hours before the week starts.

    Station-weeks with fewer than ``min_trips`` trips get no row. Index: (station, week start);
    columns: ``PROFILE_COLUMNS``, as in ``hour_of_week_profiles``.
    """
    span = weeks * HOURS_PER_WEEK
    first_week = week_start(features[TIME]).min()
    stations, rows = np.unique(features[STATION].to_numpy(), return_inverse=True)
    # the first week starts at column span + gap, so every window lies inside the matrix
    hours = ((features[TIME] - first_week) // pd.Timedelta(hours=1)).to_numpy() + span + gap
    n_weeks = (hours.max() - span - gap) // HOURS_PER_WEEK + 1
    counts = np.zeros((len(TARGETS), len(stations), hours.max() + 1), np.float32)
    for i, target in enumerate(TARGETS):
        counts[i, rows, hours] = features[target].to_numpy()

    # A window ends gap hours before a Monday, so its column j is hour of the week (j - gap)
    by_hour_of_week = (np.arange(HOURS_PER_WEEK) + gap) % HOURS_PER_WEEK
    windows = []
    for w in range(n_weeks):
        end = span + w * HOURS_PER_WEEK
        window = counts[:, :, end - span : end].reshape(
            len(TARGETS), len(stations), weeks, HOURS_PER_WEEK
        )
        windows.append(np.hstack(list(window.sum(axis=2)[:, :, by_hour_of_week])))
    profiles = np.concatenate(windows)
    index = pd.MultiIndex.from_arrays(
        [
            np.tile(stations, n_weeks),
            np.repeat(first_week + pd.to_timedelta(np.arange(n_weeks), unit="W"), len(stations)),
        ],
        names=[STATION, WEEK],
    )
    total = profiles.sum(axis=1)
    keep = total >= min_trips
    return pd.DataFrame(
        profiles[keep] / total[keep, None], index=index[keep], columns=list(PROFILE_COLUMNS)
    )


def fit_pca(profiles: pd.DataFrame, n_components: float | None = VARIANCE_TARGET) -> PCA:
    """PCA of the profiles: a float in (0, 1) keeps the components that explain that share of
    the variance, an int keeps that many, ``None`` keeps them all."""
    return PCA(n_components=n_components, svd_solver="full").fit(profiles)


def component_scores(pca: PCA, profiles: pd.DataFrame) -> pd.DataFrame:
    """The profiles' coordinates on the components, as columns ``pc1``, ``pc2``..."""
    columns = [f"pc{i}" for i in range(1, pca.n_components_ + 1)]
    return pd.DataFrame(pca.transform(profiles), index=profiles.index, columns=columns)


def loadings(pca: PCA, n: int = 3) -> pd.DataFrame:
    """Weight of every profile column on the first ``n`` components (columns ``pc1``...)."""
    return pd.DataFrame(
        pca.components_[:n].T,
        index=list(PROFILE_COLUMNS),
        columns=[f"pc{i + 1}" for i in range(n)],
    )


# --- features -> station profiles ----------------------------------------------


def create_profiles(
    src: str | Path = FEATURES_DIR,
    dst: str | Path = FEATURES_DIR,
    fit_end: str = DEFAULT_FIT_END,
    variance: float = VARIANCE_TARGET,
    min_trips: int = MIN_TRIPS,
    score_min_trips: int = MIN_TRIPS_WEEKS,
) -> dict:
    """Fits the PCA on the profiles before ``fit_end`` and writes the weekly scores of every
    station (columns station_id, week_start, pc1...).

    Summary: {path, stations, stations_without_profile (both in the fit), station_weeks,
              components, explained_variance, explained_variance_pc1..3,
              explained_variance_ratio}
    """
    features = pd.read_parquet(Path(src) / FEATURES_FILE, columns=[STATION, TIME, *TARGETS])
    profiles = hour_of_week_profiles(features, end=fit_end, min_trips=min_trips)
    pca = fit_pca(profiles, variance)
    scores = component_scores(pca, weekly_profiles(features, min_trips=score_min_trips))

    dest = Path(dst) / PROFILES_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    table = scores.reset_index()
    table.attrs[FIT_END_KEY] = fit_end
    try:
        table.to_parquet(tmp, index=False)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    ratio = pca.explained_variance_ratio_
    summary = {
        "path": str(dest),
        "stations": len(profiles),
        "stations_without_profile": int(features[STATION].nunique() - len(profiles)),
        "station_weeks": len(scores),
        "components": int(pca.n_components_),
        "explained_variance": float(ratio.sum()),
        **{f"explained_variance_pc{i + 1}": float(ratio[i]) for i in range(min(3, len(ratio)))},
        "explained_variance_ratio": [float(r) for r in ratio],
    }
    logger.success(
        "Station profiles: {} stations, {} components explain {:.1%} of the variance",
        summary["stations"],
        summary["components"],
        summary["explained_variance"],
    )
    return summary


def create_profiles_tracked(
    src: str | Path = FEATURES_DIR,
    dst: str | Path = FEATURES_DIR,
    fit_end: str = DEFAULT_FIT_END,
    variance: float = VARIANCE_TARGET,
    min_trips: int = MIN_TRIPS,
    score_min_trips: int = MIN_TRIPS_WEEKS,
) -> dict:
    """``create_profiles()`` inside an MLflow run of the project's experiment.

    Logs the station-hour features as input and the station scores as output (each with a
    digest), the profile definition and PCA settings as params, the explained variance as
    metrics and the variance of every component as artifact. The summary gets ``mlflow_run_id``.
    """
    import mlflow

    from team2_mlops2026 import tracking

    src, dst = Path(src), Path(dst)
    with tracking.start_run(RUN_NAME) as run:
        tracking.log_files_dataset(
            [src / FEATURES_FILE], src, "mibici_station_hour_features", "features"
        )
        mlflow.log_params(
            {
                "fit_end": fit_end,
                "min_trips": min_trips,
                "variance_target": variance,
                "profile": "share of trips per direction and hour of the week (336 columns)",
                "scaling": "centred, not scaled",
                "score_weeks": PROFILE_WEEKS,
                "score_gap_hours": FORECAST_GAP,
                "score_min_trips": score_min_trips,
            }
        )
        summary = create_profiles(src, dst, fit_end, variance, min_trips, score_min_trips)
        tracking.log_files_dataset(
            [dst / PROFILES_FILE], dst, "mibici_station_profiles", "profiles"
        )
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})
        mlflow.log_dict(
            {"explained_variance_ratio": summary["explained_variance_ratio"]},
            "explained_variance.json",
        )
    summary["mlflow_run_id"] = run.info.run_id
    return summary


@app.command()
def main(
    src: Annotated[
        Path, typer.Option(help="Folder with the station-hour features")
    ] = FEATURES_DIR,
    dst: Annotated[Path, typer.Option(help="Destination folder")] = FEATURES_DIR,
    fit_end: Annotated[
        str, typer.Option(help="YYYY-MM: fit on the hours before this month")
    ] = DEFAULT_FIT_END,
    variance: Annotated[
        float, typer.Option(help="Share of variance the kept components explain")
    ] = VARIANCE_TARGET,
    min_trips: Annotated[
        int, typer.Option(help="Minimum trips for a station profile")
    ] = MIN_TRIPS,
    score_min_trips: Annotated[
        int, typer.Option(help="Minimum trips in a station's last weeks for a weekly score")
    ] = MIN_TRIPS_WEEKS,
    track: Annotated[bool, typer.Option(help="Log the run to the MLflow server")] = True,
):
    (create_profiles_tracked if track else create_profiles)(
        src, dst, fit_end, variance, min_trips, score_min_trips
    )


if __name__ == "__main__":
    app()
