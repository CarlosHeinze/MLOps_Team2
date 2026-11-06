"""
Station-hour demand features and their preprocessing:
data/processed/mibici/viajes/ -> data/processed/mibici/features/

The ML Canvas (``references/ml-canvas.md``) frames the task as forecasting the hourly departures
and arrivals of every station. This module turns the cleaned trips into that table and defines
how its columns are transformed before modelling:

- One row per station and hour while the station is active, zero hours included: most
  station-hours have no trips, and a model must learn those zeros too.
- Lagged demand only from hours already known when the next-day forecast is made (>= 48 h back).
  Hours before a station opened are unknown (NaN), not zero demand.
- Calendar features with a cyclical encoding, so 23:00 sits next to 00:00 and December next to
  January (notebook 1.03: hour and weekday drive demand non-linearly).
- ``make_preprocessor``: log1p + standard scaling for the skewed counts, one-hot for the weekday,
  target encoding for the ~480 station ids, imputation + scaling for the station attributes and
  for the PCA profile scores when they are joined. It is fitted on the training split only
  (``time_split``), so no test statistic leaks in.

Usage (CLI):
    python -m team2_mlops2026.features
    python -m team2_mlops2026.features --start 2024-01 --no-track

Usage (notebook):
    from team2_mlops2026.features import create_features_tracked, make_preprocessor, time_split
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
import typer

from team2_mlops2026.data.raw_data_normalization import INTERIM_MIBICI_DIR
from team2_mlops2026.data.trip_data import (
    PROCESSED_MIBICI_DIR,
    latest_station_file,
    load_trips,
    trip_files,
)

FEATURES_DIR = PROCESSED_MIBICI_DIR / "features"
FEATURES_FILE = "station_hour_features.parquet"
RUN_NAME = "build-station-hour-features"
SEED = 42

# The post-COVID plateau (notebook 1.03); 2020-2021 is a regime change left out of training
DEFAULT_START = "2023-01"

TRIP_COLUMNS = ("Origen_Id", "Destino_Id", "Inicio_del_viaje", "Fin_del_viaje")

STATION, TIME = "station_id", "hour_start"
TARGETS = ("departures", "arrivals")
# Hours back. The forecast for day D+1 is made on the night of day D, so the latest hour it can
# use is >= 24 h before any target hour; 48 h keeps a margin for late data.
LAGS = {"lag_2d": 48, "lag_7d": 168}
SAME_HOUR_WEEKS = 4  # mean of the same hour of the week over the previous 4 weeks
WARMUP_HOURS = SAME_HOUR_WEEKS * 168  # history needed before ``start`` for every lag

DEMAND_FEATURES = tuple(
    f"{target}_{name}" for target in TARGETS for name in [*LAGS, "same_hour_4w"]
)
WEEKDAY = "weekday"  # Monday = 0
CYCLICAL = ("hour_sin", "hour_cos", "month_sin", "month_cos")
STATION_ATTRS = ("latitude", "longitude", "capacity")
PROFILE_SCORES = r"^pc\d+$"  # the station_profiles scores pc1, pc2... joined for training
FEATURE_COLUMNS = (STATION, TIME, *TARGETS, *DEMAND_FEATURES, WEEKDAY, *CYCLICAL, *STATION_ATTRS)

app = typer.Typer()


# --- station-hour table --------------------------------------------------------


def _hour_index(times: pd.Series, first_hour: pd.Timestamp) -> np.ndarray:
    hours = times.dt.floor("h").to_numpy("datetime64[s]") - np.datetime64(first_hour, "s")
    return hours // np.timedelta64(1, "h")


def hourly_counts(
    station_ids: pd.Series, times: pd.Series, stations: np.ndarray, first_hour, n_hours: int
) -> np.ndarray:
    """Trips per station (rows, in ``stations`` order) and hour (columns) from ``first_hour``."""
    rows = pd.Index(stations).get_indexer(station_ids.to_numpy("int64"))
    cols = _hour_index(times, first_hour)
    keep = (rows >= 0) & (cols >= 0) & (cols < n_hours)
    flat = rows[keep] * n_hours + cols[keep]
    counts = np.bincount(flat, minlength=len(stations) * n_hours)
    return counts.reshape(len(stations), n_hours).astype(np.int32)


def active_hours(*counts: np.ndarray) -> np.ndarray:
    """True from each station's first to its last hour with any trip.

    Outside that span the station did not exist (yet), so its zeros are not demand.
    """
    any_trip = np.logical_or.reduce([c > 0 for c in counts])
    n_hours = any_trip.shape[1]
    first = any_trip.argmax(axis=1)
    last = n_hours - 1 - any_trip[:, ::-1].argmax(axis=1)
    hour = np.arange(n_hours)
    return (hour >= first[:, None]) & (hour <= last[:, None])


def _shift(values: np.ndarray, hours: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    out[:, hours:] = values[:, :-hours]
    return out


def lag_features(counts: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
    """Lagged demand per station-hour; NaN where the lagged hour predates the station."""
    known = np.where(active, counts, np.nan).astype(np.float32)
    lags = {name: _shift(known, hours) for name, hours in LAGS.items()}
    same_hour = np.stack([_shift(known, 168 * w) for w in range(1, SAME_HOUR_WEEKS + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all four weeks unknown -> NaN
        lags["same_hour_4w"] = np.nanmean(same_hour, axis=0)
    return lags


def calendar_features(hours: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    hour_angle = 2 * np.pi * hours.hour.to_numpy() / 24
    month_angle = 2 * np.pi * (hours.month.to_numpy() - 1) / 12
    return {
        WEEKDAY: hours.day_of_week.to_numpy().astype(np.int8),
        "hour_sin": np.sin(hour_angle),
        "hour_cos": np.cos(hour_angle),
        "month_sin": np.sin(month_angle),
        "month_cos": np.cos(month_angle),
    }


def station_attributes(stations: np.ndarray, nomenclature: pd.DataFrame) -> pd.DataFrame:
    """Location and capacity per station; NaN for ids missing from the nomenclature.

    Retired stations ("ex-...") are listed with capacity 0, which is not a real capacity.
    """
    attrs = (
        nomenclature.set_index("id")[["latitude", "longitude", "dpcapacity"]]
        .rename(columns={"dpcapacity": "capacity"})
        .astype("float64")
        .reindex(stations)
    )
    attrs["capacity"] = attrs["capacity"].replace(0, np.nan)
    return attrs


def build_features(
    trips: pd.DataFrame,
    nomenclature: pd.DataFrame,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Station-hour table with targets and features, from ``start`` to ``end`` (inclusive).

    ``trips`` needs the columns in ``TRIP_COLUMNS``; the ``WARMUP_HOURS`` before ``start``
    only feed the lags. ``end`` defaults to the last hour with a departure: monthly files hold
    the trips that *start* in the month, so later hours only see arrivals of trips still running.
    """
    start = pd.Timestamp(start)
    last_departure = trips["Inicio_del_viaje"].max().floor("h")
    end = pd.Timestamp(end) if end is not None else last_departure
    first_hour = start - pd.Timedelta(hours=WARMUP_HOURS)
    hours = pd.date_range(first_hour, end, freq="h")
    stations = np.union1d(
        trips["Origen_Id"].to_numpy("int64"), trips["Destino_Id"].to_numpy("int64")
    )

    counts = {
        "departures": hourly_counts(
            trips["Origen_Id"], trips["Inicio_del_viaje"], stations, first_hour, len(hours)
        ),
        "arrivals": hourly_counts(
            trips["Destino_Id"], trips["Fin_del_viaje"], stations, first_hour, len(hours)
        ),
    }
    active = active_hours(*counts.values())
    keep = np.flatnonzero((active & (hours >= start)).ravel())

    def flat(matrix: np.ndarray) -> np.ndarray:
        return matrix.ravel()[keep]

    columns = {
        STATION: np.repeat(stations, len(hours))[keep],
        TIME: np.tile(hours.to_numpy(), len(stations))[keep],
        **{target: flat(counts[target]) for target in TARGETS},
    }
    for target in TARGETS:
        for name, values in lag_features(counts[target], active).items():
            columns[f"{target}_{name}"] = flat(values)
    for name, values in calendar_features(hours).items():
        columns[name] = np.tile(values, len(stations))[keep]
    attrs = station_attributes(stations, nomenclature)
    for name in STATION_ATTRS:
        columns[name] = np.repeat(attrs[name].to_numpy(), len(hours))[keep]
    return pd.DataFrame(columns, columns=list(FEATURE_COLUMNS))


def time_split(
    features: pd.DataFrame, test_start: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train before ``test_start``, test from it on: a random split would leak future demand."""
    is_test = features[TIME] >= pd.Timestamp(test_start)
    return features[~is_test], features[is_test]


# --- preprocessing -------------------------------------------------------------


def make_preprocessor(random_state: int = SEED) -> ColumnTransformer:
    """Column transformations for the station-hour features (fit on the training split only).

    - Station id (~480 values): target encoding. One-hot would add ~480 sparse columns; the
      encoder is cross-fitted on the training rows, so a row never sees its own target.
    - Weekday (7 values): one-hot, since its effect is not ordinal (weekend vs weekday).
    - Hour and month: already sin/cos in [-1, 1], passed through.
    - Lagged demand: right-skewed counts -> median imputation for new stations, log1p, scaling.
    - Station attributes: median imputation for ids missing from the nomenclature, scaling.
    - Profile scores (``pc1``...), if joined: a station without a profile gets 0, the average
      profile (PCA centres the scores), then scaling, plus one indicator column that flags the
      stations without a profile. A frame without the scores selects no columns.
    """
    counts = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    attributes = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    profiles = make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler())
    folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
    return ColumnTransformer(
        [
            ("station", TargetEncoder(target_type="continuous", cv=folds), [STATION]),
            ("weekday", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [WEEKDAY]),
            ("cyclical", "passthrough", list(CYCLICAL)),
            ("demand", counts, list(DEMAND_FEATURES)),
            ("station_attrs", attributes, list(STATION_ATTRS)),
            ("profiles", profiles, make_column_selector(pattern=PROFILE_SCORES)),
            # A station has all its scores or none, so one indicator is enough
            (
                "no_profile",
                MissingIndicator(features="all"),
                make_column_selector(pattern="^pc1$"),
            ),
        ]
    ).set_output(transform="pandas")


# --- processed -> features -----------------------------------------------------


def create_features(
    src: str | Path = PROCESSED_MIBICI_DIR,
    stations_src: str | Path = INTERIM_MIBICI_DIR,
    dst: str | Path = FEATURES_DIR,
    start: str = DEFAULT_START,
    end: str | None = None,
) -> dict:
    """Builds the station-hour table from the processed trips and writes it atomically.

    Summary: {path, rows, stations, hours, start, end, zero_departures_share,
              zero_arrivals_share, stations_without_attributes}
    """
    trips = load_trips(src, TRIP_COLUMNS)
    nomenclature = pd.read_parquet(latest_station_file(stations_src))
    features = build_features(trips, nomenclature, start, end)
    del trips

    dest = Path(dst) / FEATURES_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    try:
        features.to_parquet(tmp, index=False)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    summary = {
        "path": str(dest),
        "rows": len(features),
        "stations": int(features[STATION].nunique()),
        "hours": int(features[TIME].nunique()),
        "start": str(features[TIME].min()),
        "end": str(features[TIME].max()),
        **{f"zero_{t}_share": float((features[t] == 0).mean()) for t in TARGETS},
        "stations_without_attributes": int(
            features.loc[features["latitude"].isna(), STATION].nunique()
        ),
    }
    logger.success(
        "Features: {:,} station-hours, {} stations, {} -> {}",
        summary["rows"],
        summary["stations"],
        summary["start"],
        summary["end"],
    )
    return summary


def create_features_tracked(
    src: str | Path = PROCESSED_MIBICI_DIR,
    stations_src: str | Path = INTERIM_MIBICI_DIR,
    dst: str | Path = FEATURES_DIR,
    start: str = DEFAULT_START,
    end: str | None = None,
) -> dict:
    """``create_features()`` inside an MLflow run of the project's experiment.

    Logs the processed trips and stations as inputs and the feature table as output (each with
    a digest), the feature definition as params and the table's shape as metrics. The summary
    gets ``mlflow_run_id``.
    """
    import mlflow

    from team2_mlops2026 import tracking

    src, dst = Path(src), Path(dst)
    station_file = latest_station_file(stations_src)
    with tracking.start_run(RUN_NAME) as run:
        tracking.log_files_dataset(
            trip_files(src), src / "viajes", "mibici_trips_processed", "processed"
        )
        tracking.log_dataset(
            pd.read_parquet(station_file), station_file, "mibici_stations", "stations"
        )
        mlflow.log_params(
            {
                "start": start,
                "end": end or "last hour",
                "targets": ", ".join(TARGETS),
                "lags_hours": ", ".join(f"{k}={v}" for k, v in LAGS.items()),
                "same_hour_weeks": SAME_HOUR_WEEKS,
                "station_nomenclature": station_file.name,
            }
        )
        summary = create_features(src, stations_src, dst, start, end)
        tracking.log_files_dataset(
            [dst / FEATURES_FILE], dst, "mibici_station_hour_features", "features"
        )
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})
    summary["mlflow_run_id"] = run.info.run_id
    return summary


@app.command()
def main(
    src: Annotated[Path, typer.Option(help="Processed trips folder")] = PROCESSED_MIBICI_DIR,
    dst: Annotated[Path, typer.Option(help="Destination features folder")] = FEATURES_DIR,
    start: Annotated[str, typer.Option(help="YYYY-MM: first month of the table")] = DEFAULT_START,
    end: Annotated[str | None, typer.Option(help="Last hour (default: end of data)")] = None,
    track: Annotated[bool, typer.Option(help="Log the run to the MLflow server")] = True,
):
    (create_features_tracked if track else create_features)(
        src, INTERIM_MIBICI_DIR, dst, start, end
    )


if __name__ == "__main__":
    app()
