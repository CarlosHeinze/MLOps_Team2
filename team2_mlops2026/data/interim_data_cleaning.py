"""
Value cleaning of MiBici trips: data/interim/mibici/ -> data/processed/mibici/

The rules come from the data contract tests in notebook ``1.02-team2-eda-and-data-cleaning``:
- Exact duplicate rows are removed; ``Viaje_Id`` must then be unique.
- Rows that are not real trips are removed: under 60 s (false starts / re-docks, the
  Divvy and Citi Bike convention), over 24 h (bike not docked) or faster than 30 km/h
  on the straight line between stations.
- Demographic values outside the contract (gender not M/F, age at trip time outside
  16-90) become missing; the trip itself is kept.
- Missing gender / birth year is recovered from the same user's other trips. What cannot
  be recovered stays <NA>: missingness depends on the user, so dropping those trips would
  remove whole riders from the demand history, and a global fill would distort the
  demographics. Any further imputation belongs to the model pipeline (fit on train only).

Output keeps the interim layout and schema, one Parquet per month sorted by start time:
    viajes/2016/viajes_2016_01.parquet
plus ``cleaning_report.json`` with the rows each step touched per file (lineage).

Usage (CLI):
    python -m team2_mlops2026.data.interim_data_cleaning
    python -m team2_mlops2026.data.interim_data_cleaning --force --workers 8

Usage (notebook):
    from team2_mlops2026.data.interim_data_cleaning import clean
    summary = clean()
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from team2_mlops2026.config import PROCESSED_DATA_DIR
from team2_mlops2026.data.raw_data_normalization import INTERIM_MIBICI_DIR, TRIP_COLUMNS

PROCESSED_MIBICI_DIR = PROCESSED_DATA_DIR / "mibici"
REPORT_NAME = "cleaning_report.json"
DEFAULT_WORKERS = 4

BIRTH_YEAR = "Año_de_nacimiento"
DEMOGRAPHIC_COLUMNS = ("Genero", BIRTH_YEAR)
# Every other column identifies the trip and may never be null
REQUIRED_COLUMNS = tuple(c for c in TRIP_COLUMNS if c not in DEMOGRAPHIC_COLUMNS)
# Enough to build user profiles without loading the whole history
PROFILE_COLUMNS = ("Usuario_Id", "Genero", BIRTH_YEAR, "Inicio_del_viaje")

# Data contract; notebook 1.02 shows the evidence behind each bound
GENDERS = ("M", "F")
MIN_DURATION_S = 60
MAX_DURATION_S = 24 * 3600
MAX_SPEED_KMH = 30.0
MIN_AGE, MAX_AGE = 16, 90  # MiBici registers riders from 16; older ages are placeholder years

TRIP_RULES = ("too_short", "too_long", "too_fast")  # the row is not a real trip -> drop it
FIELD_RULES = {"invalid_gender": "Genero", "invalid_age": BIRTH_YEAR}  # only the value is wrong

EARTH_RADIUS_KM = 6371.0088

app = typer.Typer()


class ContractError(ValueError):
    """A cleaned frame breaks the data contract."""


@dataclass
class CleaningLog:
    """Rows and cells each cleaning step touched, in execution order."""

    steps: list[dict] = field(default_factory=list)

    def record(
        self, step: str, detail: str, rows_before: int, rows_after: int, cells_changed: int = 0
    ) -> None:
        self.steps.append(
            {
                "step": step,
                "detail": detail,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "cells_changed": cells_changed,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def _record(log, step, detail, before: pd.DataFrame, after: pd.DataFrame, cells: int = 0):
    if log is not None:
        log.record(step, detail, len(before), len(after), cells)


# --- loading -----------------------------------------------------------------


def trip_files(src: str | Path = INTERIM_MIBICI_DIR) -> list[Path]:
    return sorted((Path(src) / "viajes").rglob("viajes_*.parquet"))


def latest_station_file(src: str | Path = INTERIM_MIBICI_DIR) -> Path:
    files = sorted((Path(src) / "estaciones").glob("nomenclatura_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No station nomenclature under {src}")
    return files[-1]


def load_trips(
    src: str | Path = INTERIM_MIBICI_DIR, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Every month under ``src`` in one frame, with ``Genero`` as category to save memory."""
    files = [str(p) for p in trip_files(src)]
    # A month whose column is all null stores it as type null; unify instead of failing
    schema = pa.unify_schemas([pq.read_schema(f) for f in files], promote_options="permissive")
    df = ds.dataset(files, schema=schema, format="parquet")
    df = df.to_table(columns=columns and list(columns)).to_pandas()
    if "Genero" in df:
        df["Genero"] = df["Genero"].astype("category")
    return df


def load_stations(src: str | Path = INTERIM_MIBICI_DIR) -> pd.DataFrame:
    """Latest station nomenclature (MiBici only publishes the current one)."""
    return pd.read_parquet(latest_station_file(src))


# --- derived quantities ------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def age_at_trip(trips: pd.DataFrame) -> pd.Series:
    return trips["Inicio_del_viaje"].dt.year - trips[BIRTH_YEAR]


def trip_metrics(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Quantities the contract is written on: duration, distance, speed and age.

    Distance is the straight line between stations, so distance and speed are lower bounds
    of the real ones; both are NaN for stations missing from the nomenclature.
    """
    duration_s = (trips["Fin_del_viaje"] - trips["Inicio_del_viaje"]).dt.total_seconds()
    coords = stations.set_index("id")[["latitude", "longitude"]].astype("float64")
    origin = coords.reindex(trips["Origen_Id"].to_numpy("int64", na_value=-1)).to_numpy()
    dest = coords.reindex(trips["Destino_Id"].to_numpy("int64", na_value=-1)).to_numpy()
    distance_km = haversine_km(origin[:, 0], origin[:, 1], dest[:, 0], dest[:, 1])
    hours = duration_s.to_numpy() / 3600
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_kmh = np.where(hours > 0, distance_km / hours, np.nan)
    return pd.DataFrame(
        {
            "duration_s": duration_s.to_numpy(),
            "distance_km": distance_km,
            "speed_kmh": speed_kmh,
            "age": age_at_trip(trips).astype("float64").to_numpy(),
        },
        index=trips.index,
    ).astype("float32")


# --- contract rules ----------------------------------------------------------


def trip_violations(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rules that make the whole row invalid (True = broken)."""
    return pd.DataFrame(
        {
            "too_short": metrics["duration_s"] < MIN_DURATION_S,
            "too_long": metrics["duration_s"] > MAX_DURATION_S,
            "too_fast": metrics["speed_kmh"] > MAX_SPEED_KMH,
        }
    )


def field_violations(trips: pd.DataFrame) -> pd.DataFrame:
    """Rules that invalidate a single demographic value (True = broken)."""
    gender = trips["Genero"]
    return pd.DataFrame(
        {
            "invalid_gender": (gender.notna() & ~gender.isin(GENDERS)).astype(bool),
            "invalid_age": ~age_at_trip(trips).between(MIN_AGE, MAX_AGE).fillna(True).astype(bool),
        }
    )


def rule_violations(trips: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Every contract rule as a boolean column: True where the trip breaks it."""
    return pd.concat(
        [
            trips.duplicated().rename("duplicate"),
            trip_violations(metrics),
            field_violations(trips),
        ],
        axis=1,
    )


def overlapping_trips(trips: pd.DataFrame) -> pd.Series:
    """True where a user starts a trip before their previous one ended (diagnostic only)."""
    order = trips[["Usuario_Id", "Inicio_del_viaje", "Fin_del_viaje"]].sort_values(
        ["Usuario_Id", "Inicio_del_viaje"]
    )
    previous_end = order.groupby("Usuario_Id")["Fin_del_viaje"].shift()
    return (order["Inicio_del_viaje"] < previous_end).reindex(trips.index)


# --- cleaning steps ----------------------------------------------------------


def drop_duplicates(trips: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    out = trips.drop_duplicates()
    _record(log, "drop_duplicates", "exact duplicate rows", trips, out)
    return out


def drop_invalid_trips(
    trips: pd.DataFrame, stations: pd.DataFrame, log: CleaningLog | None = None
) -> pd.DataFrame:
    """Removes rows that are not real trips: too short, too long or too fast."""
    broken = trip_violations(trip_metrics(trips, stations))
    out = trips[~broken.any(axis=1)]
    counts = ", ".join(f"{k}={v:,}" for k, v in broken.sum().items())
    _record(log, "drop_invalid_trips", f"rules overlap: {counts}", trips, out)
    return out


def invalidate_demographics(trips: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Gender / birth year outside the contract become <NA>; the trip is kept."""
    broken = field_violations(trips)
    out = trips.assign(**{col: trips[col].mask(broken[rule]) for rule, col in FIELD_RULES.items()})
    counts = ", ".join(f"{k}={v:,}" for k, v in broken.sum().items())
    _record(log, "invalidate_demographics", counts, trips, out, int(broken.to_numpy().sum()))
    return out


def _mode_by_user(users: pd.Series, values: pd.Series) -> pd.Series:
    """Most frequent value per user; ties go to the smallest value so it is deterministic."""
    counts = (
        pd.DataFrame({"user": users, "value": values})
        .dropna()
        .groupby(["user", "value"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["user", "n", "value"], ascending=[True, False, True])
    )
    return counts.drop_duplicates("user").set_index("user")["value"]


def build_user_profiles(trips: pd.DataFrame) -> pd.DataFrame:
    """Most frequent valid gender and birth year of each user, indexed by ``Usuario_Id``.

    Only values that pass the contract vote. Needs the columns in ``PROFILE_COLUMNS``.
    """
    broken = field_violations(trips)
    gender = trips["Genero"].mask(broken["invalid_gender"])
    birth_year = trips[BIRTH_YEAR].mask(broken["invalid_age"])
    profiles = pd.concat(
        {
            "Genero": _mode_by_user(trips["Usuario_Id"], gender).astype("str"),
            BIRTH_YEAR: _mode_by_user(trips["Usuario_Id"], birth_year).astype("Int64"),
        },
        axis=1,
    )
    return profiles.rename_axis("Usuario_Id")


def impute_from_profiles(
    trips: pd.DataFrame, profiles: pd.DataFrame, log: CleaningLog | None = None
) -> pd.DataFrame:
    """Fills missing demographics from the user's profile; the rest stays <NA>."""
    cols = list(DEMOGRAPHIC_COLUMNS)
    out = trips.assign(
        **{col: trips[col].fillna(trips["Usuario_Id"].map(profiles[col])) for col in cols}
    )
    # A birth year valid for a user's recent trips can be out of range for older ones
    out[BIRTH_YEAR] = out[BIRTH_YEAR].mask(field_violations(out)["invalid_age"])
    filled = int(trips[cols].isna().to_numpy().sum() - out[cols].isna().to_numpy().sum())
    _record(
        log, "impute_from_profiles", "missing demographics <- user's own trips", trips, out, filled
    )
    return out


def drop_incomplete_demographics(
    trips: pd.DataFrame, log: CleaningLog | None = None
) -> pd.DataFrame:
    """Listwise deletion of trips without gender or birth year (alternative strategy)."""
    out = trips.dropna(subset=list(DEMOGRAPHIC_COLUMNS))
    _record(log, "drop_incomplete_demographics", "listwise deletion", trips, out)
    return out


def impute_global(trips: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Global mode (gender) and median (birth year) imputation (alternative strategy)."""
    fills = {
        "Genero": trips["Genero"].mode().iloc[0],
        BIRTH_YEAR: round(trips[BIRTH_YEAR].median()),
    }
    out = trips.fillna(fills)
    filled = int(trips[list(fills)].isna().to_numpy().sum())
    _record(log, "impute_global", f"fills: {fills}", trips, out, filled)
    return out


def clean_trips(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    profiles: pd.DataFrame,
    log: CleaningLog | None = None,
) -> pd.DataFrame:
    """The full cleaning pipeline, in the order the contract tests require."""
    return (
        trips.pipe(drop_duplicates, log=log)
        .pipe(drop_invalid_trips, stations, log=log)
        .pipe(invalidate_demographics, log=log)
        .pipe(impute_from_profiles, profiles, log=log)
    )


def validate(trips: pd.DataFrame, stations: pd.DataFrame) -> None:
    """Raises ContractError listing every rule the frame breaks."""
    if tuple(trips.columns) != TRIP_COLUMNS:
        raise ContractError(f"Expected columns {list(TRIP_COLUMNS)}, got {list(trips.columns)}")
    failures = []
    nulls = trips[list(REQUIRED_COLUMNS)].isna().sum()
    if (nulls := nulls[nulls > 0]).any():
        failures.append(f"nulls in required columns {nulls.to_dict()}")
    if n := int(trips["Viaje_Id"].duplicated().sum()):
        failures.append(f"{n:,} repeated Viaje_Id")
    broken = rule_violations(trips, trip_metrics(trips, stations)).sum()
    if (broken := broken[broken > 0]).any():
        failures.append(f"rule violations {broken.to_dict()}")
    if failures:
        raise ContractError("; ".join(failures))


# --- interim -> processed ----------------------------------------------------


def clean_file(src_file: Path, dest: Path, stations: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """Cleans one interim month, validates it and writes it atomically as Parquet."""
    log = CleaningLog()
    trips = pd.read_parquet(src_file)
    out = clean_trips(trips, stations, profiles, log)
    validate(out, stations)
    out = out.sort_values(["Inicio_del_viaje", "Viaje_Id"], ignore_index=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    try:
        out.to_parquet(tmp, index=False)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "name": src_file.name,
        "rows_in": len(trips),
        "rows_out": len(out),
        "missing_demographics": out[list(DEMOGRAPHIC_COLUMNS)].isna().sum().to_dict(),
        "steps": log.steps,
    }


def _update_report(dst: Path, reports: list[dict]) -> None:
    path = dst / REPORT_NAME
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cleaned_at = datetime.now(UTC).isoformat(timespec="seconds")
    report.update({r["name"]: {**r, "cleaned_at": cleaned_at} for r in reports})
    path.write_text(
        json.dumps(dict(sorted(report.items())), indent=2, ensure_ascii=False), "utf-8"
    )


def clean(
    src: str | Path = INTERIM_MIBICI_DIR,
    dst: str | Path = PROCESSED_MIBICI_DIR,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Cleans every interim trips file into ``dst`` (same layout), in parallel processes.

    User profiles are built from the whole history, so any newer interim file or station
    list makes every output stale. A file that fails is reported and does not stop the rest.

    Summary: {found, skipped, cleaned: [reports], failed: [{name, error}]}
    """
    src, dst = Path(src), Path(dst)
    files = trip_files(src)
    summary = {"found": len(files), "skipped": 0, "cleaned": [], "failed": []}
    if not files:
        return summary

    station_file = latest_station_file(src)
    inputs_mtime = max(p.stat().st_mtime for p in [*files, station_file])
    targets = {p: dst / p.relative_to(src) for p in files}
    pending = [
        p
        for p, dest in targets.items()
        if force or not (dest.exists() and dest.stat().st_mtime >= inputs_mtime)
    ]
    summary["skipped"] = len(files) - len(pending)
    if not pending:
        logger.success("Cleaning: every processed file is up to date")
        return summary

    stations = pd.read_parquet(station_file)
    profiles = build_user_profiles(load_trips(src, PROFILE_COLUMNS))
    logger.info("Built profiles for {:,} users", len(profiles))

    def record(path: Path, run) -> None:
        try:
            summary["cleaned"].append(run())
        except Exception as e:  # noqa: BLE001
            logger.error("Failed {}: {}", path.name, e)
            summary["failed"].append({"name": path.name, "error": str(e)})

    if workers == 1:
        for p in tqdm(pending, desc="Cleaning"):
            record(p, lambda p=p: clean_file(p, targets[p], stations, profiles))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(clean_file, p, targets[p], stations, profiles): p for p in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning"):
                record(futures[future], future.result)

    summary["cleaned"].sort(key=lambda r: r["name"])
    _update_report(dst, summary["cleaned"])
    logger.success(
        "Cleaning done: {} cleaned, {} skipped, {} failed",
        len(summary["cleaned"]),
        summary["skipped"],
        len(summary["failed"]),
    )
    return summary


@app.command()
def main(
    src: Annotated[Path, typer.Option(help="Source interim folder")] = INTERIM_MIBICI_DIR,
    dst: Annotated[Path, typer.Option(help="Destination processed folder")] = PROCESSED_MIBICI_DIR,
    force: Annotated[bool, typer.Option(help="Re-clean even if up to date")] = False,
    workers: Annotated[int, typer.Option(help="Parallel processes")] = DEFAULT_WORKERS,
):
    summary = clean(src, dst, force, workers)
    raise typer.Exit(code=1 if summary["failed"] else 0)


if __name__ == "__main__":
    app()
