"""
Shared access to the MiBici trips at any stage (interim or processed) and the quantities
derived from them (duration, distance, speed, age).

Nothing here encodes a cleaning decision, so the cleaning rules
(``interim_data_cleaning``) can change without touching the EDA that reads its output.

Usage (notebook):
    from team2_mlops2026.data.trip_data import PROCESSED_MIBICI_DIR, load_trips
    trips = load_trips(PROCESSED_MIBICI_DIR)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from team2_mlops2026.config import PROCESSED_DATA_DIR
from team2_mlops2026.data.raw_data_normalization import INTERIM_MIBICI_DIR, TRIP_COLUMNS

PROCESSED_MIBICI_DIR = PROCESSED_DATA_DIR / "mibici"

BIRTH_YEAR = "Año_de_nacimiento"
DEMOGRAPHIC_COLUMNS = ("Genero", BIRTH_YEAR)
# Every other column identifies the trip and may never be null
REQUIRED_COLUMNS = tuple(c for c in TRIP_COLUMNS if c not in DEMOGRAPHIC_COLUMNS)

EARTH_RADIUS_KM = 6371.0088


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
    """Duration, distance, speed and age of each trip.

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
