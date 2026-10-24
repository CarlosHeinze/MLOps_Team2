"""Trip and station frames with the interim schema, shared by the test modules."""

import pandas as pd

from team2_mlops2026.data.raw_data_normalization import TRIP_COLUMNS
from team2_mlops2026.data.trip_data import BIRTH_YEAR

BY = BIRTH_YEAR
START = pd.Timestamp("2024-03-10 08:00:00")

# Station 2 is ~1.0 km north of station 1; station 99 is not in the nomenclature
STATIONS = pd.DataFrame(
    {
        "id": pd.array([1, 2], dtype="Int64"),
        "name": ["A", "B"],
        "latitude": [20.0, 20.009],
        "longitude": [-103.0, -103.0],
        "dpcapacity": pd.array([10, 10], dtype="Int64"),
    }
)


def make_trips(*rows: dict) -> pd.DataFrame:
    """Trips frame with the interim schema; every row defaults to a valid 10 min trip."""
    records = []
    for i, row in enumerate(rows):
        seconds = row.pop("seconds", 600)
        start = row.pop("start", START)
        records.append(
            {
                "Viaje_Id": i + 1,
                "Usuario_Id": 7,
                "Genero": "M",
                BY: 1990,
                "Inicio_del_viaje": start,
                "Fin_del_viaje": start + pd.Timedelta(seconds=seconds),
                "Origen_Id": 1,
                "Destino_Id": 2,
                **row,
            }
        )
    df = pd.DataFrame(records, columns=list(TRIP_COLUMNS))
    for col in ("Viaje_Id", "Usuario_Id", BY, "Origen_Id", "Destino_Id"):
        df[col] = df[col].astype("Int64")
    for col in ("Inicio_del_viaje", "Fin_del_viaje"):
        df[col] = df[col].astype("datetime64[ms]")
    return df
