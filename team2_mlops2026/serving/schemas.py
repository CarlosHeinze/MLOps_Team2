"""
Request and response schemas of the prediction service (``serving.app``).

- A request carries feature rows, one per station-hour, with the columns that
  ``features.forecast_rows`` builds and ``predict.join_profiles`` completes. The service then
  needs only the models, not the feature table.
- The bounds come from the data contract: non-negative demand, weekday 0-6 matching the hour,
  sin/cos in [-1, 1], coordinates inside Guadalajara and a positive capacity. Hours are local
  time without a time zone, on the hour, as in the feature table.
- ``null`` stands for an unknown value, as NaN does in the table: lags from before a station
  opened, the attributes of a station missing from the nomenclature. The pipeline imputes them.
  The profile scores ``pc1``-``pc5`` can be left out; a station-week without them gets the
  average profile and the indicator that flags it.
- ``to_frame`` casts the rows to the feature table's dtypes (``ROW_DTYPES``), so the models see
  exactly what the batch forecast gives them. ``to_payload`` is the inverse, for clients.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from team2_mlops2026.features import (
    CYCLICAL,
    DEMAND_FEATURES,
    FEATURE_COLUMNS,
    STATION,
    STATION_ATTRS,
    TARGETS,
    TIME,
    WEEKDAY,
)
from team2_mlops2026.modeling.train import PROFILE_COMPONENTS

# A night's forecast is ~11,000 rows (~470 active stations x 24 h); this leaves room to grow
MAX_ROWS = 25_000
# The MiBici stations lie in 20.63-20.74 N, 103.30-103.44 W; the box adds a margin around them
LATITUDE = (20.5, 20.9)
LONGITUDE = (-103.6, -103.1)

PROFILE_COLUMNS = tuple(f"pc{i}" for i in range(1, PROFILE_COMPONENTS + 1))
ROW_COLUMNS = (*(c for c in FEATURE_COLUMNS if c not in TARGETS), *PROFILE_COLUMNS)
ROW_DTYPES = {
    STATION: "int64",
    TIME: "datetime64[us]",
    **dict.fromkeys(DEMAND_FEATURES, "float32"),
    WEEKDAY: "int8",
    **dict.fromkeys(CYCLICAL, "float64"),
    **dict.fromkeys(STATION_ATTRS, "float64"),
    **dict.fromkeys(PROFILE_COLUMNS, "float32"),
}

Demand = Annotated[
    float | None,
    Field(ge=0, description="Trips (null if unknown, e.g. before the station opened)"),
]
Cyclical = Annotated[float, Field(ge=-1, le=1)]
Score = Annotated[float | None, Field(description="Weekly profile score (station_profiles)")]


class FeatureRow(BaseModel):
    """The features of one station-hour, as ``features.forecast_rows`` builds them."""

    model_config = ConfigDict(extra="forbid")

    station_id: int = Field(gt=0, description="MiBici station id")
    hour_start: datetime = Field(description="Hour to forecast, local time without time zone")
    departures_lag_2d: Demand
    departures_lag_7d: Demand
    departures_same_hour_4w: Demand
    departures_mean_7d: Demand
    arrivals_lag_2d: Demand
    arrivals_lag_7d: Demand
    arrivals_same_hour_4w: Demand
    arrivals_mean_7d: Demand
    weekday: int = Field(ge=0, le=6, description="Day of the week of hour_start, Monday = 0")
    hour_sin: Cyclical
    hour_cos: Cyclical
    month_sin: Cyclical
    month_cos: Cyclical
    latitude: float | None = Field(ge=LATITUDE[0], le=LATITUDE[1])
    longitude: float | None = Field(ge=LONGITUDE[0], le=LONGITUDE[1])
    capacity: float | None = Field(gt=0, description="Docks of the station")
    pc1: Score = None
    pc2: Score = None
    pc3: Score = None
    pc4: Score = None
    pc5: Score = None

    @field_validator("hour_start")
    @classmethod
    def _local_hour(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("must be local time without a time zone, as in the feature table")
        if (value.minute, value.second, value.microsecond) != (0, 0, 0):
            raise ValueError("must be the start of an hour (minutes and seconds 0)")
        return value

    @model_validator(mode="after")
    def _weekday_of_the_hour(self) -> FeatureRow:
        if self.weekday != self.hour_start.weekday():
            raise ValueError(
                f"weekday {self.weekday} does not match hour_start {self.hour_start:%Y-%m-%d}, "
                f"a weekday {self.hour_start.weekday()} (Monday = 0)"
            )
        return self


class PredictRequest(BaseModel):
    rows: list[FeatureRow] = Field(
        min_length=1,
        max_length=MAX_ROWS,
        description=f"Station-hours to forecast, 1 to {MAX_ROWS:,}",
    )


class Prediction(BaseModel):
    station_id: int
    hour_start: datetime
    departures: float = Field(description="Expected departures in the hour")
    arrivals: float = Field(description="Expected arrivals in the hour")
    net_flow: float = Field(description="arrivals - departures: < 0 empties the station")


class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str
    uri: str = Field(description="Versioned model URI, e.g. models:/mibici-departures/1")


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    models: dict[str, ModelInfo] = Field(description="The model version of each target")


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    models_loaded: bool
    source: str = Field(description="Where the models load from: local or registry")


class ModelCardEntry(ModelInfo):
    run_id: str = Field(description="MLflow run that logged the model")
    tags: dict[str, str] = Field(description="Registry tags: metrics, digests, git commit...")


class ModelCard(BaseModel):
    source: str
    exported_at: str | None = Field(None, description="When modeling.export wrote the folder")
    git_commit: str | None = Field(None, description="Commit of the export")
    models: dict[str, ModelCardEntry]


def to_frame(rows: list[FeatureRow]) -> pd.DataFrame:
    """The rows as a frame with the feature table's columns and dtypes (null -> NaN)."""
    frame = pd.DataFrame([row.model_dump() for row in rows], columns=list(ROW_COLUMNS))
    return frame.astype(ROW_DTYPES)


def _json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def to_payload(rows: pd.DataFrame) -> dict:
    """A ``PredictRequest`` body from feature rows such as ``forecast_rows`` returns
    (NaN -> null). Columns outside the contract are left out.

    A float32 value becomes the float64 that equals it, so ``to_frame`` restores it exactly."""
    columns = [c for c in ROW_COLUMNS if c in rows]
    records = rows[columns].to_dict("records")
    return {"rows": [{k: _json_value(v) for k, v in record.items()} for record in records]}
