"""
Format normalization of MiBici raw CSVs: data/raw/mibici/ -> data/interim/mibici/

MiBici publishes its monthly files with inconsistent formats (see notebook
``1.01-team2-data-format-normalization``):
- Encoding alternates between UTF-8 and Latin-1 from month to month.
- Some files have the birth year column name corrupted at the source ("A}äe_nacimiento").
- Some files use day-first dates without seconds ("01/11/2023 0:01"), which a naive
  parse silently reads as month-first.

This module only fixes FORMAT so every month shares one typed schema; it does not clean
values (duplicates, outliers, nulls), which belongs to a later step. raw stays untouched.

Output (typed Parquet), keeping the source data's naming:
    datos_abiertos_2016_01-1.csv -> viajes/2016/viajes_2016_01.parquet
    nomenclatura_2026_09.csv     -> estaciones/nomenclatura_2026_09.parquet

Every CLI run is tracked in MLflow (``normalize_tracked``): the raw CSVs as input and the
interim trips and stations as outputs, each with the same digest the neighbouring steps log, so
the runs chain into one lineage. ``--no-track`` skips it.

Usage (CLI):
    python -m team2_mlops2026.data.raw_data_normalization
    python -m team2_mlops2026.data.raw_data_normalization --force --workers 8

Usage (notebook):
    from team2_mlops2026.data.raw_data_normalization import normalize_tracked
    summary = normalize_tracked()
"""

from __future__ import annotations

import codecs
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Annotated

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from team2_mlops2026.config import INTERIM_DATA_DIR, RAW_DATA_DIR

RAW_MIBICI_DIR = RAW_DATA_DIR / "mibici"
INTERIM_MIBICI_DIR = INTERIM_DATA_DIR / "mibici"
DEFAULT_WORKERS = 4
RUN_NAME = "normalize-raw-data"

RAW_NAME_RE = re.compile(
    r"(datos_abiertos|nomenclatura)_(\d{4})_(\d{2})(?:-\d+)?\.csv", re.IGNORECASE
)
KIND_BY_PREFIX = {"datos_abiertos": "viajes", "nomenclatura": "estaciones"}

TRIP_COLUMNS = (
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
)
# Names known to be corrupted in some source files; matched by position only
TRIP_UNTRUSTED_COLUMNS = frozenset({"Año_de_nacimiento"})
TRIP_INT_COLUMNS = ("Viaje_Id", "Usuario_Id", "Año_de_nacimiento", "Origen_Id", "Destino_Id")
TRIP_DATETIME_COLUMNS = ("Inicio_del_viaje", "Fin_del_viaje")

STATION_COLUMNS = ("id", "name", "latitude", "longitude", "dpcapacity")
STATION_INT_COLUMNS = ("id", "dpcapacity")
STATION_FLOAT_COLUMNS = ("latitude", "longitude")

# Tried in order; a file must parse entirely with a single format
DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M")
# Short names for MLflow metrics, which cannot contain "%" or "/"
DATETIME_FORMAT_NAMES = dict(zip(DATETIME_FORMATS, ("iso", "day_first", "day_first_no_seconds")))

app = typer.Typer()


class SchemaError(ValueError):
    """A raw file does not match the expected columns."""


@dataclass(frozen=True)
class RawFile:
    path: Path
    kind: str  # viajes (trips) | estaciones (stations)
    year: str
    month: str

    @property
    def dest_rel(self) -> str:
        if self.kind == "viajes":
            return f"viajes/{self.year}/viajes_{self.year}_{self.month}.parquet"
        return f"estaciones/nomenclatura_{self.year}_{self.month}.parquet"


def classify(path: Path) -> RawFile | None:
    """Parses a raw MiBici name; None if it is not a known CSV."""
    if not (m := RAW_NAME_RE.fullmatch(path.name)):
        return None
    prefix, year, month = m.groups()
    return RawFile(path, KIND_BY_PREFIX[prefix.lower()], year, month)


def detect_encoding(path: Path, chunk_size: int = 1 << 20) -> str:
    """Returns "utf-8" if the whole file decodes as UTF-8, otherwise "latin-1"."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8"


def _reader_encoding(encoding: str) -> str:
    # utf-8-sig also strips a BOM if present
    return "utf-8-sig" if encoding == "utf-8" else encoding


def parse_datetimes(values: pd.Series) -> tuple[pd.Series, str]:
    """Parses with the first known format that accepts every non-null value."""
    present = values.notna()
    for fmt in DATETIME_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed[present].notna().all():
            # Fixed resolution so every month writes the same Parquet schema
            return parsed.astype("datetime64[s]"), fmt
    sample = values[present].iloc[0] if present.any() else None
    raise ValueError(f"No single known datetime format fits the values, e.g. {sample!r}")


def normalize_columns(
    columns: Sequence[str],
    expected: Sequence[str],
    untrusted: frozenset[str] = frozenset(),
) -> list[str]:
    """Maps source columns to the canonical names by position.

    Every name must match its expected one, except those in ``untrusted`` (known to be
    corrupted at the source), so a reordered or different file is still rejected.
    """
    if len(columns) != len(expected):
        raise SchemaError(f"Expected {len(expected)} columns, got {len(columns)}: {columns}")
    mismatches = [
        (got, want)
        for got, want in zip(columns, expected)
        if want not in untrusted and got.strip() != want
    ]
    if mismatches:
        raise SchemaError(f"Unexpected columns (got, expected): {mismatches}")
    return list(expected)


def _to_int(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").astype("Int64")


def normalize_trips(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Canonical columns and dtypes for a trips frame read as strings.

    Returns (frame, datetime_format).
    """
    df = df.set_axis(normalize_columns(df.columns, TRIP_COLUMNS, TRIP_UNTRUSTED_COLUMNS), axis=1)
    formats = set()
    for col in TRIP_DATETIME_COLUMNS:
        df[col], fmt = parse_datetimes(df[col])
        formats.add(fmt)
    if len(formats) > 1:
        raise ValueError(f"Start and end use different datetime formats: {formats}")
    for col in TRIP_INT_COLUMNS:
        df[col] = _to_int(df[col])
    return df, formats.pop()


def normalize_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical columns and dtypes for a stations frame read as strings."""
    df = df.set_axis(normalize_columns(df.columns, STATION_COLUMNS), axis=1)
    for col in STATION_INT_COLUMNS:
        df[col] = _to_int(df[col])
    for col in STATION_FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    return df


def inspect_raw_file(path: Path) -> dict:
    """Format fingerprint of a raw trips CSV: encoding, header, quoting, line ending, dates.

    Used by notebook 1.01 to show why normalization is needed.
    """
    encoding = detect_encoding(path)
    with path.open("rb") as f:
        head = f.read(4096)
    lines = head.decode(_reader_encoding(encoding), errors="replace").splitlines()
    sample = next(csv.reader(lines[1:2]), [])
    start = sample[TRIP_COLUMNS.index("Inicio_del_viaje")] if len(sample) > 4 else None
    try:
        datetime_format = parse_datetimes(pd.Series([start]))[1] if start else None
    except ValueError:
        datetime_format = "unknown"
    return {
        "name": path.name,
        "encoding": encoding,
        "header": lines[0] if lines else "",
        "quoted": head.startswith(b'"'),
        "line_ending": "CRLF" if b"\r\n" in head else "LF",
        "datetime_format": datetime_format,
        "sample_start": start,
    }


def normalize_file(raw: RawFile, dst: Path) -> dict:
    """Reads one raw CSV, normalizes it and writes it atomically as Parquet."""
    encoding = detect_encoding(raw.path)
    df = pd.read_csv(raw.path, encoding=_reader_encoding(encoding), dtype=str)
    datetime_format = None
    if raw.kind == "viajes":
        df, datetime_format = normalize_trips(df)
    else:
        df = normalize_stations(df)

    dest = dst / raw.dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "name": raw.path.name,
        "kind": raw.kind,
        "encoding": encoding,
        "datetime_format": datetime_format,
        "rows": len(df),
        "dest_rel": raw.dest_rel,
    }


def is_up_to_date(src: Path, dest: Path) -> bool:
    return dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime


def normalize(
    src: str | Path = RAW_MIBICI_DIR,
    dst: str | Path = INTERIM_MIBICI_DIR,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Normalizes every raw CSV in ``src`` into typed Parquet files under ``dst``.

    Files are processed in parallel processes (``workers=1`` runs inline). A file that
    fails is reported and does not stop the rest.

    Summary: {found, skipped, normalized: [reports], failed: [{name, error}]}
    Each report: {name, kind, encoding, datetime_format, rows, dest_rel}
    """
    src, dst = Path(src), Path(dst)
    raw_files = [f for p in sorted(src.glob("*.csv")) if (f := classify(p))]

    # Two raw files for the same month would silently overwrite each other
    duplicated = [d for d, n in Counter(f.dest_rel for f in raw_files).items() if n > 1]
    if duplicated:
        raise ValueError(f"Several raw files map to the same output: {duplicated}")

    pending = [f for f in raw_files if force or not is_up_to_date(f.path, dst / f.dest_rel)]
    summary = {
        "found": len(raw_files),
        "skipped": len(raw_files) - len(pending),
        "normalized": [],
        "failed": [],
    }

    def record(raw: RawFile, run) -> None:
        try:
            summary["normalized"].append(run())
        except Exception as e:  # noqa: BLE001
            logger.error("Failed {}: {}", raw.path.name, e)
            summary["failed"].append({"name": raw.path.name, "error": str(e)})

    if workers == 1:
        for raw in tqdm(pending, desc="Normalizing"):
            record(raw, lambda raw=raw: normalize_file(raw, dst))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(normalize_file, raw, dst): raw for raw in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing"):
                record(futures[future], future.result)

    summary["normalized"].sort(key=lambda r: r["dest_rel"])
    logger.success(
        "Normalization done: {} normalized, {} skipped, {} failed",
        len(summary["normalized"]),
        summary["skipped"],
        len(summary["failed"]),
    )
    return summary


def normalize_tracked(
    src: str | Path = RAW_MIBICI_DIR,
    dst: str | Path = INTERIM_MIBICI_DIR,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """``normalize()`` inside an MLflow run of the project's experiment.

    Logs the raw CSVs as input and the interim trips and latest stations as outputs, exactly
    as the extraction and cleaning steps log them, so the digests match across runs. The
    options are logged as params, the file and row counts as metrics (encodings and date
    formats of the files normalized in this run) and the per-file report as artifact. The
    summary gets ``mlflow_run_id``.
    """
    # Imported here: mlflow takes seconds to import and the worker processes never need it
    import mlflow

    from team2_mlops2026 import tracking
    from team2_mlops2026.data.raw_data_extraction import RAW_DATASET, raw_csv_files
    from team2_mlops2026.data.trip_data import latest_station_file, trip_files

    src, dst = Path(src), Path(dst)
    with tracking.start_run(RUN_NAME) as run:
        tracking.log_files_dataset(raw_csv_files(src), src, RAW_DATASET, "raw")
        mlflow.log_params({"force": force, "workers": workers})

        summary = normalize(src, dst, force, workers)

        tracking.log_files_dataset(
            trip_files(dst), dst / "viajes", "mibici_trips_interim", "interim"
        )
        station_file = latest_station_file(dst)
        tracking.log_dataset(
            pd.read_parquet(station_file), station_file, "mibici_stations", "stations"
        )
        reports = summary["normalized"]
        encodings = Counter(r["encoding"] for r in reports)
        formats = Counter(r["datetime_format"] for r in reports if r["datetime_format"])
        mlflow.log_metrics(
            {
                "files_found": summary["found"],
                "files_skipped": summary["skipped"],
                "files_normalized": len(reports),
                "files_failed": len(summary["failed"]),
                "rows_normalized": sum(r["rows"] for r in reports),
                **{f"files_{enc.replace('-', '')}": n for enc, n in encodings.items()},
                **{f"files_datetime_{DATETIME_FORMAT_NAMES[f]}": n for f, n in formats.items()},
            }
        )
        if reports:
            mlflow.log_dict(reports, "normalized_files.json")
        if summary["failed"]:
            mlflow.log_dict(summary["failed"], "failed_files.json")
    summary["mlflow_run_id"] = run.info.run_id
    return summary


@app.command()
def main(
    src: Annotated[Path, typer.Option(help="Source raw folder")] = RAW_MIBICI_DIR,
    dst: Annotated[Path, typer.Option(help="Destination interim folder")] = INTERIM_MIBICI_DIR,
    force: Annotated[bool, typer.Option(help="Re-normalize even if up to date")] = False,
    workers: Annotated[int, typer.Option(help="Parallel processes")] = DEFAULT_WORKERS,
    track: Annotated[bool, typer.Option(help="Log the run to the MLflow server")] = True,
):
    summary = (normalize_tracked if track else normalize)(src, dst, force, workers)
    raise typer.Exit(code=1 if summary["failed"] else 0)


if __name__ == "__main__":
    app()
