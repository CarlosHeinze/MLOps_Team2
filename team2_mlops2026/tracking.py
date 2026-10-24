"""Helpers for MLflow dataset tracking (https://mlflow.org/docs/latest/ml/dataset/).

Every data step logs the datasets it reads and writes with ``mlflow.log_input``, so each run
records their name, digest, schema and source.

The MiBici trips are folders of monthly Parquet files (~38M rows), so they are logged with
``log_files_dataset`` as metadata plus a digest of the file contents. ``mlflow.data.from_pandas``
would need the whole frame in memory and its digest only covers the first 10,000 rows, so a
change in a later month could go unnoticed. ``log_dataset`` (from_pandas) remains for small frames.
"""

from collections.abc import Iterable
import hashlib
import os
from pathlib import Path
import warnings

import mlflow
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.meta_dataset import MetaDataset
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import pandas as pd
import pyarrow.parquet as pq
import requests

from team2_mlops2026.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, PROJ_ROOT

DIGEST_LENGTH = 32  # hex characters of the SHA-256
SCHEMA_SAMPLE_ROWS = 1_000


def check_tracking_server(uri: str | None = None) -> None:
    """Fail fast with a clear message if the MLflow tracking server is not running."""
    uri = uri or MLFLOW_TRACKING_URI
    if not uri.startswith("http"):
        return  # a local file/database store needs no server
    try:
        requests.get(f"{uri.rstrip('/')}/health", timeout=3).raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"MLflow tracking server not reachable at {uri}. Start it from the project root with:\n"
            "    mlflow server --host 127.0.0.1 --port 5000"
        ) from exc


def start_run(run_name: str | None = None):
    """Start (or resume) the MLflow run for a pipeline step.

    Under `mlflow run` the run already exists (MLFLOW_RUN_ID is set) and is reused. When a step
    runs from its CLI or a notebook, a new run goes to the project's experiment on the
    configured tracking server.
    """
    under_mlflow_run = "MLFLOW_RUN_ID" in os.environ
    if under_mlflow_run and not mlflow.get_tracking_uri().startswith("http"):
        raise RuntimeError(
            "`mlflow run` is not using the MLflow server. Set the tracking URI in this terminal "
            "and run it again:\n"
            '    PowerShell:   $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"\n'
            "    macOS/Linux:  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000"
        )
    if under_mlflow_run:
        check_tracking_server(mlflow.get_tracking_uri())
        return mlflow.start_run()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    check_tracking_server(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return mlflow.start_run(run_name=run_name)


def dataset_source(path) -> str:
    """Return a portable source string: relative to the project root when possible."""
    path_str = str(path)
    if ":/" in path_str and not Path(path_str).drive:  # a URI such as s3://... or file:///...
        return path_str
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def files_digest(files: Iterable[Path], root: Path) -> str:
    """Digest of every file's path (relative to ``root``) and bytes.

    It changes whenever any file is added, removed, renamed or modified.
    """
    sha = hashlib.sha256()
    for path in sorted(Path(p) for p in files):
        sha.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
    return sha.hexdigest()[:DIGEST_LENGTH]


def parquet_rows(files: Iterable[Path]) -> int:
    """Row count from the Parquet footers, without reading the data."""
    return sum(pq.ParquetFile(p).metadata.num_rows for p in files)


def _parquet_schema(path: Path):
    sample = pq.ParquetFile(path).read_row_group(0).slice(0, SCHEMA_SAMPLE_ROWS).to_pandas()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # hint that integer columns cannot hold NaN
        return infer_signature(sample).inputs


def log_files_dataset(files: Iterable[Path], root, name: str, context: str) -> MetaDataset:
    """Log a folder of files as an input of the active MLflow run, without loading it.

    The schema is read from a sample of the first file when the files are Parquet.
    """
    files = sorted(Path(p) for p in files)
    if not files:
        raise ValueError(f"No files to log for dataset {name!r} under {root}")
    schema = _parquet_schema(files[0]) if files[0].suffix == ".parquet" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "source can be interpreted in multiple ways"
        source = resolve_dataset_source(dataset_source(root))
    dataset = MetaDataset(source, name=name, digest=files_digest(files, Path(root)), schema=schema)
    mlflow.log_input(dataset, context=context)
    return dataset


def log_dataset(df: pd.DataFrame, source, name: str, context: str) -> None:
    """Log a small ``df`` as an input of the active MLflow run."""
    with warnings.catch_warnings():
        # Ambiguous-source and integer-column hints; the schema is only inferred at log_input
        warnings.simplefilter("ignore")
        try:
            dataset = mlflow.data.from_pandas(df, source=dataset_source(source), name=name)
        except MlflowException:
            # MLflow cannot interpret this kind of source: log the dataset (digest, schema) without it
            dataset = mlflow.data.from_pandas(df, name=name)
        mlflow.log_input(dataset, context=context)
