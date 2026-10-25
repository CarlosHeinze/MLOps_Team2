import mlflow
import pandas as pd
import pytest
import requests
from trip_factory import make_trips

from team2_mlops2026 import tracking
from team2_mlops2026.config import PROJ_ROOT


class FakeHealth:
    def raise_for_status(self):
        pass


@pytest.fixture
def parquet_dir(tmp_path):
    root = tmp_path / "viajes"
    for month in ("01", "02"):
        path = root / "2024" / f"viajes_2024_{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_trips({}, {"Viaje_Id": 2, "Genero": None}).to_parquet(path, index=False)
    return root


def dataset_inputs(run_id: str) -> dict:
    """{context: dataset} of the inputs logged by a run."""
    inputs = mlflow.get_run(run_id).inputs.dataset_inputs
    return {
        next(t.value for t in i.tags if t.key == "mlflow.data.context"): i.dataset for i in inputs
    }


# --- server ------------------------------------------------------------------


def test_check_tracking_server_skips_local_stores(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("no request expected"))
    tracking.check_tracking_server("sqlite:///mlflow.db")


def test_check_tracking_server_accepts_healthy_server(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "get", lambda url, **k: calls.append(url) or FakeHealth())
    tracking.check_tracking_server("http://127.0.0.1:5000/")
    assert calls == ["http://127.0.0.1:5000/health"]


def test_check_tracking_server_explains_how_to_start_it(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    with pytest.raises(RuntimeError, match="make mlflow"):
        tracking.check_tracking_server("http://127.0.0.1:5000")


def test_start_run_rejects_mlflow_run_without_server(mlflow_store, monkeypatch):
    mlflow.set_tracking_uri(mlflow_store)
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc")
    with pytest.raises(RuntimeError, match="not using the MLflow server"):
        tracking.start_run()


def test_start_run_uses_project_experiment(mlflow_store):
    with tracking.start_run("step") as run:
        pass
    assert run.info.run_name == "step"
    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name == tracking.MLFLOW_EXPERIMENT_NAME


# --- sources and digests -----------------------------------------------------


def test_dataset_source_is_portable(tmp_path):
    assert tracking.dataset_source(PROJ_ROOT / "data" / "x.csv") == "data/x.csv"
    assert tracking.dataset_source(tmp_path / "x.csv") == (tmp_path / "x.csv").resolve().as_posix()
    assert tracking.dataset_source("s3://bucket/x.csv") == "s3://bucket/x.csv"


def test_files_digest_tracks_content_names_and_membership(parquet_dir):
    files = sorted(parquet_dir.rglob("*.parquet"))
    digest = tracking.files_digest(files, parquet_dir)
    assert digest == tracking.files_digest(reversed(files), parquet_dir)
    assert len(digest) == tracking.DIGEST_LENGTH

    assert tracking.files_digest(files[:1], parquet_dir) != digest
    renamed = files[1].with_name("viajes_2024_03.parquet")
    files[1].rename(renamed)
    assert tracking.files_digest([files[0], renamed], parquet_dir) != digest
    renamed.write_bytes(renamed.read_bytes() + b"x")
    assert tracking.files_digest([files[0], renamed], parquet_dir) != digest


def test_parquet_rows_reads_footers(parquet_dir):
    assert tracking.parquet_rows(parquet_dir.rglob("*.parquet")) == 4


# --- logging -----------------------------------------------------------------


def test_log_files_dataset_logs_digest_schema_and_context(mlflow_store, parquet_dir):
    files = list(parquet_dir.rglob("*.parquet"))
    with tracking.start_run() as run:
        tracking.log_files_dataset(files, parquet_dir, "trips", "interim")

    dataset = dataset_inputs(run.info.run_id)["interim"]
    assert dataset.name == "trips"
    assert dataset.digest == tracking.files_digest(files, parquet_dir)
    assert "Viaje_Id" in dataset.schema and "Genero" in dataset.schema


def test_log_files_dataset_rejects_empty_folders(mlflow_store, tmp_path):
    with tracking.start_run(), pytest.raises(ValueError, match="No files"):
        tracking.log_files_dataset([], tmp_path, "trips", "interim")


def test_log_dataset_logs_small_frames(mlflow_store, tmp_path):
    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    with tracking.start_run() as run:
        tracking.log_dataset(df, tmp_path / "stations.parquet", "stations", "stations")
    dataset = dataset_inputs(run.info.run_id)["stations"]
    assert dataset.name == "stations" and dataset.digest
