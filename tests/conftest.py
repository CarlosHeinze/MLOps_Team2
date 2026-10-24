import mlflow
import pytest

from team2_mlops2026 import tracking


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Tracks into a throwaway SQLite store instead of the MLflow server."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.chdir(tmp_path)  # the default artifact root (./mlruns) lands in tmp_path
    yield uri
    mlflow.end_run()
