import mlflow
import mlflow.models
import pytest

from team2_mlops2026 import tracking


@pytest.fixture(autouse=True, scope="session")
def no_pip_requirements_inference():
    """Every logged model gets its flavor's default pip requirements.

    MLflow infers them by loading the model in a subprocess, ~6 s per model on Windows, which was
    most of the suite's time. No test reads them.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            mlflow.models, "infer_pip_requirements", lambda *_, fallback=None, **__: fallback or []
        )
        yield


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Tracks into a throwaway SQLite store instead of the MLflow server."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.chdir(tmp_path)  # the default artifact root (./mlruns) lands in tmp_path
    yield uri
    mlflow.end_run()
