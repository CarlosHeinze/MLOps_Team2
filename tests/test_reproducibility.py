"""Reproducibility: the golden reference set, the lock file, and a snapshot and its comparison on
the synthetic champions of ``test_registry``."""

import json
from pathlib import Path
import re

import mlflow
import numpy as np
import pytest
from test_backtest import FIRST_TEST_MONTH
from test_registry import registered  # noqa: F401  (fixture: champions in a throwaway store)

from team2_mlops2026 import features as ft
from team2_mlops2026.config import MODELS_DIR, PROJ_ROOT
from team2_mlops2026.modeling import export as ex
from team2_mlops2026.modeling import predict as pr
from team2_mlops2026.modeling import registry as reg
from team2_mlops2026.modeling import reproducibility as rp
from team2_mlops2026.serving import app as api
from team2_mlops2026.serving.schemas import PredictRequest, PredictResponse, to_frame

REFERENCE = rp.read_reference()


# --- golden reference set -----------------------------------------------------------


def test_the_reference_request_still_fits_the_api_and_matches_its_forecasts():
    request = PredictRequest.model_validate(REFERENCE["request"])
    response = PredictResponse.model_validate(REFERENCE["response"])
    assert [(r.station_id, r.hour_start) for r in request.rows] == [
        (p.station_id, p.hour_start) for p in response.predictions
    ]
    for p in response.predictions:
        assert p.departures >= 0 and p.arrivals >= 0
        assert p.net_flow == pytest.approx(p.arrivals - p.departures, abs=1e-12)
    assert {t: m.uri for t, m in response.models.items()} == {
        t: m["uri"] for t, m in REFERENCE["models"].items()
    }


def test_the_reference_is_the_documented_example_of_the_champions():
    """docs/api and tests/data come from the same champions: regenerate both (make api-docs,
    make snapshot) when the champion changes."""
    examples = {
        name: json.loads((api.API_DOCS_DIR / name).read_text(encoding="utf-8"))
        for name in (api.EXAMPLE_REQUEST, api.EXAMPLE_RESPONSE)
    }
    assert REFERENCE["request"] == examples[api.EXAMPLE_REQUEST]
    assert REFERENCE["response"] == examples[api.EXAMPLE_RESPONSE]
    for target, entry in REFERENCE["models"].items():
        assert entry["uri"] == f"models:/{reg.model_name(target)}/{entry['version']}"
        assert entry["test_metrics"]["overall_rows"] > 0 and entry["features_digest"]
    assert REFERENCE["snapshot_run"] and REFERENCE["snapshot_digest"]


def _exported_reference_versions() -> bool:
    try:
        card = ex.read_card(MODELS_DIR)
    except FileNotFoundError:
        return False
    return {t: m["uri"] for t, m in card["models"].items()} == {
        t: m["uri"] for t, m in REFERENCE["models"].items()
    }


@pytest.mark.skipif(
    not _exported_reference_versions(),
    reason="models/ does not hold the reference champions (make export or make docker-models)",
)
def test_the_exported_champions_give_the_golden_forecasts():
    models, _ = ex.load_exported(MODELS_DIR)
    rows = to_frame(PredictRequest.model_validate(REFERENCE["request"]).rows)
    served = pr.forecast(models, rows)
    expected = REFERENCE["response"]["predictions"]
    for column in [*ft.TARGETS, pr.NET_FLOW]:
        np.testing.assert_allclose(
            served[column], [p[column] for p in expected], rtol=api.CHECK_RTOL, atol=api.CHECK_ATOL
        )


# --- lock file --------------------------------------------------------------------------


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, pinned = line.split("==", 1)
            pins[re.sub(r"[-_.]+", "-", name.strip()).lower()] = pinned.strip()
    return pins


def test_the_lock_file_holds_every_pin_at_its_version():
    lock = _pins(PROJ_ROOT / "requirements-lock.txt")
    for requirements in ("requirements.txt", "requirements-serving.txt"):
        for name, pinned in _pins(PROJ_ROOT / requirements).items():
            assert lock.get(name) == pinned, f"{name} in {requirements}: run `make lock`"
    assert len(lock) > 100  # the transitive dependencies too


# --- comparisons ----------------------------------------------------------------------


def test_metric_differences_are_relative_and_skip_what_was_not_reproduced():
    reference = {"wape": 0.5, "mae": 0.0, "rows": 10.0, "only_reference": 1.0}
    reproduced = {"wape": 0.5005, "mae": 0.0, "rows": 10.0}
    differences = rp.metric_differences(reproduced, reference)
    assert differences == pytest.approx({"wape": 1e-3, "mae": 0.0, "rows": 0.0})
    assert rp.metric_rtol("seasonal_naive") == 1e-9
    assert rp.metric_rtol("lgbm_tweedie") < rp.metric_rtol("hgb_poisson")
    assert rp.metric_rtol("a_new_model") == rp.DEFAULT_METRIC_RTOL


def test_the_environment_names_the_machine_and_the_packages():
    env = rp.environment()
    assert {"os", "cpu", "cpu_count", "python", "lightgbm", "scikit-learn", "git_commit"} <= set(
        env
    )
    assert env["lightgbm"] != "not installed"


def test_an_unreachable_service_fails_the_check():
    result = rp.check_service("http://127.0.0.1:9", REFERENCE)
    assert result["passed"] is False and result["error"]


def test_a_snapshot_is_versioned_and_reproduced_from_its_tables(registered, tmp_path):  # noqa: F811
    src, models_dir = registered["src"], tmp_path / "models"
    ex.export_models(dst=models_dir)
    reference = rp.snapshot_tracked(models_dir, src, tmp_path / "reference.json")

    # The snapshot run holds the tables and the reference, and the versions point to it
    run = mlflow.get_run(reference["snapshot_run"])
    assert run.info.run_name == f"snapshot-{reference['snapshot_digest'][:12]}"
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id, "snapshot")}
    assert artifacts == {f"snapshot/{ft.FEATURES_FILE}", "snapshot/station_profiles.parquet"}
    assert reference["test_start"] == FIRST_TEST_MONTH
    assert json.loads((tmp_path / "reference.json").read_text()) == reference
    for target in ft.TARGETS:
        version = mlflow.MlflowClient().get_model_version_by_alias(
            reg.model_name(target), reg.CHAMPION
        )
        assert version.tags[rp.SNAPSHOT_TAG] == reference["snapshot_run"]
        assert reference["models"][target]["features_digest"] == version.tags["features_digest"]
    assert {r["model"] for r in reference["train_runs"].values()} == {
        "seasonal_naive",
        "historical_average",
    }

    # Same machine, same tables: the refit and the train runs reproduce the reference
    result = rp.compare_tracked(
        tmp_path / "reference.json", src=src, models_dir=models_dir, train_runs=True
    )
    assert result["passed"], rp.summary(result)
    for r in result["retrain"].values():
        assert r["max_metric_rel_diff"] <= 1e-9 and r["exported_max_metric_rel_diff"] <= 1e-9
    assert set(result["train_runs"]) == set(reference["train_runs"])
    run = mlflow.get_run(result["run_id"])
    assert run.info.run_name == "reproduce-lgbm_tweedie"
    assert run.data.tags["reproduced"] == "yes"
    assert run.data.params["env_python"] == rp.environment()["python"]


def test_a_changed_table_is_not_compared(registered, tmp_path):  # noqa: F811
    models_dir = tmp_path / "models"
    ex.export_models(dst=models_dir)
    reference = rp.build_reference(models_dir, registered["src"])
    reference["snapshot_run"] = "run"
    reference["models"]["departures"]["features_digest"] = "0" * 32
    with pytest.raises(ValueError, match="are not the snapshot"):
        rp.retrain("departures", reference, registered["src"], models_dir)
