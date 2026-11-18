"""
Reproducibility of the deployable model: snapshot, reference outputs and their comparison.
models/ + data/processed/mibici/features/ -> a ``snapshot-<digest>`` run and tests/data/reference.json
-> ``reproduce-<model>`` runs in other environments

- ``snapshot`` versions what the champions need to be rebuilt and checked, in one MLflow run:
  - the feature table and the station profiles that trained them, as Parquet files with their
    digests. Their digests must be the ones the registered versions record.
  - the reference set: the request rows of ``docs/api/`` and the champions' forecasts for them,
    the champions' test metrics, the metrics of the latest ``train-<model>-<target>`` runs and
    the environment. It is also written to ``tests/data/reference.json`` for the golden test,
    so other machines can compare against it without the MLflow server.
  The registered versions get the tag ``snapshot_run`` with the run id.
- ``compare`` checks another environment against the reference:
  - ``--retrain``: refits each champion with its own params (an unfitted ``clone`` of the
    exported model) on the training split of the local feature table, once its digests match,
    and compares the test metrics with the reference. The exported champion is scored too, so a
    difference in the data and one in the training show up apart.
  - ``--train-runs``: compares the latest local ``train-<model>-<target>`` runs (``make
    train``) with the reference ones.
  - ``--url``: sends the reference rows to a running service (local or a container) and
    compares the forecasts with the expected ones.
  - It logs a ``reproduce-<model>`` run with the environment (OS, CPU, Python, packages,
    commit) and the differences, unless ``--no-track``.
- Tolerances (``METRIC_RTOL``, canvas §8): the baselines are deterministic. LightGBM sums its
  histograms per thread, so a refit with another number of threads moves the forecasts by up to
  ~0.003 trips and the metrics by ~1e-5 (notebook 3.02). scikit-learn's histogram boosting
  differed in the third decimal between two laptops in Phase 2.

Usage (CLI):
    python -m team2_mlops2026.modeling.reproducibility snapshot
    python -m team2_mlops2026.modeling.reproducibility compare --no-track
    python -m team2_mlops2026.modeling.reproducibility compare --no-retrain --url http://127.0.0.1:8000
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Annotated

from loguru import logger
import numpy as np
from sklearn.base import clone
import typer

from team2_mlops2026.config import MODELS_DIR, PROJ_ROOT
from team2_mlops2026.features import FEATURES_DIR, FEATURES_FILE, TARGETS
from team2_mlops2026.modeling import export
from team2_mlops2026.modeling.evaluate import TEST_START, segment_report
from team2_mlops2026.modeling.models import BASELINES, MODELS
from team2_mlops2026.modeling.train import RUN_PREFIX as TRAIN_PREFIX
from team2_mlops2026.modeling.train import (
    load_training_data,
    log_inputs,
    model_inputs,
    report_metrics,
)
from team2_mlops2026.station_profiles import PROFILES_FILE

REFERENCE_FILE = PROJ_ROOT / "tests" / "data" / "reference.json"
SNAPSHOT_PREFIX, REPRODUCE_PREFIX = "snapshot", "reproduce"
SNAPSHOT_TAG = "snapshot_run"
SNAPSHOT_ARTIFACT, REFERENCE_ARTIFACT = "snapshot", "reference"
# Relative tolerance of a reproduced test metric (canvas §8, notebook 3.02)
METRIC_RTOL = {
    **dict.fromkeys(BASELINES, 1e-9),  # deterministic arithmetic
    "poisson_glm": 1e-6,  # L-BFGS converges to the same optimum up to its tolerance
    "lgbm_poisson": 1e-4,  # the thread count changes the order of LightGBM's histogram sums
    "lgbm_tweedie": 1e-4,
    "hgb_poisson": 5e-3,  # differed in the third decimal between two laptops (Phase 2)
    "hgb_squared_error": 5e-3,
}
DEFAULT_METRIC_RTOL = 5e-3
# A refit with 4 threads instead of 12 moved the champion's forecasts by up to 0.003 trips
PREDICTION_ATOL = 0.01
TRAIN_METRICS = ("overall_wape", "peak_wape", "overall_mae", "overall_poisson_deviance")
PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "lightgbm",
    "skops",
    "mlflow-skinny",
    "fastapi",
)

app = typer.Typer()


# --- environment and digests ------------------------------------------------------


def _cpu() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
            )
            return out.stdout.strip() or platform.processor()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def environment() -> dict[str, str]:
    """Where a result was produced: OS, CPU, Python, package versions and commit."""
    from team2_mlops2026 import tracking

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu(),
        "cpu_count": str(os.cpu_count()),
        "container": str(Path("/.dockerenv").exists() or "KUBERNETES_SERVICE_HOST" in os.environ),
        "python": platform.python_version(),
        **{package: _version(package) for package in PACKAGES},
        "git_commit": tracking.git_commit(),
    }


def table_digests(src: str | Path = FEATURES_DIR) -> dict[str, str]:
    """The digests of the feature table and the profiles, as the model runs record them."""
    from team2_mlops2026 import tracking

    src = Path(src)
    return {
        "features_digest": tracking.files_digest([src / FEATURES_FILE], src),
        "profiles_digest": tracking.files_digest([src / PROFILES_FILE], src),
    }


# --- comparisons ------------------------------------------------------------------


def metric_differences(reproduced: dict, reference: dict) -> dict[str, float]:
    """The relative difference of every metric present in both (0 when both are 0)."""
    return {
        key: abs(reproduced[key] - value) / abs(value) if value else abs(reproduced[key])
        for key, value in reference.items()
        if key in reproduced
    }


def metric_rtol(model: str) -> float:
    return METRIC_RTOL.get(model, DEFAULT_METRIC_RTOL)


def read_reference(path: str | Path = REFERENCE_FILE) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No reference at {path}: create it with `make snapshot`")
    return json.loads(path.read_text(encoding="utf-8"))


def retrain(target: str, reference: dict, src=FEATURES_DIR, models_dir=MODELS_DIR) -> dict:
    """Refits the exported champion of ``target`` on the local training split and compares its
    test metrics, and those of the exported model itself, with the reference."""
    entry = reference["models"][target]
    local = table_digests(src)
    expected = {k: entry[k] for k in local}
    if local != expected:
        raise ValueError(
            f"The local tables in {src} are not the snapshot {reference['snapshot_run']}: "
            f"{local} instead of {expected}. Rebuild them with `make features` or download "
            "the snapshot run's artifacts."
        )
    models, card = export.load_exported(models_dir, [target])
    if card["models"][target]["uri"] != entry["uri"]:
        raise ValueError(
            f"models/ holds {card['models'][target]['uri']}, the reference {entry['uri']}: "
            "run `make export` or `make docker-models` for that version"
        )
    champion = models[target]
    train, test = load_training_data(src, reference["test_start"])
    X_test = model_inputs(test)
    start = time.perf_counter()
    refit = clone(champion).fit(model_inputs(train), train[target])
    fit_seconds = time.perf_counter() - start

    champion_pred, refit_pred = champion.predict(X_test), refit.predict(X_test)
    scored = report_metrics(segment_report(test, champion_pred, target, train))
    reproduced = report_metrics(segment_report(test, refit_pred, target, train))
    rtol = metric_rtol(entry["model"])
    differences = metric_differences(reproduced, entry["test_metrics"])
    exported_differences = metric_differences(scored, entry["test_metrics"])
    prediction_diff = float(np.abs(refit_pred - champion_pred).max())
    result = {
        "model": entry["model"],
        "uri": entry["uri"],
        "fit_seconds": fit_seconds,
        "rtol": rtol,
        "metrics": reproduced,
        "max_metric_rel_diff": max(differences.values()),
        "exported_max_metric_rel_diff": max(exported_differences.values()),
        "max_prediction_diff": prediction_diff,
        "identical_predictions": bool(np.array_equal(refit_pred, champion_pred)),
    }
    result["passed"] = (
        result["max_metric_rel_diff"] <= rtol
        and result["exported_max_metric_rel_diff"] <= rtol
        and prediction_diff <= PREDICTION_ATOL
    )
    return result


def compare_train_runs(reference: dict) -> dict[str, dict]:
    """The latest local ``train-<model>-<target>`` runs against the reference ones."""
    from team2_mlops2026 import tracking

    results = {}
    for name, expected in reference["train_runs"].items():
        try:
            run = tracking.latest_run(name)
        except LookupError:
            continue
        model = expected["model"]
        differences = metric_differences(run.data.metrics, expected["metrics"])
        worst = max(differences.values())
        results[name] = {
            "model": model,
            "max_metric_rel_diff": worst,
            "rtol": metric_rtol(model),
            "passed": worst <= metric_rtol(model),
        }
    return results


def check_service(url: str, reference: dict) -> dict:
    """The reference rows through a running service, against the expected forecasts."""
    from team2_mlops2026.serving.app import _http_json, compare_predictions

    try:
        served = _http_json(f"{url}/predict", reference["request"], timeout=60)
        diff = compare_predictions(reference["response"], served)
    except (OSError, ValueError) as error:
        return {"url": url, "passed": False, "error": str(error)}
    return {"url": url, "passed": True, "max_diff": diff, "rows": len(served["predictions"])}


# --- snapshot -------------------------------------------------------------------------


def _test_metrics(run_id: str) -> dict[str, float]:
    """The segment metrics of a run, without its timing."""
    import mlflow

    from team2_mlops2026 import tracking

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    metrics = mlflow.get_run(run_id).data.metrics
    return {k: v for k, v in sorted(metrics.items()) if k != "fit_seconds"}


def _latest_train_runs() -> dict[str, dict]:
    import mlflow

    from team2_mlops2026 import tracking

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[tracking.MLFLOW_EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName LIKE '{TRAIN_PREFIX}-%' and status = 'FINISHED'",
        order_by=["start_time DESC"],
        output_format="list",
    )
    latest = {}
    for run in runs:
        name = run.info.run_name
        model = name.removeprefix(f"{TRAIN_PREFIX}-").rsplit("-", 1)[0]
        if name in latest or model not in MODELS:
            continue
        latest[name] = {
            "model": model,
            "run_id": run.info.run_id,
            "metrics": {m: run.data.metrics[m] for m in TRAIN_METRICS if m in run.data.metrics},
        }
    return dict(sorted(latest.items()))


def build_reference(
    models_dir: str | Path = MODELS_DIR,
    src: str | Path = FEATURES_DIR,
    examples: str | Path | None = None,
) -> dict:
    """The reference set of the exported champions (without the snapshot run id). The test
    split starts where the champions' runs started it."""
    from fastapi.testclient import TestClient
    import mlflow

    from team2_mlops2026 import tracking
    from team2_mlops2026.serving import app as api

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    examples = Path(examples or api.API_DOCS_DIR)
    card = export.read_card(models_dir)
    digests = table_digests(src)
    models, test_starts = {}, set()
    for target, entry in card["models"].items():
        test_starts.add(mlflow.get_run(entry["run_id"]).data.params.get("test_start", TEST_START))
        tags = entry["tags"]
        recorded = {k: tags.get(k) for k in digests}
        if recorded != digests:
            raise ValueError(
                f"{entry['uri']} was trained on {recorded}, but {src} holds {digests}: the "
                "snapshot must be the tables that trained the champion"
            )
        models[target] = {
            "uri": entry["uri"],
            "name": entry["name"],
            "version": entry["version"],
            "model": tags["model"],
            "run_id": entry["run_id"],
            "test_metrics_run": tags["test_metrics_run"],
            "test_metrics": _test_metrics(tags["test_metrics_run"]),
            **digests,
        }
    if len(test_starts) != 1:
        raise ValueError(f"The champions were tested from different months: {test_starts}")
    request = json.loads((examples / api.EXAMPLE_REQUEST).read_text(encoding="utf-8"))
    with TestClient(api.create_app(api.LOCAL, models_dir)) as client:
        response = client.post("/predict", json=request)
    response.raise_for_status()
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        "test_start": test_starts.pop(),
        "models": models,
        "train_runs": _latest_train_runs(),
        "request": request,
        "response": response.json(),
    }


def snapshot_tracked(
    models_dir: str | Path = MODELS_DIR,
    src: str | Path = FEATURES_DIR,
    reference_file: str | Path = REFERENCE_FILE,
    examples: str | Path | None = None,
) -> dict:
    """Logs the ``snapshot-<digest>`` run (tables, digests, reference set), tags the registered
    versions with it and writes the reference to ``reference_file``. Returns the reference."""
    import mlflow

    from team2_mlops2026 import tracking
    from team2_mlops2026.modeling.registry import _client

    src, reference_file = Path(src), Path(reference_file)
    reference = build_reference(models_dir, src, examples)
    digest = tracking.files_digest([src / FEATURES_FILE, src / PROFILES_FILE], src)
    with tracking.start_run(f"{SNAPSHOT_PREFIX}-{digest[:12]}") as run:
        log_inputs(src)
        mlflow.log_artifact(str(src / FEATURES_FILE), SNAPSHOT_ARTIFACT)
        mlflow.log_artifact(str(src / PROFILES_FILE), SNAPSHOT_ARTIFACT)
        mlflow.log_params(
            {
                "snapshot_digest": digest,
                **{k: v for k, v in reference["models"][TARGETS[0]].items() if "digest" in k},
                "test_start": reference["test_start"],
                **{f"{t}_model": m["uri"] for t, m in reference["models"].items()},
            }
        )
        mlflow.set_tag("git_commit", tracking.git_commit())
        reference["snapshot_run"] = run.info.run_id
        reference["snapshot_digest"] = digest
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        reference_file.write_text(
            json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(reference_file), REFERENCE_ARTIFACT)

    client = _client()
    for entry in reference["models"].values():
        client.set_model_version_tag(
            entry["name"], entry["version"], SNAPSHOT_TAG, run.info.run_id
        )
    logger.success(
        "Snapshot {} ({}) logged for {}; reference written to {}",
        run.info.run_name,
        run.info.run_id,
        ", ".join(m["uri"] for m in reference["models"].values()),
        reference_file,
    )
    return reference


# --- compare ------------------------------------------------------------------------


def compare(
    reference_file: str | Path = REFERENCE_FILE,
    targets: Sequence[str] = TARGETS,
    src: str | Path = FEATURES_DIR,
    models_dir: str | Path = MODELS_DIR,
    retrain_models: bool = True,
    train_runs: bool = False,
    url: str | None = None,
) -> dict:
    """Every comparison asked for, with the environment. ``passed`` is True if all passed."""
    reference = read_reference(reference_file)
    result: dict = {"environment": environment(), "snapshot_run": reference["snapshot_run"]}
    if retrain_models:
        result["retrain"] = {t: retrain(t, reference, src, models_dir) for t in targets}
    if train_runs:
        result["train_runs"] = compare_train_runs(reference)
    if url:
        result["service"] = check_service(url, reference)
    checks = [
        *result.get("retrain", {}).values(),
        *result.get("train_runs", {}).values(),
        *([result["service"]] if "service" in result else []),
    ]
    result["passed"] = all(check["passed"] for check in checks)
    return result


def _flat_metrics(result: dict) -> dict[str, float]:
    metrics = {}
    for target, r in result.get("retrain", {}).items():
        metrics |= {
            f"{target}_{k}": float(r[k])
            for k in (
                "max_metric_rel_diff",
                "exported_max_metric_rel_diff",
                "max_prediction_diff",
                "fit_seconds",
            )
        }
        metrics |= {f"{target}_{k}": r["metrics"][k] for k in TRAIN_METRICS}
    if result.get("train_runs"):
        metrics["train_runs_max_metric_rel_diff"] = max(
            r["max_metric_rel_diff"] for r in result["train_runs"].values()
        )
    if "max_diff" in result.get("service", {}):
        metrics["service_max_diff"] = result["service"]["max_diff"]
    return metrics


def compare_tracked(reference_file: str | Path = REFERENCE_FILE, **kwargs) -> dict:
    """``compare()`` inside a ``reproduce-<model>`` run with the environment as params, the
    differences as metrics and the full result as ``reproduction.json``."""
    import mlflow

    from team2_mlops2026 import tracking

    reference = read_reference(reference_file)
    names = sorted({m["model"] for m in reference["models"].values()})
    with tracking.start_run(f"{REPRODUCE_PREFIX}-{'-'.join(names)}") as run:
        result = compare(reference_file, **kwargs)
        mlflow.log_params({f"env_{k}": v for k, v in result["environment"].items()})
        mlflow.log_param("snapshot_run", reference["snapshot_run"])
        mlflow.log_metrics(_flat_metrics(result))
        mlflow.set_tags(
            {
                "reproduced": "yes" if result["passed"] else "no",
                "git_commit": tracking.git_commit(),
            }
        )
        mlflow.log_dict(result, "reproduction.json")
        result["run_id"] = run.info.run_id
    return result


def summary(result: dict) -> str:
    """One line per check, for the CLI."""
    lines = []
    for target, r in result.get("retrain", {}).items():
        lines.append(
            f"{'OK  ' if r['passed'] else 'FAIL'} retrain {target} ({r['uri']}): metrics within "
            f"{r['max_metric_rel_diff']:.2g} (exported model {r['exported_max_metric_rel_diff']:.2g}, "
            f"tolerance {r['rtol']:g}), forecasts within {r['max_prediction_diff']:.2g} trips, "
            f"fitted in {r['fit_seconds']:.0f} s"
        )
    for name, r in result.get("train_runs", {}).items():
        lines.append(
            f"{'OK  ' if r['passed'] else 'FAIL'} {name}: metrics within "
            f"{r['max_metric_rel_diff']:.2g} (tolerance {r['rtol']:g})"
        )
    if "service" in result:
        s = result["service"]
        detail = f"within {s['max_diff']:.2g}" if s["passed"] else s["error"]
        lines.append(f"{'OK  ' if s['passed'] else 'FAIL'} service {s['url']}: {detail}")
    return "\n".join(lines)


@app.command("snapshot")
def snapshot_cmd(
    models_dir: Annotated[Path, typer.Option(help="Exported champions")] = MODELS_DIR,
    src: Annotated[Path, typer.Option(help="Feature table and profiles")] = FEATURES_DIR,
    reference: Annotated[Path, typer.Option(help="Reference file to write")] = REFERENCE_FILE,
):
    """Log the snapshot run of the exported champions and write the reference set."""
    snapshot_tracked(models_dir, src, reference)


@app.command("compare")
def compare_cmd(
    reference: Annotated[Path, typer.Option(help="Reference file")] = REFERENCE_FILE,
    target: Annotated[
        list[str] | None, typer.Option(help="Target, repeatable (default: both)")
    ] = None,
    src: Annotated[Path, typer.Option(help="Feature table and profiles")] = FEATURES_DIR,
    models_dir: Annotated[Path, typer.Option(help="Exported champions")] = MODELS_DIR,
    retrain_models: Annotated[
        bool, typer.Option("--retrain/--no-retrain", help="Refit the champions")
    ] = True,
    train_runs: Annotated[
        bool, typer.Option("--train-runs", help="Compare the local train-* runs (make train)")
    ] = False,
    url: Annotated[str | None, typer.Option(help="Also check this running service")] = None,
    track: Annotated[bool, typer.Option(help="Log a reproduce-<model> run")] = True,
    save: Annotated[
        Path | None, typer.Option(help="Also write the full result to this JSON file")
    ] = None,
):
    """Compare this environment with the reference: retrained champions, train runs, service."""
    kwargs = {
        "targets": target or TARGETS,
        "src": src,
        "models_dir": models_dir,
        "retrain_models": retrain_models,
        "train_runs": train_runs,
        "url": url,
    }
    result = compare_tracked(reference, **kwargs) if track else compare(reference, **kwargs)
    print(summary(result))
    if save:
        save.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
