"""
Export of the registered models for serving:
models:/mibici-<target>@champion -> models/mibici-<target>/ + models/model_card.json

- The prediction service (``serving.app``) and its Docker image load the models from this
  folder, so they need no MLflow server.
- Each target's folder is its registered version exactly as the Registry stores it (MLflow model
  format, skops file). The alias is resolved once and the pinned version is downloaded, so a
  move of the alias during the export cannot mix versions.
- ``model_card.json`` records, for each target, the registered name, version and alias, the
  versioned URI (``models:/mibici-departures/1``), the source run and the version's tags that
  ``modeling.registry`` sets: model, algorithm, training window, test and backtest metrics,
  input digests and git commit. It also records when and from which commit it was exported.
- The folder's contents are git-ignored, since the versions live in the Registry. Exporting
  again replaces them.

Usage (CLI):
    python -m team2_mlops2026.modeling.export
    python -m team2_mlops2026.modeling.export --alias challenger --dst /tmp/models
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile
from typing import Annotated

from loguru import logger
from sklearn.base import BaseEstimator
import typer

from team2_mlops2026.config import MODELS_DIR
from team2_mlops2026.features import TARGETS
from team2_mlops2026.modeling.registry import CHAMPION, model_name

MODEL_CARD = "model_card.json"

app = typer.Typer()


def model_dir(target: str, root: str | Path = MODELS_DIR) -> Path:
    return Path(root) / model_name(target)


def version_uri(version) -> str:
    """The URI of that exact version, such as ``models:/mibici-departures/1``."""
    return f"models:/{version.name}/{version.version}"


def card_entry(version, alias: str) -> dict:
    """What the model card records about a ``ModelVersion`` that holds ``alias``."""
    return {
        "name": version.name,
        "version": str(version.version),
        "alias": alias,
        "uri": version_uri(version),
        "run_id": version.run_id,
        "tags": dict(sorted(version.tags.items())),
    }


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def export_models(
    alias: str = CHAMPION, dst: str | Path = MODELS_DIR, targets: Sequence[str] = TARGETS
) -> dict:
    """Downloads the version of each target that holds ``alias`` into ``dst`` and writes the
    model card. Returns the card."""
    import mlflow.artifacts

    from team2_mlops2026 import tracking
    from team2_mlops2026.modeling.registry import _client

    client = _client()
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    entries = {}
    for target in targets:
        version = client.get_model_version_by_alias(model_name(target), alias)
        final = model_dir(target, dst)
        with tempfile.TemporaryDirectory(dir=dst, prefix=".export-") as tmp:
            download = Path(tmp) / final.name
            mlflow.artifacts.download_artifacts(version_uri(version), dst_path=str(download))
            shutil.rmtree(final, ignore_errors=True)
            download.replace(final)
        entries[target] = card_entry(version, alias)
        logger.info("{} -> {}", version_uri(version), final)

    card = {
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": tracking.git_commit(),
        "models": entries,
    }
    _write_atomic(dst / MODEL_CARD, json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    logger.success(
        "Exported {} to {}", ", ".join(e["uri"] for e in entries.values()), dst / MODEL_CARD
    )
    return card


def read_card(src: str | Path = MODELS_DIR) -> dict:
    path = Path(src) / MODEL_CARD
    if not path.exists():
        raise FileNotFoundError(
            f"No {MODEL_CARD} in {Path(src)}: export the models first with `make export`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_exported(
    src: str | Path = MODELS_DIR, targets: Sequence[str] = TARGETS
) -> tuple[dict[str, BaseEstimator], dict]:
    """The exported model of each target, and the model card."""
    import mlflow.sklearn

    card = read_card(src)
    missing = [t for t in targets if t not in card["models"]]
    if missing:
        raise ValueError(f"The model card in {Path(src)} has no model for {', '.join(missing)}")
    models = {t: mlflow.sklearn.load_model(str(model_dir(t, src))) for t in targets}
    return models, card


@app.command()
def main(
    alias: Annotated[str, typer.Option(help="Registry alias of the models")] = CHAMPION,
    dst: Annotated[Path, typer.Option(help="Destination folder")] = MODELS_DIR,
):
    """Export the registered models of every target to a folder, for the prediction service."""
    export_models(alias, dst)


if __name__ == "__main__":
    app()
