"""The tracked steps form one lineage: each step's output digest is the next step's input."""

import mlflow
from mlflow.tracking import MlflowClient
from test_raw_data_extraction import FakeResponse, FakeSession
from test_raw_data_normalization import raw_dir  # noqa: F401  (fixture: every raw variant)
from test_train import TEST_START, lineage  # noqa: F401  (fixture: tracked features, profiles)

from team2_mlops2026 import features as ft
from team2_mlops2026.data import interim_data_cleaning as idc
from team2_mlops2026.data import raw_data_extraction as rde
from team2_mlops2026.data import raw_data_normalization as rdn
from team2_mlops2026.modeling import export as ex
from team2_mlops2026.modeling import registry as reg
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.tracking import input_digests

FEATURES, PROFILES = "mibici_station_hour_features", "mibici_station_profiles"


def digests(summary: dict) -> dict[str, str]:
    return input_digests(summary["mlflow_run_id"])


def test_tracked_steps_chain_from_raw_to_processed(raw_dir, tmp_path, mlflow_store, monkeypatch):  # noqa: F811
    monkeypatch.delenv("MIBICI_PAGE_URL", raising=False)
    # A fake MiBici site that publishes the raw fixture files
    files = rde.raw_csv_files(raw_dir)
    page = "".join(f'<a href="/files/{p.name}">{p.name}</a>' for p in files)
    site = {rde.PAGE_URL: FakeResponse(page.encode())}
    site.update(
        {f"https://mibici.net/files/{p.name}": FakeResponse(p.read_bytes()) for p in files}
    )
    raw, interim, processed = tmp_path / "downloaded", tmp_path / "interim", tmp_path / "processed"

    extracted = digests(rde.ingest_tracked(raw, session_factory=lambda: FakeSession(site)))
    normalized = digests(rdn.normalize_tracked(raw, interim, workers=1))
    cleaned = digests(idc.clean_tracked(interim, processed, workers=1))

    assert extracted[rde.RAW_DATASET] == normalized[rde.RAW_DATASET]
    assert normalized["mibici_trips_interim"] == cleaned["mibici_trips_interim"]
    assert normalized["mibici_stations"] == cleaned["mibici_stations"]


def test_champion_export_preserves_run_lineage(lineage, tmp_path):  # noqa: F811
    features = digests(lineage["features"])
    profiles = digests(lineage["profiles"])
    trained_models = tr.train_tracked(["seasonal_naive"], ft.TARGETS, lineage["dir"], TEST_START)

    assert profiles[FEATURES] == features[FEATURES]
    for model in trained_models:
        run = mlflow.get_run(model.run_id)
        reg.register(reg.Entry(model.name, model.target, reg.CHAMPION, run, run, run, None))

    dst = tmp_path / "models"
    ex.export_models(dst=dst)
    card = ex.read_card(dst)
    assert set(card["models"]) == set(ft.TARGETS)
    client = MlflowClient()
    for model in trained_models:
        name = reg.model_name(model.target)
        version = client.get_model_version_by_alias(name, reg.CHAMPION)
        entry = card["models"][model.target]
        assert entry["alias"] == reg.CHAMPION
        assert entry["name"] == name
        assert entry["version"] == str(version.version)
        assert entry["uri"] == f"models:/{name}/{version.version}"
        assert entry["run_id"] == version.run_id == model.run_id
        inputs = input_digests(entry["run_id"])
        assert entry["tags"]["features_digest"] == version.tags["features_digest"]
        assert entry["tags"]["features_digest"] == inputs[FEATURES] == features[FEATURES]
        assert entry["tags"]["profiles_digest"] == version.tags["profiles_digest"]
        assert entry["tags"]["profiles_digest"] == inputs[PROFILES] == profiles[PROFILES]
