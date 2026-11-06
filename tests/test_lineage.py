"""The tracked steps form one lineage: each step's output digest is the next step's input."""

from test_raw_data_extraction import FakeResponse, FakeSession
from test_raw_data_normalization import raw_dir  # noqa: F401  (fixture: every raw variant)

from team2_mlops2026.data import interim_data_cleaning as idc
from team2_mlops2026.data import raw_data_extraction as rde
from team2_mlops2026.data import raw_data_normalization as rdn
from team2_mlops2026.tracking import input_digests


def digests(summary: dict) -> dict[str, str]:
    return input_digests(summary["mlflow_run_id"])


def test_tracked_steps_chain_from_raw_to_processed(raw_dir, tmp_path, mlflow_store, monkeypatch):  # noqa: F811
    monkeypatch.delenv("MIBICI_PAGE_URL", raising=False)
    # A fake MiBici site that publishes the raw fixture files
    files = rde.raw_csv_files(raw_dir)
    page = "".join(f'<a href="/files/{p.name}">{p.name}</a>' for p in files)
    site = {rde.PAGE_URL: FakeResponse(page.encode())}
    site.update({f"https://mibici.net/files/{p.name}": FakeResponse(p.read_bytes()) for p in files})
    raw, interim, processed = tmp_path / "downloaded", tmp_path / "interim", tmp_path / "processed"

    extracted = digests(rde.ingest_tracked(raw, session_factory=lambda: FakeSession(site)))
    normalized = digests(rdn.normalize_tracked(raw, interim, workers=1))
    cleaned = digests(idc.clean_tracked(interim, processed, workers=1))

    assert extracted[rde.RAW_DATASET] == normalized[rde.RAW_DATASET]
    assert normalized["mibici_trips_interim"] == cleaned["mibici_trips_interim"]
    assert normalized["mibici_stations"] == cleaned["mibici_stations"]
