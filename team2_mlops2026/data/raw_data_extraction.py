"""
Raw data extraction from MiBici open data -> data/raw/mibici/

Single responsibility: populate the raw folder with the CSVs exactly as MiBici publishes
them, UNMODIFIED and with their original names. Name and structure normalization lives
in ``raw_data_normalization.py``.

- Reads the open data page and extracts the .csv links (URLs contain unpredictable
  folder IDs, so they CANNOT be built).
- Downloads only what is missing (idempotent), in parallel, with retries and atomic writes.
- Keeps a ``manifest.json`` with source URL, sha256, size and download date
  (lineage for DVC / later validations).

Usage (CLI):
    python -m team2_mlops2026.data.raw_data_extraction                    # everything missing
    python -m team2_mlops2026.data.raw_data_extraction --since 2024-01    # only from that month
    python -m team2_mlops2026.data.raw_data_extraction --force            # re-download all
    python -m team2_mlops2026.data.raw_data_extraction --dry-run          # list only

Usage (notebook):
    from team2_mlops2026.data.raw_data_extraction import ingest
    summary = ingest(since="2024-01")
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Annotated
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup
from loguru import logger
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import typer
from urllib3.util.retry import Retry

from team2_mlops2026.config import RAW_DATA_DIR

PAGE_URL = "https://mibici.net/es/datos-abiertos/"
FALLBACK_PAGE_URL = "https://mibici.net/en/open-data/"

RAW_MIBICI_DIR = RAW_DATA_DIR / "mibici"
MANIFEST_NAME = "manifest.json"
DEFAULT_WORKERS = 4

# Only recognizes MiBici CSVs and their period (for --since); does not rename anything.
CSV_RE = re.compile(
    r"(?:datos_abiertos|nomenclatura)_(\d{4})_(\d{2})(?:-\d+)?\.csv$", re.IGNORECASE
)
PERIOD_RE = re.compile(r"\d{4}-\d{2}")

app = typer.Typer()


@dataclass(frozen=True)
class RemoteFile:
    name: str  # original name published by MiBici
    url: str
    period: str  # YYYY-MM


def make_session() -> requests.Session:
    """Session with retries. Not thread-safe: create one per thread (see ``ingest``)."""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "mibici-mlops-ingest/1.0 (+open data)"
    return s


def fetch_page(session: requests.Session) -> tuple[str, str]:
    """Returns (html, base_url) of the open data page, with fallback."""
    page = os.environ.get("MIBICI_PAGE_URL", PAGE_URL)  # override useful for testing
    for url in (page, FALLBACK_PAGE_URL):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text, url
        except requests.RequestException as e:
            logger.warning("Could not read {}: {}", url, e)
    raise RuntimeError("Could not read the open data page")


def discover_files(session: requests.Session) -> list[RemoteFile]:
    """Extracts the MiBici CSVs linked on the page, sorted by name."""
    html, base = fetch_page(session)
    found: dict[str, RemoteFile] = {}
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        url = urljoin(base, a["href"])
        # Decode BEFORE taking the basename
        path = unquote(urlsplit(url).path).replace("\\", "/")
        name = path.rsplit("/", 1)[-1]
        if m := CSV_RE.fullmatch(name):
            found[name] = RemoteFile(name, url, f"{m[1]}-{m[2]}")

    if not found:
        raise RuntimeError("No CSVs found: did the page structure change?")
    return sorted(found.values(), key=lambda f: f.name)


def download(session: requests.Session, url: str, dest: Path) -> tuple[str, int]:
    """Streams to a .part file and renames it when done. Returns (sha256, bytes)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h, size = hashlib.sha256(), 0
    try:
        with session.get(url, stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        if size == 0:
            raise RuntimeError(f"Empty file: {url}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return h.hexdigest(), size


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(path: Path, manifest: dict) -> None:
    """Atomic write so the manifest is not corrupted if the process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def is_up_to_date(dest: Path, prev: dict | None, url: str) -> bool:
    if not (dest.exists() and prev is not None and prev.get("source_url") == url):
        return False
    if "sha256" not in prev:
        return True  # Backward-compatible with manifests created before hashes.
    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == prev["sha256"] and dest.stat().st_size == prev.get("bytes", -1)


def ingest(
    out: str | Path = RAW_MIBICI_DIR,
    since: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    workers: int = DEFAULT_WORKERS,
    session_factory: Callable[[], requests.Session] = make_session,
) -> dict:
    """Runs the extraction and returns a summary (used by the CLI, the notebook and MLflow).

    ``session_factory`` is called once for page discovery and once per worker thread,
    since a ``requests.Session`` must not be shared across threads.

    Summary: {discovered, selected, skipped, new: [entries], failed: [urls],
              pending: [files, dry_run only], manifest_path}
    Each new entry: {name, path, source_url, sha256, bytes, downloaded_at}
    """
if since:
        try:
            datetime.strptime(since, "%Y-%m")
        except ValueError as e:
            raise ValueError(f"--since must have format YYYY-MM, got {since!r}") from e

    out = Path(out)
    manifest_path = out / MANIFEST_NAME
    manifest = load_manifest(manifest_path)

    discovery_session = session_factory()
    try:
        files = discover_files(discovery_session)
    finally:
        discovery_session.close()
    discovered = len(files)
    logger.info("Found {} files on the page", discovered)
    if since:
        files = [f for f in files if f.period >= since]

    pending = [
        f for f in files if force or not is_up_to_date(out / f.name, manifest.get(f.name), f.url)
    ]
    summary = {
        "discovered": discovered,
        "selected": len(files),
        "skipped": len(files) - len(pending),
        "new": [],
        "failed": [],
        "pending": [],
        "manifest_path": str(manifest_path),
    }

    if dry_run:
        for f in pending:
            logger.info("[dry-run] {} <- {}", f.name, f.url)
        summary["pending"] = [asdict(f) for f in pending]
        return summary

    # One session per worker thread, created lazily and closed when the pool finishes
    local, sessions = threading.local(), []

    def download_in_worker(url: str, dest: Path) -> tuple[str, int]:
        if not hasattr(local, "session"):
            local.session = session_factory()
            sessions.append(local.session)
        return download(local.session, url, dest)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_in_worker, f.url, out / f.name): f for f in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                f = futures[future]
                try:
                    sha, size = future.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("Failed {}: {}", f.url, e)
                    summary["failed"].append(f.url)
                    continue
                entry = {
                    "source_url": f.url,
                    "sha256": sha,
                    "bytes": size,
                    "downloaded_at": datetime.now(UTC).isoformat(),
                }
                manifest[f.name] = entry
                summary["new"].append({**entry, "name": f.name, "path": str(out / f.name)})
                # Save after each file so progress is not lost
                save_manifest(manifest_path, manifest)
    finally:
        for s in sessions:
            s.close()

    logger.success(
        "Done: {} new, {} skipped, {} failed",
        len(summary["new"]),
        summary["skipped"],
        len(summary["failed"]),
    )
    return summary


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Destination raw folder")] = RAW_MIBICI_DIR,
    since: Annotated[
        str | None, typer.Option(help="YYYY-MM: only download from this month on")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download even if it exists")] = False,
    dry_run: Annotated[bool, typer.Option(help="Only list what would be downloaded")] = False,
    workers: Annotated[int, typer.Option(help="Parallel downloads")] = DEFAULT_WORKERS,
):
    summary = ingest(out, since, force, dry_run, workers)
    raise typer.Exit(code=1 if summary["failed"] else 0)


if __name__ == "__main__":
    app()
