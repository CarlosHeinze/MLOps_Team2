import hashlib
import json
import threading
import time

import pytest
import requests

from team2_mlops2026.data import raw_data_extraction as rde

TRIPS_URL = "https://mibici.net/site/assets/files/1/datos_abiertos_2024_01.csv"
TRIPS_REV_URL = "https://mibici.net/site/assets/files/2/datos_abiertos_2016_01-1.csv"
STATIONS_URL = "https://mibici.net/site/assets/files/3/nomenclatura_2024_01.csv"

PAGE_HTML = """
<html><body>
  <a href="/site/assets/files/1/datos_abiertos_2024_01.csv">Enero 2024</a>
  <a href="https://mibici.net/site/assets/files/2/datos_abiertos_2016_01-1.csv">Enero 2016</a>
  <a href="/site/assets/files/3/nomenclatura_2024_01.csv?v=2">Estaciones</a>
  <a href="/site/assets/files/1/datos_abiertos_2024_01.csv">Duplicated link</a>
  <a href="/site/assets/files/4/otro_archivo.csv">Unrelated CSV</a>
  <a href="/site/assets/files/5/reporte.pdf">PDF</a>
  <a>No href</a>
</body></html>
"""


class FakeResponse:
    def __init__(self, content: bytes = b"", status: int = 200):
        self.content = content
        self.status_code = status

    @property
    def text(self) -> str:
        return self.content.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Minimal stand-in for requests.Session: maps URL -> response or exception."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []
        self.threads: set[int] = set()
        self.closed = False

    def close(self):
        self.closed = True

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        self.threads.add(threading.get_ident())
        route = self.routes.get(url, FakeResponse(status=404))
        if isinstance(route, Exception):
            raise route
        return route


@pytest.fixture(autouse=True)
def default_page_url(monkeypatch):
    monkeypatch.delenv("MIBICI_PAGE_URL", raising=False)


@pytest.fixture
def site():
    """Fake MiBici site with one page and three downloadable CSVs."""
    return {
        rde.PAGE_URL: FakeResponse(PAGE_HTML.encode()),
        TRIPS_URL: FakeResponse(b"Viaje_Id\n1\n2\n"),
        TRIPS_REV_URL: FakeResponse(b"Viaje_Id\n3\n"),
        STATIONS_URL + "?v=2": FakeResponse(b"id,name\n2,GDL-001\n"),
    }


# --- fetch_page --------------------------------------------------------------


def test_fetch_page_returns_html_and_base_url(site):
    html, base = rde.fetch_page(FakeSession(site))
    assert base == rde.PAGE_URL
    assert "datos_abiertos" in html


def test_fetch_page_falls_back_when_primary_fails():
    session = FakeSession(
        {
            rde.PAGE_URL: requests.ConnectionError("down"),
            rde.FALLBACK_PAGE_URL: FakeResponse(b"<html></html>"),
        }
    )
    _, base = rde.fetch_page(session)
    assert base == rde.FALLBACK_PAGE_URL
    assert session.calls == [rde.PAGE_URL, rde.FALLBACK_PAGE_URL]


def test_fetch_page_uses_env_override(monkeypatch):
    monkeypatch.setenv("MIBICI_PAGE_URL", "https://example.com/page")
    _, base = rde.fetch_page(FakeSession({"https://example.com/page": FakeResponse(b"ok")}))
    assert base == "https://example.com/page"


def test_fetch_page_raises_when_all_pages_fail():
    with pytest.raises(RuntimeError, match="open data page"):
        rde.fetch_page(FakeSession({}))


# --- discover_files ----------------------------------------------------------


def test_discover_files_keeps_original_names_sorted_and_deduplicated(site):
    files = rde.discover_files(FakeSession(site))
    assert [f.name for f in files] == [
        "datos_abiertos_2016_01-1.csv",
        "datos_abiertos_2024_01.csv",
        "nomenclatura_2024_01.csv",
    ]


def test_discover_files_resolves_urls_and_periods(site):
    by_name = {f.name: f for f in rde.discover_files(FakeSession(site))}
    assert by_name["datos_abiertos_2024_01.csv"].url == TRIPS_URL
    assert by_name["datos_abiertos_2016_01-1.csv"].period == "2016-01"
    # The query string is kept in the URL but stripped from the file name
    assert by_name["nomenclatura_2024_01.csv"].url == STATIONS_URL + "?v=2"


def test_discover_files_decodes_url_encoded_names():
    html = b'<a href="/files/datos%5Fabiertos%5F2024%5F02.csv">x</a>'
    files = rde.discover_files(FakeSession({rde.PAGE_URL: FakeResponse(html)}))
    assert [f.name for f in files] == ["datos_abiertos_2024_02.csv"]


@pytest.mark.parametrize(
    "href",
    [
        "/files/%2Fdatos_abiertos_2024_01.csv",
        "/files/..%2F..%2Fevil%2Fdatos_abiertos_2024_01.csv",
        "/files/..%5C..%5Cdatos_abiertos_2024_01.csv",
        "/files/C:%5Cdatos_abiertos_2024_01.csv",
    ],
    ids=["leading-slash", "dot-dot-slash", "dot-dot-backslash", "drive-letter"],
)
def test_discover_files_strips_encoded_path_separators(href):
    html = f'<a href="{href}">x</a>'.encode()
    files = rde.discover_files(FakeSession({rde.PAGE_URL: FakeResponse(html)}))
    assert [f.name for f in files] == ["datos_abiertos_2024_01.csv"]


@pytest.mark.parametrize(
    "href",
    ["/files/evil_datos_abiertos_2024_01.csv", "/files/datos_abiertos_2024_01.csv.exe"],
    ids=["prefix", "suffix"],
)
def test_discover_files_requires_full_name_match(href):
    html = f'<a href="{href}">x</a><a href="/files/nomenclatura_2024_01.csv">y</a>'.encode()
    files = rde.discover_files(FakeSession({rde.PAGE_URL: FakeResponse(html)}))
    assert [f.name for f in files] == ["nomenclatura_2024_01.csv"]


def test_ingest_never_writes_outside_output_dir(tmp_path):
    url = "https://mibici.net/files/..%2F..%2Fdatos_abiertos_2024_01.csv"
    html = b'<a href="/files/..%2F..%2Fdatos_abiertos_2024_01.csv">x</a>'
    session = FakeSession({rde.PAGE_URL: FakeResponse(html), url: FakeResponse(b"data")})
    out = tmp_path / "a" / "b" / "out"

    rde.ingest(out=out, session_factory=lambda: session)

    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"a/b/out/datos_abiertos_2024_01.csv", "a/b/out/manifest.json"}


def test_discover_files_raises_when_no_csv_found():
    session = FakeSession({rde.PAGE_URL: FakeResponse(b"<a href='/x.pdf'>x</a>")})
    with pytest.raises(RuntimeError, match="No CSVs found"):
        rde.discover_files(session)


# --- download ----------------------------------------------------------------


def test_download_writes_file_and_returns_hash_and_size(tmp_path):
    content = b"a,b\n" * 1000
    dest = tmp_path / "nested" / "file.csv"
    sha, size = rde.download(FakeSession({"u": FakeResponse(content)}), "u", dest)

    assert dest.read_bytes() == content
    assert sha == hashlib.sha256(content).hexdigest()
    assert size == len(content)
    assert not dest.with_suffix(".csv.part").exists()


@pytest.mark.parametrize(
    "response, error",
    [(FakeResponse(b""), RuntimeError), (FakeResponse(status=500), requests.HTTPError)],
    ids=["empty-file", "http-error"],
)
def test_download_failure_leaves_no_files(tmp_path, response, error):
    dest = tmp_path / "file.csv"
    with pytest.raises(error):
        rde.download(FakeSession({"u": response}), "u", dest)
    assert list(tmp_path.iterdir()) == []


def test_download_does_not_overwrite_existing_file_on_failure(tmp_path):
    dest = tmp_path / "file.csv"
    dest.write_bytes(b"previous")
    with pytest.raises(requests.HTTPError):
        rde.download(FakeSession({"u": FakeResponse(status=503)}), "u", dest)
    assert dest.read_bytes() == b"previous"


# --- manifest ----------------------------------------------------------------


def test_load_manifest_returns_empty_dict_when_missing(tmp_path):
    assert rde.load_manifest(tmp_path / "manifest.json") == {}


def test_save_and_load_manifest_roundtrip(tmp_path):
    path = tmp_path / "sub" / "manifest.json"
    manifest = {"datos_abiertos_2024_01.csv": {"source_url": "u", "bytes": 1}}
    rde.save_manifest(path, manifest)

    assert rde.load_manifest(path) == manifest
    assert not path.with_suffix(".json.part").exists()


# --- is_up_to_date -----------------------------------------------------------


@pytest.mark.parametrize(
    "exists, prev, expected",
    [
        (True, {"source_url": "u"}, True),
        (False, {"source_url": "u"}, False),
        (True, None, False),
        (True, {"source_url": "other"}, False),
    ],
    ids=["current", "file-missing", "not-in-manifest", "url-changed"],
)
def test_is_up_to_date(tmp_path, exists, prev, expected):
    dest = tmp_path / "file.csv"
    if exists:
        dest.write_text("x")
    assert rde.is_up_to_date(dest, prev, "u") is expected


# --- ingest ------------------------------------------------------------------


def test_ingest_downloads_files_unmodified_with_original_names(tmp_path, site):
    summary = rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))

    assert summary["discovered"] == summary["selected"] == 3
    assert len(summary["new"]) == 3
    assert summary["failed"] == []
    assert (tmp_path / "datos_abiertos_2016_01-1.csv").read_bytes() == b"Viaje_Id\n3\n"
    assert (tmp_path / "nomenclatura_2024_01.csv").exists()


def test_ingest_writes_manifest_with_lineage(tmp_path, site):
    rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))

    manifest = json.loads((tmp_path / rde.MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = manifest["datos_abiertos_2024_01.csv"]
    assert entry["source_url"] == TRIPS_URL
    assert entry["sha256"] == hashlib.sha256(b"Viaje_Id\n1\n2\n").hexdigest()
    assert entry["bytes"] == len(b"Viaje_Id\n1\n2\n")
    assert "downloaded_at" in entry


def test_ingest_is_idempotent(tmp_path, site):
    rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))
    session = FakeSession(site)
    summary = rde.ingest(out=tmp_path, session_factory=lambda: session)

    assert summary["new"] == []
    assert summary["skipped"] == 3
    assert session.calls == [rde.PAGE_URL]  # only the page, no file downloads


def test_ingest_force_redownloads_everything(tmp_path, site):
    rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))
    summary = rde.ingest(out=tmp_path, force=True, session_factory=lambda: FakeSession(site))
    assert len(summary["new"]) == 3
    assert summary["skipped"] == 0


def test_ingest_redownloads_when_source_url_changes(tmp_path, site):
    rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))
    manifest_path = tmp_path / rde.MANIFEST_NAME
    manifest = rde.load_manifest(manifest_path)
    manifest["datos_abiertos_2024_01.csv"]["source_url"] = "https://old/url.csv"
    rde.save_manifest(manifest_path, manifest)

    summary = rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))
    assert [e["name"] for e in summary["new"]] == ["datos_abiertos_2024_01.csv"]


def test_ingest_since_filters_by_period(tmp_path, site):
    summary = rde.ingest(out=tmp_path, since="2024-01", session_factory=lambda: FakeSession(site))

    assert summary["discovered"] == 3
    assert summary["selected"] == 2
    assert not (tmp_path / "datos_abiertos_2016_01-1.csv").exists()


@pytest.mark.parametrize("since", ["2024-1", "2024_01", "202401", "enero"])
def test_ingest_rejects_invalid_since(tmp_path, site, since):
    with pytest.raises(ValueError, match="YYYY-MM"):
        rde.ingest(out=tmp_path, since=since, session_factory=lambda: FakeSession(site))


def test_ingest_dry_run_does_not_write(tmp_path, site):
    session = FakeSession(site)
    summary = rde.ingest(out=tmp_path, dry_run=True, session_factory=lambda: session)

    assert [f["name"] for f in summary["pending"]] == [
        "datos_abiertos_2016_01-1.csv",
        "datos_abiertos_2024_01.csv",
        "nomenclatura_2024_01.csv",
    ]
    assert summary["new"] == []
    assert list(tmp_path.iterdir()) == []
    assert session.calls == [rde.PAGE_URL]


def test_ingest_uses_one_session_per_thread_and_closes_them(tmp_path, site):
    class SlowSession(FakeSession):
        def get(self, url, **kwargs):
            time.sleep(0.05)  # force overlap between workers
            return super().get(url, **kwargs)

    created: list[FakeSession] = []

    def factory():
        created.append(SlowSession(site))
        return created[-1]

    summary = rde.ingest(out=tmp_path, workers=2, session_factory=factory)

    assert len(summary["new"]) == 3
    discovery, *workers = created
    assert discovery.calls == [rde.PAGE_URL]
    assert 1 <= len(workers) <= 2
    assert all(len(s.threads) == 1 for s in created)  # never shared across threads
    assert len({t for s in workers for t in s.threads}) == len(workers)
    assert all(s.closed for s in created)


def test_ingest_records_failures_and_keeps_going(tmp_path, site):
    site[TRIPS_URL] = FakeResponse(status=500)
    summary = rde.ingest(out=tmp_path, session_factory=lambda: FakeSession(site))

    assert summary["failed"] == [TRIPS_URL]
    assert len(summary["new"]) == 2
    manifest = rde.load_manifest(tmp_path / rde.MANIFEST_NAME)
    assert "datos_abiertos_2024_01.csv" not in manifest
