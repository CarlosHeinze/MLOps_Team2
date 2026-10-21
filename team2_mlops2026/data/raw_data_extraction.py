"""
Extracción de datos crudos de MiBici (datos abiertos) -> data/raw/mibici/

Única responsabilidad: poblar la carpeta raw con los CSV tal como los publica MiBici,
SIN modificarlos y con su nombre original. La normalización de nombres y estructura
vive en ``raw_data_normalization.py``.

- Lee la página de datos abiertos y extrae los enlaces .csv (las URLs tienen IDs de
  carpeta impredecibles, por lo que NO se pueden construir).
- Descarga solo lo que falta (idempotente), en paralelo, con reintentos y escritura atómica.
- Mantiene ``manifest.json`` con URL de origen, sha256, tamaño y fecha de descarga
  (linaje para DVC / validaciones posteriores).

Uso (CLI):
    python -m team2_mlops2026.data.raw_data_extraction                    # todo lo que falte
    python -m team2_mlops2026.data.raw_data_extraction --since 2024-01    # solo desde ese mes
    python -m team2_mlops2026.data.raw_data_extraction --force            # re-descarga todo
    python -m team2_mlops2026.data.raw_data_extraction --dry-run          # solo lista

Uso (notebook):
    from team2_mlops2026.data.raw_data_extraction import ingest
    summary = ingest(since="2024-01")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
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

# Solo reconoce los CSV de MiBici y su periodo (para --since); no renombra nada.
CSV_RE = re.compile(
    r"(?:datos_abiertos|nomenclatura)_(\d{4})_(\d{2})(?:-\d+)?\.csv$", re.IGNORECASE
)
PERIOD_RE = re.compile(r"\d{4}-\d{2}")

app = typer.Typer()


@dataclass(frozen=True)
class RemoteFile:
    name: str  # nombre original publicado por MiBici
    url: str
    period: str  # YYYY-MM


def make_session(pool_size: int = DEFAULT_WORKERS) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=pool_size))
    s.headers["User-Agent"] = "mibici-mlops-ingest/1.0 (+datos abiertos)"
    return s


def fetch_page(session: requests.Session) -> tuple[str, str]:
    """Devuelve (html, url_base) de la página de datos abiertos, con fallback."""
    page = os.environ.get("MIBICI_PAGE_URL", PAGE_URL)  # override útil para pruebas
    for url in (page, FALLBACK_PAGE_URL):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text, url
        except requests.RequestException as e:
            logger.warning("No se pudo leer {}: {}", url, e)
    raise RuntimeError("No se pudo leer la página de datos abiertos")


def discover_files(session: requests.Session) -> list[RemoteFile]:
    """Extrae los CSV de MiBici enlazados en la página, ordenados por nombre."""
    html, base = fetch_page(session)
    found: dict[str, RemoteFile] = {}
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        url = urljoin(base, a["href"])
        name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        if m := CSV_RE.search(name):
            found[name] = RemoteFile(name, url, f"{m[1]}-{m[2]}")

    if not found:
        raise RuntimeError("No se encontraron CSV: ¿cambió la estructura de la página?")
    return sorted(found.values(), key=lambda f: f.name)


def download(session: requests.Session, url: str, dest: Path) -> tuple[str, int]:
    """Descarga en streaming a un .part y renombra al terminar. Devuelve (sha256, bytes)."""
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
            raise RuntimeError(f"Archivo vacío: {url}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return h.hexdigest(), size


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(path: Path, manifest: dict) -> None:
    """Escritura atómica para no corromper el manifest si el proceso se interrumpe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def is_up_to_date(dest: Path, prev: dict | None, url: str) -> bool:
    # Re-descarga si no existe o si la URL de origen cambió
    # (MiBici a veces re-publica un mes corregido con otra carpeta/sufijo).
    return dest.exists() and prev is not None and prev["source_url"] == url


def ingest(
    out: str | Path = RAW_MIBICI_DIR,
    since: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    workers: int = DEFAULT_WORKERS,
    session: requests.Session | None = None,
) -> dict:
    """Ejecuta la extracción y devuelve un resumen (usado por el CLI, el notebook y MLflow).

    Resumen: {discovered, selected, skipped, new: [entradas], failed: [urls],
              pending: [archivos, solo en dry_run], manifest_path}
    Cada entrada nueva: {name, path, source_url, sha256, bytes, downloaded_at}
    """
    if since and not PERIOD_RE.fullmatch(since):
        raise ValueError(f"--since debe tener formato YYYY-MM, se recibió {since!r}")

    out = Path(out)
    manifest_path = out / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    session = session or make_session(workers)

    files = discover_files(session)
    discovered = len(files)
    logger.info("Encontrados {} archivos en la página", discovered)
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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download, session, f.url, out / f.name): f for f in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Descargando"):
            f = futures[future]
            try:
                sha, size = future.result()
            except Exception as e:  # noqa: BLE001
                logger.error("Falló {}: {}", f.url, e)
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
            # Guardar después de cada archivo para no perder progreso
            save_manifest(manifest_path, manifest)

    logger.success(
        "Listo: {} nuevos, {} omitidos, {} fallidos",
        len(summary["new"]),
        summary["skipped"],
        len(summary["failed"]),
    )
    return summary


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Carpeta raw de destino")] = RAW_MIBICI_DIR,
    since: Annotated[
        str | None, typer.Option(help="YYYY-MM: descargar solo desde este mes")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-descargar aunque exista")] = False,
    dry_run: Annotated[bool, typer.Option(help="Solo listar lo que se descargaría")] = False,
    workers: Annotated[int, typer.Option(help="Descargas en paralelo")] = DEFAULT_WORKERS,
):
    summary = ingest(out, since, force, dry_run, workers)
    raise typer.Exit(code=1 if summary["failed"] else 0)


if __name__ == "__main__":
    app()
