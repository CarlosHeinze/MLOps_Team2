"""Builds the final report: GitHub activity table, delivery checks and the PDF.

    python .claude/skills/final-report/scripts/report.py init      # start reports/Entrega_Final_Equipo02.html
    python .claude/skills/final-report/scripts/report.py activity  # fill the activity table from git and gh
    python .claude/skills/final-report/scripts/report.py check     # what the delivery requires
    python .claude/skills/final-report/scripts/report.py figures notebooks/3.02-team2-reproducibility.ipynb
    python .claude/skills/final-report/scripts/report.py pdf       # print reports/Entrega_Final_Equipo02.pdf
    python .claude/skills/final-report/scripts/report.py build     # activity, pdf, then check

Only the standard library, so it runs with the project's .venv or any Python 3.10+ on Windows,
macOS or Linux. `activity` needs git and an authenticated gh; `pdf` needs Chrome, Edge or
Chromium (CHROME_PATH or --browser to point at one).
"""

from __future__ import annotations

import argparse
import base64
from datetime import date, datetime
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO = SKILL_DIR.parents[2]
TEMPLATE = SKILL_DIR / "assets" / "template.html"
REPORT = REPO / "reports" / "Entrega_Final_Equipo02.html"
PDF = REPORT.with_suffix(".pdf")

GITHUB_URL = "https://github.com/CarlosHeinze/MLOps_Team2"
DOCKERHUB_URL = "https://hub.docker.com/r/racj90120hplmnv09/mibici-ml-service"
BRANCH = "origin/master"

# The day each phase started and ended (local dates). A commit, PR or issue belongs to the phase
# of its date; Phase 1 was delivered on 25 Oct and Phase 2 on 8 Nov 2026.
PHASES = (
    ("Fase 1", date(2000, 1, 1), date(2026, 10, 25)),
    ("Fase 2", date(2026, 10, 26), date(2026, 11, 8)),
    ("Fase 3", date(2026, 11, 9), date(2026, 12, 31)),
)
# Name, GitHub login and a pattern over the git author "name|email" (commit names vary by machine).
PEOPLE = (
    ("Carlos Antonio Heinze Mortera", "CarlosHeinze", r"heinze"),
    ("Javier Ramírez", "javieramirez90", r"javier|xavier"),
    ("Julián Malo Romero", "JulianMaloRomeroTEC", r"julian|malo"),
    ("Carlos Arturo Chavero Cruz", "chaverocarlos", r"chavero"),
)
ACTIVITY_START, ACTIVITY_END = "<!-- activity:start -->", "<!-- activity:end -->"
# Section ids of the template, one per part of the outline the instructions ask for.
SECTIONS = {
    "resumen": "resumen ejecutivo",
    "introduccion": "introducción: problema, ML Canvas, MLOps y diagrama",
    "actividades": "actividades realizadas en cada fase",
    "metodos": "métodos y resultados por etapa",
    "roles": "roles, interacción y actividad en GitHub",
    "conclusiones": "conclusiones: lecciones, mejoras, trabajo futuro",
    "ia": "declaración de uso de IA",
    "referencias": "referencias",
}
BROWSERS = {
    "win32": [
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
        r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
    ],
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
    "linux": [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
    ],
}


def run(*args: str) -> str:
    result = subprocess.run(
        args, cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if result.returncode != 0:
        sys.exit(f"{' '.join(args[:3])} failed:\n{result.stderr.strip()}")
    return result.stdout


def phase_of(day: date) -> str | None:
    return next((name for name, start, end in PHASES if start <= day <= end), None)


def local_day(timestamp: str) -> date:
    # Python 3.10 cannot parse the "Z" of GitHub's timestamps
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone().date()  # noqa: FURB162


def collect_activity() -> dict:
    """Commits, PRs, reviews, issues and comments of each member, by phase."""
    if not shutil.which("gh"):
        sys.exit("gh is not installed: https://cli.github.com/ (then `gh auth login`)")
    run("git", "fetch", "-q", "origin")
    stats = {
        (login, phase): {
            "commits": 0,
            "prs": [],
            "reviewed": set(),
            "issues_opened": [],
            "issues_assigned": [],
            "comments": 0,
        }
        for _, login, _ in PEOPLE
        for phase, _, _ in PHASES
    }

    def member(text: str) -> str | None:
        return next(
            (login for _, login, pattern in PEOPLE if re.search(pattern, text, re.IGNORECASE)),
            None,
        )

    logins = {login for _, login, _ in PEOPLE}
    for line in run(
        "git", "log", BRANCH, "--no-merges", "--format=%an|%ae|%ad", "--date=short"
    ).splitlines():
        author, day = line.rsplit("|", 1)
        login, phase = member(author), phase_of(date.fromisoformat(day))
        if login and phase:
            stats[login, phase]["commits"] += 1

    prs = json.loads(
        run(
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,author,createdAt,state,reviews,mergedBy,comments",
        )
    )
    for pr in prs:
        phase = phase_of(local_day(pr["createdAt"]))
        if not phase:
            continue
        author = pr["author"]["login"]
        if author in logins:
            stats[author, phase]["prs"].append((pr["number"], pr["state"]))
        helpers = {r["author"]["login"] for r in pr["reviews"]}
        helpers |= {pr["mergedBy"]["login"]} if pr.get("mergedBy") else set()
        for login in (helpers & logins) - {author}:
            stats[login, phase]["reviewed"].add(pr["number"])
        for comment in pr["comments"]:
            if comment["author"]["login"] in logins:
                stats[comment["author"]["login"], phase]["comments"] += 1

    issues = json.loads(
        run(
            "gh",
            "issue",
            "list",
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,author,assignees,createdAt,comments",
        )
    )
    for issue in issues:
        phase = phase_of(local_day(issue["createdAt"]))
        if not phase:
            continue
        if issue["author"]["login"] in logins:
            stats[issue["author"]["login"], phase]["issues_opened"].append(issue["number"])
        for assignee in issue["assignees"]:
            if assignee["login"] in logins:
                stats[assignee["login"], phase]["issues_assigned"].append(issue["number"])
        for comment in issue["comments"]:
            if comment["author"]["login"] in logins:
                stats[comment["author"]["login"], phase]["comments"] += 1
    return stats


def numbers(items, base: str) -> str:
    """'#17, #18' as links to the repo, '—' when empty."""
    links = [f'<a href="{GITHUB_URL}/{base}/{n}">#{n}</a>' for n in sorted(items)]
    return ", ".join(links) or "—"


def phase_label(phase: str, start: date, end: date) -> str:
    if start == PHASES[0][1]:
        return f"{phase} · hasta el {end.day}/{end.month}"
    if end >= date.today():
        return f"{phase} · desde el {start.day}/{start.month}"
    return f"{phase} · {start.day}/{start.month} – {end.day}/{end.month}"


def activity_table(stats: dict) -> str:
    """One small table per phase, so a phase never splits across pages."""
    tables = []
    for phase, start, end in PHASES:
        rows = [
            (
                f'<table class="activity">\n  <tr><th style="width:22%">{phase_label(phase, start, end)}</th>'
                '<th class="num" style="width:7%">Commits</th><th style="width:14%">PR propios</th>'
                '<th style="width:13%">PR revisados o integrados</th><th>Issues abiertos</th>'
                '<th style="width:14%">Issues asignados</th><th class="num" style="width:9%">Comentarios</th></tr>'
            )
        ]
        for name, login, _ in PEOPLE:
            s = stats[login, phase]
            own = [n for n, state in s["prs"] if state != "CLOSED"]
            rows.append(
                f'  <tr><td>{escape(name)}<div class="note">{login}</div></td>'
                f'<td class="num">{s["commits"]}</td><td>{numbers(own, "pull")}</td>'
                f"<td>{numbers(s['reviewed'], 'pull')}</td><td>{numbers(s['issues_opened'], 'issues')}</td>"
                f'<td>{numbers(s["issues_assigned"], "issues")}</td><td class="num">{s["comments"]}</td></tr>'
            )
        tables.append("\n".join([*rows, "</table>"]))
    totals = []
    for _, login, _ in PEOPLE:
        commits = sum(stats[login, phase]["commits"] for phase, _, _ in PHASES)
        prs = sum(len(stats[login, phase]["prs"]) for phase, _, _ in PHASES)
        totals.append(f"{login}, {commits} commits y {prs} PR")
    today = date.today()
    return "\n".join(
        [
            ACTIVITY_START,
            *tables,
            (
                f'<p class="note">Generado el {today.day}/{today.month}/{today.year} desde <code>{BRANCH}</code> '
                f"y la API de GitHub (<code>report.py activity</code>). En total: {'; '.join(totals)}. "
                "Commits sin merges; un PR o issue cuenta en la fase de su fecha de creación.</p>"
            ),
            ACTIVITY_END,
        ]
    )


def cmd_activity(_: argparse.Namespace) -> None:
    html = read_report()
    if ACTIVITY_START not in html or ACTIVITY_END not in html:
        sys.exit(
            f"{REPORT.name} lost the markers {ACTIVITY_START} … {ACTIVITY_END}: put them back in section 4"
        )
    stats = collect_activity()
    before, rest = html.split(ACTIVITY_START, 1)
    after = rest.split(ACTIVITY_END, 1)[1]
    REPORT.write_text(before + activity_table(stats) + after, encoding="utf-8")
    print("Activity table updated:")
    for phase, _, _ in PHASES:
        for name, login, _ in PEOPLE:
            s = stats[login, phase]
            print(
                f"  {phase} {login:20} commits={s['commits']:3} prs={[n for n, _ in s['prs']]} "
                f"reviewed={sorted(s['reviewed'])} issues={s['issues_assigned']} comments={s['comments']}"
            )


def read_report() -> str:
    if not REPORT.exists():
        sys.exit(f"{REPORT.relative_to(REPO)} does not exist yet: run `report.py init`")
    return REPORT.read_text(encoding="utf-8")


def where(html: str, position: int) -> str:
    """'L123 · 3.7 Reproducibilidad…': the line and the last heading before a position."""
    line = html.count("\n", 0, position) + 1
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html[:position], re.DOTALL)
    heading = re.sub(r"<[^>]+>", "", headings[-1]).strip() if headings else "header"
    return f"L{line} · {heading[:40]}"


def problems(html: str) -> list[str]:
    """Everything the delivery instructions and the team's rules require, as messages."""
    found = []
    todo = r'<(\w+)[^>]*class="[^"]*\btodo\b[^"]*"[^>]*>(.*?)</\1>'
    for match in re.finditer(todo, html, re.DOTALL):
        text = re.sub(r"<[^>]+>|\s+", " ", match.group(2)).strip()
        found.append(f"TODO ({where(html, match.start())}): {text[:100]}")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    for match in re.finditer(r'<a[^>]*class="xref"[^>]*href="#([^"]+)"', html):
        if match.group(1) not in ids:
            found.append(
                f"reference to a missing id ({where(html, match.start())}): #{match.group(1)}"
            )
    caption = r'<figcaption[^>]*>|<p[^>]*class="[^"]*\b(?:figcap|tabcap)\b[^"]*"[^>]*>'
    for match in re.finditer(rf"(?:{caption})\s*((?:Figura|Tabla) \d+\.)", html):
        found.append(
            f"numbered by hand ({where(html, match.start())}): drop '{match.group(1)}', "
            "the PDF numbers captions in order"
        )
    for match in re.finditer(r'<p class="note">\s*((?:Figura|Tabla) \d+\.)', html):
        found.append(
            f"caption without class ({where(html, match.start())}): use "
            f'class="note {"tabcap" if match.group(1).startswith("Tabla") else "figcap"}" '
            f"and drop '{match.group(1)}'"
        )
    for url, what in ((GITHUB_URL, "GitHub repository"), (DOCKERHUB_URL, "Docker Hub image")):
        if f'href="{url}' not in html:
            found.append(f"missing the mandatory link to the {what}: {url}")
    for name, login, _ in PEOPLE:
        if name not in html:
            found.append(f"{name} ({login}) is not named in the report")
    for section, what in SECTIONS.items():
        if f'id="{section}"' not in html:
            found.append(f'missing the section id="{section}" ({what})')
    if ACTIVITY_START in html and "<!-- activity:empty -->" in html:
        found.append("the GitHub activity table is empty: run `report.py activity`")
    for src in re.findall(r'<img[^>]+src="([^"]+)"', html):
        if re.match(r"https?:", src):
            found.append(f"remote image (download it to reports/figures/): {src}")
        elif not (REPORT.parent / src).is_file():
            found.append(f"image not found: {src}")
    return found


def cmd_check(args: argparse.Namespace) -> int:
    found = problems(read_report())
    if getattr(args, "only", None):
        found = [p for p in found if args.only.lower() in p.lower()]
    pdf_note = (
        f"; {PDF.name} has {pages(PDF)} pages" if PDF.exists() else f"; {PDF.name} not built yet"
    )
    if found:
        print(f"{len(found)} issue(s) before the delivery{pdf_note}:")
        print("\n".join(f"  - {p}" for p in found))
        return 1
    print(f'Ready to deliver{pdf_note}. Submit {PDF.name} in Canvas ("Entregar tarea").')
    return 0


def find_browser(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("CHROME_PATH")] + BROWSERS.get(
        sys.platform, BROWSERS["linux"]
    )
    for candidate in filter(None, candidates):
        path = os.path.expandvars(candidate)
        if Path(path).is_file():
            return path
        if shutil.which(path):
            return shutil.which(path)
    sys.exit(
        "Chrome, Edge or Chromium not found: install one or set CHROME_PATH to its executable"
    )


def pages(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def cmd_pdf(args: argparse.Namespace) -> None:
    read_report()
    browser = find_browser(args.browser)
    with tempfile.TemporaryDirectory() as profile:
        result = subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile}",
                "--no-pdf-header-footer",
                "--run-all-compositor-stages-before-draw",
                f"--print-to-pdf={PDF}",
                REPORT.as_uri(),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    if result.returncode != 0 or not PDF.exists():
        sys.exit(f"{Path(browser).name} could not print the PDF:\n{result.stderr.strip()[-2000:]}")
    print(
        f"{PDF.relative_to(REPO)}: {pages(PDF)} pages, {PDF.stat().st_size / 1e6:.1f} MB ({Path(browser).name})"
    )


def cmd_init(_: argparse.Namespace) -> None:
    if REPORT.exists():
        sys.exit(f"{REPORT.relative_to(REPO)} already exists: keep working on it (git pull first)")
    (REPO / "reports" / "figures" / "fase3").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, REPORT)
    print(
        f"{REPORT.relative_to(REPO)} created from the template; figures go in reports/figures/fase3/"
    )


def cmd_figures(args: argparse.Namespace) -> None:
    """Save every PNG a notebook shows as <prefix>_c<cell>_<n>.png (the Phase 1 naming)."""
    notebook = Path(args.notebook)
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    prefix = notebook.name.split("-", 1)[0]
    out = REPO / "reports" / "figures" / args.phase
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for index, cell in enumerate(cells):
        images = [
            o["data"]["image/png"]
            for o in cell.get("outputs", [])
            if "image/png" in o.get("data", {})
        ]
        for n, image in enumerate(images, start=1):
            if args.cell is not None and index != args.cell:
                continue
            path = out / f"{prefix}_c{index}_{n}.png"
            data = "".join(image) if isinstance(image, list) else image
            path.write_bytes(base64.b64decode(data))
            source = "".join(cell["source"]).strip().splitlines()
            print(f"{path.relative_to(REPO)}  ← cell {index}: {source[0][:70] if source else ''}")
            saved += 1
    if not saved:
        where = "" if args.cell is None else f" in cell {args.cell}"
        sys.exit(
            f"{notebook.name} shows no figure{where} (if it should, run the notebook and save it)"
        )


def cmd_build(args: argparse.Namespace) -> int:
    if not args.no_activity:
        cmd_activity(args)
    cmd_pdf(args)
    print("PDF built; now the delivery check:")
    return cmd_check(args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="copy the template to reports/Entrega_Final_Equipo02.html")
    sub.add_parser("activity", help="fill the GitHub activity table (git + gh)")
    p = sub.add_parser("check", help="list what the delivery still lacks")
    p.add_argument(
        "--only", help="only the messages that mention this text, e.g. a section: --only 3.7"
    )
    p = sub.add_parser("figures", help="save a notebook's figures to reports/figures/<phase>/")
    p.add_argument("notebook", help="e.g. notebooks/3.02-team2-reproducibility.ipynb")
    p.add_argument("--cell", type=int, help="only this code cell (index in the notebook)")
    p.add_argument("--phase", default="fase3", help="folder under reports/figures (default fase3)")
    for name in ("pdf", "build"):
        p = sub.add_parser(
            name, help="print the PDF" if name == "pdf" else "activity, pdf, then check"
        )
        p.add_argument("--browser", help="path to Chrome, Edge or Chromium")
        if name == "build":
            p.add_argument(
                "--no-activity", action="store_true", help="keep the activity table as it is"
            )
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # a Windows console cannot print every character
    commands = {
        "init": cmd_init,
        "activity": cmd_activity,
        "check": cmd_check,
        "figures": cmd_figures,
        "pdf": cmd_pdf,
        "build": cmd_build,
    }
    return commands[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
