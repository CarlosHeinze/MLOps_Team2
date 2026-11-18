---
name: final-report
description: Write, update, build or check the team's final project report (Entrega_Final_Equipo02.pdf, issue #44) for the MLOps course, in the same HTML → PDF format as the Phase 1 and 2 reports. It integrates phases 1 to 3, follows the rubric, and carries the GitHub activity table of every member and the mandatory links to GitHub and Docker Hub. Use it whenever someone asks for the final report, the "entrega final", the "reporte ejecutivo", or any of its sections (ML Canvas, solution diagram, methods and results, roles, conclusions, AI declaration), to regenerate its PDF, figures or activity table, or to check it before submitting, even if they only say "el reporte" or "#44".
argument-hint: "[section or task, e.g. 'sección 3.5' or 'build']"
---

# Final report (Entrega final, Equipo 02)

The final delivery of TC5044.10 is **one PDF**, `Entrega_Final_Equipo02.pdf`, due **Sunday
22 November 2026, 23:59**. It is uploaded once per team in Canvas. The rubric grades the
**report**, not the code. It is graded on six criteria worth 100 points, and it must integrate
the three phases of the project, not only Phase 3. Read `references/rubric.md` before writing
anything: it has the instructions, the suggested outline and the rubric with its "Excelente"
level.

Phases 1 and 2 were delivered as an HTML page printed to PDF with Chrome. The final report keeps
that format, so the three documents look like one project and the earlier sections can be reused
as they are.

## Files

| Path | What |
|---|---|
| `reports/Entrega_Final_Equipo02.html` | The report itself. Edit this file and commit it. |
| `reports/Entrega_Final_Equipo02.pdf` | The PDF, built from the HTML with `report.py build`. Commit it too. |
| `reports/figures/fase1/`, `fase2/`, `fase3/` | Figures. The first two hold the delivered ones; new figures go in `fase3/`. |
| `reports/Fase1_Equipo02.pdf`, `Fase2_Equipo02.pdf` | The delivered reports of phases 1 and 2. |
| `assets/template.html` | Starting point: CSS, outline, ML Canvas and solution diagram, with `todo` blocks. |
| `references/rubric.md` | Instructions, outline and rubric (Canvas, 18 Nov 2026). |
| `references/sources.md` | Where each number and figure comes from, and which ones need MLflow or data. |
| `references/fase1.html`, `fase2.html` | The delivered reports: their content, voice and level of detail. |
| `scripts/report.py` | `init`, `activity`, `figures`, `pdf`, `check`, `build` (only the standard library). |

Run the script with the project's Python: `.venv/bin/python` on macOS and Linux,
`.venv\Scripts\python` on Windows, or any Python 3.10+. Below it is written as `report.py`, short
for `python .claude/skills/final-report/scripts/report.py`.

## Workflow

1. **Start from what exists.** Run `git pull` and look for the report on other branches
   (`git branch -a`, `gh pr list`). Several people work on it, so two copies would conflict. If
   the report does not exist yet, run `report.py init` to create it from the template. Work on
   the #44 branch, and let the person who opens the PR be the one who wrote the sections, so
   their activity shows on GitHub.
2. **Read** `references/rubric.md` and `references/sources.md`. Skim `references/fase2.html` to
   pick up the voice and the level of detail. The section headings of both delivered reports
   show how each one was organized.
3. **Fill the sections in place.** Each `class="todo"` block says what goes there, which rubric
   criterion it serves and where the evidence is. Replace the block with the content; do not
   leave the instruction next to it.
   - Phases 1 and 2: condense the delivered reports (sections 3.1 to 3.3, and their columns of
     the roles table). Keep their numbers and figures as they are.
   - Phase 3: gather the evidence first (commands, docs, notebooks; see `sources.md`), then
     write.
   - Write the executive summary and the key figures last.
4. **Figures.**
   - Notebook figures: `report.py figures notebooks/<nb>.ipynb` saves every plot as
     `reports/figures/fase3/<nb>_c<cell>_<n>.png`.
   - Screenshots of Swagger, Docker Hub or MLflow: save them as PNG in `reports/figures/fase3/`.
     Use the browser tools if you have them; otherwise ask the person to take them. Crop each one
     to what the caption describes.
5. **Build and look at it.** Run `report.py build`. It refreshes the GitHub activity table, prints
   the PDF and lists what is still missing. Then read the PDF pages (the Read tool renders them)
   and fix what you see:
   - a heading alone at the bottom of a page;
   - a table or figure split across pages;
   - an image too large for its point;
   - an almost empty page.
   Use the `page-break` class, the figure widths (`w80`, `w65`, `w60`) or the `.two` grid.
6. **Before delivering:** `report.py check` must print "Ready to deliver". Every member must have
   reviewed the PDF (#44 acceptance criteria). Then commit the HTML, the figures and the PDF, and
   open the PR.

If you could not verify something (for example, the drift notebook is not merged yet), leave a
`<span class="todo">` with what is missing and tell the person. A visible gap can be fixed before
the deadline; a made-up number cannot be detected in time.

## How the earlier reports were written (keep doing it)

- **Every number comes from running the code.** It comes from an MLflow run, a notebook output, a
  command or a file in the repo, and the text or the caption says which one (for example
  "notebook 2.04, runs `backtest-<modelo>`"). Copying a number from memory or rounding it
  differently from its source is how a report contradicts itself.
- **Method → why → result → decision.** Each method is justified against the alternatives that
  were discarded. Each result ends in the decision it led to (the notebooks' "Result → action").
  This is what the rubric's "Excelente" asks for: methods argued "de forma clara y detallada" and
  results with "fundamento metodológico".
- **Tables before prose** for comparisons, rules, stages and roles. The chosen row uses
  `class="pick"` and baselines use `class="base"`. Tables and figures are numbered, and each
  caption says what to look at and where it comes from.
- **Roles as handoffs.** Each role hands a verifiable artifact to the next one (a digest, a run, a
  Registry alias, an image tag, the reference set), not a verbal agreement. Name the issues and
  PRs of each person. The activity table shows who committed what. Do not hide a member with
  little GitHub activity: describe their real work (verification, report, video, review) in the
  roles table.
- **Integrate, do not paste.** The final report tells one story: the problem (Canvas) → data →
  model → service → reproducibility → monitoring, and how each phase used the previous one. The
  earlier reports are condensed, about one page per phase, keeping their key tables and one or
  two figures. Phase 3 gets the most detail because it is new. Aim for about 14 to 18 Letter
  pages; phases 1 and 2 took 8 and 10.
- **Conclusions that reflect.** What worked and why, what failed and how it was fixed (the
  baseline that was not beaten at first, the PCA that helps only a little, the thread-dependent
  refit), which earlier "areas for improvement" were solved, and which concrete work remains.
- **Spanish, the way the earlier reports wrote it.**
  - Spanish from Mexico, with every accent.
  - Thousands with a comma and decimals with a point (`39,078,984`, `0.69`), and a space before
    `%` (`15.4 %`).
  - A non-breaking hyphen `‑` (U+2011) in compounds and dates that must not break across lines:
    `estación‑hora`, `2025‑10`.
  - Code, commands, runs and file names inside `<code>` or `class="mono"`.
  - English terms only where the field uses them (baseline, backtest, champion, drift).

## Template components

The template's CSS is the Phase 2 CSS, plus the Phase 1 canvas and a few new pieces:

- `.summary` with `.keyfigs`: the executive summary and six key figures;
- `.canvas`: the ML Canvas;
- `.arch`: the solution diagram, with lanes per phase, stages, cross-cutting bars and users;
- `.flow`: a chain of stages or roles;
- `.criterion`: a highlighted criterion;
- tables, with the classes `num`, `pick`, `base`, `flowing` (may split across pages) and `mono`;
- figures, with the widths `w80`, `w65` and `w60` and the `.two` grid;
- `.note`, `.refs` and `.page-break`.

The footer and the page numbers are in `@page`, so nothing else is needed to print. The text
between `<!-- activity:start -->` and `<!-- activity:end -->` belongs to `report.py activity`;
do not edit it by hand. The section ids (`resumen`, `introduccion`, `actividades`, `metodos`,
`roles`, `conclusiones`, `ia`, `referencias`) are checked by `report.py check`.

## Mandatory content (`report.py check` verifies it)

- The links to the GitHub repository and to the Docker Hub image.
- All four members, named.
- The GitHub activity table, filled.
- No `todo` left.
- Every image local and present.
- All the outline sections, by their ids.

Two things it cannot check, so check them by eye:

- the image tag in the header matches the last one published;
- the AI declaration covers every member and every tool used in the three phases.
