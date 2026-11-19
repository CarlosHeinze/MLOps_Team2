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
| `scripts/report.py` | `init`, `figures`, `pdf` and `check [--only TEXT]` while writing; `activity` and `build` for the final pass. Only the standard library. |

Run the script with the project's Python: `.venv/bin/python` on macOS and Linux,
`.venv\Scripts\python` on Windows, or any Python 3.10+. Below it is written as `report.py`, short
for `python .claude/skills/final-report/scripts/report.py`.

## Workflow

1. **Start from what exists.** Several people write this report, and two copies would conflict.
   - Run `git fetch` and look for a #44 branch or PR (`git branch -a`, `gh pr list`).
   - If one exists, check it out and pull. If not, create one from an up-to-date `master` (for
     example `44-write-the-final-report`) and run `report.py init` to start the report from the
     template.
   - Agree on who writes which sections. Each person commits their own sections, so their work
     shows in the GitHub activity the rubric asks for.
   - Do not commit or push unless the person asks.
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
   - Whoever writes a Phase 3 subsection also fills the todos that depend on it: its cell in the
     ML Canvas (§5 serving, §8 reproducibility, §10 monitoring), its box in the solution
     diagram, its row in section 2, and its key figure in the summary.
   - Write the executive summary and the key figures last.
   - When two sources disagree, use the record of the run (the notebook output, `docs/…`, the
     MLflow run) over a summary of it (the README), and tell the person about the mismatch.
4. **Figures.**
   - Notebook figures: `report.py figures notebooks/<nb>.ipynb` saves every plot as
     `reports/figures/fase3/<nb>_c<cell>_<n>.png`.
   - Screenshots of Swagger, Docker Hub or MLflow: save them as PNG in `reports/figures/fase3/`.
     Use the browser tools if you have them; otherwise ask the person to take them. Crop each one
     to what the caption describes.
5. **Print it and look at it.**
   - While writing, run `report.py pdf`: it is fast, offline and only prints. Then
     `report.py check --only 3.7` lists what is left in your section, with line numbers.
   - Read the PDF pages where your section lands (the Read tool renders them) and fix what you
     see: a table or figure split across pages, an image too large for its point, an unreadable
     axis.
   - Leave whole-document layout alone until every section is in, because everything reflows:
     headings alone at the bottom of a page, blank space before a table that cannot split.
6. **Final pass and delivery** (one person, on one machine).
   - Page breaks depend on the fonts installed. Avenir Next exists only on macOS; Windows uses
     Segoe UI. Do the final layout pass and build the delivered PDF on the same machine.
   - Run `report.py build`. It fetches git, calls the GitHub API to refresh the activity table,
     prints the PDF, and then runs `check`, which must print "Ready to deliver".
   - Check the layout page by page with `page-break`, the figure widths (`w80`, `w65`, `w60`),
     the `.two` grid, or `class="flowing"` on a long table so it may split.
   - Every member reviews the PDF (#44 acceptance criteria). Then commit the HTML, the figures
     and the PDF, and open the PR.

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
  `class="pick"` and baselines use `class="base"`. Each caption says what to look at and where it
  comes from.
- **Captions are numbered when the PDF prints**, in order, so sections can be written in any
  order and moved around.
  - Write no number yourself. Use `<figcaption>` inside a `<figure>`, or `<p class="note figcap">`
    under a diagram, and `<p class="note tabcap">` right after a table.
  - Put an `id` on the figure or the caption, and refer to it with
    `<a class="xref" href="#tab-tolerancias"></a>`, which prints as "Tabla 3".
  - The delivered reports in `references/` number by hand ("Figura 4.", "Tabla 2."). Drop those
    numbers when you reuse a caption; `check` flags any that remain.
- **Length.** Aim for about 14 to 18 Letter pages in total. Rough budget per part:
  | Part | Pages |
  |---|---|
  | Summary and section 1 (canvas, MLOps, diagram) | ~3 |
  | Section 2 | ½ |
  | 3.1 to 3.3 (phases 1 and 2, condensed) | ~1 each |
  | 3.4 to 3.8 (Phase 3) | 1, up to 1½ each |
  | Section 4 with the activity table | ~2 |
  | Conclusions | ~1 |
  | AI declaration and references | ~1 |
  A subsection that runs past its budget needs cutting: keep the table and the decision, drop
  the narration that repeats them.
- **Small numbers** in scientific notation with a real minus sign: `8.4 × 10<sup>−15</sup>`, not
  `8.4e-15`. Wide plots (much wider than tall) stay at full width or `w80`, so their axes stay
  readable.
- **Roles as handoffs.** Each role hands a verifiable artifact to the next one (a digest, a run, a
  Registry alias, an image tag, the reference set), not a verbal agreement. Name the issues and
  PRs of each person. The activity table shows who committed what. Do not hide a member with
  little GitHub activity: describe their real work (verification, report, video, review) in the
  roles table.
- **Integrate, do not paste.** The final report tells one story: the problem (Canvas) → data →
  model → service → reproducibility → monitoring, and how each phase used the previous one. The
  earlier reports are condensed, about one page per phase, keeping their key tables and one or
  two figures. Phase 3 gets the most detail because it is new.
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
- captions with `figcap` and `tabcap` and references with `a.xref`, numbered by a short script at
  the end of the template;
- `.note`, `.refs` and `.page-break`.

The footer and the page numbers are in `@page`, so nothing else is needed to print. The text
between `<!-- activity:start -->` and `<!-- activity:end -->` belongs to `report.py activity`;
do not edit it by hand. The section ids (`resumen`, `introduccion`, `actividades`, `metodos`,
`roles`, `conclusiones`, `ia`, `referencias`) are checked by `report.py check`.

## Mandatory content (`report.py check` verifies it)

- The links to the GitHub repository and to the Docker Hub image.
- All four members, named.
- The GitHub activity table, filled.
- No `todo` left (each one is listed with its line and section).
- Every image local and present.
- All the outline sections, by their ids.
- Every `xref` pointing to an existing id, and no caption numbered by hand.

Two things it cannot check, so check them by eye:

- the image tag in the header matches the last one published;
- the AI declaration covers every member and every tool used in the three phases.
