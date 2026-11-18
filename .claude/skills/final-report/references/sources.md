# Where each number and figure comes from

Every number in the report has to be traceable to an MLflow run, a notebook, a file in the repo or
a command output (issue #44, acceptance criteria). This file says where to find each one.

Some sources need things that are not in git. Check what the machine has before promising a
figure:

| Needs | Who has it | Without it |
|---|---|---|
| A clone of the repo | everyone | — |
| The MLflow server with the team's runs (`mlflow.db`, `mlartifacts/`) | Javier, Carlos | reuse the screenshots in `reports/figures/fase1|fase2`; ask Javier for new ones |
| `data/` (the MiBici history, ~5 GB) | Javier, Carlos, Julián (Phase 1) | `make data` downloads and rebuilds it (hours); not needed to write the report |
| Docker | anyone who installs Docker Desktop | the Docker Hub page shows the tags and sizes |

Executed notebooks are committed **with their outputs**, so tables and figures can be read from
them on any clone (`report.py figures <notebook>` saves the figures as PNG).

## Phases 1 and 2: reuse the delivered reports

`references/fase1.html` and `references/fase2.html` are the HTML of the delivered PDFs
(`reports/Fase1_Equipo02.pdf`, `reports/Fase2_Equipo02.pdf`). Their numbers were checked against
MLflow and the notebooks when they were delivered: reuse them **with the same values** and the
same figures (`reports/figures/fase1/`, `reports/figures/fase2/`, already referenced with those
paths). Do not recompute them unless the code changed since.

| Topic | Where in the delivered reports | Repo source |
|---|---|---|
| Problem, value proposition, ML Canvas | Fase 1 §1 | `references/ml-canvas.md` (updated in every phase) |
| Five data stages with digests, cleaning contract, A/B/C missing-data experiment | Fase 1 §2, §4 | notebooks 1.01–1.02, README "Data versioning" |
| EDA findings → decisions, preprocessor, PCA profiles | Fase 1 §3 | notebooks 1.03–1.05 |
| Cookiecutter structure, refactoring, OOP | Fase 1 §5, Fase 2 §2 | `team2_mlops2026/` |
| Metrics and success criterion | Fase 2 §1 | `modeling/evaluate.py`, Canvas §4 |
| Model selection, baselines, tuning, 12-month backtest, PCA ablation | Fase 2 §4 | notebooks 2.01–2.04, runs `train-*`, `tune-*`, `backtest-*`, `ablation-*` |
| Model Registry, aliases, batch forecast | Fase 2 §5 | notebook 2.05, `mibici-departures` / `mibici-arrivals` |
| Roles and activities of phases 1 and 2 | Fase 1 §6, Fase 2 §6 | `docs/roles.md` |
| AI usage of phases 1 and 2 | Fase 1 §8, Fase 2 §8 | `docs/ai-usage.md` |

## Phase 3

| Item (issue) | What to state | Source |
|---|---|---|
| Tests (#38) | Number of tests, how many are integration tests, the single command and its time | `make test` (= `pytest -q`; prints "N passed in S s"); `pytest -q -m integration`; `tests/test_pipeline_integration.py`; README §9 |
| FastAPI (#39) | Endpoints and status codes, the Pydantic schema, an example request and response, the model URIs and versions served, the batch comparison | README §5; `docs/api/openapi.json`, `docs/api/example_request.json`, `docs/api/example_response.json`; notebook 3.01; `GET /model` |
| Swagger screenshot | The `/docs` page of the service | `docker run -p 8000:8000 racj90120hplmnv09/mibici-ml-service:latest`, then http://127.0.0.1:8000/docs (no MLflow or data needed) |
| Docker (#40) | Dockerfile decisions, the exact build and run commands, the tags, the size and architectures | `Dockerfile`, README §6 (commands and tag table), `make docker-build` / `docker-run` / `docker-smoke`; https://hub.docker.com/r/racj90120hplmnv09/mibici-ml-service/tags (sizes per architecture) |
| Reproducibility (#41) | Pinned dependencies and seed, the snapshot run, reference vs reproduced metrics per environment, tolerances, why LightGBM is not bit for bit across thread counts | `docs/verificacion-fase3.md` (tables per environment), `docs/reproducibility/*.json`, notebook 3.02, ML Canvas §8, `requirements-lock.txt`, README §7 |
| Data drift (#42) | Synthetic scenarios, drift statistics, metrics against the baseline, thresholds, alerts and the proposed action | notebook 3.03 and its MLflow runs, ML Canvas §10 (once #42 is merged) |
| Makefile and README (#43) | The single entry point for the whole pipeline | `Makefile` (`make help`), README "How to run" |
| Nice-to-have (#45–#49) | Only the ones that were merged | their PRs |
| Roles in Phase 3 | Who did what, and what each role handed to the next | `gh issue list --state all` (assignees), the PRs, `docs/roles.md` (Phase 3 section, from #44); confirm with each member |
| AI usage in Phase 3 | Tool, purpose and scope per member | `docs/ai-usage.md` (Phase 3 section, from #44); ask each member |
| GitHub activity | Commits, PRs, reviews and issues per member and phase | `report.py activity` (fills the table) |

## Quick commands

```bash
git log --oneline origin/master | head -30                           # what was merged lately
gh pr list --state merged --limit 60                                  # PRs with their numbers
gh issue list --state all --limit 60                                  # issues, assignees and state
.venv/bin/python -m pytest -q                                         # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q -m integration
docker buildx imagetools inspect racj90120hplmnv09/mibici-ml-service:latest   # digest and platforms
```
