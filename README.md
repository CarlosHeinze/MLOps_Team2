# Team2_MLOps2026

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Project for MLOps Course for the Master in Applied AI from Tec de Monterrey

## Problem and value proposition

We forecast hourly bike demand per MiBici station so the operator can rebalance bikes before
stations run empty or full. See the [ML Canvas](references/ml-canvas.md) for the full
requirements analysis.

The data is the [MiBici open data](https://mibici.net/en/open-data/): every trip of the
Guadalajara public bike system since December 2014, published as one CSV per month, plus the
current station list.

Phase 1 handoff: [presentation material](docs/fase1-presentacion-julian.md),
[functional roles](docs/roles.md), [AI usage declaration](docs/ai-usage.md), and
[Julian's local verification](docs/verificacion-fase1-julian.md). The presentation material
distinguishes results reported in earlier PRs from checks performed on this branch.

## Pipeline

Each step is a module with a CLI, and a notebook that explains and checks its decisions. A step
reads only the previous step's output, so any step can be re-run on its own.

| Step | Module (CLI) | Output | Notebook |
|---|---|---|---|
| 1. Extraction | `team2_mlops2026.data.raw_data_extraction` | `data/raw/mibici/`: original CSVs plus `manifest.json` (URL, SHA-256, size) | `0.01` |
| 2. Format normalization | `team2_mlops2026.data.raw_data_normalization` | `data/interim/mibici/`: one typed Parquet per month, one schema | `1.01` |
| 3. Cleaning (data contract) | `team2_mlops2026.data.interim_data_cleaning` | `data/processed/mibici/viajes/` plus `cleaning_report.json` | `1.02` |
| 4. Exploratory analysis | — | Findings for features and modelling | `1.03` |
| 5. Features and preprocessing | `team2_mlops2026.features` | `data/processed/mibici/features/station_hour_features.parquet` | `1.04` |
| 6. Station profiles (PCA) | `team2_mlops2026.station_profiles` | `data/processed/mibici/features/station_profiles.parquet` | `1.05` |
| 7. Baselines and candidates | `team2_mlops2026.modeling.train fit` | MLflow runs `train-<model>-<target>`, each with its fitted model | `2.01`, `2.02` |
| 8. Hyperparameter tuning | `team2_mlops2026.modeling.train tune` | `tune-<model>-<target>`: one nested run per trial, and the best params refitted on the training split | `2.03` |
| 9. Backtest and PCA ablation | `team2_mlops2026.modeling.train backtest` and `... ablation` | `backtest-<model>` with one nested run per monthly fold; `ablation-<model>-<target>-pca` / `-no-pca` | `2.04` |
| 10. Model Registry | `team2_mlops2026.modeling.registry` | `mibici-departures` and `mibici-arrivals`, aliases `champion` and `challenger` | `2.05` |
| 11. Batch forecast | `team2_mlops2026.modeling.predict` | `data/processed/mibici/predictions/forecast_<YYYYMMDD_HH>h.parquet` and a `predict-<start>` run | `2.05` |
| 12. Model export | `team2_mlops2026.modeling.export` | `models/mibici-<target>/`: each champion as registered, plus `models/model_card.json` | `3.01` |
| 13. Online serving | `team2_mlops2026.serving.app` (FastAPI) | `POST /predict`, `GET /health` and `GET /model` on port 8000; schema in `docs/api/` | `3.01` |
| 14. Container | `Dockerfile` (`make docker-build`, `make docker-push`) | Image [`racj90120hplmnv09/mibici-ml-service`](https://hub.docker.com/r/racj90120hplmnv09/mibici-ml-service) on Docker Hub, for amd64 and arm64 | — |
| 15. Reproducibility | `team2_mlops2026.modeling.reproducibility` (`make snapshot`, `make reproduce`) | `snapshot-<digest>` run with the champions' tables and reference outputs, `tests/data/reference.json`, `reproduce-<model>` runs | `3.02` |
| 16. Drift monitoring | `team2_mlops2026.monitoring` (`make monitor`) | `data/processed/mibici/monitoring/` and one `drift-<scenario>-<severity>` MLflow run per case | `3.03` |

Every step except the exploratory analysis is tracked in MLflow (see
[Data versioning](#data-versioning) and [MLflow](#mlflow-comparing-runs-and-the-model-registry)).

## How to run

Requirements:
- Python 3.12.
- About 5 GB of disk for the data.
- RAM: 16 GB is enough for the CLI steps. Cleaning peaks at about 8 GB plus its workers; use
  a lower `--workers` value if memory is tight. Notebook `1.02` loads all 39M trips at once and
  needs about 20 GB.

### 1. Environment

```bash
make create_environment            # requires Python 3.12
make requirements                  # installs into .venv with requirements-lock.txt
cp .env.example .env               # local MLflow settings; .env is git-ignored
```

The `Makefile` uses `.venv/bin/python` by default. If you manage a separate Python 3.12
environment, pass `PYTHON_INTERPRETER=/path/to/python` to `make` instead.
`requirements.txt` pins every direct dependency to the versions installed in the local
Python 3.12 environment used for this branch's unit suite. It includes
`requirements-serving.txt`, the subset that the prediction service, its Docker image and the
logged models need, so each pin lives in one file only. The pandas, pyarrow, MLflow
and scikit-learn pins also match the combination reported as working in issue
[#12](https://github.com/CarlosHeinze/MLOps_Team2/issues/12).

`requirements-lock.txt` pins every package, the transitive ones too (186), as `pip install
-r requirements.txt` resolved them in a clean Python 3.12 venv. `make requirements` and the
Docker image install with it as constraints (`-c`), so every machine gets the same versions; a
package that only one OS needs (such as `colorama` on Windows) is still resolved there. After
changing a pin in `requirements.txt` or `requirements-serving.txt`, regenerate it with
`make lock` (`make lock LOCK_PYTHON=/path/to/python3.12` if `python3.12` is not on the path);
a test fails while a pin is missing from it.

### 2. MLflow tracking server

Start it from the project root and leave it running; the tracking database (`mlflow.db`) and
artifacts (`mlartifacts/`) stay local and are git-ignored.

```bash
make mlflow
```

The default is `127.0.0.1:5001` to avoid the macOS AirPlay Receiver on port 5000.
`make mlflow MLFLOW_PORT=...` and `MLFLOW_TRACKING_URI` in `.env` must agree if you use
another port. Keep the server terminal open while running tracked steps. The database and
artifacts remain local and are not a shared record for other teammates.

### 3. Data pipeline

`make data` runs extraction → normalization → cleaning. It downloads the full MiBici
history (about 3 GB of raw CSVs and about 5 GB across pipeline stages). It does not
install dependencies. Run it only when that download and disk use are intended.

```bash
make data
.venv/bin/python -m team2_mlops2026.features
.venv/bin/python -m team2_mlops2026.station_profiles
```

For a smaller extraction, use the extraction CLI with `--since YYYY-MM`, then run the
subsequent modules separately. These commands have not been validated on a partial
history; the full-history notebook totals will not match it. `--no-track` permits an
individual step without the MLflow server.

Every step is idempotent and has `--help`. Useful options:

| Option | Steps | Effect |
|---|---|---|
| `--since YYYY-MM` | Extraction | Only download from that month on |
| `--force` | Extraction, normalization, cleaning | Redo files that are up to date |
| `--workers N` | Extraction, normalization, cleaning | Parallel downloads or processes |
| `--no-track` | All | Run without the MLflow server |
| `--dry-run` | Extraction | Only list what would be downloaded (never tracked) |
| `--start YYYY-MM` | Features | First month of the feature table (default `2023-01`) |
| `--fit-end YYYY-MM` | Profiles | Fit the PCA on the hours before this month (default `2025-10`) |
| `--variance X` | Profiles | Keep the components that explain this share of the variance (default `0.9`) |

These are the times and output sizes for the full history (39.1M raw trips), measured on a
16-core, 64 GB machine:

| Step | Time | Output size |
|---|---|---|
| Extraction | ~12 min (depends on the network); ~20 s when nothing is missing | 3.0 GB |
| Normalization (8 workers) | ~40 s | 789 MB |
| Cleaning (8 workers) | ~50 s | 766 MB |
| Features | ~20 s | 78 MB |
| Station profiles | ~7 s | 105 KB |

### 4. Modelling pipeline

With `data/processed` present (step 3), `make pipeline` runs every modelling step in order
against the MLflow server: `features` → `train` → `tune` → `evaluate` → `register` →
`predict`. Each one is also a target of its own:

```bash
make features   # feature table and station profiles
make train      # baselines and candidates with their default params (2.01, 2.02)
make tune       # random search of the finalists' hyperparameters (2.03)
make evaluate   # 12-fold rolling-origin backtest, then the PCA ablation (2.04)
make register   # champion and challenger of each target in the Model Registry (2.05)
make predict    # forecast of the 24 h after the last data (2.05)
```

The long steps run for hours. On macOS, keep the machine awake with `caffeinate -i make
pipeline` (or `caffeinate -i make evaluate`): a laptop that sleeps stops the run halfway.
A remote VM avoids the problem altogether.

Useful options of the modelling CLIs (`--help` lists them all):

| Option | Steps | Effect |
|---|---|---|
| `--model NAME` | train, tune, backtest, ablation | Only that model, repeatable (names in `modeling/models.py`) |
| `--target departures` | train, tune, backtest, ablation, registry | Only that target |
| `--n-trials N` | tune | Random trials besides the defaults (default `20`) |
| `--n-folds N` | backtest | Monthly folds (default `12`, from `2025-10`) |
| `--no-tuned` | backtest, ablation | Use the catalog defaults instead of the latest tune run |
| `--champion NAME` | registry | Promote that finalist instead of the backtest's first |
| `--start "YYYY-MM-DD HH:00"` | predict | Replay the forecast of a past day (default: the hour after the data) |

Times on a 2019 MacBook Pro (8-core i9, 64 GB):

| Step | Time |
|---|---|
| `make train` | ~10 min (14 runs) |
| `make tune` | ~75 min (2 finalists × 2 targets × 21 trials, plus the refits) |
| `make evaluate` | ~3 h 10 min (backtest of 4 models × 12 folds ~3 h, ablation ~13 min) |
| `make register` | < 1 min |
| `make predict` | ~10 s |

### 5. Serving the champions (FastAPI)

The service forecasts online with the same models and the same numbers as the batch forecast.
It loads the champions from `models/`, so once they are exported it needs no MLflow server:

```bash
make export     # models:/mibici-<target>@champion -> models/ (needs the MLflow server)
make serve      # http://127.0.0.1:8000, Swagger UI at http://127.0.0.1:8000/docs
```

Served models (alias `champion`, exported on 2026-11-18):

| Target | Model URI | Registered model | Version | Algorithm |
|---|---|---|---|---|
| departures | `models:/mibici-departures/1` | `mibici-departures` | 1 | `lgbm_tweedie` (LGBMRegressor) |
| arrivals | `models:/mibici-arrivals/1` | `mibici-arrivals` | 1 | `lgbm_tweedie` (LGBMRegressor) |

`make export` resolves the alias once and downloads that pinned version. It also writes
`models/model_card.json`: version, source run, test and backtest metrics, input digests and git
commit of each model, as tagged in the Registry. `GET /model` returns that card.

| Endpoint | Returns |
|---|---|
| `POST /predict` | Expected departures, arrivals and net flow (arrivals - departures) per row, and the model version of each target |
| `GET /health` | 200 with the models loaded, 503 without them (for the container's health check) |
| `GET /model` | The model card |
| `GET /docs`, `GET /openapi.json` | Swagger UI and the OpenAPI schema ([docs/api/openapi.json](docs/api/openapi.json)) |

The request carries one feature row per station-hour: the columns of `features.forecast_rows`
(lags, weekday, cyclical hour and month, station attributes) and, optionally, the profile
scores `pc1`-`pc5`. `null` marks an unknown value, as NaN does in the feature table. Example
with three stations at the morning peak of 9 Sep 2026, the night replayed in notebook `2.05`:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @docs/api/example_request.json
```

The response ([docs/api/example_response.json](docs/api/example_response.json)), shortened:

```json
{
  "predictions": [
    {"station_id": 51, "hour_start": "2026-09-09T08:00:00", "departures": 46.84400274438855,
     "arrivals": 14.689625404127986, "net_flow": -32.15437734026056},
    ...
  ],
  "models": {
    "departures": {"name": "mibici-departures", "version": "1", "alias": "champion",
                   "uri": "models:/mibici-departures/1"},
    ...
  }
}
```

| Status | When |
|---|---|
| 422 | A row breaks the data contract: a field out of bounds, missing or unknown, an hour with a time zone or minutes, a weekday of another day. The body names the field and the reason |
| 400 | An empty batch or more than 25,000 rows (a whole night is ~11,200) |
| 503 | The models are not loaded: run `make export`, then restart |
| 500 | The forecast failed; the details stay in the service log |

| Setting | Default | Effect |
|---|---|---|
| `MODEL_SOURCE` | `local` | `local`: the exported folder. `registry`: `models:/mibici-<target>@<MODEL_ALIAS>` from the MLflow server in `MLFLOW_TRACKING_URI` |
| `MODEL_DIR` | `models/` | Folder that `make export` wrote |
| `MODEL_ALIAS` | `champion` | Alias served with `MODEL_SOURCE=registry` |
| `SERVE_HOST`, `SERVE_PORT` | `127.0.0.1`, `8000` | Address of `make serve` (`make serve SERVE_PORT=8080`) |

Notebook `3.01` sends the 11,208 rows of 9 Sep 2026 to the service and compares them with the
batch forecast: the largest difference is 0. After a change to the schemas or the app,
`make api-docs` regenerates `docs/api/`; a test fails while it is out of date.

### 6. Docker image

The image holds the service and the exported champions, so it runs with Docker alone: no
Python, no MLflow server and no data. It is published on Docker Hub for `linux/amd64`
(Windows and Intel or AMD Linux) and `linux/arm64` (Apple silicon), and `docker pull` takes
the one of the machine.

**Run the published image:**

```bash
docker run -d --name mibici-ml-service -p 8000:8000 racj90120hplmnv09/mibici-ml-service:0.1.1
curl -s http://127.0.0.1:8000/health
# {"status":"ok","models_loaded":true,"source":"local"}
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @docs/api/example_request.json
docker exec mibici-ml-service python -m team2_mlops2026.serving.app check
docker rm -f mibici-ml-service
```

`check` runs inside the container: it sends the example request kept in the image and compares
the answer with `docs/api/example_response.json`. In Windows PowerShell, write `curl.exe`
instead of `curl`, or open http://127.0.0.1:8000/docs.

**Build and run it locally:**

```bash
make export                                # or make docker-models (below)
docker build -t ml-service:latest .        # make docker-build adds the labels
docker run -p 8000:8000 ml-service:latest  # make docker-run
make docker-smoke                          # starts ml-service:latest in the background and runs check
```

The build fails with a clear message if `models/` is empty. Without access to the MLflow
server, `make docker-models` copies the models out of the published image into `models/`, and
`make docker-build` then rebuilds the same image from the repository.

**Published tags** ([Docker Hub](https://hub.docker.com/r/racj90120hplmnv09/mibici-ml-service)):

| Tag | Digest | Git commit | Models inside | Platforms | Download size |
|---|---|---|---|---|---|
| `0.1.1`, `latest` | `sha256:41eba4c22003...` | `29448be` | `models:/mibici-departures/1`, `models:/mibici-arrivals/1` (`lgbm_tweedie`, alias `champion`) | linux/amd64, linux/arm64 | 253 MB (amd64), 248 MB (arm64); about 880 MB unpacked |
| `0.1.0` | `sha256:8bb27ad0ec19...` | `f5fb9bc` | The same | linux/amd64, linux/arm64 | The same |

`0.1.1` installs every transitive dependency from `requirements-lock.txt`; `0.1.0` resolved
them at build time. Both serve the same models and give the same forecasts.

Each image also carries them as labels: `org.opencontainers.image.version`,
`org.opencontainers.image.revision` (the commit) and `team2_mlops2026.models` (the model URIs).
After a pull, `docker image inspect --format '{{json .Config.Labels}}' <image>` shows them, and
`GET /model` returns the full model card.

**Publish a new version** (the repository owner, or a teammate added as a collaborator on
Docker Hub under *Settings → Collaborators*, by Docker ID):
1. Bump `__version__` in `team2_mlops2026/__init__.py`, run `make api-docs` (the OpenAPI
   schema carries the version) and commit.
2. `docker login`, then `make docker-push`. It builds both platforms and pushes `:<version>`
   and `:latest`. It refuses uncommitted changes, since the image is labelled with the
   commit, and a version that is already published.
3. Add the row to the table above.

| Inside the image | |
|---|---|
| Base | `python:3.12.12-slim-trixie` pinned by digest (the Python of the training environment), plus `libgomp1` for LightGBM |
| Python packages | `requirements-serving.txt` with `requirements-lock.txt` as constraints, and this package, in `/opt/venv` (multi-stage build) |
| Models | `models/` in `/app/models` (`MODEL_DIR`), with `model_card.json` |
| API examples | `docs/api/example_request.json` and `example_response.json` in `/app/docs/api`, for `check` |
| User | `app` (uid 10001), not root |
| Health check | `GET /health` every 30 s; the container turns `healthy` in about 10 s |
| Command | `uvicorn team2_mlops2026.serving.app:app --host 0.0.0.0 --port 8000` |

**Same forecasts everywhere.** The 11,208 forecasts of 9 Sep 2026 are equal on the amd64 and
the arm64 image. Against the service run directly on macOS they differ by at most 1.8e-15, the
last bit of the exponential of another C library, so `check` allows a difference of 1e-9.

**The models' own requirements.** Every model logged from this version on (`make train`,
`make tune`) declares the pins of `requirements-serving.txt` without the web service, and
carries a copy of `team2_mlops2026` (MLflow's `code_paths`). `mlflow.sklearn.load_model` then
works in a clean environment without installing the package. The current champions were logged
before this change, so they load where the package is installed, as in the image.

### 7. Reproducibility

The deployable model can be rebuilt and checked in another environment:

| Piece | Where |
|---|---|
| Dependencies | `requirements-lock.txt` (§1), and the image's base pinned by digest |
| Seeds | `features.SEED = 42` reaches every random source (ML Canvas §8) |
| Model, inputs and reference outputs | The `snapshot-373aa06d7db0` run in MLflow: the feature table and the station profiles that trained the champions, with their digests, and the reference set (the champions' test metrics, request rows with their expected forecasts, the metrics of the 14 `train-*` runs). The registered versions carry the tag `snapshot_run`. A copy of the reference set is in `tests/data/reference.json` |

```bash
make snapshot     # once per champion: needs the MLflow server, models/ and the feature table
make reproduce    # on any machine: refit the champions, compare the train runs, check the service
make reproduce REPRODUCE_ARGS="--no-track --url http://127.0.0.1:8000 --save docs/reproducibility/<name>.json"
```

`make reproduce` (`python -m team2_mlops2026.modeling.reproducibility compare`):
1. checks that the local feature table has the snapshot's digests, and refuses to go on otherwise;
2. refits each champion with its own params (an unfitted clone of the exported model) on the
   training split, and compares its test metrics with the reference. It also scores the exported
   champion, so a difference in the data shows up apart from one in the training;
3. with `--train-runs`, compares the latest local `train-*` runs (`make train`) with the
   reference ones;
4. with `--url`, sends the reference rows to a running service and compares the forecasts;
5. logs a `reproduce-<model>` run with the environment (OS, CPU, Python, packages, commit) and
   the differences, unless `--no-track`, and exits with 1 if a check fails.

The tolerances are `METRIC_RTOL` in `modeling/reproducibility.py`: `1e-9` for the baselines,
`1e-4` for LightGBM, `5e-3` for the histogram boosting. The ML Canvas §8 explains them and what
is reproduced bit for bit. Results, in [docs/verificacion-fase3.md](docs/verificacion-fase3.md)
and notebook `3.02`:

| Environment | Refit: largest relative metric difference (departures / arrivals) | Service: golden forecasts |
|---|---|---|
| macOS, Apple M4 Pro (the machine of the snapshot) | 1.4e-14 / 1.1e-13 | Image `0.1.0`: difference 0 |
| Clean Linux container (aarch64): clone, `make requirements`, `make test`, `make reproduce` | 8.4e-15 / 1.5e-13 | Image `0.1.1` from Docker Hub: difference 0 |
| Another machine (a teammate) | Pending | Pending |

The golden test (`tests/test_reproducibility.py`) runs in `pytest -q`. It checks that the
reference request still fits the API and that the reference is the documented example. When
`models/` holds the reference versions (`make export` or `make docker-models`), it also checks
that they give the golden forecasts; otherwise that one test is skipped.

### 8. Drift monitoring

Use the station-hour features and profiles from the team snapshot and both exported champions.
The default uses October 2025 as held-out validation for monitoring, November 2025 as the
source for the simulations, and September 2026 for the unmodified historical replay. The
baselines and the station demand groups use the models' recorded training window.

```bash
make monitor
make monitor MONITOR_ARGS="--no-track --scenario missing-features --severity 0.10 --severity 0.25 --severity 0.50"
make monitor MONITOR_ARGS="--no-track --scenario real --real-month 2026-09"
```

Start `make mlflow` in another terminal to track the runs. The `--no-track` option writes the
same reports and figures locally without logging runs. The CLI takes `--src`, `--models-dir`,
`--dst`, window overrides and alert thresholds. Notebook `3.03` runs a small synthetic
demonstration without the historical snapshot. See [monitoring verification](docs/monitoring.md)
and [ML Canvas §10](references/ml-canvas.md#10-monitoring).

### 9. Notebooks

Run them in order: `0.01` → `1.01` → `1.02` → `1.03` → `1.04` → `1.05` → `2.01` → `2.02` →
`2.03` → `2.04` → `2.05` → `3.01` → `3.02` → `3.03`.
- `0.01`, `1.01`, `1.02`, `1.04` and `1.05` each run their pipeline step, skipping files that
  are already up to date, and validate its output.
- `1.03` only reads `data/processed`. It first checks that data against the cleaning contract,
  so it stops if the data is stale.
- `2.01` (baselines) and `2.02` (candidates) train their models with tracking. `2.03`, `2.04`
  and `2.05` read the runs of `make tune` and `make evaluate`, and `2.05` registers the
  champions and runs the batch forecast.
- `3.01` needs `make mlflow` and `make serve` running: it sends a night of rows to the service
  and compares the answer with the batch forecast.
- `3.02` needs `make mlflow` and the image running on port 8000. It refits the champions (about
  six minutes) and compares them, and the results of `docs/reproducibility/`, with the reference.
- `3.03` compares the monitoring scenarios, alerts and proposed responses.

### 10. Tests and lint

One command from the project root runs the whole suite:

```bash
pytest -q
```

It needs no historical data and no MLflow server: the tests build small synthetic trips
(`tests/trip_factory.py`) and track into a throwaway SQLite store. `make test` runs the same
command.

```bash
pytest -q -m "not integration"   # only the unit tests
pytest -q -m integration         # only the end-to-end test
pytest -q --cov                  # with the line coverage of team2_mlops2026
make lint                        # ruff format --check + ruff check
```

- **Unit tests:** one module per source module (`tests/test_<module>.py`). They cover the
  rules of each step on hand-made data: the cleaning contract, the lags and the preprocessing
  (`test_features`), the metrics and the success criterion (`test_evaluate`), the forecast and
  the profile join (`test_predict`), the tracked runs, the Registry and the lineage, and the
  export of the champions (`test_export`). `test_serving` calls the API with `TestClient` on
  small synthetic models: the forecasts equal `predict.forecast`, invalid rows get 422, an
  empty or oversized batch 400, and the service without models 503.
- **Integration test:** `tests/test_pipeline_integration.py`, marked `integration`. It runs
  the pipeline end to end on three months of synthetic commutes: raw CSVs → normalization →
  cleaning → features → station profiles → training of a baseline and a small LightGBM →
  champion in the Registry → 24 h forecast → metrics on the held-out month. At each step it
  checks the columns the next step reads, so a renamed feature column or a wrong net flow
  makes it fail.

| Command | Tests | Time |
|---|---|---|
| `pytest -q` | 292 | ~2 min |
| `pytest -q -m "not integration"` | 291 | ~2 min |
| `pytest -q -m integration` | 1 | ~15 s |

Times on a Windows laptop (Ryzen 7 4800H, 16 GB). Most of the time goes to the tests that log
runs and models to MLflow.

## Data versioning

- **Raw:** `data/raw/mibici/manifest.json` records the source URL, SHA-256, size and download
  date of every file.
- **MLflow:** each tracked step is a run in the `mibici` experiment that logs:
  - its input and output datasets, each with a digest of the files' contents, so any change in
    the data produces a new digest
  - its parameters (download options, the data contract bounds, the feature definition)
  - its metrics (files per encoding and date format, rows removed per rule, table shape)

  Each step's output digest equals the next step's input digest. This links the runs from the
  published CSVs to the model inputs:

  | Dataset | Output of | Input of |
  |---|---|---|
  | `mibici_raw` | `extract-raw-data` | `normalize-raw-data` |
  | `mibici_trips_interim` | `normalize-raw-data` | `clean-interim-trips` |
  | `mibici_stations` | `normalize-raw-data` | `clean-interim-trips`, `build-station-hour-features` |
  | `mibici_trips_processed` | `clean-interim-trips` | `build-station-hour-features` |
  | `mibici_station_hour_features` | `build-station-hour-features` | `fit-station-profiles` |
  | `mibici_station_profiles` | `fit-station-profiles` | `train-*`, `tune-*`, `backtest-*`, `ablation-*`, `predict-*` |
  | `mibici_predictions` | `predict-<start>` | the rebalancing plan |

  The model runs also read `mibici_station_hour_features`. A registered model version records
  the digests of the tables its source run read (`features_digest` and `profiles_digest`).

  The steps are deterministic: re-running one with the same code on the same input
  reproduces the same output digest, so a new digest always means the data or the code changed.
- **Changes to the data:** `data/processed/mibici/cleaning_report.json` lists, for every month,
  the rows and values each cleaning rule touched.

## MLflow: comparing runs and the Model Registry

Open the UI at http://127.0.0.1:5001 (with `make mlflow` running) and the `mibici` experiment.
The server and its database are local. To share them with the team, forward port 5001 from
VS Code (Ports tab) or run the server on a shared VM.

- **Compare models:** filter the runs with `attributes.run_name LIKE 'train-%'`, group them
  by the tag `model_family`, select them and press *Compare*. A bar chart of `peak_wape` or
  `overall_wape` across the runs reproduces the comparison of `2.02`.
- **Tuning:** open a `tune-<model>-<target>` run. Its nested runs (`defaults`, `trial-01`...)
  are the trials. Select them, *Compare*, and add a *Parallel Coordinates* chart with the
  searched params and `val_overall_poisson_deviance`.
- **Backtest:** each `backtest-<model>` run logs the key metrics of every fold as steps
  (`fold_departures_peak_wape`, ...): its *Model metrics* tab draws them per fold. The
  nested `fold-<month>` runs hold the full metrics of each month, and the parent run the mean
  and std (`departures_peak_wape`, `departures_peak_wape_std`) plus the figures under
  `backtest/`.
- **PCA ablation:** filter with `tags.experiment = 'pca_ablation'` and compare the `-pca` and
  `-no-pca` runs of each model and target.
- **Model Registry:** the *Models* tab lists `mibici-departures` and `mibici-arrivals`. Each
  version links to its source run and carries a description and tags: algorithm, training
  window, test and backtest metrics, success criterion, input digests and git commit. The
  alias `champion` marks the version that `predict` loads (`models:/mibici-<target>@champion`),
  and `challenger` the one kept for comparison. Promoting another version only moves the
  alias.

## Project Organization

```
├── Dockerfile         <- Image of the prediction service with the exported models
├── Makefile           <- environment, local MLflow, pipeline, serving, Docker, tests, lint
├── README.md
├── data               <- Git-ignored; rebuilt by the pipeline
│   ├── raw            <- MiBici CSVs exactly as published, plus manifest.json
│   ├── interim        <- Typed Parquet, one schema for every month
│   └── processed      <- Cleaned trips, cleaning_report.json, feature table, station profiles
│                         and forecasts
│
├── docs
│   ├── api            <- OpenAPI schema and an example request and response (make api-docs)
│   ├── reproducibility  <- Results of make reproduce in other environments, and the container script
│   └── verificacion-fase3.md <- The clean-environment checks of Phase 3
├── models             <- Git-ignored; the champions exported for serving, plus model_card.json
│                         (make export). The versions live in the MLflow Model Registry
├── notebooks          <- PHASE.NOTEBOOK-team2-description, run in order
│   ├── 0.01-team2-initial-data-exploration.ipynb
│   ├── 1.01-team2-data-format-normalization.ipynb
│   ├── 1.02-team2-data-cleaning.ipynb
│   ├── 1.03-team2-exploratory-data-analysis.ipynb
│   ├── 1.04-team2-preprocessing.ipynb
│   ├── 1.05-team2-station-profiles-pca.ipynb
│   ├── 2.01-team2-baselines.ipynb
│   ├── 2.02-team2-model-comparison.ipynb
│   ├── 2.03-team2-hyperparameter-tuning.ipynb
│   ├── 2.04-team2-model-evaluation.ipynb
│   ├── 2.05-team2-model-registry.ipynb
│   ├── 3.01-team2-model-serving.ipynb
│   └── 3.02-team2-reproducibility.ipynb
│
├── pyproject.toml     <- Package metadata, pytest, coverage and ruff configuration
├── references         <- ml-canvas.md: requirements analysis and value proposition
├── reports            <- Delivered reports (Fase1_Equipo02.pdf, Fase2_Equipo02.pdf) and the
│   │                     final one, written with the final-report skill (.claude/skills/)
│   └── figures        <- Figures of the reports, one folder per phase (fase1, fase2, fase3)
├── requirements.txt   <- Python dependencies (installs this package with `-e .`)
├── requirements-serving.txt <- The pinned subset for the service, the image and the logged models
├── requirements-lock.txt    <- Every package at its version, transitive ones too (make lock)
├── tests              <- pytest suite: one module per source module, plus the end-to-end
│   │                     test_pipeline_integration.py
│   └── data           <- reference.json: the golden reference set of the champions (make snapshot)
│
└── team2_mlops2026    <- Source code
    ├── config.py      <- Paths and MLflow settings (read from .env)
    ├── data
    │   ├── raw_data_extraction.py     <- Step 1: download the MiBici CSVs
    │   ├── raw_data_normalization.py  <- Step 2: encoding, column names, dates, types
    │   ├── interim_data_cleaning.py   <- Step 3: data contract, cleaning and validation
    │   └── trip_data.py               <- Shared trip loading and derived metrics
    ├── eda.py         <- EDA tables (data quality, outliers, effect sizes)
    ├── features.py    <- Step 5: station-hour features and preprocessing pipeline
    ├── station_profiles.py  <- Step 6: station usage profiles reduced with PCA
    ├── plots.py       <- Charts used by the notebooks
    ├── tracking.py    <- MLflow helpers: runs and dataset logging with digests
    ├── serving
    │   ├── schemas.py <- Step 13: Pydantic request and response schemas (the data contract)
    │   └── app.py     <- Step 13: FastAPI service, POST /predict, GET /health and /model
    └── modeling
        ├── models.py    <- Catalog of baselines and candidates, with their search spaces
        ├── evaluate.py  <- Metrics, segments, success criterion and rolling-origin folds
        ├── train.py     <- Step 7 and the CLI of steps 8-9: fit, tune, backtest, ablation
        ├── tuning.py    <- Step 8: random search with time-based validation
        ├── backtest.py  <- Step 9: rolling-origin backtest and PCA ablation
        ├── registry.py  <- Step 10: champion and challenger in the Model Registry
        ├── predict.py   <- Step 11: batch forecast with the registered champions
        ├── export.py    <- Step 12: the champions from the Registry to models/
        └── reproducibility.py <- Snapshot, reference set and comparison in another environment
```
