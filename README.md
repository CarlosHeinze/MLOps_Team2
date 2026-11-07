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
make requirements                  # installs into .venv; no implicit pip upgrade
cp .env.example .env               # local MLflow settings; .env is git-ignored
```

The `Makefile` uses `.venv/bin/python` by default. If you manage a separate Python 3.12
environment, pass `PYTHON_INTERPRETER=/path/to/python` to `make` instead.
`requirements.txt` pins every direct dependency to the versions installed in the local
Python 3.12 environment used for this branch's unit suite. The pandas, pyarrow, MLflow
and scikit-learn pins also match the combination reported as working in issue
[#12](https://github.com/CarlosHeinze/MLOps_Team2/issues/12). Transitive dependencies
are resolved by pip and are not locked; a full clean-machine/full-data reproduction
still needs verification by a teammate with the historical data.

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

### 5. Notebooks

Run them in order: `0.01` → `1.01` → `1.02` → `1.03` → `1.04` → `1.05` → `2.01` → `2.02` →
`2.03` → `2.04` → `2.05`.
- `0.01`, `1.01`, `1.02`, `1.04` and `1.05` each run their pipeline step, skipping files that
  are already up to date, and validate its output.
- `1.03` only reads `data/processed`. It first checks that data against the cleaning contract,
  so it stops if the data is stale.
- `2.01` (baselines) and `2.02` (candidates) train their models with tracking. `2.03`, `2.04`
  and `2.05` read the runs of `make tune` and `make evaluate`, and `2.05` registers the
  champions and runs the batch forecast.

### 6. Tests and lint

```bash
make test    # pytest: unit tests on small synthetic data, no historical dataset needed
make lint    # ruff format --check + ruff check
```

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
├── Makefile           <- environment, local MLflow, pipeline, tests, lint
├── README.md
├── data               <- Git-ignored; rebuilt by the pipeline
│   ├── raw            <- MiBici CSVs exactly as published, plus manifest.json
│   ├── interim        <- Typed Parquet, one schema for every month
│   └── processed      <- Cleaned trips, cleaning_report.json, feature table, station profiles
│                         and forecasts
│
├── docs               <- Default mkdocs project (not used yet)
├── models             <- Not used: the models are logged to MLflow and served by its Registry
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
│   └── 2.05-team2-model-registry.ipynb
│
├── pyproject.toml     <- Package metadata and ruff configuration
├── references         <- ml-canvas.md: requirements analysis and value proposition
├── reports
│   └── figures        <- Figures for reports and presentations
├── requirements.txt   <- Python dependencies (installs this package with `-e .`)
├── tests              <- pytest suite, one module per source module
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
    └── modeling
        ├── models.py    <- Catalog of baselines and candidates, with their search spaces
        ├── evaluate.py  <- Metrics, segments, success criterion and rolling-origin folds
        ├── train.py     <- Step 7 and the CLI of steps 8-9: fit, tune, backtest, ablation
        ├── tuning.py    <- Step 8: random search with time-based validation
        ├── backtest.py  <- Step 9: rolling-origin backtest and PCA ablation
        ├── registry.py  <- Step 10: champion and challenger in the Model Registry
        └── predict.py   <- Step 11: batch forecast with the registered champions
```
