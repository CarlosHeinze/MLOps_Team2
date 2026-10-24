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

Steps 3, 5 and 6 are tracked in MLflow (see [Data versioning](#data-versioning)).

## How to run

Requirements:
- Python 3.12.
- About 5 GB of disk for the data.
- RAM: 16 GB is enough for the CLI steps. Cleaning peaks at about 8 GB plus its workers; use
  a lower `--workers` value if memory is tight. Notebook `1.02` loads all 39M trips at once and
  needs about 20 GB.

### 1. Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # also installs this package in editable mode
```

### 2. MLflow tracking server

Start it from the project root and leave it running; the tracking database (`mlflow.db`) and
artifacts (`mlartifacts/`) stay local and are git-ignored.

```bash
mlflow server --host 127.0.0.1 --port 5000
```

> **macOS:** AirPlay Receiver already listens on port 5000 and answers 403, so the pipeline
> stops with "MLflow tracking server not reachable". Start the server on another port, for
> example `--port 5001`, and point the project to it in a `.env` file at the project root
> (git-ignored):
>
> ```bash
> MLFLOW_TRACKING_URI=http://127.0.0.1:5001
> ```

### 3. Data pipeline

```bash
python -m team2_mlops2026.data.raw_data_extraction         # ~3 GB, only downloads what is missing
python -m team2_mlops2026.data.raw_data_normalization
python -m team2_mlops2026.data.interim_data_cleaning       # MLflow run: clean-interim-trips
python -m team2_mlops2026.features                         # MLflow run: build-station-hour-features
python -m team2_mlops2026.station_profiles                 # MLflow run: fit-station-profiles
```

Every step is idempotent and has `--help`. Useful options:

| Option | Steps | Effect |
|---|---|---|
| `--since YYYY-MM` | Extraction | Only download from that month on |
| `--force` | Extraction, normalization, cleaning | Redo files that are up to date |
| `--workers N` | Extraction, normalization, cleaning | Parallel downloads or processes |
| `--no-track` | Cleaning, features, profiles | Run without the MLflow server |
| `--start YYYY-MM` | Features | First month of the feature table (default `2023-01`) |
| `--fit-end YYYY-MM` | Profiles | Fit the PCA on the hours before this month (default `2025-10`) |
| `--variance X` | Profiles | Keep the components that explain this share of the variance (default `0.9`) |

These are the times and output sizes for the full history (39.1M raw trips), measured on a
16-core, 64 GB machine:

| Step | Time | Output size |
|---|---|---|
| Extraction | ~12 min (depends on the network) | 3.0 GB |
| Normalization (8 workers) | ~40 s | 789 MB |
| Cleaning (8 workers) | ~50 s | 766 MB |
| Features | ~20 s | 78 MB |
| Station profiles | ~7 s | 105 KB |

### 4. Notebooks

Run them in order: `0.01` → `1.01` → `1.02` → `1.03` → `1.04` → `1.05`.
- `0.01`, `1.01`, `1.02`, `1.04` and `1.05` each run their pipeline step, skipping files that
  are already up to date, and validate its output.
- `1.03` only reads `data/processed`. It first checks that data against the cleaning contract,
  so it stops if the data is stale.

### 5. Tests and lint

```bash
make test    # pytest: unit tests on small synthetic data, no network or MLflow server needed
make lint    # ruff format --check + ruff check
```

## Data versioning

- **Raw:** `data/raw/mibici/manifest.json` records the source URL, SHA-256, size and download
  date of every file.
- **MLflow:** each tracked step is a run in the `mibici` experiment that logs:
  - its input and output datasets, each with a digest of the files' contents, so any change in
    the data produces a new digest
  - its parameters (the data contract bounds, the feature definition)
  - its metrics (rows removed per rule, table shape)

  Each step's output digest equals the next step's input digest, which links the runs:
  `clean-interim-trips` → `build-station-hour-features` → `fit-station-profiles`.
- **Changes to the data:** `data/processed/mibici/cleaning_report.json` lists, for every month,
  the rows and values each cleaning rule touched.

## Project Organization

```
├── Makefile           <- `make requirements`, `make test`, `make lint`
├── README.md
├── data               <- Git-ignored; rebuilt by the pipeline
│   ├── raw            <- MiBici CSVs exactly as published, plus manifest.json
│   ├── interim        <- Typed Parquet, one schema for every month
│   └── processed      <- Cleaned trips, cleaning_report.json, feature table and station profiles
│
├── docs               <- Default mkdocs project (not used yet)
├── models             <- Trained models (phase 2)
├── notebooks          <- PHASE.NOTEBOOK-team2-description, run in order
│   ├── 0.01-team2-initial-data-exploration.ipynb
│   ├── 1.01-team2-data-format-normalization.ipynb
│   ├── 1.02-team2-data-cleaning.ipynb
│   ├── 1.03-team2-exploratory-data-analysis.ipynb
│   ├── 1.04-team2-preprocessing.ipynb
│   └── 1.05-team2-station-profiles-pca.ipynb
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
    ├── dataset.py     <- Cookiecutter placeholder (not used)
    └── modeling       <- Cookiecutter placeholders for training and prediction (phase 2)
```
