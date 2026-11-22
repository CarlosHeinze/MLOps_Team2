# Drift monitoring verification

Issue [#42](https://github.com/CarlosHeinze/MLOps_Team2/issues/42) defines the scenarios
and uses the served champions from [#39](https://github.com/CarlosHeinze/MLOps_Team2/issues/39).
The threshold and response policy is in [ML Canvas §10](../references/ml-canvas.md#10-monitoring).

## Replay the snapshot

Use the station-hour table and the profiles from the
[Phase 3 snapshot](verificacion-fase3.md) under `data/processed/mibici/features/`.
Use the two trustworthy champion exports under `models/` with `model_card.json`.
Get them through the team's snapshot and model export procedure. Monitor reads only the
training columns needed by the baselines plus the three requested windows. It does not
read raw trip CSVs.

```bash
make monitor MONITOR_ARGS="--severity 0.10 --severity 0.25 --severity 0.50"
make monitor MONITOR_ARGS="--scenario real --real-month 2026-09 --no-track"
.venv/bin/python -m team2_mlops2026.monitoring --help
```

Start `make mlflow` for tracked runs. Paths can be overridden with `--src`,
`--models-dir` and `--dst`. Use different destination folders to preserve report
sets from different invocations. The local model loader matches the service's default.
The Registry loader is available with `--model-source registry`. The CLI writes
`summaries.json` plus per-case drift, performance and decision CSVs, a summary JSON and
PNG figures. MLflow records the reference and monitoring inputs with content digests,
configuration, metrics, alert flags, action and model URIs. Its artifacts include the
case's reports and plots. The digests cover the schema and every ordered row using pandas
hashes and SHA-256. The source feature tables stay in the snapshot.

October 2025 serves as the validation reference for the frozen champions. The original
tuning validation months were subsequently used in their fit. November 2025 is the clean
source for all injected cases, so paired WAPE deltas isolate the intervention from the
calendar and station mix. September 2026 is the default unmodified replay. Its new-station
segment uses IDs absent before May 2026 in the feature table. A reference/monitoring
comparison measures the combined differences between the windows. Each baseline is fitted
with the model card's training dates. The joined PCA basis must predate the reference window,
and weekly profile inputs and lags retain the pipeline's historical lookback.

## Interpret the output

- PSI uses reference quantile bins with duplicate edges removed, lower and upper tails,
  a missing bin and a probability floor of `1e-6`. A steady variable should have PSI near
  zero. Large shifts can occur if the monitoring observations leave the support of a
  discrete or concentrated variable. KS reports the largest gap between numeric empirical
  CDFs and excludes missing values. Its p-values are descriptive because station-hours are
  dependent, and some inputs have ties. Station IDs use categorical PSI.
  [SciPy KS documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- Reference-vs-monitoring PSI drives alerts. `psi_vs_control` isolates the injected change.
  The encoded month and actual demand are shown for context. Missing features and unseen IDs
  have quality checks. The model's WAPE and relative gain over the best baseline drive
  performance alerts. The signed paired delta remains visible even when the performance
  alert threshold is not crossed.
- With an undefined WAPE or zero baseline WAPE, an undefined relative improvement is shown
  as blank in CSV and `null` in JSON. Use the available absolute metrics and the number of rows.
  Imbalance precision and recall use capacity from the unmodified rows for the simulations,
  including when the scenario masks it as a model input. Historical rows with unknown
  capacity do not enter the decision metric. The decision report includes eligible station-days.
- The holiday scenario attenuates weekday peak counts and their matched historical lags.
  It approximates the share of commuting hours in the rolling mean. Station-based
  perturbations round selected station counts up, and seed 42 fixes their selection.
  These interventions stress the feature and forecast paths. They do not model individual
  trip flows or enforce system-wide balance.

## Synthetic execution on macOS

Run the [3.03 notebook](../notebooks/3.03-team2-data-drift.ipynb) from a kernel using the
repository environment. The demonstration writes a separate SQLite store and reports
under `data/processed/mibici/monitoring-demo/`. It fits two small LightGBM models on
synthetic features only. Its control contains 5,040 station-hours across seven stations,
and the reference contains 5,208. The unmodified synthetic replay has 5,952 rows across
eight stations. Its eighth station is unseen in training.

On 22 November 2026 the notebook produced 17 finished, locally tracked runs, with one
`ok` and 16 `alert` outcomes. The table shows champion WAPE and the lowest same-window
baseline WAPE for the corresponding target, both during weekday peak hours. The PSI column
uses the largest monitored-feature or forecast PSI, excluding the encoded calendar month
and demand labels.

| Run | Max PSI | Peak WAPE departures (model / baseline) | Peak WAPE arrivals (model / baseline) | Observed outcome |
|---|---:|---:|---:|---|
| `drift-control-0` | 0.0355 | 0.0043 / 0.3916 | 0.0063 / 0.3577 | No alert |
| `drift-mean-shift-0.25` | 12.2890 | 0.0501 / 0.3933 | 0.1262 / 0.3538 | PSI alert, performance within both limits |
| `drift-missing-features-0.25` | 6.9806 | 0.1152 / 0.3916 | 0.1073 / 0.3577 | Input-quality alert, review the pipeline |
| `drift-covariate-only-0.5` | 23.3742 | 0.3722 / 0.3916 | 0.3029 / 0.3577 | Departure gain fell to 4.96%, below the 10% floor |
| `drift-seasonal-0.5` | 3.5943 | 0.4724 / 0.7885 | 0.5624 / 0.7108 | Arrival WAPE exceeded the 0.561 limit |
| `drift-real-0` (synthetic inputs) | 10.2063 | 0.0683 / 0.4011 | 0.1435 / 0.3703 | Unseen row share was 12.5% |

The MLflow `Model training → Runs` view showed all 17 runs in
`mibici-monitoring-demo`. The `drift-missing-features-0.25` run rendered the PSI artifact
and listed the CSVs. The observed run set is under
`data/processed/mibici/monitoring-demo/demo-eu2qogm1/`. To reopen it locally, run:

```bash
DEMO="$PWD/data/processed/mibici/monitoring-demo/demo-eu2qogm1"
.venv/bin/mlflow server --backend-store-uri "sqlite:///$DEMO/mlflow.db" --host 127.0.0.1 --port 5012
```

The deterministic fixture's low control error demonstrates the checks on a simple signal.
The results do not estimate MiBici's production accuracy. The historical replay of
September 2026 and the golden forecasts of the exported champions remain unverified on
this machine. `make monitor MONITOR_ARGS="--no-track --scenario real"` exited with code
2 at the missing `models/model_card.json`. The snapshot's features and profiles were
also absent locally.

The final `make lint` passed. `make test` completed with 306 passed, one skipped and
34 warnings from MLflow's use of a deprecated pandas operation. The skipped test is
the existing golden-forecast check for the exported champions. The 15 monitoring
tests cover every zero-severity scenario, the perturbations, PSI and KS, the alert
boundaries, full dataset digests, offline CLI execution and tracked synthetic replay.
