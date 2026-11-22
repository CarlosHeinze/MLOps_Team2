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

## Historical replay on 22 November 2026

The team-shared export passed the schema and coverage checks. The station-hour table contains
12,538,736 rows from January 2023 through October 2026. The profile table contains 70,972
unique station-weeks and records a PCA fit cutoff of October 2025. The two exports include
October 2026, one month beyond snapshot `snapshot-373aa06d7db0`. Their digests differ from the
ones recorded for the deployed champions.

| File | Replay input digest | Champion training snapshot digest |
|---|---|---|
| `station_hour_features.parquet` | `75857847f53a78a330c347114f0c8987` | `636e73295b9df173482041fd5cd04ede` |
| `station_profiles.parquet` | `c9578c9c9963778247a15ce9afb8a9d0` | `51c85bee5c1166854a6da55a41631c28` |

The exported v1 champions pass the existing golden test on its pinned inputs. When those
same three station-hours are assembled from the shared Parquet files, their five PCA scores
differ and the forecasts move by up to 0.554 departures and 0.220 arrivals. The historical
results below describe the shared export. Exact reproduction with the original snapshot
remains a separate comparison.

The unmodified September 2026 replay used 346,320 station-hours from 481 stations.
The October 2025 reference contained 284,742 station-hours from 383 stations.
Both months are after the champions' training cutoff. The baselines used the same
recorded training window, from January 2023 up to October 2025.

| Window | Target | Champion peak WAPE | Seasonal-naive peak WAPE | Historical-average peak WAPE |
|---|---|---:|---:|---:|
| October 2025 reference | Departures | 0.4420 | 0.5521 | 0.4951 |
| October 2025 reference | Arrivals | 0.4084 | 0.5040 | 0.4510 |
| September 2026 replay | Departures | 0.5070 | 0.6125 | 0.6712 |
| September 2026 replay | Arrivals | 0.4838 | 0.5851 | 0.6519 |

- Champion overall WAPE rose from 0.5371 to 0.5892 for departures and from 0.5175 to
  0.5793 for arrivals. Overall MAE rose from 0.7724 to 0.8002 and from 0.7442 to
  0.7867, respectively. Imbalance precision and recall moved from 0.7310 and 0.4739
  to 0.6439 and 0.3901.
- Station ID PSI was 1.6387, longitude PSI was 0.4232 with KS D of 0.0666, and PC3
  PSI was 0.1031 with KS D of 0.0724. Forecast PSI was 0.0238 for departures and
  0.0274 for arrivals, both below the 0.10 warning threshold. The missing-rate
  threshold was not crossed.
- The quality alert fired because 100 unseen training IDs accounted for 20.79% of the
  replay rows. The proposed action was to review the feature pipeline and data contract.
  September peak WAPE stayed below both backtest limits. The best September baseline was
  seasonal naive, and the relative gains of 17.22% and 17.31% stayed above the 10% floor.
  The raw errors rose, while neither target met the configured performance degradation rule.

MLflow run `d65aa2c23ae943c3a68e9cbb39d79f74` (`drift-real-0`) finished in
`mibici-monitoring-historical-2026-09` with `data_kind=historical`. The raw input digests,
the champion snapshot digests and their mismatch are recorded on the run. The reports,
CSV metrics, and PSI and distribution previews were verified in the local MLflow store.
The store and artifacts are under `data/processed/mibici/monitoring-historical/2026-09/`.

```bash
HISTORICAL="$PWD/data/processed/mibici/monitoring-historical/2026-09"
.venv/bin/mlflow server --backend-store-uri "sqlite:///$HISTORICAL/mlflow.db" --host 127.0.0.1 --port 5012 --workers 1
```

### Required simulations on historical inputs

The same experiment also contains the unmodified November 2025 control and three 25%
perturbations of its 275,760 station-hours. They all compare with the October 2025
reference above and with both baselines. The table gives champion and best-baseline
peak WAPE on each case. The maximum PSI excludes calendar month and demand labels.

| Case | Max PSI | Departures model / baseline | Arrivals model / baseline | Outcome |
|---|---:|---:|---:|---|
| November control | 0.0515 | 0.4721 / 0.5417 | 0.4329 / 0.4885 | `ok` |
| Mean shift | 2.1772 | 0.4952 / 0.5653 | 0.4521 / 0.5174 | PSI alert, performance within limits |
| Missing features | 1.9142 | 0.5448 / 0.5417 | 0.5132 / 0.4885 | Input-quality alert and loss on both targets |
| Seasonal shift | 0.0515 | 0.4762 / 0.5919 | 0.4423 / 0.5451 | `ok` |

These four simulation runs are under the experiment's `reports/simulations-from-november/`
directory. They use historical source rows. The September run is the unmodified historical
replay, and the 17 runs in `mibici-monitoring-demo` used the separate synthetic fixture.

After the files were copied, `make lint test` passed with **307 tests passed** and
34 warnings from MLflow's deprecated pandas operation. The real replay reused the
exported champions.

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
The recorded historical replay is in the separate MLflow experiment described above.
The 15 monitoring tests cover zero severity, the perturbations, PSI and KS, alert
boundaries, full dataset digests, offline CLI execution and tracked synthetic replay.
