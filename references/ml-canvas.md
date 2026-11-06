# ML Canvas · MiBici station demand forecasting

Requirements analysis and value proposition for the project, following the
[ML Canvas](https://www.machinelearningcanvas.com/) by Louis Dorard. Every figure comes
from the notebooks in `notebooks/` run on the full MiBici history (December 2014 to September
2026), unless it is marked as an assumption.

## The problem

MiBici is the public bike-sharing system of the Guadalajara Metropolitan Area. A trip starts at
one station and ends at another, so bikes accumulate where people arrive and run out where
people leave. A station with no bikes loses departures, and a full station prevents riders
from returning their bike. The operator corrects this by moving bikes between stations with
trucks (rebalancing). Rebalancing works best when the operator knows **in advance** which
stations will empty or fill, and when.

The data shows the imbalance is large and structural:

- **Flows are one-directional.** Over the whole history, the Av. Vallarta / Los Arcos cluster
  (GDL-127 to GDL-130) sent about 320k more trips than it received, while López Cotilla /
  Penitenciaría (GDL-049) received 241k more than it sent (`1.03` §8).
- **A station can empty or fill within a single day.** The median station has 15 docks. Since
  2025, a station's daily net flow (arrivals minus departures) exceeds 12 bikes on 10% of
  station-days and 32 bikes on 1%.
- **Demand follows strong patterns.** Weekdays have commuter peaks at 8:00 and 18:00 and
  weekends a single midday peak. Almost no trips happen between 1:00 and 5:00 (`1.03` §3).
  These patterns are what make demand predictable.
- **Demand is concentrated.** The top 10% of the 486 stations start 39.5% of all trips
  (`1.03` §8).

## Canvas at a glance

| Block | Summary |
|---|---|
| **Value proposition** | Tell MiBici's rebalancing team, a day ahead, which stations will run out of bikes or docks and when, so trucks go where they are needed before the peaks. |
| **Prediction task** | Hourly departures and arrivals per station for the next 24 h; net flow follows from them. Multi-series count forecasting. |
| **Decisions** | Nightly and midday rebalancing plan: which stations to restock or empty, by how many bikes, and in what order. |
| **Impact simulation** | Rolling-origin backtest over the last 12 months against seasonal-naive and historical-average baselines (MAE, WAPE, Poisson deviance), plus how well it flags high-imbalance station-days. |
| **Making predictions** | Batch, nightly at about 23:00 for the next day (~480 stations × 24 h ≈ 11.5k values), with an optional midday refresh. Runs in seconds. |
| **Data collection** | Each month MiBici publishes a new CSV; the pipeline ingests, normalizes, validates and cleans it, and MLflow logs every dataset version. Labels arrive for free: next month's trips are the ground truth. |
| **Data sources** | MiBici open data: 142 monthly trip files (39.1M raw trips) and the current station nomenclature. Candidates: weather, public holidays and the school calendar. |
| **Building models** | Retrain monthly on station-hour counts with a time-based split. Baselines first, then gradient boosting with a Poisson or Tweedie objective. Tracked in MLflow. |
| **Features** | Station (id, location, capacity, usage cluster), hour × weekday with cyclical encoding, month and holidays, lagged and rolling demand, and trend. |
| **Monitoring** | Contract validation of every new month; drift in volume, hourly profile and station ids; forecast error per month and per station against the baseline. |

## 1. Value proposition

- **End user:** MiBici's operations team, who plan truck routes for rebalancing. Indirectly,
  riders benefit, because they find a bike when they want to leave and a free dock when they
  arrive.
- **Objective:** fewer station-hours with no bikes or no free docks, and more efficient use of
  rebalancing trucks.
- **What changes for the user:** instead of planning from experience or fixed routines
  (an assumption: the open data includes no rebalancing logs), the team gets a ranked list of
  stations expected to empty or fill, with the expected time and number of bikes.
- **Why ML and not simple rules:** a per-station historical average captures the average
  week. It misses the trend (growth until 2019, the 2020-2021 COVID break, a new record in
  September 2026), the hour × weekday × station interactions, and stations with little
  history. A single model trained on all stations can learn from stations with similar usage
  and adapt to the trend. The simple rules remain as the **baselines the model must beat**.

## 2. Prediction task

- **Unit:** a station and an hour.
- **Outputs:** expected departures and expected arrivals in each of the next 24 hours. Net
  flow (arrivals minus departures) and its running sum give the expected change in bikes at
  each station.
- **Type:** forecasting many related series of counts (regression on non-negative counts).
  The data is sparse. Since 2025, even counting only hours when the system had trips, 55.7%
  of station-hours have no departures, and active station-hours average 2.85 departures. That calls for a count-aware loss (Poisson or Tweedie)
  rather than plain squared error.
- **Horizon:** 1 to 24 hours ahead, made the night before. An intraday refresh can shorten it.

## 3. Decisions

1. Every night, forecast each station's hourly departures and arrivals for the next day.
2. Combine them with the current number of bikes at each station. The operator knows this in
   real time; the open data does not include it. The result is the projected bikes per station
   for each hour.
3. Flag stations projected to fall below a minimum number of bikes, or to exceed capacity
   (`dpcapacity`) minus a safety margin, before each peak.
4. Rank the flagged stations by expected unmet trips. Plan the truck route to restock the
   emptying stations from the filling ones, for example by moving bikes from the López Cotilla
   area toward the Av. Vallarta cluster before the morning peak.

Parameters the operator sets: the safety margins, the number of trucks and their capacity, and
the time windows (overnight and midday).

## 4. Impact simulation (offline evaluation)

- **Backtesting:** a rolling origin over the last 12 months of data. Train up to month *m*,
  forecast month *m + 1*, then move forward one month. Random splits are not allowed, because
  they would leak future demand (`1.03` findings).
- **Baselines:**
  - seasonal naive: same station, same hour, one week earlier
  - historical average per station × hour × weekday
- **Accuracy metrics** for departures and arrivals per station-hour: MAE, WAPE (weighted
  absolute percentage error) and Poisson deviance. Results are reported overall, at peak hours
  and for the top-50 stations.
- **Decision metric:** precision and recall in flagging *high-imbalance station-days*, defined
  as days where a station's absolute net flow exceeds half of its capacity. This approximates
  the decision in block 3 without inventory data.
- **Success criterion (proposed):** at least 10% lower WAPE than the best baseline at peak
  hours, with no station group (for example, low-demand peripheral stations) worse than the
  baseline.

`team2_mlops2026/modeling/evaluate.py` implements this protocol. It makes these choices:

- **Peak hours:** Monday to Friday at 7-9 h and 17-19 h, around the 8:00 and 18:00 commuter
  peaks (`1.03`).
- **Station groups:** stations are ranked by their mean hourly demand **in the training split**,
  so a test month never decides its own segments. The top 50 form one segment, and five
  demand quintiles form the groups for the fairness check. Ranking by the hourly mean rather
  than by total trips keeps a busy station that opened recently among the busy ones.
- **Unseen stations:** stations without training history are reported as their own segment.
  About 100 stations opened between May and August 2026, so this segment also signals drift.
- **Undefined values:** WAPE is undefined (NaN) for a segment with no demand, and that segment
  is skipped in the group comparison. Station-days without a known capacity are left out of the
  decision metric.
- **Best baseline:** the baseline with the lowest peak-hour WAPE. The model is compared with
  that baseline for both conditions of the criterion.
- **Backtest folds:** 12 monthly folds from 2025-10 to 2026-09. Each fold trains on every hour
  before its month and tests on that month.

## 5. Making predictions

- **When:** a batch job every night at about 23:00 for the next day, with an optional refresh
  at about 13:00 before the evening peak.
- **Volume:** about 480 active stations × 24 hours ≈ 11.5k forecasts per run, which takes
  seconds once the features are built.
- **Delivery:** a table or dashboard that the planning team reads, with each station's hourly
  forecasts and the ranked list of stations to act on.
- **Requirement for a real deployment:** live bike counts per station from the operator. The
  open data publishes monthly with a delay, so in this project predictions are simulated on
  historical months.

## 6. Data collection

- MiBici publishes a new trips CSV every month. `raw_data_extraction` downloads only what is
  missing and keeps a `manifest.json` with the URL, SHA-256 and size of each file.
- `raw_data_normalization` unifies encoding, column names and date formats. `interim_data_cleaning`
  applies the data contract (`1.02`) and validates every month before writing it. A month that
  breaks the contract stops the pipeline.
- MLflow records each dataset version with a digest of its files, so every model can be traced
  back to the exact data it was trained on.
- **Labels come for free:** the trips of the following month are the observed departures and
  arrivals that the forecasts are scored against.

## 7. Data sources

| Source | Content | Status |
|---|---|---|
| MiBici open data: trips | 142 monthly CSVs, 39,078,984 raw trips, 37,786,886 after cleaning (Dec 2014 to Sep 2026): trip id, user id, gender, birth year, start and end time, origin and destination station | In use |
| MiBici open data: nomenclature | Station id, name, latitude, longitude and capacity. Only the **current** list is published (484 stations, 8,171 docks) | In use |
| Weather (for example, the Open-Meteo historical API) | Hourly rain and temperature for Guadalajara | Candidate |
| Calendar | Mexican public holidays, school calendar, large events | Candidate |

**Known gaps:**
- There is no history of station locations or capacity.
- Station ids 124, 405 and 407 appear in trips but not in the nomenclature (4,036 trips,
  `1.02` T8).
- Four retired stations (`ex-ZPN-...`) are listed with capacity 0.
- There are no snapshots of bikes per station and no rebalancing logs. Inventory therefore has
  to come from the operator, and our target is demand rather than stock.

## 8. Building models

- **Training data:** trips aggregated to station × hour counts of departures and arrivals,
  including the hours with zero trips. 2020-2021 (the COVID break) is either left out or
  flagged explicitly, as `1.03` recommends.
- **Models, in order:**
  1. the baselines above
  2. gradient boosting (LightGBM or XGBoost) with a Poisson or Tweedie objective, as one model
     trained on all stations
  3. optionally, comparison against a dedicated time-series model for the top stations
- **Retraining:** monthly, when MiBici publishes a new month and it passes the contract. The
  model is also retrained when monitoring detects drift.
- **Tracking:** each training run in MLflow with its parameters, metrics, input dataset digests
  and model artifact. This is the same mechanism that already tracks the cleaning step.
- **Cost:** about 4.3M trips per year become about 4M station-hours. Building features and
  training takes minutes on a laptop.

## 9. Features

All features are computed per station-hour, using only information available at prediction
time.

- **Station:**
  - id and location (latitude, longitude) and capacity
  - a usage-profile cluster from PCA of each station's hour-of-week demand (issue #9)
- **Time:**
  - hour, weekday and their interaction; `1.03` shows the morning peak exists only on
    weekdays
  - sin/cos cyclical encoding of the hour and weekday
  - month, and holiday and school-calendar flags
- **History:**
  - demand in the same hour 1 day and 7 days before
  - rolling means over the last 7 and 28 days
  - the station's average net flow
- **Context (candidates):** rain and temperature.
- **Not used:** rider-level gender and birth year. `1.03` shows they predict trip
  characteristics only weakly, they are personal data, and a station-hour forecast does not
  need them.

## 10. Monitoring

- **Data:** `validate` gates every new month. The following are tracked for drift, as listed
  in the `1.03` findings:
  - monthly volume (September 2026 is a record)
  - the hourly and weekday profile
  - new or unknown station ids
  - the missing-gender rate
- **Model:** forecast error (WAPE and MAE) per month and per station, compared with the
  seasonal-naive baseline. An alert fires when the model stops beating the baseline or when its
  error rises more than an agreed threshold over its backtest value.
- **Business (with operator data):** station-hours with no bikes or no free docks, and
  kilometers driven by the rebalancing trucks, compared before and after adoption.

## Risks and assumptions

- **Assumption:** the operator can provide live bike counts per station. Without them, the
  forecast still ranks stations by expected imbalance, but cannot project when a station will
  be empty or full.
- **Privacy:** the trips contain a user id, gender and birth year. The forecasting model uses
  only aggregated station-hour counts.
- **Fairness:** optimizing total trips could neglect low-demand peripheral stations. The
  success criterion therefore also requires that no station group performs worse than the
  baseline.
- **Network changes:** new or relocated stations have little history. A single model trained
  on all stations, with location and usage-cluster features, mitigates this.
