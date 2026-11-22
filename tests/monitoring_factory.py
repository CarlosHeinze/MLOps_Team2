"""Small, deterministic station-hour fixture for the monitoring tests and notebook 3.03."""

import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from team2_mlops2026 import features as ft
from team2_mlops2026.modeling.export import MODEL_CARD, model_dir
from team2_mlops2026.modeling.models import CANDIDATES, TRUSTED_TYPES
from team2_mlops2026.modeling.train import load_features, model_inputs
from team2_mlops2026.station_profiles import FIT_END_KEY, PROFILES_FILE, WEEK, week_start

TRAIN_START, REFERENCE_MONTH, SIM_MONTH, REPLAY_MONTH = (
    "2026-01",
    "2026-03",
    "2026-04",
    "2026-05",
)


def write_fixture(root: Path) -> tuple[Path, Path]:
    """Generate features, synthetic profile scores and two small exported LightGBM models."""
    src, models = root / "features", root / "models"
    src.mkdir(parents=True, exist_ok=True)
    models.mkdir(exist_ok=True)
    hours = pd.date_range("2025-12-01", "2026-06-01", freq="h", inclusive="left")
    replay = hours >= pd.Timestamp(REPLAY_MONTH)
    calendar = ft.calendar_features(hours)
    frames = []
    for station in range(1, 9):
        active = replay if station == 8 else np.ones(len(hours), dtype=bool)
        parity = (hours.day_of_year % 2).to_numpy()
        departures = 3 + station + 10 * parity + 5 * (hours.hour == 8)
        arrivals = 2 + station + 9 * parity + 6 * (hours.hour == 18) + 2 * (station <= 4)
        frame = pd.DataFrame(
            {
                ft.STATION: station,
                ft.TIME: hours,
                **calendar,
                "latitude": 20.6 + station / 100,
                "longitude": -103.3 + station / 100,
                "capacity": 18 + station,
            }
        )
        for target, values in zip(ft.TARGETS, (departures, arrivals)):
            counts = np.where(replay, np.rint(values * 1.3), values).astype("int32")
            counts[~active] = 0
            frame[target] = counts
            for name, values in ft.lag_features(counts[None, :], active[None, :]).items():
                frame[f"{target}_{name}"] = values[0]
        frames.append(frame.loc[active & (hours >= pd.Timestamp(TRAIN_START))])
    features = pd.concat(frames, ignore_index=True).loc[:, list(ft.FEATURE_COLUMNS)]
    features.to_parquet(src / ft.FEATURES_FILE, index=False)
    scores = (
        features[[ft.STATION]]
        .assign(**{WEEK: week_start(features[ft.TIME])})
        .drop_duplicates(ignore_index=True)
        .assign(pc1=lambda rows: (rows[ft.STATION] - 4) / 4)
    )
    scores.attrs[FIT_END_KEY] = REFERENCE_MONTH
    scores.to_parquet(src / PROFILES_FILE, index=False)
    train = load_features(src, test_start=REFERENCE_MONTH, end=REFERENCE_MONTH)
    inputs = model_inputs(train)
    card = {"models": {}}
    for target in ft.TARGETS:
        model = (
            CANDIDATES["lgbm_tweedie"]
            .build(target, n_estimators=60, n_jobs=1)
            .fit(inputs, train[target])
        )
        mlflow.sklearn.save_model(
            model,
            str(model_dir(target, models)),
            pip_requirements=[],
            serialization_format="skops",
            skops_trusted_types=TRUSTED_TYPES,
        )
        card["models"][target] = {
            "uri": f"synthetic:/lgbm_tweedie/{target}",
            "tags": {"training_window": f"{TRAIN_START} to {REFERENCE_MONTH} (exclusive)"},
        }
    (models / MODEL_CARD).write_text(json.dumps(card), encoding="utf-8")
    return src, models
