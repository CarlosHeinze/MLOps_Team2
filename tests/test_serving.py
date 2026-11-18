"""The prediction service with ``TestClient``: two small synthetic models saved as the folder
that ``modeling.export`` writes, and the forecast rows of the next day."""

import dataclasses
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from trip_factory import STATIONS, make_trips

from team2_mlops2026 import features as ft
from team2_mlops2026 import station_profiles as sp
from team2_mlops2026.modeling import export as ex
from team2_mlops2026.modeling import predict as pr
from team2_mlops2026.modeling import registry as reg
from team2_mlops2026.modeling import train as tr
from team2_mlops2026.modeling.models import MODELS, TRUSTED_TYPES
from team2_mlops2026.serving import app as api
from team2_mlops2026.serving.schemas import (
    MAX_ROWS,
    PROFILE_COLUMNS,
    ROW_COLUMNS,
    ROW_DTYPES,
    FeatureRow,
    PredictRequest,
    PredictResponse,
    to_frame,
    to_payload,
)

DAY0 = pd.Timestamp("2024-01-01")  # a Monday
# The factory's stations moved into Guadalajara, inside the bounds the service accepts
GDL_STATIONS = STATIONS.assign(latitude=[20.67, 20.679], longitude=-103.35)
VERSION = "3"


def card_entry(target: str) -> dict:
    name = reg.model_name(target)
    return {
        "name": name,
        "version": VERSION,
        "alias": reg.CHAMPION,
        "uri": f"models:/{name}/{VERSION}",
        "run_id": f"run-{target}",
        "tags": {"model": "lgbm_tweedie", "target": target},
    }


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Six weeks of a morning trip 1 -> 2 and an evening trip 2 -> 1, with weekly profile
    scores; an lgbm_tweedie per target fitted on them and exported with a model card."""
    trips = make_trips(
        *(
            {"start": DAY0 + pd.Timedelta(days=d, hours=h), "Origen_Id": o, "Destino_Id": 3 - o}
            for d in range(42)
            for h, o in ((8, 1), (18, 2))
        )
    )
    table = ft.build_features(trips, GDL_STATIONS, DAY0)
    weeks = pd.date_range(DAY0, periods=6, freq="W-MON")
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        {
            ft.STATION: np.tile([1, 2], len(weeks)),
            sp.WEEK: weeks.repeat(2),
            **{pc: rng.normal(size=2 * len(weeks)).astype("float32") for pc in PROFILE_COLUMNS},
        }
    )
    train = pr.join_profiles(table, scores)
    models = {
        target: MODELS["lgbm_tweedie"]
        .build(target, n_estimators=10)
        .fit(tr.model_inputs(train), train[target])
        for target in ft.TARGETS
    }
    root = tmp_path_factory.mktemp("models")
    for target, model in models.items():
        mlflow.sklearn.save_model(
            model,
            ex.model_dir(target, root),
            serialization_format="skops",
            skops_trusted_types=TRUSTED_TYPES,
            pip_requirements=tr.model_requirements(),  # no inference: ~6 s per model
            code_paths=[str(tr.PACKAGE_DIR)],
        )
    card = {
        "exported_at": "2026-11-18T00:00:00+00:00",
        "git_commit": "0" * 40,
        "models": {target: card_entry(target) for target in ft.TARGETS},
    }
    (root / ex.MODEL_CARD).write_text(json.dumps(card))
    rows = pr.join_profiles(ft.forecast_rows(table), scores)
    return SimpleNamespace(models=models, root=root, train=train, rows=rows)


@pytest.fixture(scope="module")
def client(synthetic):
    with TestClient(api.create_app(api.LOCAL, synthetic.root)) as client:
        yield client


def one_row(synthetic, **change) -> dict:
    return to_payload(synthetic.rows.head(1))["rows"][0] | change


# --- contract ------------------------------------------------------------------


def test_the_request_rows_carry_the_columns_the_models_were_fitted_on(synthetic):
    assert list(FeatureRow.model_fields) == list(ROW_COLUMNS)
    assert list(ROW_COLUMNS) == list(tr.model_inputs(synthetic.train).columns)
    assert set(ROW_DTYPES) == set(ROW_COLUMNS)
    assert not set(ft.TARGETS) & set(ROW_COLUMNS)


def test_a_payload_restores_the_rows_with_the_feature_table_dtypes(synthetic):
    rows = synthetic.rows
    assert rows[list(PROFILE_COLUMNS)].notna().all().all()  # every station-week scored
    frame = to_frame(PredictRequest(**to_payload(rows)).rows)
    expected = rows[list(ROW_COLUMNS)].astype(ROW_DTYPES).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame, expected, check_exact=True)


# --- POST /predict -----------------------------------------------------------------


def test_predict_returns_a_forecast_per_row_and_the_model_versions(client, synthetic):
    response = client.post("/predict", json=to_payload(synthetic.rows))
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == len(synthetic.rows) == 2 * ft.FORECAST_HOURS
    assert list(body["predictions"][0]) == [ft.STATION, ft.TIME, *ft.TARGETS, pr.NET_FLOW]
    expected = {k: card_entry("departures")[k] for k in ("name", "version", "alias", "uri")}
    assert body["models"]["departures"] == expected
    assert body["models"]["arrivals"]["uri"] == "models:/mibici-arrivals/3"


def test_the_api_forecasts_equal_the_batch_forecast(client, synthetic):
    """The models loaded from the exported folder, behind JSON, give exactly what
    ``predict.forecast`` gives with the fitted estimators in memory."""
    expected = pr.forecast(synthetic.models, synthetic.rows)
    response = client.post("/predict", json=to_payload(synthetic.rows))
    served = pd.DataFrame(response.json()["predictions"])
    served[ft.TIME] = pd.to_datetime(served[ft.TIME]).astype(expected[ft.TIME].dtype)
    pd.testing.assert_frame_equal(served, expected, check_exact=True)


def test_rows_without_profile_scores_still_predict(client, synthetic):
    rows = synthetic.rows.drop(columns=list(PROFILE_COLUMNS))
    response = client.post("/predict", json=to_payload(rows))
    assert response.status_code == 200
    unscored = synthetic.rows.assign(**dict.fromkeys(PROFILE_COLUMNS, np.nan))
    expected = pr.forecast(synthetic.models, unscored)
    served = pd.DataFrame(response.json()["predictions"])
    for target in ft.TARGETS:
        np.testing.assert_array_equal(served[target], expected[target])


@pytest.mark.parametrize(
    ("change", "field", "message"),
    [
        ({"weekday": 9}, "weekday", "less than or equal to 6"),
        ({"departures_lag_2d": -1}, "departures_lag_2d", "greater than or equal to 0"),
        ({"hour_sin": 1.5}, "hour_sin", "less than or equal to 1"),
        ({"latitude": 19.43}, "latitude", "greater than or equal to 20.5"),  # Mexico City
        ({"capacity": 0}, "capacity", "greater than 0"),
        ({"hour_start": "2024-02-12T08:00:00Z"}, "hour_start", "without a time zone"),
        ({"hour_start": "2024-02-12T08:30:00"}, "hour_start", "start of an hour"),
        ({"departures": 4.0}, "departures", "Extra inputs are not permitted"),
    ],
)
def test_an_invalid_row_returns_422_with_the_field_and_the_reason(
    client, synthetic, change, field, message
):
    response = client.post("/predict", json={"rows": [one_row(synthetic, **change)]})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "rows", 0, field]
    assert message in error["msg"]


def test_a_missing_field_or_a_weekday_of_another_day_returns_422(client, synthetic):
    row = one_row(synthetic)
    del row["capacity"]
    [error] = client.post("/predict", json={"rows": [row]}).json()["detail"]
    assert error["loc"][-1] == "capacity" and error["msg"] == "Field required"

    row = one_row(synthetic)
    other_day = (row["weekday"] + 1) % 7
    response = client.post("/predict", json={"rows": [row | {"weekday": other_day}]})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "rows", 0] and "does not match hour_start" in error["msg"]


@pytest.mark.parametrize("n_rows", [0, MAX_ROWS + 1])
def test_an_empty_or_oversized_batch_returns_400(client, synthetic, n_rows):
    response = client.post("/predict", json={"rows": [one_row(synthetic)] * n_rows})
    assert response.status_code == 400
    assert response.json() == {"detail": f"Send between 1 and {MAX_ROWS:,} rows"}


class Broken:
    def predict(self, X):
        raise RuntimeError("internal detail that must not leak")


def test_a_failing_model_returns_500_without_the_error(synthetic):
    with TestClient(api.create_app(api.LOCAL, synthetic.root)) as client:
        loaded = client.app.state.loaded
        broken = dataclasses.replace(loaded, models={**loaded.models, "arrivals": Broken()})
        client.app.state.loaded = broken
        response = client.post("/predict", json={"rows": [one_row(synthetic)]})
    assert response.status_code == 500
    assert response.json() == {"detail": "The forecast failed; see the service log"}


# --- GET /health and /model, model sources ---------------------------------------


def test_health_and_model_report_the_loaded_models(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "models_loaded": True, "source": api.LOCAL}
    card = client.get("/model").json()
    assert card["source"] == api.LOCAL and card["git_commit"] == "0" * 40
    assert card["models"]["arrivals"] == card_entry("arrivals")


def test_without_exported_models_the_service_starts_and_answers_503(synthetic, tmp_path):
    with TestClient(api.create_app(api.LOCAL, tmp_path)) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json() == {
            "status": "unavailable",
            "models_loaded": False,
            "source": "local",
        }
        assert client.get("/model").status_code == 503
        response = client.post("/predict", json={"rows": [one_row(synthetic)]})
    assert response.status_code == 503 and "make export" in response.json()["detail"]


def test_the_registry_source_serves_the_versions_that_hold_the_alias(synthetic, monkeypatch):
    def load_model(target, alias):
        entry = card_entry(target)
        version = SimpleNamespace(**{k: entry[k] for k in ("name", "version", "run_id", "tags")})
        return synthetic.models[target], version

    monkeypatch.setattr(reg, "load_model", load_model)
    with TestClient(api.create_app(api.REGISTRY)) as client:
        assert client.get("/health").json()["source"] == api.REGISTRY
        assert client.get("/model").json()["models"]["departures"] == card_entry("departures")
        response = client.post("/predict", json=to_payload(synthetic.rows))
    expected = pr.forecast(synthetic.models, synthetic.rows)
    np.testing.assert_array_equal(
        pd.DataFrame(response.json()["predictions"])["arrivals"], expected["arrivals"]
    )
    with pytest.raises(ValueError, match="MODEL_SOURCE"):
        api.load_models("s3")


# --- docs/api ---------------------------------------------------------------------


def test_the_committed_api_docs_match_the_service():
    """Regenerate them with `make api-docs` after changing a schema or the app."""
    committed = json.loads((api.API_DOCS_DIR / api.OPENAPI_FILE).read_text(encoding="utf-8"))
    assert committed == api.app.openapi()
    request = (api.API_DOCS_DIR / api.EXAMPLE_REQUEST).read_text(encoding="utf-8")
    response = (api.API_DOCS_DIR / api.EXAMPLE_RESPONSE).read_text(encoding="utf-8")
    rows = PredictRequest.model_validate_json(request).rows
    predictions = PredictResponse.model_validate_json(response).predictions
    assert [(r.station_id, r.hour_start) for r in rows] == [
        (p.station_id, p.hour_start) for p in predictions
    ]


# --- check --------------------------------------------------------------------------


def test_check_compares_two_responses_within_the_tolerance():
    expected = json.loads((api.API_DOCS_DIR / api.EXAMPLE_RESPONSE).read_text(encoding="utf-8"))
    assert api.compare_predictions(expected, expected) == 0

    def changed(**change) -> dict:
        served = json.loads(json.dumps(expected))
        served["predictions"][0] |= change
        return served

    departures = expected["predictions"][0]["departures"]
    close = changed(departures=departures * (1 + 1e-12))  # last digits, another CPU
    assert 0 < api.compare_predictions(expected, close) < 1e-9
    with pytest.raises(ValueError, match="differ by up to"):
        api.compare_predictions(expected, changed(departures=departures + 0.01))
    with pytest.raises(ValueError, match="other station-hours"):
        api.compare_predictions(expected, changed(station_id=-1))
    other_models = json.loads(json.dumps(expected))
    other_models["models"]["arrivals"]["version"] = "2"
    with pytest.raises(ValueError, match="Served by"):
        api.compare_predictions(expected, other_models)
