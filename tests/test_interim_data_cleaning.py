import json
import os

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest
from trip_factory import BY, START, STATIONS, make_trips

from team2_mlops2026 import tracking
from team2_mlops2026.data import interim_data_cleaning as idc
from team2_mlops2026.data.trip_data import trip_files, trip_metrics


@pytest.fixture
def interim_dir(tmp_path):
    """Interim layout with two months and a station list."""
    src = tmp_path / "interim"
    (src / "estaciones").mkdir(parents=True)
    STATIONS.to_parquet(src / "estaciones" / "nomenclatura_2024_03.parquet", index=False)
    feb = make_trips({"Genero": None, BY: None, "start": pd.Timestamp("2024-02-01 08:00")})
    mar = make_trips({}, {"Viaje_Id": 2, "seconds": 5}, {"Viaje_Id": 3, BY: 1880})
    mar = pd.concat([mar, mar.iloc[[0]]], ignore_index=True)  # exact duplicate
    for year_month, df in (("2024_02", feb), ("2024_03", mar)):
        dest = src / "viajes" / "2024" / f"viajes_{year_month}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
    return src


# --- rules -------------------------------------------------------------------


@pytest.mark.parametrize(
    "row, rule",
    [
        ({"seconds": idc.MIN_DURATION_S - 1, "Destino_Id": 1}, "too_short"),
        ({"seconds": idc.MAX_DURATION_S + 1}, "too_long"),
        ({"seconds": 60}, "too_fast"),  # 1 km in 1 min = 60 km/h
    ],
)
def test_trip_violations_flag_each_rule(row, rule):
    broken = idc.trip_violations(trip_metrics(make_trips(row), STATIONS))
    assert broken.loc[0, rule] and broken.loc[0].sum() == 1


def test_trip_violations_accept_bounds_and_unknown_stations():
    trips = make_trips(
        {"seconds": idc.MIN_DURATION_S, "Destino_Id": 1},  # round trip at the minimum
        {"seconds": idc.MAX_DURATION_S},
        {"seconds": 61, "Destino_Id": 99},  # speed unknown -> cannot be proven wrong
    )
    assert not idc.trip_violations(trip_metrics(trips, STATIONS)).to_numpy().any()


@pytest.mark.parametrize(
    "row, rule",
    [
        ({"Genero": "X"}, "invalid_gender"),
        ({BY: START.year - idc.MIN_AGE + 1}, "invalid_age"),
        ({BY: START.year - idc.MAX_AGE - 1}, "invalid_age"),
        ({BY: 1}, "invalid_age"),
    ],
)
def test_field_violations_flag_each_rule(row, rule):
    broken = idc.field_violations(make_trips(row))
    assert broken.loc[0, rule] and broken.loc[0].sum() == 1


def test_field_violations_ignore_missing_values_and_bounds():
    trips = make_trips(
        {"Genero": None, BY: None},
        {BY: START.year - idc.MIN_AGE},
        {BY: START.year - idc.MAX_AGE},
    )
    assert not idc.field_violations(trips).to_numpy().any()


def test_rule_violations_has_every_rule():
    trips = make_trips({}, {})
    trips.iloc[1] = trips.iloc[0]
    broken = idc.rule_violations(trips, trip_metrics(trips, STATIONS))
    assert list(broken.columns) == ["duplicate", *idc.TRIP_RULES, *idc.FIELD_RULES]
    assert broken["duplicate"].tolist() == [False, True]


def test_overlapping_trips():
    trips = make_trips(
        {"start": START, "seconds": 600},
        {"start": START + pd.Timedelta(minutes=5)},  # starts before the first one ended
        {"start": START + pd.Timedelta(minutes=5), "Usuario_Id": 8},  # other user
    )
    assert idc.overlapping_trips(trips).tolist() == [False, True, False]


# --- steps -------------------------------------------------------------------


def test_drop_duplicates_logs_removed_rows():
    trips = make_trips({}, {})
    trips.iloc[1] = trips.iloc[0]
    log = idc.CleaningLog()
    out = idc.drop_duplicates(trips, log)
    assert len(out) == 1
    assert log.steps[0]["rows_removed"] == 1


def test_drop_invalid_trips_keeps_valid_rows_and_index():
    trips = make_trips({}, {"seconds": 5}, {})
    out = idc.drop_invalid_trips(trips, STATIONS)
    assert out.index.tolist() == [0, 2]


def test_invalidate_demographics_keeps_the_trip():
    trips = make_trips({"Genero": "X"}, {BY: 2020})
    log = idc.CleaningLog()
    out = idc.invalidate_demographics(trips, log)
    assert len(out) == 2
    assert pd.isna(out.loc[0, "Genero"]) and out.loc[0, BY] == 1990
    assert pd.isna(out.loc[1, BY]) and out.loc[1, "Genero"] == "M"
    assert log.steps[0]["cells_changed"] == 2
    assert trips.loc[0, "Genero"] == "X"  # input untouched


def test_build_user_profiles_uses_valid_mode_with_deterministic_ties():
    trips = make_trips(
        {"Usuario_Id": 1, "Genero": "F", BY: 1980},
        {"Usuario_Id": 1, "Genero": "F", BY: 1980},
        {"Usuario_Id": 1, "Genero": "M", BY: 1},  # invalid age does not vote
        {"Usuario_Id": 2, "Genero": "M", BY: 1991},
        {"Usuario_Id": 2, "Genero": "F", BY: 1990},
        {"Usuario_Id": 3, "Genero": None, BY: None},
    )
    profiles = idc.build_user_profiles(trips)
    assert profiles.loc[1].tolist() == ["F", 1980]
    assert profiles.loc[2].tolist() == ["F", 1990]  # tie -> smallest value
    assert 3 not in profiles.index


def test_impute_from_profiles_fills_only_missing_values():
    trips = make_trips(
        {"Usuario_Id": 1, "Genero": None, BY: None},
        {"Usuario_Id": 1, "Genero": "M", BY: 1985},
        {"Usuario_Id": 2, "Genero": None, BY: None},  # no profile -> stays missing
    )
    profiles = pd.DataFrame(
        {"Genero": ["F"], BY: pd.array([1980], dtype="Int64")},
        index=pd.Index([1], name="Usuario_Id"),
    )
    log = idc.CleaningLog()
    out = idc.impute_from_profiles(trips, profiles, log)
    assert out.loc[0, ["Genero", BY]].tolist() == ["F", 1980]
    assert out.loc[1, ["Genero", BY]].tolist() == ["M", 1985]
    assert out.loc[2, ["Genero", BY]].isna().all()
    assert log.steps[0]["cells_changed"] == 2


def test_impute_from_profiles_does_not_create_invalid_ages():
    old_trip = START - pd.DateOffset(years=5)
    trips = make_trips({"Genero": None, BY: None, "start": old_trip})
    born = START.year - idc.MIN_AGE  # valid now, too young five years earlier
    profiles = pd.DataFrame(
        {"Genero": ["F"], BY: pd.array([born], dtype="Int64")},
        index=pd.Index([7], name="Usuario_Id"),
    )
    out = idc.impute_from_profiles(trips, profiles)
    assert out.loc[0, "Genero"] == "F" and pd.isna(out.loc[0, BY])


def test_alternative_strategies():
    trips = make_trips({}, {"Genero": None}, {BY: None}, {BY: 2000})
    assert idc.drop_incomplete_demographics(trips).index.tolist() == [0, 3]
    imputed = idc.impute_global(trips)
    assert imputed.loc[1, "Genero"] == "M" and imputed.loc[2, BY] == 1990


def test_clean_trips_output_passes_validation():
    trips = make_trips({}, {"seconds": 5}, {"Genero": "X"}, {"Usuario_Id": 8, BY: None})
    trips = pd.concat([trips, trips.iloc[[0]]], ignore_index=True)
    profiles = idc.build_user_profiles(make_trips({"Usuario_Id": 8, BY: 1970}))
    log = idc.CleaningLog()
    out = idc.clean_trips(trips, STATIONS, profiles, log)
    idc.validate(out, STATIONS)
    assert out["Viaje_Id"].tolist() == [1, 3, 4]
    assert out.loc[3, BY] == 1970
    assert [s["step"] for s in log.steps] == [
        "drop_duplicates",
        "drop_invalid_trips",
        "invalidate_demographics",
        "impute_from_profiles",
    ]


@pytest.mark.parametrize(
    "trips, message",
    [
        (make_trips({}).drop(columns="Genero"), "Expected columns"),
        (make_trips({"Origen_Id": None}), "nulls in required columns"),
        (make_trips({}, {"Viaje_Id": 1}), "repeated Viaje_Id"),
        (make_trips({"seconds": 5}), "too_short"),
        (make_trips({BY: 1}), "invalid_age"),
    ],
    ids=["schema", "nulls", "ids", "trip-rule", "field-rule"],
)
def test_validate_rejects_contract_breaks(trips, message):
    with pytest.raises(idc.ContractError, match=message):
        idc.validate(trips, STATIONS)


# --- clean -------------------------------------------------------------------


def test_clean_writes_layout_and_report(interim_dir, tmp_path):
    dst = tmp_path / "processed"
    summary = idc.clean(interim_dir, dst, workers=1)

    assert summary["failed"] == [] and len(summary["cleaned"]) == 2
    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*.parquet")) == [
        "viajes/2024/viajes_2024_02.parquet",
        "viajes/2024/viajes_2024_03.parquet",
    ]
    mar = pd.read_parquet(dst / "viajes/2024/viajes_2024_03.parquet")
    assert mar["Viaje_Id"].tolist() == [1, 3]  # duplicate and 5 s trip dropped
    assert mar.loc[1, BY] == 1990  # invalid 1880 recovered from the user's other trips
    assert mar["Inicio_del_viaje"].is_monotonic_increasing

    feb = pd.read_parquet(dst / "viajes/2024/viajes_2024_02.parquet")
    assert feb.loc[0, ["Genero", BY]].tolist() == ["M", 1990]  # filled from March

    report = json.loads((dst / idc.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["viajes_2024_03.parquet"]["rows_in"] == 4
    assert report["viajes_2024_03.parquet"]["rows_out"] == 2


def test_clean_is_idempotent_and_reprocesses_on_new_inputs(interim_dir, tmp_path):
    dst = tmp_path / "processed"
    idc.clean(interim_dir, dst, workers=1)
    assert idc.clean(interim_dir, dst, workers=1)["skipped"] == 2

    # A newer interim month can change any user's profile -> everything is stale
    os.utime(interim_dir / "viajes/2024/viajes_2024_02.parquet", None)
    assert len(idc.clean(interim_dir, dst, workers=1)["cleaned"]) == 2

    assert len(idc.clean(interim_dir, dst, force=True, workers=1)["cleaned"]) == 2


def test_clean_records_failures_and_keeps_going(interim_dir, tmp_path):
    bad = make_trips({"Origen_Id": None})
    bad.to_parquet(interim_dir / "viajes/2024/viajes_2024_04.parquet", index=False)
    summary = idc.clean(interim_dir, tmp_path / "processed", workers=1)

    assert [f["name"] for f in summary["failed"]] == ["viajes_2024_04.parquet"]
    assert "nulls in required columns" in summary["failed"][0]["error"]
    assert len(summary["cleaned"]) == 2
    assert not list((tmp_path / "processed").rglob("*.part"))


def test_clean_in_parallel_processes(interim_dir, tmp_path):
    summary = idc.clean(interim_dir, tmp_path / "processed", workers=2)
    assert summary["failed"] == [] and len(summary["cleaned"]) == 2


def test_clean_empty_source(tmp_path):
    summary = idc.clean(tmp_path, tmp_path / "processed", workers=1)
    assert summary == {"found": 0, "skipped": 0, "cleaned": [], "failed": []}


# --- tracking ----------------------------------------------------------------


def test_report_totals_sum_every_processed_file(interim_dir, tmp_path):
    dst = tmp_path / "processed"
    idc.clean(interim_dir, dst, workers=1)
    assert idc.report_totals(dst) == {
        "files_processed": 2,
        "rows_interim": 5,
        "rows_processed": 3,
        "duplicates_removed": 1,
        "invalid_trips_removed": 1,
        "demographics_invalidated": 1,
        "demographics_imputed": 3,
        "missing_gender": 0,
        "missing_birth_year": 0,
    }


def test_clean_tracked_logs_datasets_contract_and_totals(interim_dir, tmp_path, mlflow_store):
    dst = tmp_path / "processed"
    summary = idc.clean_tracked(interim_dir, dst, workers=1, extra_metrics={"strategy_a_rows": 2})
    run = mlflow.get_run(summary["mlflow_run_id"])

    assert run.info.run_name == idc.RUN_NAME and run.info.status == "FINISHED"
    inputs = {
        next(t.value for t in i.tags if t.key == "mlflow.data.context"): i.dataset
        for i in run.inputs.dataset_inputs
    }
    assert {c: d.name for c, d in inputs.items()} == {
        "interim": "mibici_trips_interim",
        "stations": "mibici_stations",
        "processed": "mibici_trips_processed",
    }
    assert inputs["processed"].digest == tracking.files_digest(trip_files(dst), dst / "viajes")
    assert run.data.params["min_duration_s"] == str(idc.MIN_DURATION_S)
    assert run.data.params["missing_data_strategy"] == idc.MISSING_DATA_STRATEGY
    assert run.data.metrics["rows_processed"] == 3
    assert run.data.metrics["files_cleaned"] == 2
    assert run.data.metrics["strategy_a_rows"] == 2
    artifacts = [a.path for a in MlflowClient().list_artifacts(run.info.run_id)]
    assert artifacts == [idc.REPORT_NAME]


def test_clean_tracked_digest_changes_only_when_output_changes(
    interim_dir, tmp_path, mlflow_store
):
    dst = tmp_path / "processed"
    first = idc.clean_tracked(interim_dir, dst, workers=1)
    again = idc.clean_tracked(interim_dir, dst, workers=1)  # everything up to date

    def processed_digest(summary):
        inputs = mlflow.get_run(summary["mlflow_run_id"]).inputs.dataset_inputs
        return next(i.dataset.digest for i in inputs if i.dataset.name == "mibici_trips_processed")

    assert again["skipped"] == 2
    assert processed_digest(again) == processed_digest(first)

    extra = make_trips({"Viaje_Id": 9, "start": pd.Timestamp("2024-04-02 08:00")})
    extra.to_parquet(interim_dir / "viajes/2024/viajes_2024_04.parquet", index=False)
    assert processed_digest(idc.clean_tracked(interim_dir, dst, workers=1)) != processed_digest(
        first
    )
