import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest

from team2_mlops2026 import tracking
from team2_mlops2026.data import raw_data_normalization as rdn
from team2_mlops2026.data.raw_data_extraction import RAW_DATASET, raw_csv_files
from team2_mlops2026.data.trip_data import trip_files

HEADER = list(rdn.TRIP_COLUMNS)
# Same trips in both date formats (day-first files have no seconds)
ROWS_ISO = [
    ["1", "10", "M", "1990", "2023-11-01 00:01:00", "2023-11-01 00:20:00", "5", "6"],
    ["2", "11", "F", "", "2023-11-13 23:59:00", "2023-11-14 00:10:00", "7", "8"],
]
ROWS_DMY = [
    ["1", "10", "M", "1990", "01/11/2023 0:01", "01/11/2023 0:20", "5", "6"],
    ["2", "11", "F", "", "13/11/2023 23:59", "14/11/2023 0:10", "7", "8"],
]


def write_raw(
    path: Path,
    rows: list[list[str]],
    header: list[str] = HEADER,
    encoding: str = "utf-8",
    quoted: bool = False,
    newline: str = "\n",
) -> Path:
    """Writes a CSV reproducing one of the raw MiBici format variants."""

    def fmt(values):
        return ",".join(f'"{v}"' if quoted else v for v in values)

    text = newline.join(fmt(r) for r in [header, *rows]) + newline
    path.write_bytes(text.encode(encoding))
    return path


@pytest.fixture
def raw_dir(tmp_path):
    """Raw folder with the five trip format variants found in MiBici, plus stations."""
    raw = tmp_path / "raw"
    raw.mkdir()
    write_raw(raw / "datos_abiertos_2014_12.csv", ROWS_ISO)
    write_raw(raw / "datos_abiertos_2021_06.csv", ROWS_ISO, encoding="latin-1", quoted=True)
    write_raw(raw / "datos_abiertos_2022_12.csv", ROWS_ISO, quoted=True)
    write_raw(raw / "datos_abiertos_2023_11.csv", ROWS_DMY, encoding="latin-1", newline="\r\n")
    corrupted = HEADER.copy()
    corrupted[3] = "A}äe_nacimiento"
    write_raw(
        raw / "datos_abiertos_2024_02-1.csv",
        ROWS_ISO,
        header=corrupted,
        encoding="latin-1",
        quoted=True,
    )
    write_raw(
        raw / "nomenclatura_2026_09.csv",
        [["2", "(GDL-002) Av. Niños héroes", "20.66", "-103.36", "15"]],
        header=list(rdn.STATION_COLUMNS),
        encoding="latin-1",
        quoted=True,
    )
    (raw / "manifest.json").write_text("{}")
    return raw


def snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir()}


# --- classify ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("datos_abiertos_2016_01.csv", ("viajes", "2016", "01")),
        ("datos_abiertos_2016_01-1.csv", ("viajes", "2016", "01")),
        ("nomenclatura_2026_09.csv", ("estaciones", "2026", "09")),
        ("DATOS_ABIERTOS_2020_12.CSV", ("viajes", "2020", "12")),
    ],
)
def test_classify_parses_known_names(name, expected):
    f = rdn.classify(Path(name))
    assert (f.kind, f.year, f.month) == expected


@pytest.mark.parametrize(
    "name",
    ["manifest.json", "otro.csv", "datos_abiertos_2016.csv", "x_datos_abiertos_2016_01.csv"],
)
def test_classify_returns_none_for_unknown_names(name):
    assert rdn.classify(Path(name)) is None


def test_dest_rel_keeps_source_naming():
    assert (
        rdn.classify(Path("datos_abiertos_2016_01-1.csv")).dest_rel
        == "viajes/2016/viajes_2016_01.parquet"
    )
    assert (
        rdn.classify(Path("nomenclatura_2026_09.csv")).dest_rel
        == "estaciones/nomenclatura_2026_09.parquet"
    )


# --- detect_encoding ---------------------------------------------------------


@pytest.mark.parametrize(
    "content, expected",
    [
        ("Año".encode(), "utf-8"),
        ("Año".encode("latin-1"), "latin-1"),
        (b"plain ascii", "utf-8"),
        (b"", "utf-8"),
    ],
    ids=["utf-8", "latin-1", "ascii", "empty"],
)
def test_detect_encoding(tmp_path, content, expected):
    path = tmp_path / "f.csv"
    path.write_bytes(content)
    assert rdn.detect_encoding(path) == expected


def test_detect_encoding_handles_char_split_across_chunks(tmp_path):
    path = tmp_path / "f.csv"
    path.write_bytes("aÑb".encode())  # "Ñ" is 2 bytes: positions 1-2
    assert rdn.detect_encoding(path, chunk_size=2) == "utf-8"


def test_detect_encoding_finds_latin1_beyond_first_chunk(tmp_path):
    path = tmp_path / "f.csv"
    path.write_bytes(b"a" * 100 + "ñ".encode("latin-1"))
    assert rdn.detect_encoding(path, chunk_size=16) == "latin-1"


# --- parse_datetimes ---------------------------------------------------------


def test_parse_datetimes_iso():
    parsed, fmt = rdn.parse_datetimes(pd.Series(["2023-11-01 00:01:05"]))
    assert fmt == "%Y-%m-%d %H:%M:%S"
    assert parsed[0] == pd.Timestamp("2023-11-01 00:01:05")


@pytest.mark.parametrize(
    "value, fmt",
    [("01/11/2023 0:01", "%d/%m/%Y %H:%M"), ("01/11/2023 0:01:30", "%d/%m/%Y %H:%M:%S")],
)
def test_parse_datetimes_day_first(value, fmt):
    parsed, got = rdn.parse_datetimes(pd.Series([value, "13/11/2023 23:59" + value[15:]]))
    assert got == fmt
    assert parsed[0].month == 11 and parsed[0].day == 1  # not January 11
    assert parsed[1] == pd.Timestamp("2023-11-13 23:59" + value[15:])


def test_parse_datetimes_keeps_nulls():
    parsed, _ = rdn.parse_datetimes(pd.Series(["2023-11-01 00:01:05", None]))
    assert pd.isna(parsed[1])


@pytest.mark.parametrize(
    "values",
    [["2023-11-01 00:01:05", "13/11/2023 0:01"], ["not a date"]],
    ids=["mixed-formats", "unknown-format"],
)
def test_parse_datetimes_rejects_unparseable(values):
    with pytest.raises(ValueError, match="datetime format"):
        rdn.parse_datetimes(pd.Series(values))


# --- normalize_columns -------------------------------------------------------


def test_normalize_columns_accepts_exact_and_padded_names():
    assert rdn.normalize_columns([" a", "b "], ["a", "b"]) == ["a", "b"]


def test_normalize_columns_maps_untrusted_name_by_position():
    got = ["a", "A}äe_nacimiento", "c"]
    expected = ["a", "Año_de_nacimiento", "c"]
    assert rdn.normalize_columns(got, expected, frozenset({"Año_de_nacimiento"})) == expected


def test_normalize_columns_rejects_wrong_count():
    with pytest.raises(rdn.SchemaError, match="Expected 2 columns"):
        rdn.normalize_columns(["a"], ["a", "b"])


def test_normalize_columns_rejects_reordered_trusted_names():
    with pytest.raises(rdn.SchemaError, match="Unexpected columns"):
        rdn.normalize_columns(["b", "a"], ["a", "b"])


# --- normalize_trips / normalize_stations ------------------------------------


def test_normalize_trips_types_and_format():
    df = pd.DataFrame(ROWS_DMY, columns=HEADER)
    out, fmt = rdn.normalize_trips(df)

    assert fmt == "%d/%m/%Y %H:%M"
    assert list(out.columns) == HEADER
    assert str(out["Viaje_Id"].dtype) == "Int64"
    assert pd.isna(out.loc[1, "Año_de_nacimiento"])
    assert out.loc[1, "Fin_del_viaje"] == pd.Timestamp("2023-11-14 00:10")


def test_normalize_trips_rejects_non_numeric_ids():
    rows = [r.copy() for r in ROWS_ISO]
    rows[0][0] = "abc"
    with pytest.raises(ValueError):
        rdn.normalize_trips(pd.DataFrame(rows, columns=HEADER))


def test_normalize_trips_rejects_mixed_start_end_formats():
    rows = [ROWS_ISO[0][:5] + [ROWS_DMY[0][5]] + ROWS_ISO[0][6:]]
    with pytest.raises(ValueError, match="different datetime formats"):
        rdn.normalize_trips(pd.DataFrame(rows, columns=HEADER))


def test_normalize_stations_types():
    df = pd.DataFrame([["2", "GDL", "20.66", "-103.36", "15"]], columns=rdn.STATION_COLUMNS)
    out = rdn.normalize_stations(df)
    assert str(out["id"].dtype) == "Int64"
    assert out.loc[0, "longitude"] == pytest.approx(-103.36)


# --- inspect_raw_file --------------------------------------------------------


def test_inspect_raw_file_fingerprints_variants(raw_dir):
    by_name = {
        p.name: rdn.inspect_raw_file(p) for p in sorted(raw_dir.glob("datos_abiertos_*.csv"))
    }

    plain = by_name["datos_abiertos_2014_12.csv"]
    assert (plain["encoding"], plain["quoted"], plain["line_ending"]) == ("utf-8", False, "LF")
    assert plain["datetime_format"] == "%Y-%m-%d %H:%M:%S"

    dmy = by_name["datos_abiertos_2023_11.csv"]
    assert (dmy["encoding"], dmy["line_ending"]) == ("latin-1", "CRLF")
    assert dmy["datetime_format"] == "%d/%m/%Y %H:%M"
    assert "Año_de_nacimiento" in dmy["header"]  # decoded with the right encoding

    assert "A}äe_nacimiento" in by_name["datos_abiertos_2024_02-1.csv"]["header"]


# --- normalize ---------------------------------------------------------------


def test_normalize_unifies_all_variants(raw_dir, tmp_path):
    dst = tmp_path / "interim"
    summary = rdn.normalize(raw_dir, dst, workers=1)

    assert summary["failed"] == []
    assert summary["found"] == len(summary["normalized"]) == 6

    trips = [pd.read_parquet(p) for p in sorted((dst / "viajes").rglob("*.parquet"))]
    assert len(trips) == 5
    expected = trips[0]
    for df in trips[1:]:
        assert list(df.columns) == HEADER
        assert df.dtypes.equals(expected.dtypes)
        pd.testing.assert_frame_equal(df, expected)  # same data, whatever the raw format


def test_normalize_output_layout_and_report(raw_dir, tmp_path):
    dst = tmp_path / "interim"
    summary = rdn.normalize(raw_dir, dst, workers=1)

    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*.parquet")) == [
        "estaciones/nomenclatura_2026_09.parquet",
        "viajes/2014/viajes_2014_12.parquet",
        "viajes/2021/viajes_2021_06.parquet",
        "viajes/2022/viajes_2022_12.parquet",
        "viajes/2023/viajes_2023_11.parquet",
        "viajes/2024/viajes_2024_02.parquet",
    ]
    report = {r["name"]: r for r in summary["normalized"]}
    assert report["datos_abiertos_2023_11.csv"]["encoding"] == "latin-1"
    assert report["datos_abiertos_2023_11.csv"]["datetime_format"] == "%d/%m/%Y %H:%M"
    assert report["datos_abiertos_2014_12.csv"]["rows"] == 2


def test_normalize_decodes_latin1_stations(raw_dir, tmp_path):
    dst = tmp_path / "interim"
    rdn.normalize(raw_dir, dst, workers=1)
    stations = pd.read_parquet(dst / "estaciones/nomenclatura_2026_09.parquet")
    assert stations.loc[0, "name"] == "(GDL-002) Av. Niños héroes"


def test_normalize_does_not_modify_raw(raw_dir, tmp_path):
    before = snapshot(raw_dir)
    rdn.normalize(raw_dir, tmp_path / "interim", workers=1)
    assert snapshot(raw_dir) == before


def test_normalize_is_idempotent_and_force_reprocesses(raw_dir, tmp_path):
    dst = tmp_path / "interim"
    rdn.normalize(raw_dir, dst, workers=1)

    again = rdn.normalize(raw_dir, dst, workers=1)
    assert again["normalized"] == [] and again["skipped"] == 6

    forced = rdn.normalize(raw_dir, dst, force=True, workers=1)
    assert len(forced["normalized"]) == 6


def test_normalize_reprocesses_when_raw_is_newer(raw_dir, tmp_path):
    dst = tmp_path / "interim"
    rdn.normalize(raw_dir, dst, workers=1)
    out = dst / "viajes/2014/viajes_2014_12.parquet"
    raw = raw_dir / "datos_abiertos_2014_12.csv"
    os.utime(out, (0, 0))  # make the output older than its source
    summary = rdn.normalize(raw_dir, dst, workers=1)
    assert [r["name"] for r in summary["normalized"]] == [raw.name]


def test_normalize_records_failures_and_keeps_going(raw_dir, tmp_path):
    write_raw(raw_dir / "datos_abiertos_2025_01.csv", [["1", "2"]], header=["a", "b"])
    summary = rdn.normalize(raw_dir, tmp_path / "interim", workers=1)

    assert [f["name"] for f in summary["failed"]] == ["datos_abiertos_2025_01.csv"]
    assert "Expected 8 columns" in summary["failed"][0]["error"]
    assert len(summary["normalized"]) == 6
    assert not list((tmp_path / "interim").rglob("*.part"))


def test_normalize_rejects_two_raw_files_for_same_month(raw_dir, tmp_path):
    write_raw(raw_dir / "datos_abiertos_2014_12-1.csv", ROWS_ISO)
    with pytest.raises(ValueError, match="same output"):
        rdn.normalize(raw_dir, tmp_path / "interim", workers=1)


def test_normalize_in_parallel_processes(raw_dir, tmp_path):
    summary = rdn.normalize(raw_dir, tmp_path / "interim", workers=2)
    assert summary["failed"] == []
    assert len(summary["normalized"]) == 6


def test_normalize_empty_source(tmp_path):
    summary = rdn.normalize(tmp_path, tmp_path / "interim", workers=1)
    assert summary == {"found": 0, "skipped": 0, "normalized": [], "failed": []}


# --- tracked run -------------------------------------------------------------


def test_normalize_tracked_logs_raw_input_interim_outputs_and_formats(
    raw_dir, tmp_path, mlflow_store
):
    dst = tmp_path / "interim"
    summary = rdn.normalize_tracked(raw_dir, dst, workers=1)
    run = mlflow.get_run(summary["mlflow_run_id"])

    assert run.info.run_name == rdn.RUN_NAME and run.info.status == "FINISHED"
    datasets = {i.dataset.name: i.dataset for i in run.inputs.dataset_inputs}
    assert set(datasets) == {RAW_DATASET, "mibici_trips_interim", "mibici_stations"}
    assert datasets[RAW_DATASET].digest == tracking.files_digest(raw_csv_files(raw_dir), raw_dir)
    assert datasets["mibici_trips_interim"].digest == tracking.files_digest(
        trip_files(dst), dst / "viajes"
    )
    metrics = run.data.metrics
    assert metrics["files_normalized"] == 6
    assert metrics["rows_normalized"] == 5 * 2 + 1  # five trip variants and one station
    assert metrics["files_utf8"] + metrics["files_latin1"] == 6
    assert metrics["files_datetime_iso"] == 4
    assert metrics["files_datetime_day_first_no_seconds"] == 1
    artifacts = [a.path for a in MlflowClient().list_artifacts(run.info.run_id)]
    assert artifacts == ["normalized_files.json"]
