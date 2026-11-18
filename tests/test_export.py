"""Export of the registered champions to the folder the prediction service loads."""

import numpy as np
import pytest
from test_backtest import FIRST_TEST_MONTH
from test_registry import registered  # noqa: F401  (fixture: champions in a throwaway store)

from team2_mlops2026 import features as ft
from team2_mlops2026.modeling import export as ex
from team2_mlops2026.modeling import registry as reg
from team2_mlops2026.modeling import train as tr


def test_export_writes_each_champion_and_a_model_card(registered, tmp_path):  # noqa: F811
    card = ex.export_models(dst=tmp_path)
    assert ex.read_card(tmp_path) == card
    assert len(card["git_commit"]) == 40 and card["exported_at"]

    _, test = tr.load_training_data(registered["src"], FIRST_TEST_MONTH)
    sample = tr.model_inputs(test.head(500))
    models, _ = ex.load_exported(tmp_path)
    for target in ft.TARGETS:
        champion = registered["versions"][target][reg.CHAMPION]
        entry = card["models"][target]
        assert entry["name"] == reg.model_name(target) and entry["alias"] == reg.CHAMPION
        assert entry["version"] == str(champion.version)
        assert entry["uri"] == f"models:/{champion.name}/{champion.version}"
        assert entry["run_id"] == champion.run_id
        assert entry["tags"] == champion.tags  # metrics, digests and git commit of the version
        assert (ex.model_dir(target, tmp_path) / "MLmodel").exists()

        registered_model, _ = reg.load_model(target)
        assert np.array_equal(models[target].predict(sample), registered_model.predict(sample))


def test_exporting_again_replaces_the_folders(registered, tmp_path):  # noqa: F811
    ex.export_models(dst=tmp_path)
    stale = ex.model_dir("departures", tmp_path) / "stale.txt"
    stale.write_text("from an older version")
    ex.export_models(dst=tmp_path)
    assert not stale.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [ex.MODEL_CARD, *(reg.model_name(t) for t in ft.TARGETS)]
    )


def test_loading_without_an_export_asks_for_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="make export"):
        ex.load_exported(tmp_path)
