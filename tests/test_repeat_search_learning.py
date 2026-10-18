"""A new seed must not silently change the method or mix prior repeat outputs."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from imba_chess.eval.golden import file_sha256
from imba_chess.self_play.config import load_config


def test_repeat_changes_only_seed_and_keeps_original_immutable(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from repeat_search_learning import prepare_configs

    source = tmp_path / "original.toml"
    source.write_text(Path("config/self_play.toml").read_text())
    original = source.read_text()
    study = tmp_path / "study"
    study.mkdir()
    (study / "frozen-selection.json").write_text(json.dumps(dict(
        learning_arms=["gumbel"], configs=dict(gumbel=dict(
            path=str(source), sha256=file_sha256(source))))))
    output = tmp_path / "repeat"
    _, configs = prepare_configs(study, output, 43)
    expected = asdict(load_config(source))
    expected["run"]["seed"] = 43
    assert asdict(load_config(configs["gumbel"])) == expected
    assert source.read_text() == original
    assert prepare_configs(study, output, 43)[1] == configs
    with pytest.raises(ValueError, match="repeat configuration changed"):
        prepare_configs(study, output, 44)
    source.write_text(original + "\n# changed after freezing\n")
    with pytest.raises(ValueError, match="frozen configuration changed"):
        prepare_configs(study, output, 43)
