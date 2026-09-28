"""The frozen-evaluator control is opt-in and refuses incoherent settings."""

import subprocess
import sys
from pathlib import Path

import pytest

CONFIG = Path("config/self_play_5090.toml")


def run(tmp_path, *extra, value_weight):
    config = tmp_path / "control.toml"
    text = CONFIG.read_text().replace("[learning]", "[learning]\nauxiliary_value_weight = 0.0")
    assert "value_weight" in text
    config.write_text(
        "\n".join(
            f"value_weight = {value_weight}" if line.strip().startswith("value_weight")
            else line
            for line in text.splitlines()
        )
    )
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_self_play", "--config", str(config),
         "--resume", "--seeds", str(tmp_path / "seeds.json"),
         "--output", str(tmp_path / "out"), *extra],
        capture_output=True, text=True, cwd=Path.cwd(),
    )


def test_frozen_evaluator_requires_policy_only_training(tmp_path):
    evaluator = tmp_path / "frozen.pt"
    evaluator.write_bytes(b"")
    done = run(tmp_path, "--frozen-evaluator", str(evaluator), value_weight=1.0)
    assert done.returncode == 2
    assert "value_weight = 0" in done.stderr


def test_frozen_evaluator_accepts_policy_only_training(tmp_path):
    evaluator = tmp_path / "frozen.pt"
    evaluator.write_bytes(b"")
    done = run(tmp_path, "--frozen-evaluator", str(evaluator), value_weight=0.0)
    # Gets past validation and fails later, on the missing seed manifest.
    assert "value_weight = 0" not in done.stderr


def test_missing_frozen_evaluator_is_rejected(tmp_path):
    done = run(tmp_path, "--frozen-evaluator", str(tmp_path / "absent.pt"), value_weight=0.0)
    assert done.returncode == 2
    assert "does not exist" in done.stderr
