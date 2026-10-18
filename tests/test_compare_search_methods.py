"""Check study cohort integrity and separation of tuning from held-out results."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from imba_chess.eval.golden import digest


def study(tmp_path, monkeypatch, positions, labelled, extra):
    spec = importlib.util.spec_from_file_location(
        "compare_search_methods_test", Path(__file__).parents[1] / "scripts/compare_search_methods.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = dict(positions=positions, positions_sha256=digest(positions))
    roots = tmp_path / "positions.json"
    roots.write_text(json.dumps(manifest))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "manifest.json").write_text(json.dumps(dict(positions_sha256=manifest["positions_sha256"])))
    with (reference / "labels.jsonl").open("w") as stream:
        for pid in labelled:
            stream.write(json.dumps(dict(position_id=pid, root=dict(expectation=0.8),
                moves={"a2a3": dict(expectation=0.4), "a2a4": dict(expectation=0.8)})) + "\n")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"unused by the fake search")
    arms = tmp_path / "arms.toml"
    arms.write_text("[[arms]]\nname='gumbel'\n[[arms]]\nname='puct'\nalgorithm='puct'\n")
    cohorts = []

    def run_arm(args, arm, selected, *hashes):
        cohorts.append([p["position_id"] for p in selected])
        rows = []
        for p in selected:
            policy = ([0.5, 0.5] if arm["name"] == "gumbel" else
                      [1, 0] if p["split"] == "tune" else [0, 1])
            rows.append(dict(position_id=p["position_id"], variant="eval",
                legal_ucis=["a2a3", "a2a4"], policy=policy, visits=[1, 1],
                move_uci="a2a4", root_log_priors=[-0.6931471805599453] * 2))
        return rows

    monkeypatch.setattr(module, "run_arm", run_arm)
    monkeypatch.setattr(sys, "argv", ["compare_search_methods", "--positions", str(roots),
        "--reference", str(reference), "--checkpoint", str(checkpoint), "--arms", str(arms),
        "--out", str(tmp_path / "out"), *extra])
    module.main()
    return json.loads((tmp_path / "out/summary.json").read_text()), cohorts


def position(pid, split, game):
    return dict(position_id=pid, split=split, game_id=game, source="human", phase="middlegame")


def test_full_report_keeps_held_out_conclusion_separate(tmp_path, monkeypatch):
    positions = [position("t", "tune", "g0"), position("a", "test", "g1"),
                 position("b", "test", "g1"), position("c", "test", "g2")]
    result, cohorts = study(tmp_path, monkeypatch, positions, [p["position_id"] for p in positions],
                            ["--split", "all"])
    assert cohorts == [["t", "a", "b", "c"]] * 2
    assert result["comparisons"]["puct"]["target_regret"]["candidate_minus_baseline"] == pytest.approx(-0.1)
    held_out = result["by_split"]["test"]["comparisons"]["puct"]["target_regret"]
    assert held_out["candidate_minus_baseline"] == pytest.approx(-0.2)
    assert held_out["source_games"] == 2
    assert result["by_split"]["tune"]["comparisons"]["puct"]["target_regret"]["candidate_minus_baseline"] == pytest.approx(0.2)


def test_partial_labels_do_not_replace_requested_tuning_roots(tmp_path, monkeypatch):
    positions = [position(pid, "tune", pid) for pid in ("missing", "available", "later")]
    result, cohorts = study(tmp_path, monkeypatch, positions, ["available", "later"],
                            ["--limit", "2", "--allow-partial-reference"])
    assert cohorts == [["available"]] * 2
    assert result["requested_positions"] == 2
    assert result["positions"] == 1
