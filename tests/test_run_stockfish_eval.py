"""The batched SF runner builds arguments the eval script accepts."""
from pathlib import Path
import runpy
import sys
import json

import pytest
from types import SimpleNamespace
from copy import deepcopy


def batch_payload(checkpoint):
    """Use the evaluator's serializer and actual ladder envelope."""
    evaluator = runpy.run_path("scripts/eval_vs_stockfish.py")
    payload = evaluator["_summary_to_payload"](
        summary=evaluator["EvalSummary"](games=50, completed_games=50, wins=10, draws=10, losses=30),
        checkpoint_path=checkpoint, stockfish_path=Path("/usr/bin/stockfish"),
        engine_limit=evaluator["chess"].engine.Limit(nodes=40000),
        stockfish_options={"UCI_Elo": 2600}, device=evaluator["torch"].device("cuda"),
        dtype=evaluator["torch"].float32, compile_enabled=True, seed=1042,
        max_plies=512, model_move_policy="gumbel", search_lambda=0.1,
        opening_random_plies=0, search_knobs={"gumbel_simulations": 512})
    return dict(mode="ladder", aggregate=payload,
                segments=[dict(stockfish={"elo": 2600}, results=deepcopy(payload))])


def test_runner_arguments_parse_in_eval_script(monkeypatch, tmp_path):
    runner = runpy.run_path("scripts/run_stockfish_eval.py")
    evaluator = runpy.run_path("scripts/eval_vs_stockfish.py")
    args = SimpleNamespace(config=Path("config/imba_chess_v4_aux3.toml"), checkpoint=tmp_path / "actor.pt",
                           batch_games=50, elo=2600, stockfish="/usr/bin/stockfish", nodes=40000,
                           concurrent_games=8, seed_base=1042)
    extra = ["--gumbel-simulations", "512", "--gumbel-root-forcing", "--gumbel-forcing-floor",
             "--gumbel-minimax-weight", "0.5"]
    monkeypatch.setattr(sys, "argv", ["eval_vs_stockfish.py", *runner["eval_arguments"](args, 1, tmp_path, extra)])
    parsed = evaluator["_parse_args"]()
    specs = evaluator["_build_segment_specs"](parsed)
    assert len(specs) == 1 and specs[0].elo == 2600 and specs[0].games == 50 and specs[0].limit_strength
    assert parsed.seed == 1043 and parsed.gumbel_simulations == 512 and parsed.gumbel_minimax_weight == 0.5
    assert parsed.gumbel_root_forcing and parsed.gumbel_forcing_floor and parsed.gumbel_value_scale == 0.5 and not parsed.gumbel_rescale_values


@pytest.mark.parametrize("old_manifest", [True, False])
def test_runner_rejects_batches_from_old_runtime(monkeypatch, tmp_path, old_manifest):
    runner = runpy.run_path("scripts/run_stockfish_eval.py")
    checkpoint, config = tmp_path / "actor.pt", tmp_path / "config.toml"
    checkpoint.touch()
    config.touch()
    out = tmp_path / "eval"
    (out / "batch-00").mkdir(parents=True)
    manifest = dict(checkpoint=str(checkpoint), config=str(config), games=50,
                    batch_games=50, seed_base=1042, stockfish_elo=2600,
                    stockfish_nodes=40000, concurrent_games=8, search_args=[])
    if not old_manifest:
        manifest["runtime_revision"] = runner["RUNTIME_REVISION"]
    (out / "manifest.json").write_text(json.dumps(manifest))
    payload = batch_payload(checkpoint)
    payload["aggregate"]["run_config"]["runtime_revision"] = "shared-search-v1"
    (out / "batch-00/results.json").write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", ["run_stockfish_eval.py", "--checkpoint", str(checkpoint),
                                     "--config", str(config), "--out", str(out), "--games", "50"])
    with pytest.raises(SystemExit, match="runtime_revision" if old_manifest else "different runtime"):
        runner["main"]()
    assert not (out / "results.json").exists()


@pytest.mark.parametrize("stale", [None, "aggregate", "segment", "missing"])
def test_runner_resumes_real_ladder_results(monkeypatch, tmp_path, stale):
    runner = runpy.run_path("scripts/run_stockfish_eval.py")
    checkpoint, config = tmp_path / "actor.pt", tmp_path / "config.toml"
    checkpoint.touch()
    config.touch()
    out = tmp_path / "eval"
    (out / "batch-00").mkdir(parents=True)
    payload = batch_payload(checkpoint)
    if stale is not None:
        row = payload["segments"][0]["results"] if stale == "segment" else payload["aggregate"]
        if stale == "missing":
            row.pop("run_config")
        else:
            row["run_config"]["runtime_revision"] = "shared-search-v1"
    (out / "batch-00/results.json").write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", ["run_stockfish_eval.py", "--checkpoint", str(checkpoint),
                                     "--config", str(config), "--out", str(out), "--games", "50"])
    original_popen = runner["subprocess"].Popen

    def unexpected_launch(command, *args, **kwargs):
        if command[0] == "git":
            return original_popen(command, *args, **kwargs)
        raise AssertionError("completed batch must not launch new games")
    monkeypatch.setattr(runner["subprocess"], "Popen", unexpected_launch)
    if stale is not None:
        with pytest.raises(SystemExit, match="different runtime"):
            runner["main"]()
    else:
        runner["main"]()
        aggregate = json.loads((out / "results.json").read_text())["aggregate"]
        assert aggregate["games"] == 50 and aggregate["score"] == 0.3


def test_puct_runner_does_not_inject_gumbel_only_flags(monkeypatch, tmp_path):
    runner = runpy.run_path("scripts/run_stockfish_eval.py")
    evaluator = runpy.run_path("scripts/eval_vs_stockfish.py")
    args = SimpleNamespace(config=Path("config/imba_chess_v4_aux3.toml"), checkpoint=tmp_path / "actor.pt",
                           batch_games=50, elo=2600, stockfish="/usr/bin/stockfish", nodes=40000,
                           concurrent_games=8, seed_base=1042)
    extra = ["--model-move-policy", "puct_variance_rpo", "--simulations", "256",
             "--search-config", "config/self_play_puct_variance_rpo.toml"]
    argv = runner["eval_arguments"](args, 0, tmp_path, extra)
    monkeypatch.setattr(sys, "argv", ["eval_vs_stockfish.py", *argv])
    parsed = evaluator["_parse_args"]()
    assert parsed.model_move_policy == "puct_variance_rpo"
    assert parsed.gumbel_value_scale is None and parsed.gumbel_simulations is None
    assert parsed.simulations == 256


def test_runner_rejects_changed_search_file_on_resume(monkeypatch, tmp_path):
    import hashlib
    runner = runpy.run_path("scripts/run_stockfish_eval.py")
    checkpoint, config, search = (tmp_path / f for f in ("actor.pt", "model.toml", "search.toml"))
    checkpoint.touch()
    config.touch()
    search.write_text('[search]\nalgorithm="puct_variance"\ncpuct=1.75\n')
    out = tmp_path / "eval"
    out.mkdir()
    extra = ["--model-move-policy", "puct_variance", "--search-config", str(search)]
    manifest = dict(checkpoint=str(checkpoint.resolve()), config=str(config.resolve()), games=50,
                    batch_games=50, seed_base=1042, stockfish_elo=2600, stockfish_nodes=40000,
                    concurrent_games=8, search_args=extra, runtime_revision=runner["RUNTIME_REVISION"],
                    search_config_sha256=hashlib.sha256(search.read_bytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(manifest))
    search.write_text(search.read_text().replace("1.75", "2.0"))
    monkeypatch.setattr(sys, "argv", ["run_stockfish_eval.py", "--checkpoint", str(checkpoint),
                                     "--config", str(config), "--out", str(out), "--games", "50", "--", *extra])
    with pytest.raises(SystemExit, match="search_config_sha256"):
        runner["main"]()
