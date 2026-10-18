"""Public method names resolve to the tested production search configurations."""
from pathlib import Path
import runpy
import sys

import pytest

from imba_chess.config import EvalVsStockfishConfig
from imba_chess.eval.search_options import PUCT_METHODS, puct_config, search_backend
from imba_chess.self_play.config import load_config


@pytest.mark.parametrize("method", PUCT_METHODS)
def test_public_preset_matches_training_recipe(method):
    path = Path(f"config/self_play_{method}.toml")
    trained = load_config(path).search
    assert puct_config(method) == trained
    assert puct_config(method, path=path) == trained
    assert puct_config(method, path=path, simulations=19).simulations == 19
    assert search_backend(method) == trained.algorithm
    assert EvalVsStockfishConfig(model_move_policy=method).model_move_policy == method


def test_mislabeled_parameters_rejected():
    with pytest.raises(ValueError, match="does not match"):
        puct_config("puct_variance", path=Path("config/self_play_puct_variance_rpo.toml"))
    with pytest.raises(ValueError):
        puct_config("puct_variance", simulations=0)
    with pytest.raises(ValueError):
        search_backend("typo")


def test_match_can_compare_gumbel_with_variance_rpo(monkeypatch):
    module = runpy.run_path("scripts/match_two_checkpoints.py")
    monkeypatch.setattr(sys, "argv", ["match_two_checkpoints.py", "--checkpoint-a", "a.pt",
        "--checkpoint-b", "b.pt", "--model-move-policy-a", "gumbel",
        "--model-move-policy-b", "puct_variance_rpo", "--simulations", "256",
        "--search-config-b", "config/self_play_puct_variance_rpo.toml"])
    args = module["_parse_args"]()
    configs = module["_search_configs"](args, EvalVsStockfishConfig(),
                                      {"A": args.model_move_policy_a, "B": args.model_move_policy_b})
    assert configs["A"].simulations == configs["B"].simulations == 256
    assert configs["B"].target_policy == "rpo"
    assert configs["B"].rpo_regularization == 0.015625


@pytest.mark.parametrize("script", ["scripts/match_two_checkpoints.py", "scripts/eval_vs_stockfish.py"])
def test_evaluation_explicitly_disables_puct_exploration(script):
    from types import SimpleNamespace
    import chess
    module = runpy.run_path(script)
    seen = []

    def search(**kwargs):
        seen.append(kwargs)
        yield "pending"

    runtime = SimpleNamespace(algorithm="puct_variance", search=search, search_batch=search)
    if script.endswith("match_two_checkpoints.py"):
        gen = module["_select_move"](side="A", board=chess.Board(), history=None,
                                     runtime=runtime, config=puct_config("puct_variance"), rng=None)
    else:
        gen = module["_select_model_move_stepwise"](board=chess.Board(), batch={},
            runtime=runtime, config=puct_config("puct_variance"), debug_topk=0)
    # Both adapters reach the underlying runtime before awaiting any neural work.
    try:
        next(gen)
    except AttributeError:  # the fake pending request deliberately has no kind/payload
        pass
    finally:
        gen.close()
    assert seen[0]["noise"] == 0.0


def test_mixed_match_routes_each_side_and_saves_actual_settings(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.move_vocab import MoveVocab
    module = runpy.run_path("scripts/match_two_checkpoints.py")
    namespace = module["main"].__globals__
    vocab = MoveVocab.build_static()
    seen, algorithms = [], []
    def load(**kwargs):
        algorithms.append(kwargs["algorithm"])
        return SimpleNamespace(move_vocab=vocab, encoder=BoardStateEncoder(), executors={},
            options={"runtime_revision": "test"}, clear_caches=lambda: None), 128
    def select(**kwargs):
        seen.append((kwargs["side"], kwargs["config"]))
        if False:
            yield
        return next(iter(kwargs["board"].legal_moves)).uci()
    monkeypatch.setitem(namespace, "load_runtime", load)
    monkeypatch.setitem(namespace, "_select_move", select)
    monkeypatch.setitem(namespace, "LichessDataset", lambda **kwargs: None)
    monkeypatch.setitem(namespace, "_opening_iter", lambda *args, **kwargs: [[]])
    monkeypatch.setattr(namespace["os"], "_exit", lambda code: None)
    output = tmp_path / "match.json"
    monkeypatch.setattr(sys, "argv", ["match_two_checkpoints.py", "--checkpoint-a", "a.pt",
        "--checkpoint-b", "b.pt", "--model-move-policy-a", "gumbel",
        "--model-move-policy-b", "puct_variance_rpo", "--simulations-a", "16",
        "--simulations-b", "32", "--games", "2", "--max-plies", "4", "--device", "cpu",
        "--inference-dtype-b", "bfloat16", "--output-json", str(output)])
    module["main"]()
    assert algorithms == ["gumbel", "puct_variance"]
    assert {side for side, _ in seen} == {"A", "B"}
    assert all(cfg.simulations == (16 if side == "A" else 32) for side, cfg in seen)
    assert all(cfg.target_policy == "rpo" for side, cfg in seen if side == "B")
    data = json.loads(output.read_text())
    assert data["algorithm"] == "mixed"
    assert data["search_b"]["rpo_regularization"] == 0.015625
    assert data["budget_a"] == 16 and data["budget_b"] == 32
    assert data["precision_a"] == "float32" and data["precision_b"] == "bfloat16"
    assert data["exploration_b"] == "zero_noise"
