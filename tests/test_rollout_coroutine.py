"""CPU-only test of the Task 4 game coroutine (_process_game).

Drives one game coroutine by hand with fake executors -- scripted root
outputs / PositionEvals, no real model, no GPU, no BatchScheduler -- to
pin the yield contract (WorkRequest("root_eval", ...) before any
WorkRequest("decode_wave", ...)) and confirm the rows it produces still
match the schema _arm_rows_to_dicts/RolloutRow expect. The merged executors'
own GPU-path correctness (root_eval batching, forward_decode_grouped) is
exercised elsewhere (tests/test_generate_search_rollouts.py's tiny-model
tests, and Task 5's GPU gates) -- this file only needs the coroutine itself,
which is why it can stay torch-light and fully offline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import chess
import pytest
import torch

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.search import PositionEval


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_search_rollouts.py"
    spec = importlib.util.spec_from_file_location(
        "generate_search_rollouts_coroutine_test", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load generate_search_rollouts.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GAME = {
    "game_id": "fake-game",
    "result": "1-0",
    "plays": [
        {"move_uci": "e2e4"},
        {"move_uci": "e7e5"},
        {"move_uci": "g1f3"},
        {"move_uci": "b8c6"},
    ],
}


def _fake_root_output(move_vocab: MoveVocab) -> dict:
    """A structurally valid (all-zero) root_eval response: one token's worth
    of logits/value_logits plus a one-layer kv_caches the coroutine stores
    on its CachedPositionEvaluator but never runs through a model (the
    decode_wave side is fully faked below, so build_decode_request/evaluate
    is never invoked on it)."""
    vocab_size = len(move_vocab)
    return {
        "logits": torch.zeros(1, vocab_size),
        "value_logits": torch.zeros(1, 3),
        "kv_caches": [(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
    }


def _fake_decode_wave(batch: list[tuple[object, chess.Board]]) -> list[PositionEval]:
    """Scripted PositionEvals: real legal moves off the real board (so
    search's forcing-move/refutation logic has something sane to chew on)
    with made-up uniform-ish priors, and a constant value -- no model call
    anywhere in this path."""
    results = []
    for _, board in batch:
        moves = list(board.legal_moves)[:2]
        priors = [-0.1 * (i + 1) for i in range(len(moves))]
        results.append(PositionEval(value_stm=0.0, legal_moves=moves, legal_log_priors=priors))
    return results


def _drive_game(gen, move_vocab):
    """Manual next()/send() driver recording every yielded WorkRequest's
    kind, answering root_eval/decode_wave with the fakes above."""
    kinds: list[str] = []
    try:
        request = next(gen)
        while True:
            kinds.append(request.kind)
            if request.kind == "root_eval":
                response = _fake_root_output(move_vocab)
            elif request.kind == "decode_wave":
                _evaluator, batch = request.payload
                response = _fake_decode_wave(batch)
            else:
                raise AssertionError(f"unexpected WorkRequest kind: {request.kind!r}")
            request = gen.send(response)
    except StopIteration as stop:
        return kinds, stop.value


def _make_game_coroutine(module, game, *, every_n_plies=1, halving_config=None):
    move_vocab = MoveVocab.build_static()
    board_state_encoder = BoardStateEncoder()
    halving_config = halving_config or module.HalvingConfig(budget=8, top_m=4, max_depth=2)
    gen = module._process_game(
        game,
        model=None,  # never called: decode_wave is fully faked, no real evaluate()
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        halving_config=halving_config,
        every_n_plies=every_n_plies,
        sample_seed=1,
        checkpoint_path="fake.pt",
    )
    return gen, move_vocab


def test_process_game_yields_root_eval_before_any_decode_wave():
    module = _load_script_module()
    gen, move_vocab = _make_game_coroutine(module, _GAME)

    kinds, rows = _drive_game(gen, move_vocab)

    assert kinds[0] == "root_eval"
    # Every decode_wave must be preceded by at least one root_eval (the root
    # forward has to exist before a search wave can be built from it) --
    # equivalently: no decode_wave appears before the first root_eval.
    first_decode_wave = next((i for i, k in enumerate(kinds) if k == "decode_wave"), None)
    first_root_eval = kinds.index("root_eval")
    assert first_decode_wave is None or first_root_eval < first_decode_wave
    # 4 plies, every_n_plies=1 -> 4 sampled positions -> 4 root_eval yields.
    assert kinds.count("root_eval") == 4
    assert len(rows) >= 1


def test_process_game_rows_match_arm_rows_to_dicts_schema():
    module = _load_script_module()
    gen, move_vocab = _make_game_coroutine(module, _GAME)

    _kinds, rows = _drive_game(gen, move_vocab)

    assert len(rows) >= 1
    for row in rows:
        assert row.game_id == "fake-game"
        assert 0 <= row.ply < len(_GAME["plays"])
        assert isinstance(row.human_move_uci, str)
        assert row.real_outcome_stm in (-1, 0, 1)
        assert isinstance(row.best_arm_move_uci, str)
        assert len(row.root_wdl_unsearched) == 3
        assert abs(sum(row.root_wdl_unsearched) - 1.0) < 1e-6
        # Same tuple-length invariant _arm_rows_to_dicts/RolloutRow rely on:
        # every per-arm field is aligned by index, no padding/truncation.
        assert len(row.arm_move_uci) == len(row.arm_backed_value)
        assert len(row.arm_move_uci) == len(row.arm_evals_spent)
        assert len(row.arm_move_uci) == len(row.arm_log_prior)
        assert row.best_arm_move_uci in row.arm_move_uci
        assert row.search_budget == module.HalvingConfig(budget=8, top_m=4, max_depth=2).budget


def test_process_game_zero_sampled_plies_returns_immediately_without_yielding():
    module = _load_script_module()
    # every_n_plies larger than the game's ply count with a seed/offset that
    # samples nothing: _sample_ply_indices returns [] and _process_game must
    # return [] via StopIteration on the very first next(), with no
    # WorkRequest yielded at all (mirrors the zero-yield-game case the batch
    # scheduler (Task 2) already special-cases).
    tiny_game = {"game_id": "g", "result": "1-0", "plays": []}
    gen, move_vocab = _make_game_coroutine(module, tiny_game)

    with pytest.raises(StopIteration) as excinfo:
        next(gen)
    assert excinfo.value.value == []
