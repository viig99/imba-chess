from __future__ import annotations

import importlib.util
import multiprocessing
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import chess
import chess.engine
import pytest

torch = pytest.importorskip("torch")

from imba_chess.config import BoardStateConfig, ModelConfig, RepoConfig
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.eval.position_evaluator import _forward_model
from imba_chess.model import HSTUChessModel, build_hstu_chess_config

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_VOCAB_PATH = REPO_ROOT / "artifacts" / "move_vocab_static_uci.json"


def _load_eval_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_vs_stockfish.py"
    spec = importlib.util.spec_from_file_location("eval_vs_stockfish_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_vs_stockfish.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dummy_kv(total_tokens: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # One fake layer; the evaluator only threads shapes through.
    return [(torch.zeros(1, total_tokens, 1), torch.zeros(1, total_tokens, 1))]


def _dummy_decode_kv(batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.zeros(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))]


class _DummyValueRerankModel(torch.nn.Module):
    """Root prefers e2e4; value head favors positions reached via d2d4."""

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        value_logits[last, 1] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        prev_ids = new_token_batch["prev_move_id"]
        for row in range(batch_size):
            move_id = int(prev_ids[row].item())
            if move_id == self.move_vocab.token_to_id["e2e4"]:
                value_logits[row] = torch.tensor([0.0, 0.0, 4.0])
            elif move_id == self.move_vocab.token_to_id["d2d4"]:
                value_logits[row] = torch.tensor([4.0, 0.0, 0.0])
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }


class _DummyNoValueModel(torch.nn.Module):
    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 1.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 0.5
        out = {"logits": logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out


class _DummyValueSearchD2Model(torch.nn.Module):
    """Depth-1 nodes get opponent priors; depth-2 values depend on the line.

    The root move is recovered from the node's board (piece_ids): after e2e4
    a white pawn (id 1) sits on e4 (square 28); after d2d4, on d4 (27).
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        value_logits[last, 1] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        prev_ids = new_token_batch["prev_move_id"]
        piece_ids = new_token_batch["piece_ids"]
        root_moves = {
            self.move_vocab.token_to_id["e2e4"],
            self.move_vocab.token_to_id["d2d4"],
        }
        for row in range(batch_size):
            prev = int(prev_ids[row].item())
            if prev in root_moves:
                # Depth 1: opponent to move after our root move.
                logits[row, self.move_vocab.token_to_id["e7e5"]] = 3.0
                logits[row, self.move_vocab.token_to_id["d7d5"]] = 2.5
                value_logits[row, 1] = 1.0
                continue
            # Depth 2: root move recovered from the board.
            root_is_e4 = int(piece_ids[row, chess.E4].item()) == 1
            if prev == self.move_vocab.token_to_id["e7e5"]:
                value_logits[row] = (
                    torch.tensor([4.0, 0.0, 0.0])
                    if root_is_e4
                    else torch.tensor([0.0, 1.0, 2.0])
                )
            elif prev == self.move_vocab.token_to_id["d7d5"]:
                value_logits[row] = (
                    torch.tensor([2.0, 1.0, 0.0])
                    if root_is_e4
                    else torch.tensor([0.0, 0.0, 4.0])
                )
            else:
                value_logits[row, 1] = 1.0
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }


def _mini_repo_config() -> RepoConfig:
    return RepoConfig(
        model=ModelConfig(
            model_dim=64,
            linear_hidden_dim=16,
            attention_dim=16,
            num_heads=1,
            num_layers=0,
            dropout=0.0,
            max_position_embeddings=128,
            enable_value_head=False,
        )
    )


def _mini_vocab() -> MoveVocab:
    return MoveVocab.build(
        ["e2e4", "d2d4", "e7e5", "d7d5"],
        config=MoveVocabConfig(include_unk=False),
    )


def test_value_rerank_selects_move_using_batched_value_lookahead():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    model = _DummyValueRerankModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    move, debug = module._select_model_move(
        model=model,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_rerank",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    assert move.uci() == "d2d4"
    assert model.forward_calls == 2  # prefill + 1 decode wave
    assert debug["policy"] == "value_rerank"
    assert len(debug["value_rerank_candidates"]) == 2


def test_value_rerank_requires_value_logits():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    model = _DummyNoValueModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    with pytest.raises(
        RuntimeError, match="model_move_policy=value_rerank requires a checkpoint with value head enabled"
    ):
        module._select_model_move(
            model=model,
            batch=batch,
            board=board,
            move_vocab=move_vocab,
            board_state_encoder=BoardStateEncoder(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            policy="value_rerank",
            value_rerank_top_k=2,
            value_rerank_lambda=1.0,
            debug_topk=0,
        )


def test_load_model_fails_fast_when_value_rerank_requested_without_value_head(
    tmp_path: Path,
):
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    repo_config = _mini_repo_config()
    model_cfg = build_hstu_chess_config(repo_config.model, move_vocab_size=len(move_vocab))
    model = HSTUChessModel(model_cfg)
    checkpoint_path = tmp_path / "policy_only_ckpt.pt"
    torch.save(model.state_dict(), checkpoint_path)

    with pytest.raises(
        ValueError, match="requires a checkpoint with value_head parameters"
    ):
        module.load_hstu_checkpoint(
            checkpoint_path=checkpoint_path,
            repo_config=repo_config,
            move_vocab=move_vocab,
            device=torch.device("cpu"),
            compile_model=False,
            require_value_head=True,
        )


def test_value_search_d2_selects_move_using_opponent_best_reply():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    model = _DummyValueSearchD2Model(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    move, debug = module._select_model_move(
        model=model,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_d2",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    assert move.uci() == "d2d4"
    assert model.forward_calls == 3  # prefill + depth-1 decode wave + depth-2 decode wave
    assert debug["policy"] == "value_search_d2"
    rows = debug["value_search_d2_candidates"]
    assert len(rows) == 2
    by_move = {str(row["move_uci"]): row for row in rows}
    assert float(by_move["d2d4"]["worst_reply_value"]) > float(
        by_move["e2e4"]["worst_reply_value"]
    )


class _DummyMatePreferenceModel(torch.nn.Module):
    """Prefers a quiet move by policy logit; only the value modes should find mate."""

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["a1b1"]] = 4.0
        logits[last, self.move_vocab.token_to_id["a1a8"]] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        return {
            "logits": torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32),
            "value_logits": torch.zeros((batch_size, 3), dtype=torch.float32),
            "kv": _dummy_decode_kv(batch_size),
        }


def _mate_in_one_setup():
    module = _load_eval_script_module()
    move_vocab = MoveVocab.build(
        ["a1a8", "a1b1"],
        config=MoveVocabConfig(include_unk=False),
    )
    model = _DummyMatePreferenceModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    batch = history.build_batch_for_current_position(board)
    return module, move_vocab, model, history, board, batch


def test_value_rerank_prefers_mate_in_one_over_higher_logit_move():
    module, move_vocab, model, history, board, batch = _mate_in_one_setup()

    move, _ = module._select_model_move(
        model=model,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_rerank",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    assert move.uci() == "a1a8"
    # prefill + 0 decode waves: finding the mate short-circuits before any
    # candidate batch, so the value head is never consulted.
    assert model.forward_calls == 1


def test_value_search_d2_plays_mate_in_one_immediately():
    module, move_vocab, model, history, board, batch = _mate_in_one_setup()

    move, _ = module._select_model_move(
        model=model,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_d2",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    assert move.uci() == "a1a8"
    # prefill + 0 decode waves: mate short-circuits before any depth-1/depth-2
    # decode waves.
    assert model.forward_calls == 1


def test_value_search_d2_requires_value_logits():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    model = _DummyNoValueModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    with pytest.raises(
        RuntimeError,
        match="model_move_policy=value_search_d2 requires a checkpoint with value head enabled",
    ):
        module._select_model_move(
            model=model,
            batch=batch,
            board=board,
            move_vocab=move_vocab,
            board_state_encoder=BoardStateEncoder(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            policy="value_search_d2",
            value_rerank_top_k=2,
            value_rerank_lambda=1.0,
            debug_topk=0,
        )


def test_stockfish_label_formats_limited_and_full_strength():
    module = _load_eval_script_module()

    assert module._stockfish_label(limit_strength=True, elo=1400) == "Stockfish (elo=1400)"
    assert (
        module._stockfish_label(limit_strength=False, elo=None)
        == "Stockfish (full strength)"
    )


def test_outcome_label_covers_all_cases():
    module = _load_eval_script_module()

    assert (
        module._outcome_label(completed=False, result="*", model_color=chess.WHITE)
        == "incomplete"
    )
    assert (
        module._outcome_label(
            completed=True, result="1/2-1/2", model_color=chess.WHITE
        )
        == "draw"
    )
    assert (
        module._outcome_label(completed=True, result="1-0", model_color=chess.WHITE)
        == "model_win"
    )
    assert (
        module._outcome_label(completed=True, result="1-0", model_color=chess.BLACK)
        == "model_loss"
    )
    assert (
        module._outcome_label(completed=True, result="0-1", model_color=chess.BLACK)
        == "model_win"
    )


def test_save_traced_game_writes_pgn_and_html(tmp_path):
    module = _load_eval_script_module()
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5"]:
        board.push_uci(move_uci)
    save_games_dir = tmp_path / "games"

    module._save_traced_game(
        board=board,
        model_color=chess.BLACK,
        result="*",
        completed=False,
        segment_name="sf_elo_1400",
        stockfish_label="Stockfish (elo=1400)",
        game_idx=1,
        save_games_dir=save_games_dir,
    )

    pgn_text = (save_games_dir / "sf_elo_1400_game002_incomplete.pgn").read_text(
        encoding="utf-8"
    )
    assert '[Event "sf_elo_1400"]' in pgn_text
    assert '[White "Stockfish (elo=1400)"]' in pgn_text
    assert '[Black "imba-chess"]' in pgn_text
    assert '[Result "*"]' in pgn_text
    assert "1. e4 e5" in pgn_text
    html_path = save_games_dir / "sf_elo_1400_game002_incomplete.html"
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


class _DummyHalvingModel(torch.nn.Module):
    """Root policy prefers e2e4; value head says the d2d4 subtree is winning.

    Value is read from the side-to-move POV, so the sign is keyed on the new
    token's turn_id; the root move is recovered from the board's d4 square.
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        piece_ids = new_token_batch["piece_ids"]
        turn_ids = new_token_batch["turn_id"]
        for row in range(batch_size):
            logits[row, self.move_vocab.token_to_id["e2e4"]] = 4.0
            logits[row, self.move_vocab.token_to_id["d2d4"]] = 3.0
            good_for_white = int(piece_ids[row, chess.D4].item()) == 1
            stm_is_white = int(turn_ids[row].item()) == 0
            if good_for_white == stm_is_white:
                value_logits[row] = torch.tensor([0.0, 0.0, 3.0])
            else:
                value_logits[row] = torch.tensor([3.0, 0.0, 0.0])
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }


def test_value_search_halving_end_to_end_picks_value_backed_move():
    module = _load_eval_script_module()
    from imba_chess.eval.search import HalvingConfig

    move_vocab = _mini_vocab()
    model = _DummyHalvingModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    move, debug = module._select_model_move(
        model=model,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_halving",
        value_rerank_top_k=2,
        value_rerank_lambda=0.05,
        debug_topk=0,
        halving_config=HalvingConfig(budget=6, top_m=2, rounds=2, lam=0.05),
    )

    assert move.uci() == "d2d4"  # higher root logit is e2e4; value flips it
    assert debug["policy"] == "value_search_halving"
    rows = debug["value_search_halving_candidates"]
    assert {row["move_uci"] for row in rows} == {"e2e4", "d2d4"}


# ---------------------------------------------------------------------------
# Task 3: _select_model_move_stepwise / batched scheduler driver
# ---------------------------------------------------------------------------
#
# Two layers of coverage below:
#   1. Equivalence tests -- _select_model_move_stepwise, driven manually
#      (answering root_eval via _forward_model(..., return_kv=True) and
#      decode_wave via evaluator.evaluate(batch) directly: exactly the
#      single-item codepath the real merged executors fall back to at
#      len(payloads)==1) -- must select the SAME move, for EVERY policy, as
#      the existing synchronous _select_model_move. This is the "request
#      sequence per game must equal today's call sequence" requirement,
#      proven per policy rather than only for one.
#   2. A real BatchScheduler + EnginePool + _play_game integration test
#      (fake Stockfish engine, fake root-eval model, no torch autograd
#      surprises) covering summary aggregation, color alternation, engine
#      reuse per slot at --concurrent-games 1 vs distinct engines at
#      --concurrent-games 2, and fail-fast on an engine exception.


def _drive_model_move_stepwise(gen, *, model, device, dtype):
    """Manual next()/send() driver for _select_model_move_stepwise.

    Answers WorkRequest("root_eval", batch) via _forward_model(...,
    return_kv=True) -- the merged root_eval executor's own hardcoded
    contract (imba_chess.eval.merged_executors._make_root_eval_executor) --
    and WorkRequest("decode_wave", (evaluator, batch)) via evaluator.
    evaluate(batch) directly -- the merged decode_wave executor's
    len(payloads)==1 passthrough. This is exactly the single-game codepath
    BatchScheduler exercises at --concurrent-games 1, without needing a real
    BatchScheduler in these equivalence tests.
    """
    try:
        request = next(gen)
        while True:
            if request.kind == "root_eval":
                response = _forward_model(
                    model=model,
                    batch=request.payload,
                    device=device,
                    dtype=dtype,
                    return_kv=True,
                )
            elif request.kind == "decode_wave":
                evaluator, batch = request.payload
                response = evaluator.evaluate(batch)
            else:
                raise AssertionError(f"unexpected WorkRequest kind: {request.kind!r}")
            request = gen.send(response)
    except StopIteration as stop:
        return stop.value


def test_select_model_move_stepwise_matches_sync_for_greedy():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    move_sync, debug_sync = module._select_model_move(
        model=_DummyNoValueModel(move_vocab),
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="greedy",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    gen = module._select_model_move_stepwise(
        model=_DummyNoValueModel(move_vocab),
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="greedy",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )
    move_stepwise, debug_stepwise = _drive_model_move_stepwise(
        gen, model=_DummyNoValueModel(move_vocab), device=torch.device("cpu"), dtype=torch.float32
    )

    assert move_stepwise.uci() == move_sync.uci() == "e2e4"
    assert debug_stepwise["policy"] == debug_sync["policy"] == "greedy"


def test_select_model_move_stepwise_matches_sync_for_value_rerank():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    model_sync = _DummyValueRerankModel(move_vocab)
    move_sync, debug_sync = module._select_model_move(
        model=model_sync,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_rerank",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    model_stepwise = _DummyValueRerankModel(move_vocab)
    gen = module._select_model_move_stepwise(
        model=model_stepwise,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_rerank",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )
    move_stepwise, debug_stepwise = _drive_model_move_stepwise(
        gen, model=model_stepwise, device=torch.device("cpu"), dtype=torch.float32
    )

    assert move_stepwise.uci() == move_sync.uci() == "d2d4"
    assert model_stepwise.forward_calls == model_sync.forward_calls
    assert len(debug_stepwise["value_rerank_candidates"]) == len(
        debug_sync["value_rerank_candidates"]
    )


def test_select_model_move_stepwise_matches_sync_for_value_search_d2():
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    model_sync = _DummyValueSearchD2Model(move_vocab)
    move_sync, debug_sync = module._select_model_move(
        model=model_sync,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_d2",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )

    model_stepwise = _DummyValueSearchD2Model(move_vocab)
    gen = module._select_model_move_stepwise(
        model=model_stepwise,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_d2",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
    )
    move_stepwise, debug_stepwise = _drive_model_move_stepwise(
        gen, model=model_stepwise, device=torch.device("cpu"), dtype=torch.float32
    )

    assert move_stepwise.uci() == move_sync.uci() == "d2d4"
    assert model_stepwise.forward_calls == model_sync.forward_calls
    assert len(debug_stepwise["value_search_d2_candidates"]) == len(
        debug_sync["value_search_d2_candidates"]
    )


def test_select_model_move_stepwise_matches_sync_for_value_search_halving():
    module = _load_eval_script_module()
    from imba_chess.eval.search import HalvingConfig

    move_vocab = _mini_vocab()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)
    halving_config = HalvingConfig(budget=6, top_m=2, rounds=2, lam=0.05)

    model_sync = _DummyHalvingModel(move_vocab)
    move_sync, debug_sync = module._select_model_move(
        model=model_sync,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_halving",
        value_rerank_top_k=2,
        value_rerank_lambda=0.05,
        debug_topk=0,
        halving_config=halving_config,
    )

    model_stepwise = _DummyHalvingModel(move_vocab)
    gen = module._select_model_move_stepwise(
        model=model_stepwise,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_halving",
        value_rerank_top_k=2,
        value_rerank_lambda=0.05,
        debug_topk=0,
        halving_config=halving_config,
    )
    move_stepwise, debug_stepwise = _drive_model_move_stepwise(
        gen, model=model_stepwise, device=torch.device("cpu"), dtype=torch.float32
    )

    assert move_stepwise.uci() == move_sync.uci() == "d2d4"
    assert model_stepwise.forward_calls == model_sync.forward_calls
    assert {row["move_uci"] for row in debug_stepwise["value_search_halving_candidates"]} == {
        row["move_uci"] for row in debug_sync["value_search_halving_candidates"]
    }


class _GreedyZeroLogitModel(torch.nn.Module):
    """All-zero logits/value_logits for ANY position -- select_greedy then
    deterministically picks the first legal-move index python-chess's own
    move generator yields (Python's max() returns the first max on ties),
    regardless of which side is to move or what the position looks like.
    Used only with model_move_policy="greedy", so value_logits is never
    read -- included anyway because the merged root_eval executor's
    multi-payload path (_split_root_output, --concurrent-games > 1) always
    slices "value_logits" unconditionally, independent of the eval policy.
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        total_tokens = int(batch["total_tokens"])
        out = {
            "logits": torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32),
            "value_logits": torch.zeros((total_tokens, 3), dtype=torch.float32),
        }
        if return_kv:
            out["kv_caches"] = [
                (torch.zeros(1, total_tokens, 1), torch.zeros(1, total_tokens, 1))
            ]
        return out


class _FakeSFEngine:
    """Fake `chess.engine.SimpleEngine` double for the scheduler-driver
    tests below: no subprocess, no real UCI protocol. Records every
    `configure`/`play`/`quit` call; `play` always returns the board's first
    legal move (deterministic, works for any position) unless `play_exc` is
    set, in which case every `play` call raises it instead.
    """

    def __init__(self, *, play_exc: BaseException | None = None) -> None:
        self.configure_calls: list[dict] = []
        self.play_calls: list[tuple[str, object]] = []
        self.quit_calls = 0
        self._play_exc = play_exc

    def configure(self, options):  # type: ignore[no-untyped-def]
        self.configure_calls.append(dict(options))

    def play(self, board, limit):  # type: ignore[no-untyped-def]
        self.play_calls.append((board.fen(), limit))
        if self._play_exc is not None:
            raise self._play_exc
        move = next(iter(board.legal_moves))
        return chess.engine.PlayResult(move, None)

    def quit(self):  # type: ignore[no-untyped-def]
        self.quit_calls += 1


def _patch_fake_stockfish(monkeypatch, *, play_exc: BaseException | None = None):
    """Monkeypatch `chess.engine.SimpleEngine.popen_uci` (called by
    `_run_segment`'s `_spawn_engine` closure) to hand out `_FakeSFEngine`
    instances instead of spawning a real Stockfish subprocess. Returns the
    list of spawned fake engines (append-order == EnginePool spawn order,
    i.e. slot index) so tests can assert on per-slot call counts.
    """
    spawned: list[_FakeSFEngine] = []

    def _fake_popen_uci(command, **kwargs):  # type: ignore[no-untyped-def]
        engine = _FakeSFEngine(play_exc=play_exc)
        spawned.append(engine)
        return engine

    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", staticmethod(_fake_popen_uci)
    )
    return spawned


def _run_fake_segment(module, *, games: int, concurrent_games: int, spawned, model=None):
    move_vocab = MoveVocab.build_static()
    board_state_encoder = BoardStateEncoder()
    model = model if model is not None else _GreedyZeroLogitModel(move_vocab)
    return module._run_segment(
        stockfish_path=Path("fake-stockfish-binary"),
        segment_options={"Threads": 1},
        segment_name="fake-segment",
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        games=games,
        max_plies=2,
        engine_limit=chess.engine.Limit(time=0.01),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_move_policy="greedy",
        value_rerank_top_k=1,
        value_rerank_lambda=0.0,
        opening_random_plies=0,
        debug_trace_games=0,
        debug_trace_max_plies=0,
        debug_topk=0,
        stockfish_label="fake",
        save_games_dir=None,
        concurrent_games=concurrent_games,
        halving_config=None,
    )


def test_run_segment_scheduler_g1_aggregates_alternates_colors_and_reuses_engine(
    monkeypatch,
):
    module = _load_eval_script_module()
    spawned = _patch_fake_stockfish(monkeypatch)

    summary = _run_fake_segment(module, games=3, concurrent_games=1, spawned=spawned)

    # Exactly one engine spawned and configured -- concurrent_games=1 means
    # every game reuses the single pool slot's engine sequentially, matching
    # today's one-engine-for-all-games-in-a-segment behavior.
    assert len(spawned) == 1
    assert spawned[0].configure_calls == [{"Threads": 1}]
    assert spawned[0].quit_calls == 1

    # 3 games, max_plies=2 -> every game has exactly one model turn and one
    # engine turn (order depends on model_color) -> 3 sf_move calls total on
    # the single reused engine.
    assert len(spawned[0].play_calls) == 3

    # Summary aggregation: 3 games, all incomplete (max_plies cuts them off
    # before any decisive/drawn result), color alternates by game_idx % 2.
    assert summary.games == 3
    assert summary.incomplete_games == 3
    assert summary.completed_games == 0
    assert summary.games_as_white == 2  # game_idx 0, 2
    assert summary.games_as_black == 1  # game_idx 1
    assert summary.total_plies == 3 * 2
    assert summary.model_turns == 3  # one model turn per game


def test_run_segment_rejects_concurrent_games_greater_than_one(monkeypatch):
    """Task 3 cutover: `_run_segment`'s in-process `EnginePool`/
    `BatchScheduler`/`make_sf_move_executor` generality used to ALSO serve
    `concurrent_games > 1` (this was the exact scenario the deleted
    `test_run_segment_scheduler_g2_uses_distinct_engines_per_slot` covered);
    that capability is gone, not merely unreachable from `main()` --
    `_run_segment` itself now fails fast on `concurrent_games > 1` so it can
    never again be scaled past one live game, from any caller."""
    module = _load_eval_script_module()
    spawned = _patch_fake_stockfish(monkeypatch)

    with pytest.raises(ValueError, match="only supports concurrent_games=1"):
        _run_fake_segment(module, games=2, concurrent_games=2, spawned=spawned)


def test_run_segment_scheduler_engine_exception_aborts_run(monkeypatch):
    module = _load_eval_script_module()
    spawned = _patch_fake_stockfish(
        monkeypatch, play_exc=RuntimeError("engine crashed mid-game")
    )

    with pytest.raises(RuntimeError, match="engine crashed mid-game"):
        _run_fake_segment(module, games=3, concurrent_games=1, spawned=spawned)

    # Fail-fast: the run must not silently continue past a failed game (no
    # partial/degraded summary is returned -- the exception propagates all
    # the way out of _run_segment). The engine is still spawned once (pool
    # construction happens before any game runs).
    assert len(spawned) == 1
    # This exception originates inside the sf_move EXECUTOR call (the fake
    # engine's play()), not inside _play_game's own code -- so it bypasses
    # _release_engine_on_finish's per-slot pool.release entirely (see that
    # function's docstring: executor-phase exceptions abandon the tick's
    # suspended game generator(s) without resuming/releasing them). The
    # actual cleanup path is _run_segment's unconditional
    # `finally: pool.close()`, which must still quit every pool engine even
    # though this run aborted mid-game -- regression-pin that here.
    assert spawned[0].quit_calls == 1


# ---------------------------------------------------------------------------
# Task 3: actor-mode orchestration (--concurrent-games > 1).
#
# Real `multiprocessing.get_context("spawn")` integration tests: actual
# worker PROCESSES, not threads and not direct in-process calls (unlike
# tests/test_actor_worker.py's own worker tests). This means
# `worker_config["engine"]["fake_engine_factory"]` must be picklable end to
# end -- a lambda/closure (fine for Task 1's in-process tests) is not; the
# factories below are module-level functions instead, resolvable by the
# freshly-spawned child process via a normal `import
# tests.test_eval_vs_stockfish` (works under `.venv/bin/pytest`, which puts
# the repo root on sys.path and multiprocessing's spawn bootstrap propagates
# that to the child -- verified empirically; a bare `python script.py`
# invocation would NOT have `tests` importable, but these tests only ever
# run under pytest).
# ---------------------------------------------------------------------------


class _ActorModeFakeEngine:
    """Always plays the board's first legal move; never raises."""

    def play(self, board, limit):  # type: ignore[no-untyped-def]
        move = next(iter(board.legal_moves))
        return SimpleNamespace(move=move)

    def quit(self) -> None:
        pass


def _actor_mode_fake_engine_factory() -> _ActorModeFakeEngine:
    """Top-level (picklable-under-spawn) fake-engine factory. See this
    section's module docstring for why a lambda closure -- Task 1's own
    `tests/test_actor_worker.py` fixture -- does not work for a REAL spawn
    integration test: `actor_worker._build_engine`'s docstring documents the
    same worker-side design note."""
    return _ActorModeFakeEngine()


class _ActorModeCrashingEngine:
    """Every `.play()` call raises -- used to make a worker process crash
    deterministically for the dead-worker fail-fast test below."""

    def play(self, board, limit):  # type: ignore[no-untyped-def]
        raise RuntimeError("actor mode test: engine crashed")

    def quit(self) -> None:
        pass


def _actor_mode_crashing_engine_factory() -> _ActorModeCrashingEngine:
    return _ActorModeCrashingEngine()


def _tiny_actor_mode_model(move_vocab: MoveVocab) -> HSTUChessModel:
    torch.manual_seed(3)
    config = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=8,
            attention_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            max_position_embeddings=64,
            enable_value_head=True,  # ActorInferenceServer requires this.
        ),
        move_vocab_size=len(move_vocab),
    )
    return HSTUChessModel(config).eval()


def test_actor_mode_two_workers_play_two_short_games_end_to_end():
    """2 REAL worker processes play 2 short (max_plies=2) games each against
    a fake Stockfish engine, served by a real (tiny CPU) `ActorInferenceServer`
    living in this test process -- summaries end up aggregated in game-index
    order via the hold-back buffer, and shutdown is clean: both workers join
    with exitcode 0 and no child process is left behind."""
    module = _load_eval_script_module()
    move_vocab = MoveVocab.build_static()
    model = _tiny_actor_mode_model(move_vocab)

    summary = module._run_segment_actor_mode(
        stockfish_path=Path("unused-fake-engine-path"),
        segment_options={},
        segment_name="actor-mode-test",
        model=model,
        games=2,
        max_plies=2,
        engine_limit=chess.engine.Limit(time=0.01),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_move_policy="greedy",
        value_rerank_top_k=1,
        value_rerank_lambda=0.0,
        opening_random_plies=0,
        seed=0,
        concurrent_games=2,
        vocab_path=STATIC_VOCAB_PATH,
        vocab_include_unk=False,
        board_state_config=asdict(BoardStateConfig()),
        halving_config=None,
        fake_engine_factory=_actor_mode_fake_engine_factory,
    )

    # max_plies=2 cuts both games off before either can complete; game_idx 0
    # is model=white (0 % 2 == 0), game_idx 1 is model=black -- one model
    # turn and one SF turn each, opposite order.
    assert summary.games == 2
    assert summary.incomplete_games == 2
    assert summary.completed_games == 0
    assert summary.games_as_white == 1
    assert summary.games_as_black == 1
    assert summary.total_plies == 4
    assert summary.model_turns == 2

    # Clean shutdown: no worker process left behind.
    assert multiprocessing.active_children() == []


def test_actor_mode_dead_worker_fails_fast_and_terminates_all_workers():
    """A crashing engine kills a worker process (an uncaught exception
    inside `run_eval_worker` exits that process nonzero); the orchestrator
    must detect this via pipe EOF, raise, and leave NO worker process alive
    behind it -- --concurrent-games > 1's counterpart to
    `test_run_segment_scheduler_engine_exception_aborts_run` above."""
    module = _load_eval_script_module()
    move_vocab = MoveVocab.build_static()
    model = _tiny_actor_mode_model(move_vocab)

    with pytest.raises(RuntimeError):
        module._run_segment_actor_mode(
            stockfish_path=Path("unused-fake-engine-path"),
            segment_options={},
            segment_name="actor-mode-crash-test",
            model=model,
            games=2,
            max_plies=4,
            engine_limit=chess.engine.Limit(time=0.01),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model_move_policy="greedy",
            value_rerank_top_k=1,
            value_rerank_lambda=0.0,
            opening_random_plies=0,
            seed=0,
            concurrent_games=2,
            vocab_path=STATIC_VOCAB_PATH,
            vocab_include_unk=False,
            board_state_config=asdict(BoardStateConfig()),
            halving_config=None,
            fake_engine_factory=_actor_mode_crashing_engine_factory,
        )

    # Fail-fast supervision: the run dies (nonzero, via the caller's
    # existing hard-exit wrapper in production) AND every worker process is
    # gone -- no leaked Stockfish-hosting process left behind.
    assert multiprocessing.active_children() == []


def test_run_segment_actor_mode_rejects_concurrent_games_one():
    module = _load_eval_script_module()
    move_vocab = MoveVocab.build_static()
    model = _tiny_actor_mode_model(move_vocab)

    with pytest.raises(ValueError, match="requires concurrent_games > 1"):
        module._run_segment_actor_mode(
            stockfish_path=Path("unused-fake-engine-path"),
            segment_options={},
            segment_name="actor-mode-guard-test",
            model=model,
            games=1,
            max_plies=2,
            engine_limit=chess.engine.Limit(time=0.01),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model_move_policy="greedy",
            value_rerank_top_k=1,
            value_rerank_lambda=0.0,
            opening_random_plies=0,
            seed=0,
            concurrent_games=1,
            vocab_path=STATIC_VOCAB_PATH,
            vocab_include_unk=False,
            board_state_config=asdict(BoardStateConfig()),
            halving_config=None,
            fake_engine_factory=_actor_mode_fake_engine_factory,
        )


class _FakeWorkerProcess:
    """Minimal `multiprocessing.Process`-shaped stand-in for
    `_join_and_verify_workers`/`_terminate_worker_processes` unit tests --
    deterministic, no real subprocess/scheduling race needed to exercise the
    "a later-indexed worker is still alive when an earlier one's check
    fails" scenario those functions must handle."""

    def __init__(self, *, pid: int, exitcode: int = 0, stays_alive: bool = False) -> None:
        self.pid = pid
        self.exitcode = exitcode
        self._stays_alive = stays_alive
        self._alive = True
        self.join_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout=None):  # type: ignore[no-untyped-def]
        self.join_calls += 1
        if not self._stays_alive:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False


def test_join_and_verify_workers_terminates_all_on_late_nonzero_exitcode():
    """Regression test for the code-review fix: `_join_and_verify_workers`
    walks `processes` in order and must terminate EVERY worker -- not just
    stop at the one whose check failed -- the moment ANY worker's exitcode
    is nonzero. `bad` (index 0) fails its check first; `still_running`
    (index 1) is never reached by the plain join loop, so it must be
    cleaned up via `_terminate_worker_processes`'s own full sweep, not by
    ever having its own `.join()`/exitcode check run.

    This is exactly what the ORIGINAL (pre-fix) code got wrong: its
    `exitcode != 0` branch raised directly without calling
    `_terminate_worker_processes` at all, so `still_running` would never
    have `.terminate()` called on it -- this test fails against that
    version (`still_running.terminate_calls == 0`) and passes against the
    fix.
    """
    module = _load_eval_script_module()
    bad = _FakeWorkerProcess(pid=101, exitcode=1)
    still_running = _FakeWorkerProcess(pid=102, exitcode=0, stays_alive=True)
    processes = [bad, still_running]

    with pytest.raises(RuntimeError, match="nonzero code"):
        module._join_and_verify_workers(processes)

    assert still_running.terminate_calls == 1
    assert not still_running.is_alive()


def test_join_and_verify_workers_terminates_all_when_worker_never_exits():
    """Same regression coverage for the sibling is_alive() branch (which
    already called _terminate_worker_processes before this fix, but is
    covered here too so both branches of _join_and_verify_workers stay
    pinned side by side)."""
    module = _load_eval_script_module()
    stuck = _FakeWorkerProcess(pid=201, exitcode=0, stays_alive=True)
    other = _FakeWorkerProcess(pid=202, exitcode=0, stays_alive=True)
    processes = [stuck, other]

    with pytest.raises(RuntimeError, match="did not exit within the grace period"):
        module._join_and_verify_workers(processes)

    assert stuck.terminate_calls == 1
    assert other.terminate_calls == 1

