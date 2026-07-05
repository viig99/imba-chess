from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import chess
import pytest

torch = pytest.importorskip("torch")

from imba_chess.config import ModelConfig, RepoConfig
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


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
        module._load_model(
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


class _StubValueNet(torch.nn.Module):
    """Constant WDL logits: every position looks equally winning (stm POV)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch):  # type: ignore[no-untyped-def]
        self.calls += 1
        batch_size = int(batch["turn_id"].numel())
        logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        logits[:, 2] = 10.0  # p(win) ~ 1 for the side to move, always
        return logits


def _rerank_move_with_net(value_net, alpha):
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    model = _DummyValueRerankModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)
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
        value_net=value_net,
        value_net_alpha=alpha,
    )
    return move


def test_value_net_alpha_zero_reproduces_model_head():
    net = _StubValueNet()
    # alpha=0: model head decides (d2d4 per the dummy's value scheme), and
    # the net must not even be called.
    move = _rerank_move_with_net(net, alpha=0.0)
    assert move.uci() == "d2d4"
    assert net.calls == 0


def test_value_net_alpha_one_uses_net_values():
    net = _StubValueNet()
    # alpha=1: the constant net makes all candidates equal; the policy-prior
    # tiebreak picks the higher-prior move e2e4 instead of the model head's
    # d2d4 preference.
    move = _rerank_move_with_net(net, alpha=1.0)
    assert move.uci() == "e2e4"
    assert net.calls == 1


def test_value_net_alpha_half_blends():
    module = _load_eval_script_module()
    # Blend math is checked directly on the evaluator with a stub decode
    # model: model scalar -1 (loss), net scalar ~+1 (win) -> ~0 at alpha=.5.
    move_vocab = _mini_vocab()
    model = _DummyValueRerankModel(move_vocab)
    encoder = BoardStateEncoder()
    board = chess.Board()
    history = module._SequenceHistory(move_vocab=move_vocab, board_state_encoder=encoder)
    root_batch = history.build_batch_for_current_position(board)
    prefill = model(root_batch, return_loss=False, return_kv=True)
    net = _StubValueNet()
    evaluator = module.CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=prefill["kv_caches"],
        prefix_len=int(root_batch["total_tokens"]),
        value_net=net,
        value_net_alpha=0.5,
    )
    move = chess.Move.from_uci("d2d4")
    handle = evaluator.extend(None, board, move)
    board1 = board.copy()
    board1.push(move)
    (result,) = evaluator.evaluate([(handle, board1)])
    # Dummy model head: d2d4 node -> value_logits [4,0,0] -> scalar ~ -0.96.
    # Stub net: scalar ~ +1. Blend at 0.5 ~ 0.02.
    assert -0.2 < result.value_stm < 0.2
