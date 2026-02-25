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


class _DummyValueRerankModel(torch.nn.Module):
    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        vocab_size = len(self.move_vocab)
        logits = torch.zeros((total_tokens, vocab_size), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)

        if int(batch["num_games"]) == 1:
            last = total_tokens - 1
            logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
            logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
            value_logits[last, 1] = 1.0
        else:
            seq_offsets = batch["seq_offsets"]
            last_positions = seq_offsets[1:] - 1
            prev_ids = batch["prev_move_id"][last_positions]
            for row_idx, last_pos in enumerate(last_positions.tolist()):
                move_id = int(prev_ids[row_idx].item())
                if move_id == self.move_vocab.token_to_id["e2e4"]:
                    value_logits[last_pos] = torch.tensor([0.0, 0.0, 4.0])
                elif move_id == self.move_vocab.token_to_id["d2d4"]:
                    value_logits[last_pos] = torch.tensor([4.0, 0.0, 0.0])

        return {"logits": logits, "value_logits": value_logits}


class _DummyNoValueModel(torch.nn.Module):
    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab

    def forward(self, batch, *, block_mask=None, return_loss=False):  # type: ignore[no-untyped-def]
        total_tokens = int(batch["total_tokens"])
        vocab_size = len(self.move_vocab)
        logits = torch.zeros((total_tokens, vocab_size), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 1.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 0.5
        return {"logits": logits}


class _DummyValueSearchD2Model(torch.nn.Module):
    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        vocab_size = len(self.move_vocab)
        logits = torch.zeros((total_tokens, vocab_size), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        seq_offsets = batch["seq_offsets"]
        last_positions = seq_offsets[1:] - 1
        prev_move_id = batch["prev_move_id"]

        if int(batch["num_games"]) == 1:
            last = int(last_positions[0].item())
            logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
            logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
            value_logits[last, 1] = 1.0
            return {"logits": logits, "value_logits": value_logits}

        for game_idx, last_pos_tensor in enumerate(last_positions):
            start = int(seq_offsets[game_idx].item())
            end = int(seq_offsets[game_idx + 1].item())
            last_pos = int(last_pos_tensor.item())
            seq_prev = prev_move_id[start:end]
            last_prev = int(seq_prev[-1].item())

            if last_prev in {
                self.move_vocab.token_to_id["e2e4"],
                self.move_vocab.token_to_id["d2d4"],
            }:
                logits[last_pos, self.move_vocab.token_to_id["e7e5"]] = 3.0
                logits[last_pos, self.move_vocab.token_to_id["d7d5"]] = 2.5
                value_logits[last_pos, 1] = 1.0
                continue

            if last_prev in {
                self.move_vocab.token_to_id["e7e5"],
                self.move_vocab.token_to_id["d7d5"],
            } and int(seq_prev.numel()) >= 2:
                root_prev = int(seq_prev[-2].item())
                if (
                    root_prev == self.move_vocab.token_to_id["e2e4"]
                    and last_prev == self.move_vocab.token_to_id["e7e5"]
                ):
                    value_logits[last_pos] = torch.tensor([4.0, 0.0, 0.0])
                elif (
                    root_prev == self.move_vocab.token_to_id["e2e4"]
                    and last_prev == self.move_vocab.token_to_id["d7d5"]
                ):
                    value_logits[last_pos] = torch.tensor([2.0, 1.0, 0.0])
                elif (
                    root_prev == self.move_vocab.token_to_id["d2d4"]
                    and last_prev == self.move_vocab.token_to_id["e7e5"]
                ):
                    value_logits[last_pos] = torch.tensor([0.0, 1.0, 2.0])
                elif (
                    root_prev == self.move_vocab.token_to_id["d2d4"]
                    and last_prev == self.move_vocab.token_to_id["d7d5"]
                ):
                    value_logits[last_pos] = torch.tensor([0.0, 0.0, 4.0])
                else:
                    value_logits[last_pos, 1] = 1.0

        return {"logits": logits, "value_logits": value_logits}


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
        history=history,
        board=board,
        move_vocab=move_vocab,
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_rerank",
        sample_temperature=1.0,
        sample_top_k=0,
        sample_top_p=1.0,
        value_rerank_top_k=2,
        value_rerank_lambda=1.0,
        debug_topk=0,
    )

    assert move.uci() == "d2d4"
    assert model.forward_calls == 2  # current-state + single batched candidate eval
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
            history=history,
            board=board,
            move_vocab=move_vocab,
            device=torch.device("cpu"),
            dtype=torch.float32,
            policy="value_rerank",
            sample_temperature=1.0,
            sample_top_k=0,
            sample_top_p=1.0,
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
        history=history,
        board=board,
        move_vocab=move_vocab,
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_d2",
        sample_temperature=1.0,
        sample_top_k=0,
        sample_top_p=1.0,
        value_rerank_top_k=2,
        value_rerank_lambda=2.0,
        debug_topk=0,
    )

    assert move.uci() == "d2d4"
    assert model.forward_calls == 3  # root + batched depth-1 + batched depth-2
    assert debug["policy"] == "value_search_d2"
    rows = debug["value_search_d2_candidates"]
    assert len(rows) == 2
    by_move = {str(row["move_uci"]): row for row in rows}
    assert float(by_move["d2d4"]["worst_reply_value"]) > float(
        by_move["e2e4"]["worst_reply_value"]
    )


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
            history=history,
            board=board,
            move_vocab=move_vocab,
            device=torch.device("cpu"),
            dtype=torch.float32,
            policy="value_search_d2",
            sample_temperature=1.0,
            sample_top_k=0,
            sample_top_p=1.0,
            value_rerank_top_k=2,
            value_rerank_lambda=1.0,
            debug_topk=0,
        )
