from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from imba_chess.config import ModelConfig
from imba_chess.data.collate import collate_jagged_batch
from imba_chess.data.event_builder import EventBuilder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.rollout_store import RolloutRow, load_rollout_lookup, write_rollout_parquet
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


def _row():
    return {
        "Event": "Rated Blitz game",
        "Site": "https://lichess.org/e2e-example",
        "UTCDate": "2026-01-01",
        "UTCTime": "12:00:00",
        "White": "Alice",
        "Black": "Bob",
        "WhiteElo": "2200",
        "BlackElo": "2200",
        "Result": "1-0",
        "TimeControl": "300+0",
        "Termination": "Normal",
        "ECO": "C20",
        "Opening": "King's Pawn Game",
        "movetext": "1. e4 e5 2. Nf3 Nc6 1-0",
    }


def test_end_to_end_training_step_with_rollout_targets(tmp_path):
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])
    game_id = game["game_id"]

    rollout_row = RolloutRow(
        game_id=game_id,
        ply=1,
        human_move_uci=game["plays"][1]["move_uci"],
        human_move_backed_value=0.1,
        real_outcome_stm=-1,
        best_arm_move_uci=game["plays"][1]["move_uci"],
        best_arm_backed_value=0.4,
        root_wdl_unsearched=(0.3, 0.3, 0.4),
        arm_move_uci=(game["plays"][1]["move_uci"],),
        arm_backed_value=(0.4,),
        arm_evals_spent=(64,),
        arm_log_prior=(-0.2,),
        search_budget=128,
        search_top_m=1,
        search_max_depth=4,
        checkpoint="dummy.pt",
    )
    rollout_path = tmp_path / "rollouts.parquet"
    write_rollout_parquet([rollout_row], rollout_path)
    lookup = load_rollout_lookup(rollout_path)

    builder = EventBuilder(vocab, rollout_lookup=lookup, beta=0.7)
    sample = builder.build_game(game)
    assert sum(sample["has_rollout_value_target"]) == 1

    batch = collate_jagged_batch([sample])
    assert "value_target_soft" in batch
    assert "has_rollout_value_target" in batch

    model_cfg = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=16,
            attention_dim=16,
            num_heads=2,
            num_layers=1,
            max_position_embeddings=32,
            enable_value_head=True,
            value_loss_weight=0.2,
        ),
        move_vocab_size=len(vocab),
    )
    model = HSTUChessModel(model_cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(3):
        optimizer.zero_grad()
        out = model(batch, return_loss=True)
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["value_loss"])
        out["loss"].backward()
        optimizer.step()

    # The value head must actually have received gradients through the
    # soft-CE path (not silently skipped): check at least one of its
    # parameters moved.
    grad_norms = [
        p.grad.norm().item() for p in model.value_head.parameters() if p.grad is not None
    ]
    assert any(norm > 0.0 for norm in grad_norms)
