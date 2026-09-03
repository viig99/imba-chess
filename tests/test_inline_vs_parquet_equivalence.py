from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from imba_chess.data.event_builder import EventBuilder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.rollout_store import RolloutRow, load_rollout_lookup, write_rollout_parquet
from imba_chess.data.stockfish_evals import CpToWdlCalibration


def _rows():
    common = {
        "WhiteElo": "2200",
        "BlackElo": "2200",
        "Termination": "Normal",
        "TimeControl": "600+0",
    }
    return [
        {
            **common,
            "Site": "https://lichess.org/equivalence-a",
            "Result": "1-0",
            "movetext": (
                "1. e4 { [%eval 0.20] } 1... e5 { [%eval 0.10] } "
                "2. Nf3 { [%eval #4] } 2... Nc6 { [%eval 0.25] } 1-0"
            ),
        },
        {
            **common,
            "Site": "https://lichess.org/equivalence-b",
            "Result": "1/2-1/2",
            "movetext": (
                "1. d4 { [%eval -0.10] } 1... d5 { [%eval 0.00] } "
                "2. c4 { [%eval #-3] } 2... e6 { [%eval 0.05] } 1/2-1/2"
            ),
        },
    ]


def _calibration():
    return CpToWdlCalibration(
        centers=(-100.0, 0.0, 100.0),
        probs=((0.7, 0.2, 0.1), (0.35, 0.3, 0.35), (0.1, 0.2, 0.7)),
        mate_probs={-1: (0.95, 0.03, 0.02), 1: (0.02, 0.03, 0.95)},
    )


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_inline_targets_match_existing_parquet_route(tmp_path, beta):
    dataset = LichessDataset(min_avg_elo=2000, parse_stockfish_evals=True)
    games = list(dataset.stream_from_rows(_rows()))
    calibration = _calibration()
    vocab = MoveVocab.build_from_games(games)

    rollout_rows = []
    for game in games:
        for ply, play in enumerate(game["plays"]):
            cp_stm = play.get("eval_cp_stm")
            mate_stm = play.get("eval_mate_stm")
            if cp_stm is None and mate_stm is None:
                continue
            p_loss, p_draw, p_win = calibration.wdl(cp_stm, mate_stm)
            stm_white = int(play["state"]["turn_id"]) == 0
            result_white = 1 if game["result"] == "1-0" else 0
            real_outcome_stm = result_white if stm_white else -result_white
            rollout_rows.append(
                RolloutRow(
                    game_id=game["game_id"],
                    ply=ply,
                    human_move_uci="",
                    human_move_backed_value=None,
                    real_outcome_stm=real_outcome_stm,
                    best_arm_move_uci="",
                    best_arm_backed_value=p_win - p_loss,
                    root_wdl_unsearched=(p_loss, p_draw, p_win),
                    arm_move_uci=(),
                    arm_backed_value=(),
                    arm_evals_spent=(),
                    arm_log_prior=(),
                    search_budget=0,
                    search_top_m=0,
                    search_max_depth=0,
                    checkpoint="external:test",
                )
            )

    target_path = tmp_path / "targets.parquet"
    write_rollout_parquet(rollout_rows, target_path)
    lookup = load_rollout_lookup(target_path)
    parquet_builder = EventBuilder(vocab, rollout_lookup=lookup, beta=beta)
    inline_builder = EventBuilder(vocab, eval_calibration=calibration, beta=beta)

    for game in games:
        parquet_sample = parquet_builder.build_game(game)
        inline_sample = inline_builder.build_game(game)
        assert inline_sample["has_rollout_value_target"] == parquet_sample[
            "has_rollout_value_target"
        ]
        np.testing.assert_allclose(
            inline_sample["value_target_soft"],
            parquet_sample["value_target_soft"],
            atol=1e-12,
            rtol=0.0,
        )
