import pytest

from imba_chess.data.event_builder import BOS_TOKEN_ID, EventBuilder, TARGET_IGNORE_INDEX
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.rollout_store import RolloutRow


def _row():
    return {
        "Event": "Rated Blitz game",
        "Site": "https://lichess.org/example",
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


def test_event_builder_builds_bos_plus_plies():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])

    builder = EventBuilder(vocab)
    sample = builder.build_game(game)

    # 4 plies + BOS
    assert len(sample["seq_token_id"]) == 5
    assert sample["game_result_white"] == 1
    assert sample["seq_token_id"][0] == BOS_TOKEN_ID
    assert sample["target_move_id"][0] == TARGET_IGNORE_INDEX
    assert all(token_id != TARGET_IGNORE_INDEX for token_id in sample["target_move_id"][1:])
    assert sample["prev_move_id"][1] == vocab.start_id
    assert sample["played_by_elo"][0] == 0
    assert len(sample["played_by_elo"]) == len(sample["seq_token_id"])
    assert len(sample["piece_ids"][1]) == 64


def test_event_builder_without_rollout_lookup_omits_new_keys():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])

    builder = EventBuilder(vocab)
    sample = builder.build_game(game)

    assert "value_target_soft" not in sample
    assert "has_rollout_value_target" not in sample
    assert "policy_kl_arm_ids" not in sample
    assert "policy_kl_arm_qhat" not in sample
    assert "policy_kl_arm_mask" not in sample
    assert "has_rollout_policy_target" not in sample


def test_event_builder_with_rollout_lookup_blends_value_target():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])
    game_id = game["game_id"]

    # Rollout for ply 1 (the second play, token index 2) only.
    rollout_row = RolloutRow(
        game_id=game_id,
        ply=1,
        human_move_uci=game["plays"][1]["move_uci"],
        human_move_backed_value=0.2,
        real_outcome_stm=1,
        best_arm_move_uci=game["plays"][1]["move_uci"],
        best_arm_backed_value=0.6,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=(game["plays"][1]["move_uci"],),
        arm_backed_value=(0.6,),
        arm_evals_spent=(100,),
        arm_log_prior=(-0.1,),
        search_budget=256,
        search_top_m=1,
        search_max_depth=4,
        checkpoint="dummy.pt",
    )
    lookup = {(game_id, 1): rollout_row}

    builder = EventBuilder(vocab, rollout_lookup=lookup, beta=1.0)
    sample = builder.build_game(game)

    assert len(sample["value_target_soft"]) == len(sample["seq_token_id"])
    assert len(sample["has_rollout_value_target"]) == len(sample["seq_token_id"])
    # Only token 2 (== ply 1) has a rollout row; every other token (BOS at 0,
    # ply 0 at 1, ply 2 at 3, ply 3 at 4) must be untouched.
    for token_idx in range(len(sample["seq_token_id"])):
        if token_idx == 2:
            continue
        assert sample["has_rollout_value_target"][token_idx] == 0
        assert sample["value_target_soft"][token_idx] == [0.0, 0.0, 0.0]
    # Token 2 == ply 1 gets the blended target (beta=1.0 -> pure searched_vec).
    assert sample["has_rollout_value_target"][2] == 1
    assert abs(sum(sample["value_target_soft"][2]) - 1.0) < 1e-9
    assert sample["value_target_soft"][2][1] == pytest.approx(0.3)


def test_event_builder_with_rollout_lookup_builds_policy_kl_arm_targets():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])
    game_id = game["game_id"]

    rollout_row = RolloutRow(
        game_id=game_id,
        ply=1,
        human_move_uci=game["plays"][1]["move_uci"],
        human_move_backed_value=0.2,
        real_outcome_stm=1,
        best_arm_move_uci=game["plays"][1]["move_uci"],
        best_arm_backed_value=0.6,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=(game["plays"][1]["move_uci"], game["plays"][0]["move_uci"]),
        arm_backed_value=(0.6, -0.2),
        arm_evals_spent=(100, 50),
        arm_log_prior=(-0.1, -0.4),
        search_budget=256,
        search_top_m=2,
        search_max_depth=4,
        checkpoint="dummy.pt",
    )
    lookup = {(game_id, 1): rollout_row}

    builder = EventBuilder(vocab, rollout_lookup=lookup, beta=1.0)
    sample = builder.build_game(game)

    seq_len = len(sample["seq_token_id"])
    assert len(sample["policy_kl_arm_ids"]) == seq_len
    assert len(sample["policy_kl_arm_qhat"]) == seq_len
    assert len(sample["policy_kl_arm_mask"]) == seq_len
    assert len(sample["has_rollout_policy_target"]) == seq_len

    # Only token 2 (== ply 1) has a rollout row.
    for token_idx in range(seq_len):
        if token_idx == 2:
            continue
        assert sample["has_rollout_policy_target"][token_idx] == 0
        assert sample["policy_kl_arm_mask"][token_idx] == [False] * 24

    assert sample["has_rollout_policy_target"][2] == 1
    assert sample["policy_kl_arm_mask"][2][:2] == [True, True]
    assert sample["policy_kl_arm_mask"][2][2:] == [False] * 22
    expected_id_0 = vocab.token_to_id[game["plays"][1]["move_uci"]]
    expected_id_1 = vocab.token_to_id[game["plays"][0]["move_uci"]]
    assert sample["policy_kl_arm_ids"][2][0] == expected_id_0
    assert sample["policy_kl_arm_ids"][2][1] == expected_id_1
    assert sample["policy_kl_arm_qhat"][2][0] == pytest.approx(0.6)
    assert sample["policy_kl_arm_qhat"][2][1] == pytest.approx(-0.2)
