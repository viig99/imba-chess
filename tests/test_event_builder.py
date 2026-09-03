import pytest

from imba_chess.data.event_builder import BOS_TOKEN_ID, EventBuilder, TARGET_IGNORE_INDEX
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.stockfish_evals import winpercent_wdl


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


def _annotated_game():
    row = _row()
    row["movetext"] = (
        "1. e4 { [%eval 0.20] } 1... e5 { [%eval 0.10] } "
        "2. Nf3 { [%eval 0.30] } 2... Nc6 { [%eval 0.25] } 1-0"
    )
    dataset = LichessDataset(min_avg_elo=2000, parse_stockfish_evals=True)
    return list(dataset.stream_from_rows([row]))[0]


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


def test_event_builder_masks_value_target_without_evals():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])

    sample = EventBuilder(vocab).build_game(game)

    assert sample["has_value_target"] == [0, 0, 0, 0, 0]
    assert sample["value_target"] == [[0.0, 0.0, 0.0]] * 5


def test_event_builder_builds_winpercent_value_targets_from_evals():
    game = _annotated_game()
    vocab = MoveVocab.build_from_games([game])

    sample = EventBuilder(vocab).build_game(game)

    # BOS and the first ply never carry an eval (the comment after move k
    # targets the state before move k+1); plies 2..4 do.
    assert sample["has_value_target"] == [0, 0, 1, 1, 1]
    assert sample["value_target"][0] == [0.0, 0.0, 0.0]
    assert sample["value_target"][1] == [0.0, 0.0, 0.0]
    # Ply 2 (Black to move) sees White's +0.20 as -20 cp from its own side.
    assert sample["value_target"][2] == pytest.approx(list(winpercent_wdl(-20.0, None)))
    # Ply 3 (White to move) sees the +0.10 after 1...e5 as +10 cp.
    assert sample["value_target"][3] == pytest.approx(list(winpercent_wdl(10.0, None)))
    assert sample["value_target"][4] == pytest.approx(list(winpercent_wdl(-30.0, None)))
    for token in sample["value_target"]:
        assert token[1] == 0.0
    assert len(sample["value_target"]) == len(sample["seq_token_id"])
    assert len(sample["has_value_target"]) == len(sample["seq_token_id"])


def test_event_builder_maps_mate_evals_to_ceiling():
    row = _row()
    row["movetext"] = "1. e4 { [%eval #5] } 1... e5 { [%eval #-3] } 2. Nf3 Nc6 1-0"
    dataset = LichessDataset(min_avg_elo=2000, parse_stockfish_evals=True)
    game = list(dataset.stream_from_rows([row]))[0]
    vocab = MoveVocab.build_from_games([game])

    sample = EventBuilder(vocab).build_game(game)

    assert sample["has_value_target"] == [0, 0, 1, 1, 0]
    # White mates in 5 -> Black to move at ply 2 is losing.
    assert sample["value_target"][2] == pytest.approx(list(winpercent_wdl(None, -1)))
    # Black mates in 3 after 1...e5 -> White to move at ply 3 is losing.
    assert sample["value_target"][3] == pytest.approx(list(winpercent_wdl(None, -1)))
