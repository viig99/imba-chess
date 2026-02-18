import datetime as dt
import json

import pytest

from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.models import GameRecord


def _row(**overrides):
    base = {
        "Event": "Rated Blitz game",
        "Site": "https://lichess.org/example",
        "UTCDate": "2026-01-01",
        "UTCTime": "12:00:00",
        "White": "Alice",
        "Black": "Bob",
        "WhiteElo": "2100",
        "BlackElo": "1900",
        "Result": "1-0",
        "TimeControl": "300+0",
        "Termination": "Normal",
        "ECO": "C20",
        "Opening": "King's Pawn Game",
        "movetext": (
            "1. e4 {[%clk 0:05:00]} e5 {[%clk 0:05:00]} "
            "2. Nf3 {[%clk 0:04:50]} Nc6 {[%clk 0:04:58]} 1-0"
        ),
    }
    base.update(overrides)
    return base


def test_elo_filter_uses_average_threshold():
    dataset = LichessDataset(min_avg_elo=2000)
    rows = [
        _row(WhiteElo="2200", BlackElo="2100"),
        _row(Site="https://lichess.org/low", WhiteElo="1700", BlackElo="1800"),
    ]
    games = list(dataset.stream_from_rows(rows))
    assert len(games) == 1
    assert games[0]["game_id"] != "https://lichess.org/low"


def test_move_and_clock_parsing():
    dataset = LichessDataset(min_avg_elo=2000)
    games = list(dataset.stream_from_rows([_row()]))
    assert len(games) == 1

    game = games[0]
    plays = game["plays"]
    assert game["num_plies"] == 4
    assert plays[0]["play_id"] == 1
    assert plays[0]["move_uci"] == "e2e4"
    assert len(plays[0]["state"]["piece_ids"]) == 64
    assert plays[0]["state"]["turn_id"] == 0
    assert plays[0]["state"]["castle_id"] == 15
    assert plays[0]["state"]["ep_file_id"] == 0
    assert plays[0]["outcome_for_player"] == "win"
    assert plays[1]["outcome_for_player"] == "loss"
    assert plays[2]["time_taken_seconds"] == 10.0
    assert game["winner_player"] == "Alice"
    assert game["loser_player"] == "Bob"


def test_typed_date_time_values_are_supported_and_json_serializable():
    dataset = LichessDataset(min_avg_elo=2000)
    games = list(
        dataset.stream_from_rows(
            [
                _row(
                    UTCDate=dt.date(2026, 1, 1),
                    UTCTime=dt.time(12, 0, 0),
                )
            ]
        )
    )

    assert games
    assert games[0]["metadata"]["utc_date"] == "2026-01-01"
    assert games[0]["metadata"]["utc_time"] == "12:00:00"
    json.dumps(games[0])


def test_can_return_dataclasses():
    dataset = LichessDataset(min_avg_elo=2000, return_dataclasses=True)
    games = list(dataset.stream_from_rows([_row(WhiteElo="2200", BlackElo="2200")]))

    assert games
    assert isinstance(games[0], GameRecord)
    assert games[0].plays[0].state.turn_id == 0


def test_respects_max_seq_len_truncation():
    dataset = LichessDataset(min_avg_elo=2000, max_seq_len=2)
    games = list(dataset.stream_from_rows([_row()]))

    assert len(games) == 1
    assert games[0]["num_plies"] == 2
    assert len(games[0]["plays"]) == 2


def test_invalid_max_seq_len_raises():
    with pytest.raises(ValueError, match="max_seq_len must be >= 1"):
        LichessDataset(min_avg_elo=2000, max_seq_len=0)


def test_skips_games_with_zero_parsed_plies():
    dataset = LichessDataset(min_avg_elo=2000)
    rows = [_row(Site="https://lichess.org/bad", movetext="BAD_MOVE 1-0")]
    games = list(dataset.stream_from_rows(rows))

    assert games == []
