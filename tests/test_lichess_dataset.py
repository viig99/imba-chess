import datetime as dt
import json

import pytest

from imba_chess.data.lichess_dataset import LichessDataset


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


def test_move_parsing_yields_lean_records():
    dataset = LichessDataset(min_avg_elo=2000)
    games = list(dataset.stream_from_rows([_row()]))
    assert len(games) == 1

    game = games[0]
    assert game["game_id"] == "https://lichess.org/example"
    assert game["result"] == "1-0"
    assert game["white_elo"] == 2100
    assert game["black_elo"] == 1900

    plays = game["plays"]
    assert len(plays) == 4
    assert [play["move_uci"] for play in plays] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    # played_by_elo alternates white/black
    assert [play["played_by_elo"] for play in plays] == [2100, 1900, 2100, 1900]
    assert len(plays[0]["state"]["piece_ids"]) == 64
    assert plays[0]["state"]["turn_id"] == 0
    assert plays[1]["state"]["turn_id"] == 1
    assert plays[0]["state"]["castle_id"] == 15
    assert plays[0]["state"]["ep_file_id"] == 0


def test_output_is_json_serializable_with_typed_row_values():
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
    json.dumps(games[0])


def test_rejects_games_with_corrupt_movetext():
    dataset = LichessDataset(min_avg_elo=2000)
    # Qxe5 is illegal on move 2; the parser truncates there. The truncated
    # prefix must be rejected, not emitted with the full-game result label.
    rows = [_row(movetext="1. e4 e5 2. Qxe5 Nf6 1-0")]
    games = list(dataset.stream_from_rows(rows))

    assert games == []


def test_respects_max_seq_len_truncation():
    dataset = LichessDataset(min_avg_elo=2000, max_seq_len=2)
    games = list(dataset.stream_from_rows([_row()]))

    assert len(games) == 1
    assert len(games[0]["plays"]) == 2


def test_invalid_max_seq_len_raises():
    with pytest.raises(ValueError, match="max_seq_len must be >= 1"):
        LichessDataset(min_avg_elo=2000, max_seq_len=0)


def test_skips_games_with_zero_parsed_plies():
    dataset = LichessDataset(min_avg_elo=2000)
    rows = [_row(Site="https://lichess.org/bad", movetext="BAD_MOVE 1-0")]
    games = list(dataset.stream_from_rows(rows))

    assert games == []


def test_stream_from_rows_respects_max_games():
    dataset = LichessDataset(min_avg_elo=2000)
    rows = [
        _row(Site="https://lichess.org/g1", WhiteElo="2200", BlackElo="2200"),
        _row(Site="https://lichess.org/g2", WhiteElo="2200", BlackElo="2200"),
    ]
    games = list(dataset.stream_from_rows(rows, max_games=1))

    assert len(games) == 1
    assert games[0]["game_id"] == "https://lichess.org/g1"


def test_stream_applies_val_max_games(monkeypatch):
    rows = [
        _row(Site="https://lichess.org/v1", WhiteElo="2200", BlackElo="2200"),
        _row(Site="https://lichess.org/v2", WhiteElo="2200", BlackElo="2200"),
    ]

    def _fake_load_dataset(**kwargs):
        return rows

    monkeypatch.setattr("imba_chess.data.lichess_dataset.load_dataset", _fake_load_dataset)
    dataset = LichessDataset(
        min_avg_elo=2000,
        split="val",
        val_start_month="2025-08",
        val_end_month="2025-08",
        val_max_games=1,
    )
    games = list(dataset.stream())

    assert len(games) == 1
    assert games[0]["game_id"] == "https://lichess.org/v1"


class _FakeStream:
    """Minimal stand-in for a HF streaming dataset with filter/shuffle."""

    def __init__(self, rows):
        self.rows = rows
        self.shuffle_calls: list[dict] = []

    def filter(self, fn, input_columns=None):
        return self

    def shuffle(self, *, seed, buffer_size):
        self.shuffle_calls.append({"seed": seed, "buffer_size": buffer_size})
        return self

    def __iter__(self):
        return iter(self.rows)


def test_stream_shuffles_train_split_with_buffer(monkeypatch):
    fake = _FakeStream([_row(WhiteElo="2200", BlackElo="2200")])
    monkeypatch.setattr(
        "imba_chess.data.lichess_dataset.load_dataset", lambda **kwargs: fake
    )
    dataset = LichessDataset(
        min_avg_elo=2000,
        split="train",
        train_start_month="2025-07",
        train_end_month="2025-07",
        train_shuffle_buffer_size=5000,
        train_month_shuffle_seed=123,
    )
    games = list(dataset.stream())

    assert games
    assert fake.shuffle_calls == [{"seed": 123, "buffer_size": 5000}]


def test_stream_does_not_shuffle_val_split(monkeypatch):
    fake = _FakeStream([_row(WhiteElo="2200", BlackElo="2200")])
    monkeypatch.setattr(
        "imba_chess.data.lichess_dataset.load_dataset", lambda **kwargs: fake
    )
    dataset = LichessDataset(
        min_avg_elo=2000,
        split="val",
        val_start_month="2025-08",
        val_end_month="2025-08",
        train_shuffle_buffer_size=5000,
    )
    games = list(dataset.stream())

    assert games
    assert fake.shuffle_calls == []


def test_temporal_mode_uses_reverse_month_order(monkeypatch):
    captured: dict[str, object] = {}
    rows = [_row(Site="https://lichess.org/t1", WhiteElo="2200", BlackElo="2200")]

    def _fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return rows

    monkeypatch.setattr("imba_chess.data.lichess_dataset.load_dataset", _fake_load_dataset)
    dataset = LichessDataset(
        min_avg_elo=2000,
        split="train",
        train_start_month="2025-07",
        train_end_month="2025-09",
    )
    games = list(dataset.stream())

    assert games
    assert captured["path"] == "parquet"
    assert captured["split"] == "train"
    data_files = captured["data_files"]
    assert isinstance(data_files, dict)
    train_files = data_files["train"]
    assert "year=2025/month=09" in train_files[0]
    assert "year=2025/month=08" in train_files[1]
    assert "year=2025/month=07" in train_files[2]


def test_stream_shards_parquet_file_list(monkeypatch):
    captured: dict[str, object] = {}
    rows = [_row(Site="https://lichess.org/s1", WhiteElo="2200", BlackElo="2200")]

    def _fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return rows

    monkeypatch.setattr("imba_chess.data.lichess_dataset.load_dataset", _fake_load_dataset)
    dataset = LichessDataset(
        min_avg_elo=2000,
        split="train",
        train_start_month="2025-07",
        train_end_month="2025-10",
    )
    games = list(dataset.stream(shard_id=1, num_shards=2))

    assert games
    data_files = captured["data_files"]
    assert isinstance(data_files, dict)
    train_files = data_files["train"]
    # global reverse-month order is [10, 09, 08, 07]; shard 1 gets [09, 07]
    assert "year=2025/month=09" in train_files[0]
    assert "year=2025/month=07" in train_files[1]
    assert len(train_files) == 2


def test_time_control_filter_disabled_by_default():
    dataset = LichessDataset(min_avg_elo=2000)
    games = list(dataset.stream_from_rows([_row(TimeControl="60+0")]))
    assert len(games) == 1


def test_time_control_filter_drops_fast_and_unknown_games():
    dataset = LichessDataset(min_avg_elo=2000, min_time_control_sec=180)
    rows = [
        _row(Site="https://lichess.org/bullet", TimeControl="60+0"),
        _row(Site="https://lichess.org/bullet_inc", TimeControl="120+1"),
        _row(Site="https://lichess.org/blitz", TimeControl="180+0"),
        _row(Site="https://lichess.org/blitz_inc", TimeControl="60+3"),
        _row(Site="https://lichess.org/rapid", TimeControl="600+5"),
        _row(Site="https://lichess.org/corr", TimeControl="-"),
        _row(Site="https://lichess.org/unknown", TimeControl="?"),
    ]
    games = list(dataset.stream_from_rows(rows))
    kept = {game["game_id"] for game in games}
    assert kept == {
        "https://lichess.org/blitz",
        "https://lichess.org/blitz_inc",  # 60 + 40*3 = 180
        "https://lichess.org/rapid",
    }


def test_time_control_column_prefilter_matches_row_filter():
    dataset = LichessDataset(min_avg_elo=2000, min_time_control_sec=180)
    assert dataset._game_filter_from_columns("2100", "2100", "300+0")
    assert not dataset._game_filter_from_columns("2100", "2100", "60+0")
    assert not dataset._game_filter_from_columns("2100", "2100", "-")
    assert not dataset._game_filter_from_columns("1700", "1800", "300+0")
