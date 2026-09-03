"""The one-ply shift and the perspective flip in scripts/extract_lichess_evals.py.

Both are silent-corruption bugs: a target attached one ply early, or with the
wrong sign, still trains and still reports a plausible loss. It just teaches
the value head the opposite of what Stockfish said.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import chess
import chess.pgn
import pytest

_spec = importlib.util.spec_from_file_location(
    "extract_lichess_evals",
    Path(__file__).resolve().parent.parent / "scripts" / "extract_lichess_evals.py",
)
extract_lichess_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_lichess_evals)
extract_game = extract_lichess_evals.extract_game

PGN = (
    '1. e4 { [%eval 0.2] } 1... e5 { [%eval 0.1] } '
    '2. Nf3 { [%eval 0.3] } 2... Nc6 { [%eval 0.25] } '
    '3. Bb5 { [%eval 0.35] } 3... a6 { [%eval 0.4] } 1-0'
)


def test_eval_is_attached_one_ply_later_than_the_move_it_follows():
    """`1. e4 { [%eval 0.2] }` evaluates the position AFTER e4, which is the
    position stored at ply 1 (before Black's reply) -- not ply 0."""
    rows = extract_game(PGN, "g1", "1-0", max_seq_len=None)
    by_ply = {r["ply"]: r for r in rows}
    # Six moves -> six trailing evals, but the last describes the final
    # position, which has no ply of its own.
    assert sorted(by_ply) == [1, 2, 3, 4, 5]
    assert by_ply[1]["cp_white"] == 20.0    # the eval printed after e4
    assert by_ply[2]["cp_white"] == 10.0    # ...after e5
    assert by_ply[5]["cp_white"] == 35.0    # ...after Bb5
    assert 0 not in by_ply                  # ply 0 is the start position


def test_side_to_move_and_sign_match_a_real_board_replay():
    """Independent oracle: replay the game and assert the recorded
    side-to-move -- and therefore the sign -- agrees with python-chess."""
    rows = extract_game(PGN, "g1", "1-0", max_seq_len=None)
    game = chess.pgn.read_game(io.StringIO(PGN))
    boards = []
    board = game.board()
    node = game
    while node.variations:
        boards.append(board.turn)   # side to move BEFORE this ply
        node = node.variations[0]
        board.push(node.move)

    for r in rows:
        expected_white_to_move = boards[r["ply"]] == chess.WHITE
        assert r["stm_is_white"] == expected_white_to_move, r
        expected_sign = 1.0 if expected_white_to_move else -1.0
        assert r["cp_stm"] == r["cp_white"] * expected_sign, r


def test_outcome_is_recorded_from_the_side_to_move_pov():
    rows = extract_game(PGN, "g1", "1-0", max_seq_len=None)
    for r in rows:
        assert r["real_outcome_stm"] == (1 if r["stm_is_white"] else -1)
    drawn = extract_game(PGN, "g1", "1/2-1/2", max_seq_len=None)
    assert all(r["real_outcome_stm"] == 0 for r in drawn)


def test_mate_scores_are_kept_separate_and_signed():
    pgn = '1. e4 { [%eval 0.2] } 1... f6 { [%eval #3] } 2. Nc3 { [%eval 1.0] } 1-0'
    rows = {r["ply"]: r for r in extract_game(pgn, "g", "1-0", max_seq_len=None)}
    # "#3" trails Black's 1...f6 (ply 1) so it lands on ply 2, White to move.
    assert rows[2]["mate_white"] == 3
    assert rows[2]["mate_stm"] == 3
    assert rows[2]["cp_white"] is None      # never folded into the cp scale
    # And ply 1 (Black to move) flips a White-POV cp score.
    assert rows[1]["cp_white"] == 20.0 and rows[1]["cp_stm"] == -20.0


def test_unannotated_and_broken_games_yield_nothing():
    assert extract_game("1. e4 e5 2. Nf3 1-0", "g", "1-0", max_seq_len=None) == []
    assert extract_game("1. e4 { [%eval 0.2] } 1... Qxq9 1-0", "g", "1-0",
                        max_seq_len=None) == []
    assert extract_game(PGN, "g", "*", max_seq_len=None) == []


def test_max_seq_len_truncation_matches_the_dataset_walk():
    """LichessDataset._extract_plays stops at max_seq_len; ply indices past
    the cut do not exist, so no eval may be emitted for them."""
    rows = extract_game(PGN, "g", "1-0", max_seq_len=3)
    assert sorted(r["ply"] for r in rows) == [1, 2]


def _annotated_row(movetext=PGN):
    return {
        "Site": "https://lichess.org/annotated",
        "Result": "1-0",
        "WhiteElo": "2200",
        "BlackElo": "2200",
        "Termination": "Normal",
        "TimeControl": "600+0",
        "movetext": movetext,
    }


@pytest.mark.parametrize("max_seq_len", [None, 3])
def test_inline_dataset_capture_matches_offline_extractor(max_seq_len):
    from imba_chess.data.lichess_dataset import LichessDataset

    dataset = LichessDataset(
        min_avg_elo=2000,
        max_seq_len=max_seq_len,
        parse_stockfish_evals=True,
    )
    game = list(dataset.stream_from_rows([_annotated_row()]))[0]
    expected = {
        row["ply"]: row
        for row in extract_game(PGN, game["game_id"], "1-0", max_seq_len)
    }
    for ply, play in enumerate(game["plays"]):
        row = expected.get(ply)
        assert play["eval_cp_stm"] == (None if row is None else row["cp_stm"])
        assert play["eval_mate_stm"] == (
            None if row is None else row["mate_stm"]
        )


def test_inline_eval_capture_is_opt_in():
    from imba_chess.data.lichess_dataset import LichessDataset

    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_annotated_row()]))[0]
    assert all("eval_cp_stm" not in play for play in game["plays"])
    assert all("eval_mate_stm" not in play for play in game["plays"])


# ── dataset.local_corpus_path routing ───────────────────────────────────────

def _tiny_corpus(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"Site": "https://lichess.org/aaa", "Result": "1-0",
         "WhiteElo": "2100", "BlackElo": "2100", "Termination": "Normal",
         "TimeControl": "600+0", "movetext": "1. e4 e5 2. Nf3 Nc6 1-0"},
    ]), path)
    return path


def _dataset(tmp_path, **kwargs):
    from imba_chess.data.lichess_dataset import LichessDataset
    return LichessDataset(
        min_avg_elo=2000, split="train",
        local_corpus_path=str(_tiny_corpus(tmp_path)), **kwargs
    )


def test_local_corpus_path_replaces_the_hf_stream(tmp_path):
    games = list(_dataset(tmp_path).stream())
    assert len(games) == 1
    assert games[0]["game_id"] == "https://lichess.org/aaa"
    assert [p["move_uci"] for p in games[0]["plays"]] == [
        "e2e4", "e7e5", "g1f3", "b8c6"
    ]


def test_sharding_a_local_corpus_is_refused(tmp_path):
    """A materialized corpus is one captured stream. Slicing it per worker
    would hand each a different subsequence than upstream sharding, silently
    mis-aligning the (game_id, ply) keys -- the 2026-07-25 sharding bug's
    failure mode. It must raise, not quietly return a shard."""
    import pytest as _pytest
    ds = _dataset(tmp_path)
    with _pytest.raises(ValueError, match="cannot be sharded"):
        list(ds.stream(shard_id=0, num_shards=4))
    # num_shards == 1 is the num_workers=0 case and must still work.
    assert len(list(ds.stream(shard_id=0, num_shards=1))) == 1
