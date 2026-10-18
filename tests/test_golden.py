import json

import chess
import pytest

from imba_chess.eval.golden import read_labels, sample_games, score_info, validate_labels


def test_game_split_history_and_deduplication():
    import random

    def games():
        for seed in range(5):
            rng, board, moves = random.Random(seed), chess.Board(), []
            for _ in range(70):
                if board.is_game_over(claim_draw=True):
                    break
                move = rng.choice(list(board.legal_moves))
                moves.append(move.uci())
                board.push(move)
            yield dict(source="fixture", source_id=str(seed), metadata={}, moves=moves)
    a = sample_games([(games(), 3)], per_game=3)
    b = sample_games([(games(), 3)], per_game=3)
    assert a == b
    assert len(a["positions"]) == 9
    splits, seen = {}, set()
    for p in a["positions"]:
        board = chess.Board()
        for move in p["prefix_moves"]:
            board.push_uci(move)
        assert board.fen() == p["fen"]
        assert not board.is_game_over(claim_draw=True)
        key = " ".join(board.fen().split()[:4])
        assert key not in seen
        seen.add(key)
        assert splits.setdefault(p["game_id"], p["split"]) == p["split"]


def test_recovery_rejects_duplicates(tmp_path):
    path = tmp_path / "labels.jsonl"
    row = dict(position_id="a")
    path.write_text(json.dumps(row) + '\n{"position_id":')
    assert read_labels(path) == {"a": row}
    assert path.read_text().endswith(':')  # Read-only observers never truncate a live writer.
    assert read_labels(path, recover_partial=True) == {"a": row}
    assert path.read_text().endswith("\n")
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_labels(path)


def test_native_wdl_perspective_and_bound_rejection():
    import chess.engine

    info = dict(score=chess.engine.PovScore(chess.engine.Cp(50), chess.BLACK),
                wdl=chess.engine.PovWdl(chess.engine.Wdl(400, 500, 100), chess.BLACK))
    row = score_info(info, chess.WHITE)
    assert row["cp"] == -50
    assert row["wdl"] == [0.4, 0.5, 0.1]
    assert row["expectation"] == 0.35
    with pytest.raises(ValueError, match="bounded"):
        score_info(info | dict(lowerbound=True), chess.WHITE)
    with pytest.raises(ValueError, match="incomplete"):
        validate_labels(dict(positions=[dict(position_id="missing")]), {})


def test_analysis_does_not_keep_stale_bound_flags():
    import chess.engine
    from imba_chess.eval.golden import analyse_exact

    exact = dict(score=chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE),
                 wdl=chess.engine.PovWdl(chess.engine.Wdl(100, 850, 50), chess.WHITE))
    class Reports:
        def __enter__(self):
            return iter([exact | dict(lowerbound=True), exact])
        def __exit__(self, *args):
            pass
    class Engine:
        def configure(self, settings):
            pass
        def analysis(self, *args, **kwargs):
            return Reports()
    assert analyse_exact(Engine(), chess.Board(), 100)["cp"] == 30


@pytest.mark.extended
def test_stockfish_label_smoke():
    from imba_chess.eval.golden import start_engine, close_engines, label_position

    board = chess.Board()
    position = dict(position_id="start", prefix_moves=[], fen=board.fen())
    try:
        start_engine("/usr/bin/stockfish", 16)
        row = label_position((position, 1000, 1000))
        validate_labels(dict(positions=[position]), {"start": row})
    finally:
        close_engines()
