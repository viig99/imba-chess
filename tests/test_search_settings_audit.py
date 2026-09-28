import chess
import chess.pgn
import pytest

from scripts.audit_search_settings import paired_summary, sample_positions, setting_key


def _write(tmp_path, name, white, moves):
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = "Stockfish" if white == "imba-chess" else "imba-chess"
    node = game
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    path = tmp_path / name
    path.write_text(str(game))
    return str(path)


def test_samples_only_model_to_move_after_min_ply_and_dedupes(tmp_path):
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]
    a = _write(tmp_path, "a.pgn", "imba-chess", moves)
    b = _write(tmp_path, "b.pgn", "imba-chess", moves)
    black = _write(tmp_path, "c.pgn", "Stockfish", moves)
    white_positions = sample_positions([a, b], per_game=2, min_ply=2)
    assert len(white_positions) == 2  # second game duplicates the first
    assert all(len(p["prefix"]) % 2 == 0 and len(p["prefix"]) >= 2 for p in white_positions)
    black_positions = sample_positions([black], per_game=10, min_ply=0)
    assert all(len(p["prefix"]) % 2 == 1 for p in black_positions)


def test_paired_difference_is_against_baseline_by_position():
    base, other = setting_key(50, 16), setting_key(10, 5)
    m = lambda e, s: dict(selected_expectation=e, selected_regret=1 - e, changed_move=0.0, target_gain=0.0, selected=s)
    rows = [dict(position_id="p", setting=base, metrics=m(0.5, "a")),
            dict(position_id="q", setting=base, metrics=m(0.2, "b")),
            dict(position_id="p", setting=other, metrics=m(0.7, "c")),
            dict(position_id="q", setting=other, metrics=m(0.2, "b"))]
    s = paired_summary(rows, base)
    assert s[other]["diff_vs_baseline"] == pytest.approx(0.1)
    assert s[other]["same_move_as_baseline"] == pytest.approx(0.5)
    assert s[base]["diff_vs_baseline"] == 0
