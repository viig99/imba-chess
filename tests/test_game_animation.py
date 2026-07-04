from __future__ import annotations

import json

import chess
import chess.pgn

from imba_chess.eval.game_animation import GameAnimator


def _short_game() -> chess.pgn.Game:
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(move_uci)
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "imba-chess"
    game.headers["Black"] = "Stockfish (elo=1400)"
    game.headers["Result"] = "*"
    return game


def _metadata() -> dict[str, str]:
    return {
        "white": "imba-chess",
        "black": "Stockfish (elo=1400)",
        "result": "*",
        "segment": "sf_elo_1400",
        "ply_count": "3",
    }


def _extract_frames(html: str) -> list[dict]:
    start = html.index("const FRAMES = ") + len("const FRAMES = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def test_render_html_embeds_frames_moves_and_metadata():
    game = _short_game()
    html = GameAnimator().render_html(game, metadata=_metadata())

    assert html.startswith("<!doctype html>")
    assert "imba-chess vs Stockfish (elo=1400)" in html
    assert "Segment: sf_elo_1400" in html
    assert "Plies: 3" in html

    frames = _extract_frames(html)
    assert len(frames) == 4  # start position + 3 plies
    assert frames[0]["san"] is None
    assert [frame["san"] for frame in frames[1:]] == ["e4", "e5", "Nf3"]
    assert all("<svg" in frame["svg"] for frame in frames)


def test_render_html_handles_game_with_no_moves():
    board = chess.Board()
    game = chess.pgn.Game.from_board(board)
    metadata = {
        "white": "imba-chess",
        "black": "Stockfish (full strength)",
        "result": "*",
        "segment": "sf_full_strength",
        "ply_count": "0",
    }

    html = GameAnimator().render_html(game, metadata=metadata)

    frames = _extract_frames(html)
    assert len(frames) == 1
    assert frames[0]["san"] is None


def test_save_writes_html_file_and_creates_parent_dirs(tmp_path):
    game = _short_game()
    metadata = _metadata()
    output_path = tmp_path / "nested" / "game.html"

    GameAnimator().save(output_path, game, metadata=metadata)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == GameAnimator().render_html(
        game, metadata=metadata
    )
