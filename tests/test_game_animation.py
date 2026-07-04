from __future__ import annotations

import json

import chess
import chess.pgn

from imba_chess.eval.game_animation import render_game_html


def _short_game() -> chess.pgn.Game:
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(move_uci)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "sf_elo_1400"
    game.headers["White"] = "imba-chess"
    game.headers["Black"] = "Stockfish (elo=1400)"
    game.headers["Result"] = "*"
    return game


def _extract_frames(html: str) -> list[dict]:
    start = html.index("const FRAMES = ") + len("const FRAMES = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def test_render_game_html_embeds_frames_moves_and_metadata():
    html = render_game_html(_short_game())

    assert html.startswith("<!doctype html>")
    assert "imba-chess vs Stockfish (elo=1400)" in html
    assert "Event: sf_elo_1400" in html
    assert "Plies: 3" in html

    frames = _extract_frames(html)
    assert len(frames) == 4  # start position + 3 plies
    assert frames[0]["san"] is None
    assert [frame["san"] for frame in frames[1:]] == ["e4", "e5", "Nf3"]
    assert all("<svg" in frame["svg"] for frame in frames)


def test_render_game_html_handles_game_with_no_moves():
    game = chess.pgn.Game.from_board(chess.Board())

    html = render_game_html(game)

    frames = _extract_frames(html)
    assert len(frames) == 1
    assert frames[0]["san"] is None
