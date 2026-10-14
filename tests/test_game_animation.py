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


def test_render_game_html_shows_players_result_and_search_analysis():
    board = chess.Board()
    for move_uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(move_uci)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "sf_elo_2600"
    game.headers["White"] = "imba-chess"
    game.headers["Black"] = "Stockfish (elo=2600)"
    game.headers["Result"] = "0-1"
    game.headers["Termination"] = "checkmate"
    analysis = dict(
        wdl=[0.5, 0.3, 0.2], search_wdl=[0.6, 0.3, 0.1], simulations=512, legal_moves=20,
        candidates=[
            dict(uci="g2g4", prior=0.4, visits=300, q=-0.25, policy=0.6),
            dict(uci="d2d4", prior=0.3, visits=0, q=0.0, policy=0.1),
        ],
    )

    html = render_game_html(game, analyses={3: analysis})

    assert "White: imba-chess" in html and "Black: Stockfish (elo=2600)" in html
    assert "Stockfish (elo=2600) (Black) wins by checkmate · 0-1" in html
    frames = _extract_frames(html)
    assert [f["color"] for f in frames[1:]] == ["White", "Black", "White", "Black"]
    assert [f["move_number"] for f in frames[1:]] == [1, 1, 2, 2]
    assert frames[1]["analysis"] is None
    assert [c["san"] for c in frames[3]["analysis"]["candidates"]] == ["g4", "d4"]
    assert frames[3]["uci"] == "g2g4" and frames[3]["player"] == "imba-chess"
