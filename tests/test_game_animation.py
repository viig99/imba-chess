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
    assert frames[0]["fen"] == chess.STARTING_BOARD_FEN
    assert frames[3]["fen"] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    assert frames[1]["slides"] == [[chess.E2, chess.E4]]


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
    # The eval bar holds the model's search expected score (White's view) until it moves again.
    assert [f["white_score"] for f in frames] == [None, None, None, 0.25, 0.25]
    assert frames[4]["check"] == chess.E1


def test_render_game_html_slides_both_castling_pieces():
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]:
        board.push_uci(move_uci)
    frames = _extract_frames(render_game_html(chess.pgn.Game.from_board(board)))
    assert frames[-1]["san"] == "O-O"
    assert frames[-1]["slides"] == [[chess.E1, chess.G1], [chess.H1, chess.F1]]


def test_render_game_gif_frames_and_timing(tmp_path):
    from PIL import Image

    from imba_chess.eval.game_gif import render_game_gif

    path = tmp_path / "game.gif"
    render_game_gif(_short_game(), path)
    with Image.open(path) as gif:
        assert gif.n_frames == 1 + 1 + 3 * 6 + 1
        durations = []
        for i in range(gif.n_frames):
            gif.seek(i)
            durations.append(gif.info["duration"])
    assert durations[0] == 3000 and durations[1] == 1000  # title card, start position
    # GIF delays are whole centiseconds, so a 1/24 s frame is stored as 40 ms.
    assert durations[2:7] == [40] * 5 and durations[7] == 1040  # glide, then land and pause
    assert durations[-2] == 3040 and durations[-1] == 2000  # final hold, result card
