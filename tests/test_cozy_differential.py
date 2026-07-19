"""Permanent differential harness: python-chess is the oracle for every
cozy-backed primitive used by search.py. Covers perft-suite edge positions
(castling, en-passant pins/discoveries, promotions) plus seeded random games.
"""

import random

import chess
import pytest

from imba_chess.eval.cozy_bridge import (
    board_to_cozy,
    gives_check,
    py_move_to_cozy,
)
from tests.test_cozy_bridge import EDGE_FENS, _random_boards

# Hand-built (position, move, expected gives_check) cases the random sweep is
# unlikely to hit. All five verified against python-chess 1.11.2 on 2026-07-18.
CURATED_CASES = [
    # En passant capture giving direct check (pawn lands on d3, attacks Ke2).
    ("k7/8/8/8/3Pp3/8/4K3/8 b - d3 0 1", "e4d3", True),
    # En passant capture giving DISCOVERED check (vacating e4 opens Re8-Ke1).
    ("k3r3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3", True),
    # Castling that gives check (rook lands f1, black king on f-file).
    ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", True),
    # Knight under-promotion with check (Ne8 attacks Kg7).
    ("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1", "e7e8n", True),
    # Quiet discovered check (Nd5 vacates the a1-h8 diagonal onto Kh8).
    ("7k/8/8/8/8/2N5/8/B3K3 w - - 0 1", "c3d5", True),
]


def _all_boards() -> list[chess.Board]:
    return [chess.Board(f) for f in EDGE_FENS] + _random_boards(200, seed=1234)


def test_gives_check_matches_python_chess_everywhere():
    checked = 0
    for board in _all_boards():
        cozy = board_to_cozy(board)
        for move in board.legal_moves:
            assert gives_check(cozy, py_move_to_cozy(board, move)) == board.gives_check(
                move
            ), (board.fen(), move.uci())
            checked += 1
    assert checked > 50_000


@pytest.mark.parametrize("fen,uci,expected", CURATED_CASES)
def test_gives_check_curated_edge_cases(fen, uci, expected):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, "test fixture is broken: move not legal"
    assert board.gives_check(move) == expected, "test fixture is broken: oracle disagrees"
    assert gives_check(board_to_cozy(board), py_move_to_cozy(board, move)) == expected


def test_legal_move_sets_match_python_chess_everywhere():
    from imba_chess.eval.cozy_bridge import cozy_move_to_uci

    for board in _all_boards():
        cozy = board_to_cozy(board)
        assert sorted(m.uci() for m in board.legal_moves) == sorted(
            cozy_move_to_uci(cozy, m) for m in cozy.generate_moves()
        ), board.fen()


def test_terminal_value_fast_matches_terminal_value_for_color():
    from imba_chess.eval.cozy_bridge import terminal_value_fast
    from imba_chess.eval.search import terminal_value_for_color

    terminal_seen = 0
    # Random games REPLAYED so boards carry real move stacks -- repetition and
    # 50-move claims need history, bare FENs can't exercise them.
    rng = random.Random(99)
    for g in range(400):
        board = chess.Board()
        # Shuffle-heavy move choice to actually reach repetitions/50-move claims.
        for _ in range(200):
            moves = list(board.legal_moves)
            if not moves:
                break
            quiet = [m for m in moves if not board.is_capture(m) and m.promotion is None]
            move = rng.choice(quiet if (quiet and rng.random() < 0.8) else moves)
            board.push(move)
            expected = terminal_value_for_color(board, color=chess.WHITE)
            got = terminal_value_fast(board_to_cozy(board), board, chess.WHITE)
            assert got == expected, (board.fen(), expected, got)
            if expected is not None:
                terminal_seen += 1
                break
    assert terminal_seen >= 30  # sweep must actually hit terminal states


def test_terminal_value_fast_curated_insufficient_material():
    """K+B vs K, halfmove 0: hits the _no_heavy_pieces insufficient-material
    branch specifically (not the draw-claim path, which needs halfmove >= 7).

    The random-game sweep above is not guaranteed to reach insufficient
    material, so this curated case exercises that branch deterministically.
    """
    from imba_chess.eval.cozy_bridge import terminal_value_fast
    from imba_chess.eval.search import terminal_value_for_color

    board = chess.Board("8/8/3k4/8/8/3KB3/8/8 w - - 0 1")
    expected = terminal_value_for_color(board, color=chess.WHITE)
    assert expected == 0.0
    assert terminal_value_fast(board_to_cozy(board), board, chess.WHITE) == expected


def test_encode_cozy_matches_encode_on_conversions_and_played_lines():
    import random

    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.models import BoardTokenConfig

    for mode in ("legal", "fen", "xfen"):
        enc = BoardStateEncoder(BoardTokenConfig(en_passant=mode))
        # Conversion equivalence on edge FENs + random boards
        for board in [chess.Board(f) for f in EDGE_FENS] + _random_boards(30, seed=21):
            assert vars(enc.encode(board)) == vars(enc.encode_cozy(board_to_cozy(board))), (
                mode,
                board.fen(),
            )
        # Played-line equivalence: cozy board reached via play(), NOT conversion —
        # catches ep-semantics drift (cozy reports the ep file even with no capturer).
        rng = random.Random(31)
        for _ in range(40):
            pyb = chess.Board()
            cb = board_to_cozy(pyb)
            for _ in range(rng.randrange(10, 80)):
                moves = list(pyb.legal_moves)
                if not moves:
                    break
                mv = rng.choice(moves)
                cb2 = __import__("copy").copy(cb)
                cb2.play(py_move_to_cozy(pyb, mv))
                pyb.push(mv)
                cb = cb2
                assert vars(enc.encode(pyb)) == vars(enc.encode_cozy(cb)), (mode, pyb.fen())
                if pyb.is_game_over():
                    break


def test_insufficient_material_matches_python_chess():
    fens = [
        "8/8/3k4/8/8/3KB3/8/8 w - - 0 1",  # KB vs K -> True
        "8/8/3k4/8/8/3KN3/8/8 w - - 0 1",  # KN vs K -> True
        "8/8/3k4/8/8/3K4/8/8 w - - 0 1",  # K vs K -> True
        "8/2b5/3k4/8/8/3KB3/8/8 w - - 0 1",  # KB vs KB (same/opposite bishop colors)
        "8/2n5/3k4/8/8/3KN3/8/8 w - - 0 1",  # KN vs KN -> oracle decides
        "8/8/3k4/8/8/3KP3/8/8 w - - 0 1",  # pawn -> False
        "8/8/3k4/8/8/2NKN3/8/8 w - - 0 1",  # two knights same side -> oracle decides
    ]
    from imba_chess.eval.cozy_bridge import insufficient_material

    for fen in fens:
        b = chess.Board(fen)
        assert insufficient_material(board_to_cozy(b)) == b.is_insufficient_material(), fen
    for board in _random_boards(60, seed=41):
        assert insufficient_material(board_to_cozy(board)) == board.is_insufficient_material(), (
            board.fen()
        )


def test_terminal_value_native_matches_oracle_on_replayed_games():
    import copy as copymod
    import random

    from imba_chess.eval.cozy_bridge import repetition_hash, terminal_value_native
    from imba_chess.eval.search import terminal_value_for_color

    rng = random.Random(77)
    terminal_seen = draw_claims_seen = 0
    for g in range(800):
        pyb = chess.Board()
        cb = board_to_cozy(pyb)
        # repetition_hash() of prior positions since the last irreversible
        # (zeroing: capture/pawn-move) move -- see terminal_value_native's
        # docstring for why zeroing-only is a sufficient reset condition.
        hash_history = []
        for _ in range(300):
            moves = list(pyb.legal_moves)
            if not moves:
                break
            quiet = [m for m in moves if not pyb.is_capture(m) and m.promotion is None]
            mv = rng.choice(quiet if (quiet and rng.random() < 0.92) else moves)
            prev_hash = repetition_hash(cb)
            prev_halfmove = pyb.halfmove_clock
            cb2 = copymod.copy(cb)
            cb2.play(py_move_to_cozy(pyb, mv))
            pyb.push(mv)
            hash_history = [] if pyb.halfmove_clock <= prev_halfmove else hash_history + [prev_hash]
            cb = cb2
            expected = terminal_value_for_color(pyb, color=pyb.turn)
            got = terminal_value_native(cb, color_is_stm=True, hash_history=hash_history)
            assert got == expected, (pyb.fen(), len(hash_history), expected, got)
            if expected is not None:
                terminal_seen += 1
                if expected == 0.0 and not pyb.is_stalemate() and not pyb.is_insufficient_material():
                    draw_claims_seen += 1
                break
    assert terminal_seen >= 30
    assert draw_claims_seen >= 5  # repetition/50-move path must actually be exercised


def test_terminal_value_native_curated_phantom_ep_repetition():
    """Deterministic king-shuffle threefold repetition where one leg is a
    capturer-less double pawn push -- exercises exactly the phantom-ep
    divergence documented in the Task 2 handoff: cozy's Board.hash() folds
    in the ep flag unconditionally (even with no legal capturer), while
    python-chess's transposition key (what real repetition claims key off
    of) excludes it. A naive `cb.hash()` repetition counter would fail to
    recognize the position immediately after the double push as "the same"
    as its later king-shuffle revisits, undercounting the threefold.
    """
    import copy as copymod

    from imba_chess.eval.cozy_bridge import repetition_hash, terminal_value_native
    from imba_chess.eval.search import terminal_value_for_color

    # Kings far apart, single white pawn free to double-push with no black
    # pawn anywhere near it -- the resulting ep flag is unconditionally
    # phantom (no legal capturer can possibly exist).
    pyb = chess.Board("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1")
    cb = board_to_cozy(pyb)

    moves = [
        "d2d4",  # White: capturer-less double push -> phantom ep on d3 (P1, halfmove resets to 0)
        "e8f8",
        "e1f1",
        "f8e8",
        "f1e1",  # occurrence #2 of P1's occupancy (no ep flag; several plies elapsed)
        "e8f8",
        "e1f1",
        "f8e8",
        "f1e1",  # occurrence #3 -> threefold repetition claim
    ]

    hash_history: list[int] = []
    saw_phantom_ep = False
    hashes_after_occurrence_1 = None
    for uci in moves:
        mv = chess.Move.from_uci(uci)
        assert mv in pyb.legal_moves, (pyb.fen(), uci)
        prev_hash = repetition_hash(cb)
        prev_halfmove = pyb.halfmove_clock
        cb2 = copymod.copy(cb)
        cb2.play(py_move_to_cozy(pyb, mv))
        pyb.push(mv)
        hash_history = [] if pyb.halfmove_clock <= prev_halfmove else hash_history + [prev_hash]
        cb = cb2
        if cb.en_passant() is not None:
            saw_phantom_ep = True
            hashes_after_occurrence_1 = (cb.hash(), repetition_hash(cb))
        expected = terminal_value_for_color(pyb, color=pyb.turn)
        got = terminal_value_native(cb, color_is_stm=True, hash_history=hash_history)
        assert got == expected, (pyb.fen(), uci, len(hash_history), expected, got)

    assert saw_phantom_ep, "fixture is broken: expected a phantom ep flag after d2d4"
    assert pyb.is_repetition(3)
    assert terminal_value_for_color(pyb, color=pyb.turn) == 0.0
    # The regression this test targets: cozy's raw hash() at P1 (with the
    # phantom ep flag) differs from its own hash_without_ep(), proving the
    # divergence is real and repetition_hash is the thing bridging it.
    raw_hash_at_p1, canonical_hash_at_p1 = hashes_after_occurrence_1
    assert raw_hash_at_p1 != canonical_hash_at_p1


def test_repetition_hash_matches_raw_hash_when_ep_is_legally_capturable():
    """Counterpart to the phantom-ep curated test above: when the ep flag
    DOES have a legal capturer, python-chess's transposition key includes
    ep (Board.has_legal_en_passant() is true), so repetition_hash must NOT
    strip it -- it should be byte-for-byte cozy's own Board.hash(), not
    hash_without_ep(). Seed verified via REPL: black's d7d5 leaves white's
    e5 pawn able to capture en passant on d6.
    """
    import copy as copymod

    from imba_chess.eval.cozy_bridge import ep_has_legal_capturer, repetition_hash

    pyb = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    cb = board_to_cozy(pyb)
    mv = chess.Move.from_uci("d7d5")
    assert mv in pyb.legal_moves, pyb.fen()

    cb2 = copymod.copy(cb)
    cb2.play(py_move_to_cozy(pyb, mv))
    pyb.push(mv)

    assert cb2.en_passant() is not None
    assert ep_has_legal_capturer(cb2), "fixture is broken: expected a legally capturable ep"
    assert pyb.has_legal_en_passant()
    assert repetition_hash(cb2) == cb2.hash()
    assert repetition_hash(cb2) != cb2.hash_without_ep()


def test_search_dual_boards_stay_in_sync(monkeypatch):
    """Enable the opt-in _dual_push verification and run a real search: every
    tree edge must keep the python-chess/cozy-chess board pair in sync.
    """
    from imba_chess.eval import search
    from tests.test_search import _MaterialEvaluator

    monkeypatch.setattr(search, "_DUAL_PUSH_VERIFY", True)
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
    legal_moves = list(board.legal_moves)
    legal_log_priors = [-1.0] * len(legal_moves)
    evaluator = _MaterialEvaluator()
    search.select_value_search_halving(
        evaluator=evaluator,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=search.HalvingConfig(budget=64, top_m=8, max_depth=3),
    )
