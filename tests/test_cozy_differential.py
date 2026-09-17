"""Permanent differential harness: python-chess is the oracle for every
cozy-backed primitive used by search.py. Covers perft-suite edge positions
(castling, en-passant pins/discoveries, promotions) plus seeded random games.
"""

import os
import random
from typing import Any

import chess
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval.cozy_bridge import (
    board_to_cozy,
    gives_check,
    py_move_to_cozy,
)
from tests.chess_positions import EDGE_FENS, _random_boards, differential_positions

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


@pytest.mark.parametrize("n_games", [3, pytest.param(200, marks=pytest.mark.extended)])
def test_gives_check_matches_python_chess_everywhere(n_games):
    checked = 0
    for board in differential_positions(n_games, seed=1234):
        cozy = board_to_cozy(board)
        for move in board.legal_moves:
            assert gives_check(cozy, py_move_to_cozy(board, move)) == board.gives_check(
                move
            ), (board.fen(), move.uci())
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("fen,uci,expected", CURATED_CASES)
def test_gives_check_curated_edge_cases(fen, uci, expected):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, "test fixture is broken: move not legal"
    assert board.gives_check(move) == expected, "test fixture is broken: oracle disagrees"
    assert gives_check(board_to_cozy(board), py_move_to_cozy(board, move)) == expected


@pytest.mark.parametrize("n_games", [3, pytest.param(200, marks=pytest.mark.extended)])
def test_is_capture_cozy_matches_python_chess_everywhere(n_games):
    """cozy_bridge.is_capture_cozy (Stage 3 Task 5) is the capture test
    _forcing_index_set_tree uses at cozy-only tree nodes -- python-chess is_capture
    remains the oracle, castling included (cozy's king-takes-own-rook
    encoding must NOT read as a capture despite the destination being
    occupied by the player's own rook)."""
    from imba_chess.eval.cozy_bridge import is_capture_cozy

    checked = 0
    for board in differential_positions(n_games, seed=1234):
        cozy = board_to_cozy(board)
        for move in board.legal_moves:
            cozy_move = py_move_to_cozy(board, move)
            assert is_capture_cozy(cozy, cozy_move) == board.is_capture(move), (
                board.fen(), move.uci(),
            )
            checked += 1
    assert checked > 100


def test_is_capture_cozy_curated_castling_is_not_a_capture():
    from imba_chess.eval.cozy_bridge import is_capture_cozy

    for fen, uci in [
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1"),
        ("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", "e8c8"),
    ]:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, "test fixture is broken: move not legal"
        assert not board.is_capture(move), "test fixture is broken: oracle disagrees"
        assert is_capture_cozy(board_to_cozy(board), py_move_to_cozy(board, move)) is False


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


def _python_terminal_value(board, color):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == color else -1.0


@pytest.mark.parametrize("fen,moves,termination", [
    (chess.STARTING_FEN, "e2e4 e7e5", None),
    (chess.STARTING_FEN, "f2f3 e7e5 g2g4 d8h4", chess.Termination.CHECKMATE),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "", chess.Termination.STALEMATE),
    ("7k/8/8/8/8/8/8/KB6 w - - 0 1", "", chess.Termination.INSUFFICIENT_MATERIAL),
    ("7k/8/8/8/8/8/8/K5R1 w - - 98 60", "", None),
    ("7k/8/8/8/8/8/8/K5R1 w - - 99 60", "", chess.Termination.FIFTY_MOVES),
    ("7k/8/8/8/8/8/8/K5R1 w - - 100 60", "", chess.Termination.FIFTY_MOVES),
    ("8/8/8/8/8/5k2/6q1/7K w - - 100 60", "", chess.Termination.CHECKMATE),
    (chess.STARTING_FEN, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1", chess.Termination.THREEFOLD_REPETITION),
    (chess.STARTING_FEN, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8", chess.Termination.THREEFOLD_REPETITION),
])
def test_terminal_value_native_matches_independent_outcome(fen, moves, termination):
    import copy

    board = chess.Board(fen)
    native = board_to_cozy(board)
    history = []
    for uci in [None, *moves.split()]:
        if uci is not None:
            move = chess.Move.from_uci(uci)
            history.append(cozy_bridge.repetition_hash(native))
            child = copy.copy(native)
            child.play(py_move_to_cozy(board, move))
            board.push(move)
            native = child
            if board.halfmove_clock == 0:
                history = []
        for color in (chess.WHITE, chess.BLACK):
            assert cozy_bridge.terminal_value_native(
                native, color_is_stm=color == board.turn, hash_history=history,
            ) == _python_terminal_value(board, color), (board.fen(), color)
    outcome = board.outcome(claim_draw=True)
    assert (outcome.termination if outcome else None) == termination


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
        expected = _python_terminal_value(pyb, pyb.turn)
        got = terminal_value_native(cb, color_is_stm=True, hash_history=hash_history)
        assert got == expected, (pyb.fen(), uci, len(hash_history), expected, got)

    assert saw_phantom_ep, "fixture is broken: expected a phantom ep flag after d2d4"
    assert pyb.is_repetition(3)
    assert _python_terminal_value(pyb, pyb.turn) == 0.0
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


def test_root_hash_seed_matches_incremental_history_replayed_games():
    """search._root_hash_seed(board) must reconstruct exactly the
    hash_history an incrementally-maintained tracker (the Task-3 test
    pattern: reset to () on a zeroing move, else append the pre-move
    repetition_hash) would hold at that same position -- the contract
    push_and_classify expects a tree walk to seed itself with.
    Replays real games and checks the seed at MANY cut points per game (not
    just the final ply), so both the "mid-window" and "empty after zeroing"
    cases get exercised, not just whatever halfmove_clock the game happens
    to end on.
    """
    import copy as copymod

    from imba_chess.eval.cozy_bridge import repetition_hash
    from imba_chess.eval.search import _root_hash_seed

    rng = random.Random(2024)
    checked = nonempty_seen = 0
    for g in range(150):
        pyb = chess.Board()
        cb = board_to_cozy(pyb)
        hash_history: list[int] = []
        n_plies = rng.randrange(1, 120)
        for ply in range(n_plies):
            moves = list(pyb.legal_moves)
            if not moves:
                break
            quiet = [m for m in moves if not pyb.is_capture(m) and m.promotion is None]
            mv = rng.choice(quiet if (quiet and rng.random() < 0.85) else moves)
            prev_hash = repetition_hash(cb)
            prev_halfmove = pyb.halfmove_clock
            cb2 = copymod.copy(cb)
            cb2.play(py_move_to_cozy(pyb, mv))
            pyb.push(mv)
            hash_history = (
                [] if pyb.halfmove_clock <= prev_halfmove else hash_history + [prev_hash]
            )
            cb = cb2
            if pyb.is_game_over():
                break
            # Check at every ply (not just game end): exercises both the
            # empty-after-zeroing case and growing/mid-window cases.
            got = _root_hash_seed(pyb)
            assert list(got) == list(hash_history), (pyb.fen(), ply, hash_history, got)
            checked += 1
            if hash_history:
                nonempty_seen += 1
    assert checked >= 500
    assert nonempty_seen >= 100  # must actually exercise non-trivial seeds


def test_root_hash_seed_empty_stack_is_empty_tuple():
    from imba_chess.eval.search import _root_hash_seed

    assert _root_hash_seed(chess.Board()) == ()
    # A board built straight from a FEN with a nonzero halfmove_clock but no
    # move_stack has no history to reconstruct -- n = min(hmc, len(stack))
    # caps at the (empty) stack, matching pre-Task-5 stackless behavior.
    assert _root_hash_seed(chess.Board("8/8/3k4/8/8/3K4/8/8 w - - 42 30")) == ()


class _ShadowVerifyingEvaluator:
    """Test-only PositionEvaluator wrapper for the opt-in cozy-tree-integrity
    check (IMBA_COZY_TREE_VERIFY): maintains an independent python-chess
    shadow board per handle (rebuilt via extend()'s move_uci chain from a
    fixed root board) and, when enabled, asserts every evaluate() batch's
    cozy board matches board_to_cozy(shadow) by repetition_hash.

    This replaces the pre-Task-5 `_dual_push` verify assert (which compared
    an incrementally cozy-played board against a python-chess-driven
    board_to_cozy conversion at every tree edge, INSIDE search.py): now that
    the cozy-only tree carries no python-chess board at all, that oracle
    check has nowhere to live in production code and moves here, into a
    test-side wrapper that reconstructs the missing python-chess twin
    independently and drives a real search with it.
    """

    def __init__(self, inner, root_handle, root_board, *, verify=None):
        self.inner = inner
        self._shadow: dict[Any, chess.Board] = {root_handle: root_board.copy()}
        self._verify = (
            os.environ.get("IMBA_COZY_TREE_VERIFY") == "1" if verify is None else verify
        )
        self.verified_count = 0

    def extend(self, handle, move_uci, move_vocab_id=None):
        new_handle = self.inner.extend(handle, move_uci)
        child_board = self._shadow[handle].copy()
        child_board.push_uci(move_uci)
        self._shadow[new_handle] = child_board
        return new_handle

    def evaluate(self, batch):
        if self._verify:
            for handle, cozy_board in batch:
                shadow_board = self._shadow[handle]
                shadow_cozy_hash = cozy_bridge.repetition_hash(cozy_bridge.board_to_cozy(shadow_board))
                assert cozy_bridge.repetition_hash(cozy_board) == shadow_cozy_hash, (
                    shadow_board.fen(), cozy_board.fen(),
                )
                self.verified_count += 1
        return self.inner.evaluate(batch)


def test_search_cozy_tree_hash_history_matches_shadow_py_replay(monkeypatch):
    """Enable IMBA_COZY_TREE_VERIFY and run a real search: every cozy tree
    node's board (reached purely through _cozy_push chains, root through
    several plies of descent) must match an independently-replayed
    python-chess shadow board's own cozy conversion -- proving the
    hash_history-threaded cozy-only tree stays in sync with what a
    from-scratch python-chess replay of the same move sequence would reach.
    """
    from imba_chess.eval import search
    from tests.test_search import _MaterialEvaluator

    monkeypatch.setenv("IMBA_COZY_TREE_VERIFY", "1")
    # A few reversible half-moves on top of an already-played opening so
    # halfmove_clock/move_stack are non-trivial (_root_hash_seed exercised,
    # not just the trivial empty-history case) and the search tree actually
    # explores several plies of hash_history threading.
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
    for uci in ("c6b8", "b1c3", "b8c6", "c3b1"):
        board.push_uci(uci)
    legal_moves = list(board.legal_moves)
    legal_log_priors = [-1.0] * len(legal_moves)
    evaluator = _ShadowVerifyingEvaluator(
        _MaterialEvaluator(), root_handle=None, root_board=board
    )
    search.select_value_search_halving(
        evaluator=evaluator,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=search.HalvingConfig(budget=64, top_m=8, max_depth=3),
    )
    assert evaluator.verified_count > 0  # the check actually ran, not a no-op
