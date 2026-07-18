# Cozy-Chess Hot-Path Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace python-chess with cozy-chess (Rust, via cozy-chess-py) in `search.py`'s measured hot paths — forcing-move detection, terminal detection, node expansion — behind the existing seams, with hard cutover per stage.

**Architecture:** A single new boundary module `src/imba_chess/eval/cozy_bridge.py` owns all python-chess ↔ cozy-chess interop (board conversion via raw bitboards, castling-UCI translation, gives-check). `search.py` threads a cozy board alongside each python-chess board through the search tree (dual-board: python-chess stays the interface currency and what evaluators/encoders consume; cozy does the chess-rules math). A permanent differential-test harness uses python-chess as the oracle. Spec: `docs/superpowers/specs/2026-07-18-rollout-cpu-hotpath-optimization-design.md`.

**Tech Stack:** Python 3.13, cozy-chess-py 0.1.1 (pinned), python-chess 1.11.2, pytest, uv.

## Global Constraints

- python-chess remains the public interface currency of `search.py` (`chess.Board`/`chess.Move` in signatures) and the correctness oracle in tests.
- **No leftover flags or dead code:** each stage's final commit deletes the old code path, its validation flag, and any A/B scaffolding test. The differential harness (`tests/test_cozy_differential.py`) is permanent and exempt.
- `search.py` stays torch-free (its module docstring requires it; `cozy_chess` import is fine).
- Pin `cozy-chess-py==0.1.1` exactly.
- Test command: `.venv/bin/pytest` from repo root (suite currently 154 tests; all must pass at every commit).
- Profile gate command (GPU must be idle, ~2.5 min):
  `.venv/bin/python scripts/generate_search_rollouts.py --config config/imba_chess_exit_full.toml --checkpoint "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt" --output-path <scratch>/prof.parquet --max-games 20 --profile`
  Baseline (2026-07-18): total 107.9s; search_bookkeeping 46.2%, search_gpu 36.6%, root_eval 17.2%.
- Rollout equivalence gate: two runs of the profile command (old path via flag vs new path) must produce identical parquet rows (`--sample-seed` defaults to 42; generation is fully deterministic given the seed).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## Verified API facts (probed 2026-07-18, do not re-derive)

- `cc.Board.status()` returns `cc.GameStatus.Won` when the **side to move is checkmated** (winner = the side NOT to move), `Drawn` for stalemate only, `Ongoing` otherwise. No repetition/50-move/insufficient-material coverage.
- cozy generates castling as king-takes-own-rook (`e1h1`/`e1a1`); python-chess UCI is `e1g1`/`e1c1`.
- `cc.Square` has **no int constructor**; make squares by iterating `cc.BitBoard(mask)` over a raw u64.
- `copy.copy(cozy_board)` + `board.play(move)` ≈ 196 ns; `board.checkers()` returns a BitBoard; `play()` raises `ValueError` on illegal moves.
- Bitboard-builder conversion (`board_to_cozy` below) verified equal to `cc.Board.from_fen(pyboard.fen())` on all 2,087 real training FENs, ~6.0 µs vs ~16.4 µs.
- `cc.Piece`/`cc.Color` enums support `==`. `board.pieces(cc.Piece.Pawn)` → BitBoard; `int(bitboard)` → raw u64.

---

### Task 1: Dependency + cozy_bridge conversion and move translation

**Files:**
- Modify: `pyproject.toml` (+ `uv.lock`, via `uv add`)
- Create: `src/imba_chess/eval/cozy_bridge.py`
- Test: `tests/test_cozy_bridge.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `board_to_cozy(board: chess.Board) -> cc.Board`; `py_move_to_cozy(board: chess.Board, move: chess.Move) -> cc.Move`; `cozy_move_to_uci(cozy_board: cc.Board, move: cc.Move) -> str`.

- [ ] **Step 1: Add the pinned dependency**

```bash
uv add "cozy-chess-py==0.1.1"
.venv/bin/python -c "import cozy_chess as cc; print(len(cc.Board().generate_moves()))"
```
Expected: `20`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_cozy_bridge.py
import random

import chess
import cozy_chess as cc
import pytest

from imba_chess.eval.cozy_bridge import (
    board_to_cozy,
    cozy_move_to_uci,
    py_move_to_cozy,
)

# Perft-suite positions: kiwipete, ep-pin, promotion-heavy, castling-rich.
EDGE_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def _random_boards(n_games: int = 50, seed: int = 7) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    for g in range(n_games):
        board = chess.Board()
        for _ in range(rng.randrange(10, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            boards.append(board.copy())
            if board.is_game_over():
                break
    return boards


@pytest.mark.parametrize("fen", EDGE_FENS)
def test_board_to_cozy_matches_fen_roundtrip(fen):
    board = chess.Board(fen)
    assert board_to_cozy(board).fen() == cc.Board.from_fen(fen).fen()


def test_board_to_cozy_matches_fen_roundtrip_on_random_games():
    for board in _random_boards():
        assert board_to_cozy(board).fen() == cc.Board.from_fen(board.fen()).fen()


def test_move_translation_roundtrips_all_legal_moves():
    for board in [chess.Board(f) for f in EDGE_FENS] + _random_boards(20, seed=11):
        cozy = board_to_cozy(board)
        # py -> cozy: every python-chess legal move maps to a cozy-legal move
        for move in board.legal_moves:
            assert cozy.is_legal(py_move_to_cozy(board, move)), (board.fen(), move)
        # cozy -> uci: the translated set equals python-chess's uci set
        py_ucis = sorted(m.uci() for m in board.legal_moves)
        cc_ucis = sorted(cozy_move_to_uci(cozy, m) for m in cozy.generate_moves())
        assert py_ucis == cc_ucis, board.fen()


def test_castling_translation_both_directions():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    cozy = board_to_cozy(board)
    kingside = chess.Move.from_uci("e1g1")
    assert str(py_move_to_cozy(board, kingside)) == "e1h1"
    queenside = chess.Move.from_uci("e1c1")
    assert str(py_move_to_cozy(board, queenside)) == "e1a1"
    ucis = {cozy_move_to_uci(cozy, m) for m in cozy.generate_moves()}
    assert "e1g1" in ucis and "e1c1" in ucis
    assert "e1h1" not in ucis and "e1a1" not in ucis
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cozy_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.eval.cozy_bridge'`

- [ ] **Step 4: Implement cozy_bridge**

```python
# src/imba_chess/eval/cozy_bridge.py
"""python-chess <-> cozy-chess interop for search hot paths.

cozy-chess (Rust, via cozy-chess-py) is an internal acceleration detail of
search.py; python-chess remains the interface currency and the correctness
oracle (tests/test_cozy_differential.py). Convention differences owned here:

- Castling: python-chess UCI moves the king two files (e1g1); cozy-chess
  represents castling as king-takes-own-rook (e1h1).
- cozy Board.status() covers checkmate/stalemate only; draw claims and
  insufficient material remain the caller's job.
"""

from __future__ import annotations

import copy

import chess
import cozy_chess as cc

_PIECES = (
    (cc.Piece.Pawn, "pawns"),
    (cc.Piece.Knight, "knights"),
    (cc.Piece.Bishop, "bishops"),
    (cc.Piece.Rook, "rooks"),
    (cc.Piece.Queen, "queens"),
    (cc.Piece.King, "kings"),
)


def board_to_cozy(board: chess.Board) -> cc.Board:
    """Convert via raw bitboard ints (~6us; python-chess .fen() alone is ~16us)."""
    builder = cc.BoardBuilder.empty()
    occ_white = board.occupied_co[chess.WHITE]
    for piece, attr in _PIECES:
        bitboard = getattr(board, attr)
        for color, mask in (
            (cc.Color.White, bitboard & occ_white),
            (cc.Color.Black, bitboard & ~occ_white & bitboard),
        ):
            if mask:
                for square in cc.BitBoard(mask):
                    builder.set_piece(square, piece, color)
    if board.turn == chess.BLACK:
        builder.set_side_to_move(cc.Color.Black)
    rights = board.castling_rights
    for color, kingside_bb, queenside_bb in (
        (cc.Color.White, chess.BB_H1, chess.BB_A1),
        (cc.Color.Black, chess.BB_H8, chess.BB_A8),
    ):
        short = cc.File.H if rights & kingside_bb else None
        long = cc.File.A if rights & queenside_bb else None
        if short is not None or long is not None:
            builder.set_castle_rights(color, short=short, long=long)
    if board.ep_square is not None and board.has_legal_en_passant():
        for square in cc.BitBoard(1 << board.ep_square):
            builder.set_en_passant(square)
    builder.set_halfmove_clock(board.halfmove_clock)
    builder.set_fullmove_number(board.fullmove_number)
    return builder.build()


def py_move_to_cozy(board: chess.Board, move: chess.Move) -> cc.Move:
    uci = move.uci()
    if board.is_castling(move):
        rook_file = (
            "h"
            if chess.square_file(move.to_square) > chess.square_file(move.from_square)
            else "a"
        )
        uci = uci[0] + uci[1] + rook_file + uci[1]
    return cc.Move.from_str(uci)


def cozy_move_to_uci(cozy_board: cc.Board, move: cc.Move) -> str:
    uci = str(move)
    if (
        cozy_board.piece_on(move.from_square) == cc.Piece.King
        and cozy_board.color_on(move.to_square) == cozy_board.side_to_move()
    ):
        # King "capturing" its own rook = castling; emit standard two-file UCI.
        new_file = "g" if uci[2] > uci[0] else "c"
        return uci[0] + uci[1] + new_file + uci[3]
    return uci
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cozy_bridge.py -v`
Expected: 4 test items (one parametrized), all PASS.

- [ ] **Step 6: Run the full suite, then commit**

Run: `.venv/bin/pytest`
Expected: 154 + new tests pass.

```bash
git add pyproject.toml uv.lock src/imba_chess/eval/cozy_bridge.py tests/test_cozy_bridge.py
git commit -m "feat: cozy_bridge — python-chess/cozy-chess conversion and move translation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: gives_check + permanent differential harness (Stage 0 gate)

**Files:**
- Modify: `src/imba_chess/eval/cozy_bridge.py`
- Create: `tests/test_cozy_differential.py`

**Interfaces:**
- Consumes: Task 1's `board_to_cozy`, `py_move_to_cozy`.
- Produces: `gives_check(cozy_board: cc.Board, cozy_move: cc.Move) -> bool`.

- [ ] **Step 1: Write the failing differential harness**

```python
# tests/test_cozy_differential.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cozy_differential.py -v`
Expected: FAIL with `ImportError: cannot import name 'gives_check'`.
Note: if any CURATED_CASES fixture asserts "move not legal", fix the FEN/move pair (the intent of each case is in its comment) — do not delete the case.

- [ ] **Step 3: Implement gives_check in cozy_bridge**

```python
def gives_check(cozy_board: cc.Board, cozy_move: cc.Move) -> bool:
    """Does this legal move give check? Simulate-in-Rust (~240ns vs ~3us
    for python-chess's Python-level push/check/pop)."""
    after = copy.copy(cozy_board)
    after.play(cozy_move)
    return bool(after.checkers())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cozy_differential.py tests/test_cozy_bridge.py -v`
Expected: all PASS (the >50k-move sweep takes a few seconds).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/cozy_bridge.py tests/test_cozy_differential.py
git commit -m "feat: cozy gives_check + permanent python-chess differential harness

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Stage 1 — forcing-move floor on cozy, validate, cut over

**Files:**
- Modify: `src/imba_chess/eval/search.py` (functions `_is_forcing`, `_push_children`, `select_value_search_d2` lines ~289-301, `select_value_search_halving` lines ~513-518)
- Test: `tests/test_search.py` (existing tests must pass unchanged), temporary A/B test inside `tests/test_cozy_differential.py`

**Interfaces:**
- Consumes: `cozy_bridge.board_to_cozy`, `cozy_bridge.py_move_to_cozy`, `cozy_bridge.gives_check`.
- Produces: `search._forcing_index_set(board: chess.Board, legal_moves: list[chess.Move]) -> set[int]` (module-private; Task 5 modifies it).

- [ ] **Step 1: Add `_forcing_index_set` with a validation-only flag, keeping `_is_forcing`**

In `search.py`, add `import os` and `from imba_chess.eval import cozy_bridge` to the imports, then below `_is_forcing`:

```python
# Validation-only switch for Stage 1 A/B equivalence runs; deleted at cutover.
_FORCING_IMPL = os.environ.get("IMBA_SEARCH_FORCING", "cozy")


def _forcing_index_set(board: chess.Board, legal_moves: list[chess.Move]) -> set[int]:
    """Indices of forcing moves (promotion/capture/check), one cozy board per node.

    Promotion and capture stay on python-chess (cheap bitboard tests); only
    check detection crosses to cozy (~240ns/move vs ~3us -- the profiled ~15%
    hot spot, 1M+ calls per 20-game rollout run).
    """
    if _FORCING_IMPL == "py":
        return {
            idx for idx, move in enumerate(legal_moves) if _is_forcing(board, move)
        }
    forcing: set[int] = set()
    cozy_board = None
    for idx, move in enumerate(legal_moves):
        if move.promotion is not None or board.is_capture(move):
            forcing.add(idx)
            continue
        if cozy_board is None:
            cozy_board = cozy_bridge.board_to_cozy(board)
        if cozy_bridge.gives_check(cozy_board, cozy_bridge.py_move_to_cozy(board, move)):
            forcing.add(idx)
    return forcing
```

- [ ] **Step 2: Refactor the three `_is_forcing` call sites to use the set**

In `select_value_search_halving` (root floor, currently lines ~513-518):

```python
    picks = list(order[: min(config.top_m, len(order))])
    seen = set(picks)
    forcing = _forcing_index_set(board, legal_moves)
    for idx in range(len(legal_moves)):
        if idx not in seen and idx in forcing:
            picks.append(idx)
            seen.add(idx)
```

In `select_value_search_d2` (reply floor, currently lines ~295-301):

```python
        opp_seen = set(opp_indices)
        opp_forcing = _forcing_index_set(board1, board1_eval.legal_moves)
        for opp_idx in range(len(board1_eval.legal_moves)):
            if opp_idx in opp_seen:
                continue
            if opp_idx in opp_forcing:
                opp_indices.append(opp_idx)
                opp_seen.add(opp_idx)
```

In `_push_children` (refutation floor + `floor_pick`, currently lines ~448-468):

```python
    opponent_to_move = node.board.turn != root_color
    order = _prior_order(position_eval.legal_log_priors)
    if opponent_to_move:
        forcing = _forcing_index_set(node.board, position_eval.legal_moves)
        # Refutation floor: top-r replies by prior plus ALL forcing replies.
        picks = list(order[: config.refutation_top_r])
        seen = set(picks)
        for idx in range(len(position_eval.legal_moves)):
            if idx not in seen and idx in forcing:
                picks.append(idx)
                seen.add(idx)
    else:
        forcing = set()
        picks = list(order[: config.expand_top])
```
and inside the expansion loop replace `floor_pick = opponent_to_move and _is_forcing(node.board, move)` with:
```python
        floor_pick = opponent_to_move and idx in forcing
```

- [ ] **Step 3: Add the temporary A/B equivalence test**

Append to `tests/test_cozy_differential.py` (marked for deletion at cutover):

```python
# TEMPORARY Stage-1 A/B scaffolding -- deleted at Stage 1 cutover.
def test_forcing_index_set_cozy_matches_py_impl(monkeypatch):
    from imba_chess.eval import search

    for board in _all_boards():
        moves = list(board.legal_moves)
        monkeypatch.setattr(search, "_FORCING_IMPL", "py")
        expected = search._forcing_index_set(board, moves)
        monkeypatch.setattr(search, "_FORCING_IMPL", "cozy")
        assert search._forcing_index_set(board, moves) == expected, board.fen()
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest`
Expected: all pass, including all existing `tests/test_search.py` behavior tests (they exercise the refutation floor with `_MaterialEvaluator` — no test changes allowed).

- [ ] **Step 5: Rollout equivalence gate (A/B on real data, GPU)**

```bash
SCRATCH=$(mktemp -d)
IMBA_SEARCH_FORCING=py .venv/bin/python scripts/generate_search_rollouts.py --config config/imba_chess_exit_full.toml --checkpoint "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt" --output-path $SCRATCH/py.parquet --max-games 20 --profile
IMBA_SEARCH_FORCING=cozy .venv/bin/python scripts/generate_search_rollouts.py --config config/imba_chess_exit_full.toml --checkpoint "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt" --output-path $SCRATCH/cozy.parquet --max-games 20 --profile
.venv/bin/python -c "
import pandas as pd
a = pd.read_parquet('$SCRATCH/py.parquet'); b = pd.read_parquet('$SCRATCH/cozy.parquet')
pd.testing.assert_frame_equal(a, b)
print('EQUAL:', len(a), 'rows')
"
```
Expected: `EQUAL: <n> rows`, and the cozy run's `search_bookkeeping` share visibly below the py run's (baseline 46.2%; expect roughly 35-40%). Record both profile blocks for Task 6.

- [ ] **Step 6: Cutover — delete the old path, flag, and scaffolding**

- Delete `_is_forcing` and the `_FORCING_IMPL` env switch (and the `import os` if now unused) from `search.py`; `_forcing_index_set` keeps only the cozy implementation.
- Delete `test_forcing_index_set_cozy_matches_py_impl` from `tests/test_cozy_differential.py`.
- Run: `.venv/bin/pytest` — all pass. Run: `grep -rn "IMBA_SEARCH_FORCING\|_is_forcing" src scripts tests` — no hits.

- [ ] **Step 7: Commit**

```bash
git add -A src/imba_chess/eval/search.py tests/test_cozy_differential.py
git commit -m "perf: forcing-move floor via cozy-chess (Stage 1 cutover, ~15% hot spot)

Fixed-seed 20-game rollout parquet identical py vs cozy; old path and
validation flag removed per no-leftover-flags policy.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Stage 2a — cozy-backed terminal detection (not yet wired)

**Files:**
- Modify: `src/imba_chess/eval/cozy_bridge.py`
- Modify: `tests/test_cozy_differential.py`

**Interfaces:**
- Consumes: Task 1-2 bridge functions.
- Produces: `terminal_value_fast(cozy_board: cc.Board, board: chess.Board, color: chess.Color) -> Optional[float]` — exact drop-in semantics of `search.terminal_value_for_color(board, color=color)`.

- [ ] **Step 1: Write the failing differential test**

Append to `tests/test_cozy_differential.py`:

```python
def test_terminal_value_fast_matches_terminal_value_for_color():
    from imba_chess.eval.cozy_bridge import terminal_value_fast
    from imba_chess.eval.search import terminal_value_for_color

    terminal_seen = 0
    # Random games REPLAYED so boards carry real move stacks -- repetition and
    # 50-move claims need history, bare FENs can't exercise them.
    rng = random.Random(99)
    for g in range(300):
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cozy_differential.py::test_terminal_value_fast_matches_terminal_value_for_color -v`
Expected: FAIL with `ImportError: cannot import name 'terminal_value_fast'`.

- [ ] **Step 3: Implement in cozy_bridge**

```python
def _no_heavy_pieces(cozy_board: cc.Board) -> bool:
    return not (
        int(cozy_board.pieces(cc.Piece.Pawn))
        | int(cozy_board.pieces(cc.Piece.Rook))
        | int(cozy_board.pieces(cc.Piece.Queen))
    )


def terminal_value_fast(
    cozy_board: cc.Board, board: chess.Board, color: chess.Color
) -> float | None:
    """Drop-in for search.terminal_value_for_color, cozy fast path.

    cozy status() decides checkmate/stalemate (~50ns vs ~4.3us). python-chess
    stays the oracle on the rare paths: insufficient material (only reachable
    when no pawn/rook/queen exists -- cheap cozy pre-filter) and draw claims
    (only reachable at halfmove_clock >= 7, the pre-existing guard).
    """
    status = cozy_board.status()
    if status == cc.GameStatus.Won:
        # cozy 'Won' == side to move is checkmated; winner is the other side.
        winner = not board.turn
        return 1.0 if winner == color else -1.0
    if status == cc.GameStatus.Drawn:
        return 0.0  # stalemate
    if _no_heavy_pieces(cozy_board) and board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 7:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            return 1.0 if outcome.winner == color else -1.0
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cozy_differential.py -v`
Expected: all PASS. If the `terminal_seen >= 30` assertion fails, raise the game count (400), don't weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/cozy_bridge.py tests/test_cozy_differential.py
git commit -m "feat: cozy terminal_value_fast, differentially tested incl. repetition/50-move claims

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Stage 2b — thread dual boards through the search tree, cut over

**Files:**
- Modify: `src/imba_chess/eval/search.py` (`_TreeNode`, `_RootCandidate`, `_expand_root_candidates`, `select_value_search_d2`, `_push_children`, `select_value_search_halving`, `terminal_value_for_color`)
- Modify: `tests/test_search.py` (only if `terminal_value_for_color`'s signature change requires it — see Step 1)
- Test: temporary A/B via rollout gate (as Task 3 Step 5)

**Interfaces:**
- Consumes: `cozy_bridge.board_to_cozy`, `py_move_to_cozy`, `gives_check`, `terminal_value_fast`.
- Produces: `search.terminal_value_for_color(board, *, color, cozy_board=None)` — unchanged behavior when `cozy_board` is None (public callers unaffected); `_forcing_index_set(board, legal_moves, cozy_board)` now takes the node's cozy board (no internal conversion).

- [ ] **Step 1: Survey external callers**

Run: `grep -rn "terminal_value_for_color\|_forcing_index_set" src scripts tests --include=*.py`
Expected callers of `terminal_value_for_color` outside `search.py`: `tests/test_search.py` (and possibly `scripts/eval_vs_stockfish.py`). The public signature gains an optional keyword-only `cozy_board=None`; when None it computes `board_to_cozy(board)` itself — one implementation, no fork:

```python
def terminal_value_for_color(
    board: chess.Board, *, color: chess.Color, cozy_board: "cc.Board | None" = None
) -> Optional[float]:
    if cozy_board is None:
        cozy_board = cozy_bridge.board_to_cozy(board)
    return cozy_bridge.terminal_value_fast(cozy_board, board, color)
```
(The old `board.outcome(claim_draw=board.halfmove_clock >= 7)` body is deleted here — `terminal_value_fast` is its differentially-proven replacement; the `>= 7` guard comment moves to `cozy_bridge`.) External callers need no edits.

- [ ] **Step 2: Thread cozy boards through the tree structures**

`_TreeNode` and `_RootCandidate` gain a cozy field:

```python
@dataclass
class _TreeNode:
    board: chess.Board
    cozy_board: "cc.Board"
    ...  # existing fields unchanged
```
```python
@dataclass
class _RootCandidate:
    ...
    board1: chess.Board
    cozy1: "cc.Board"
    ...
```
Import at top of `search.py`: `import cozy_chess as cc` (type refs) alongside the existing `cozy_bridge` import.

Every `_search_copy(x); .push(move)` pair gets a parallel cozy line. The pattern, applied at all four sites (`_expand_root_candidates`, `select_value_search_d2` board2 loop, `_push_children` child loop, `select_value_search_halving` arm loop):

```python
        board1 = _search_copy(board)
        board1.push(move)
        cozy1 = copy.copy(root_cozy)
        cozy1.play(cozy_bridge.py_move_to_cozy(board, move))
        terminal_value = terminal_value_for_color(board1, color=root_color, cozy_board=cozy1)
```
where `root_cozy = cozy_bridge.board_to_cozy(board)` is computed **once** at the top of each `select_*` entry point (and passed into `_expand_root_candidates` as a parameter) and each node's children copy from their parent's `cozy_board`/`cozy1`. Add `import copy` to `search.py`'s imports.

- [ ] **Step 3: Update `_forcing_index_set` to take the node's cozy board**

```python
def _forcing_index_set(
    board: chess.Board, legal_moves: list[chess.Move], cozy_board: "cc.Board"
) -> set[int]:
    forcing: set[int] = set()
    for idx, move in enumerate(legal_moves):
        if move.promotion is not None or board.is_capture(move):
            forcing.add(idx)
        elif cozy_bridge.gives_check(cozy_board, cozy_bridge.py_move_to_cozy(board, move)):
            forcing.add(idx)
    return forcing
```
Call sites pass the already-threaded board: `_forcing_index_set(board, legal_moves, root_cozy)` (halving root), `_forcing_index_set(board1, board1_eval.legal_moves, candidate.cozy1)` (d2), `_forcing_index_set(node.board, position_eval.legal_moves, node.cozy_board)` (`_push_children`). The lazy in-function conversion from Task 3 is deleted (no dead code).

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest`
Expected: all pass. `tests/test_search.py`'s strategy tests construct boards and evaluators only through public entry points, so threading is invisible to them.

- [ ] **Step 5: Rollout equivalence gate vs Stage 1**

Same procedure as Task 3 Step 5, but the A side is the Stage-1 cutover commit and the B side is the working tree:

```bash
git worktree add /tmp/stage1-ab HEAD   # run BEFORE committing Stage 2 changes, from the Stage-1 commit
# A side: cd /tmp/stage1-ab && uv sync && run the profile command -> a.parquet
# B side: repo root, working tree with Stage 2 changes -> b.parquet
# compare with pd.testing.assert_frame_equal as in Task 3; then:
git worktree remove /tmp/stage1-ab
```
Rows must be identical; record the new profile block. Expected: `search_bookkeeping` drops to roughly 20-30% of a smaller total (outcome ~10% slice + copy/push slice collapse).

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/eval/search.py tests
git commit -m "perf: thread cozy boards through search tree; cozy terminal detection (Stage 2 cutover)

Fixed-seed 20-game rollout parquet identical to Stage 1; single-implementation
terminal_value_for_color, no flags left behind.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Final measurement, docs, and Stage-3 go/no-go

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-rollout-cpu-hotpath-optimization-design.md` (append Results section)
- Modify: `docs/superpowers/notes/2026-07-15-rollout-generation-throughput-investigation.md` (closing pointer)
- Modify: memory (`imba-chess-elo-goal.md` workstream pointer)

- [ ] **Step 1: Final profile + games/hr comparison**

Run the profile gate command once more on the final tree. Compute the end-to-end speedup vs the 107.9s / 5.9s-per-game baseline. Also re-run `.venv/bin/pytest` one final time.

- [ ] **Step 2: Append a Results section to the spec**

Record: per-stage profile blocks (baseline / Stage 1 / Stage 2), total games/hr change, and the Stage-3 decision: justified only if `search_bookkeeping` remains the largest bucket; otherwise the next lever is cross-game batched search per the spec's Amdahl note. Add a closing line to the 2026-07-15 notes file pointing at the spec's Results.

- [ ] **Step 3: Update memory and commit**

Update the `imba-chess-elo-goal` memory file's status line to mention the cozy-chess hot-path work and its measured outcome, with the spec path.

```bash
git add docs
git commit -m "docs: cozy-chess adoption results — per-stage profiles and Stage-3 decision

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
