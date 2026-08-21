# Imba Chess Native Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the external Python chess binding with an owned PyO3 binding and cut legal move projection's movegen/mapping/sort cost by at least 2x without changing rollout labels or search work.

**Architecture:** Vendor the MIT-licensed `cosy-chess-py` 0.1.1 wrapper at pinned commit `60f663ed4f4f8f95276453245bbc121314ad533f` into an in-repository maturin package named `imba_chess_native`, preserving its API while the old and new bindings coexist for differential validation. Add a native `MoveProjector`, prove parity and speed in isolation, then perform one clean application-wide type/dependency cutover and delete the Python hot-path implementation.

**Tech Stack:** Rust 2021, `cozy-chess` 0.3.4, PyO3 0.23, maturin, Python 3.12, uv, pytest, pandas/pyarrow rollout gates.

**Spec:** `docs/superpowers/specs/2026-08-21-imba-chess-native-design.md`

## Global Constraints

- The owned binding distribution and import module are both named `imba_chess_native`.
- Vendor source only from `kaajjaak/cosy-chess-py` commit `60f663ed4f4f8f95276453245bbc121314ad533f` and preserve its MIT license attribution.
- Pin `cozy-chess = "0.3.4"` and `pyo3 = "0.23"` initially.
- Keep `cozy-chess-py==0.1.1` as production until parity, exactness, and the isolated 2x projector gate pass.
- Do not add Board conversion shims, compatibility aliases, scheduler changes, or Torch/RNG/search ports.
- `MoveProjector` receives only canonical UCI move tokens; Python filters pad/start/unknown tokens once during projector construction.
- Generated cozy castling moves remain king-to-own-rook for play; lookup/output UCI is normalized to king-to-g/c from board state.
- Stage C is forbidden unless native movegen + mapping + sorting is at most 5.56 microseconds/node against the current 11.12 microseconds/node baseline.
- The clean cutover migrates every caller and removes `cozy-chess-py`, `_MOVE_ID_MEMO`, `_cozy_move_id_and_uci`, and actor worker's duplicated projection.
- The final 20-game G=8 budget-2048 parquet, `arm_evals_spent`, waves, and evaluations must be bit-identical.
- Execute in an isolated worktree created with the `using-git-worktrees` skill; the current main workspace contains unrelated user experiments.

## File Structure

### Native package

- `native/imba_chess_native/Cargo.toml`: Rust package, library name, pinned dependencies.
- `native/imba_chess_native/pyproject.toml`: maturin build metadata for distribution `imba-chess-native`.
- `native/imba_chess_native/LICENSE`: vendored MIT license attribution.
- `native/imba_chess_native/imba_chess_native.pyi`: public Python types, including `MoveProjector`.
- `native/imba_chess_native/src/*.rs`: thin Board/Move API parity copied from the pinned wrapper.
- `native/imba_chess_native/src/move_projector.rs`: project-specific immutable UCI-to-ID projector.
- `native/imba_chess_native/tests/test_imba_chess_native.py`: permanent wrapper parity and projector behavior tests.

### Application integration

- `pyproject.toml`: side-by-side path dependency first; old dependency removal only at cutover.
- `uv.lock`: reproducible native path build.
- `src/imba_chess/eval/cozy_bridge.py`: native projector cache and the single torch-free projection function.
- `src/imba_chess/eval/position_evaluator.py`: consume shared native projection; remove Python memo/projection loop.
- `src/imba_chess/eval/actor_worker.py`: consume the same torch-free native projection.
- `src/imba_chess/data/board_state.py`, `src/imba_chess/eval/search.py`: binding import cutover only.
- `tests/test_native_binding_parity.py`: temporary old/new cross-binding differential, deleted only with the old dependency after passing.
- `tests/test_native_projection.py`: permanent projector cache/order/caller contract tests.
- Existing cozy/actor/batched projection tests: import migration and independent-oracle retention.
- `scripts/bench_native_move_projector.py`: interleaved old/new projection benchmark and performance gate.
- `docs/GENERATION_PERF_HANDOFF.md`: measured results and new native build constraint.

---

### Task 1: Vendor and build the owned binding with API parity

**Files:**
- Create: `native/imba_chess_native/Cargo.toml`
- Create: `native/imba_chess_native/pyproject.toml`
- Create: `native/imba_chess_native/LICENSE`
- Create: `native/imba_chess_native/imba_chess_native.pyi`
- Create: `native/imba_chess_native/src/{lib,board,board_builder,bitboard,castle_rights,chess_move,enums,functions,piece_moves}.rs`
- Create: `native/imba_chess_native/tests/test_imba_chess_native.py`
- Create: `tests/test_native_binding_parity.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: pinned wrapper source commit `60f663ed4f4f8f95276453245bbc121314ad533f`.
- Produces: importable module `imba_chess_native` with Board/Move API parity; no `MoveProjector` yet.

- [ ] **Step 1: Add failing native-module tests before source code**

Copy the pinned upstream `tests/test_cozy_chess.py` to `native/imba_chess_native/tests/test_imba_chess_native.py` and change only:

```python
import cozy_chess as cc
```

to:

```python
import imba_chess_native as cc
```

Add `tests/test_native_binding_parity.py` with an independently comparable snapshot rather than cross-extension object equality:

```python
from __future__ import annotations

import chess
import cozy_chess as old_cc
import imba_chess_native as new_cc
import pytest

from tests.test_cozy_bridge import EDGE_FENS, _random_boards


def _move_fields(move) -> tuple[int, int, str | None, str]:
    promotion = None if move.promotion is None else str(move.promotion)
    return int(move.from_square), int(move.to_square), promotion, str(move)

_RANDOM_BOARDS = _random_boards(200, seed=731)
_RANDOM_POSITIONS = _RANDOM_BOARDS[
    :: max(1, len(_RANDOM_BOARDS) // 200)
][:200]


@pytest.mark.parametrize(
    "board",
    [chess.Board(fen) for fen in EDGE_FENS] + _RANDOM_POSITIONS,
)
def test_native_binding_matches_pinned_binding(board: chess.Board) -> None:
    old = old_cc.Board.from_fen(board.fen())
    new = new_cc.Board.from_fen(board.fen())

    assert new.fen() == old.fen()
    assert new.hash() == old.hash()
    assert str(new.status()) == str(old.status())
    assert [
        _move_fields(move) for move in new.generate_moves()
    ] == [
        _move_fields(move) for move in old.generate_moves()
    ]
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run:

```bash
.venv/bin/pytest -q \
  native/imba_chess_native/tests/test_imba_chess_native.py \
  tests/test_native_binding_parity.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'imba_chess_native'`.

- [ ] **Step 3: Vendor the pinned source and license**

Clone the exact source into a temporary directory and copy only the owned source files, stub, test, and license into `native/imba_chess_native/`. Do not add a submodule or preserve upstream `.git` metadata.

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/kaajjaak/cosy-chess-py.git /tmp/cosy-chess-py-v011
git -C /tmp/cosy-chess-py-v011 checkout \
  60f663ed4f4f8f95276453245bbc121314ad533f
```

The copied `LICENSE` remains unmodified. Rename `cozy_chess.pyi` to `imba_chess_native.pyi` and replace only module-import examples/names required by the new module.

- [ ] **Step 4: Rename package and module metadata**

Use this `native/imba_chess_native/Cargo.toml`:

```toml
[package]
name = "imba_chess_native"
version = "0.1.0"
edition = "2021"
license = "MIT"

[lib]
name = "imba_chess_native"
crate-type = ["cdylib"]

[dependencies]
cozy-chess = "0.3.4"
pyo3 = { version = "0.23", features = ["extension-module", "generate-import-lib"] }
```

Use this `native/imba_chess_native/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "imba-chess-native"
version = "0.1.0"
description = "Owned native chess primitives for imba-chess"
requires-python = ">=3.10"
license = { text = "MIT" }

[tool.maturin]
features = ["pyo3/extension-module"]
```

Change `src/lib.rs` registration to:

```rust
#[pymodule]
fn imba_chess_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    enums::register(m)?;
    bitboard::register(m)?;
    chess_move::register(m)?;
    piece_moves::register(m)?;
    castle_rights::register(m)?;
    board::register(m)?;
    board_builder::register(m)?;
    functions::register(m)?;
    Ok(())
}
```

- [ ] **Step 5: Add the side-by-side path dependency**

In root `pyproject.toml`, retain `cozy-chess-py==0.1.1` and add:

```toml
"imba-chess-native",
```

and:

```toml
[tool.uv.sources]
imba-chess-native = { path = "native/imba_chess_native" }
```

Run:

```bash
uv lock
uv sync
```

Expected: uv builds the maturin path package into `.venv`; both `cozy_chess` and `imba_chess_native` import successfully.

- [ ] **Step 6: Run parity tests**

Run:

```bash
.venv/bin/pytest -q \
  native/imba_chess_native/tests/test_imba_chess_native.py \
  tests/test_native_binding_parity.py \
  tests/test_cozy_bridge.py \
  tests/test_cozy_differential.py
```

Expected: all pass. Fix wrapper parity only; do not add projection behavior in this task.

- [ ] **Step 7: Commit the parity package**

```bash
git add native/imba_chess_native pyproject.toml uv.lock \
  tests/test_native_binding_parity.py
git commit -m "feat: add owned native chess binding"
```

### Task 2: Implement `MoveProjector` with exact castling semantics

**Files:**
- Create: `native/imba_chess_native/src/move_projector.rs`
- Modify: `native/imba_chess_native/src/lib.rs`
- Modify: `native/imba_chess_native/imba_chess_native.pyi`
- Modify: `native/imba_chess_native/tests/test_imba_chess_native.py`

**Interfaces:**
- Consumes: native `Board(pub cozy_chess::Board)` and `ChessMove(pub cozy_chess::Move)` from Task 1.
- Produces: `MoveProjector(mapping: dict[str, int])`; `project(board) -> tuple[list[int], list[Move], list[str], int]`.

- [ ] **Step 1: Write failing projector tests**

Append tests covering ordering, unmapped moves, independent vocabularies, validation, and castling:

```python
import pytest
import imba_chess_native as cc


def test_move_projector_returns_aligned_canonical_lists():
    ucis = [
        "a2a3", "a2a4", "b1a3", "b1c3", "b2b3", "b2b4",
        "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4",
        "f2f3", "f2f4", "g1f3", "g1h3", "g2g3", "g2g4",
        "h2h3", "h2h4",
    ]
    mapping = {uci: 1000 - index for index, uci in enumerate(reversed(ucis))}
    projector = cc.MoveProjector(mapping)

    ids, moves, got_ucis, total = projector.project(cc.Board())

    assert total == 20
    assert got_ucis == sorted(ucis)
    assert ids == [mapping[uci] for uci in got_ucis]
    assert [str(move) for move in moves] == got_ucis


def test_move_projector_normalizes_castling_but_returns_playable_raw_moves():
    board = cc.Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    projector = cc.MoveProjector({"e1c1": 7, "e1g1": 9})

    ids, moves, ucis, total = projector.project(board)

    assert total > 2
    assert ids == [7, 9]
    assert ucis == ["e1c1", "e1g1"]
    assert [str(move) for move in moves] == ["e1a1", "e1h1"]
    for move in moves:
        copy = board.__copy__()
        copy.play(move)


def test_move_projector_keeps_vocabularies_isolated():
    board = cc.Board()
    first = cc.MoveProjector({"e2e4": 11})
    second = cc.MoveProjector({"e2e4": 29})

    assert first.project(board)[0] == [11]
    assert second.project(board)[0] == [29]


def test_move_projector_rejects_non_uci_keys():
    with pytest.raises(ValueError, match="invalid UCI move"):
        cc.MoveProjector({"<pad>": 0})


def test_move_projector_normalizes_chess960_castling_from_the_board():
    board = cc.Board.from_fen(
        "4k3/8/8/8/8/8/8/RK5R w AH - 0 1",
        shredder=True,
    )
    projector = cc.MoveProjector({"b1c1": 31, "b1g1": 37})

    ids, moves, ucis, total = projector.project(board)

    assert total > 2
    assert ids == [31, 37]
    assert ucis == ["b1c1", "b1g1"]
    assert [str(move) for move in moves] == ["b1a1", "b1h1"]
    for move in moves:
        copy = board.__copy__()
        copy.play(move)
```

- [ ] **Step 2: Run projector tests and verify failure**

Run:

```bash
.venv/bin/pytest -q native/imba_chess_native/tests/test_imba_chess_native.py \
  -k move_projector
```

Expected: FAIL with `AttributeError: module 'imba_chess_native' has no attribute 'MoveProjector'`.

- [ ] **Step 3: Implement the immutable Rust projector**

Create `move_projector.rs` around this exact structure:

```rust
use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::board::Board;
use crate::chess_move::ChessMove;

struct ProjectionEntry {
    id: i64,
    sort_key: String,
    py_uci: Py<PyString>,
}

#[pyclass]
pub struct MoveProjector {
    entries: HashMap<cozy_chess::Move, ProjectionEntry>,
}

fn lookup_move(board: &cozy_chess::Board, mv: cozy_chess::Move) -> cozy_chess::Move {
    let is_castle = board.piece_on(mv.from) == Some(cozy_chess::Piece::King)
        && board.color_on(mv.to) == Some(board.side_to_move());
    if !is_castle {
        return mv;
    }
    let file = if mv.to.file() > mv.from.file() {
        cozy_chess::File::G
    } else {
        cozy_chess::File::C
    };
    cozy_chess::Move {
        from: mv.from,
        to: cozy_chess::Square::new(file, mv.from.rank()),
        promotion: mv.promotion,
    }
}
```

Implement `#[new]` by iterating a `PyDict`, extracting `String` and `i64`, parsing every key as `cozy_chess::Move`, rejecting malformed keys with `PyValueError("invalid UCI move: ...")`, and storing one `PyString` reference per entry.

Implement `project(py, board)` by generating all legal moves once, mapping through `lookup_move`, collecting only mapped entries, sorting by `sort_key`, and returning separate vectors. Clone Python string references with `clone_ref(py)`; return raw generated `ChessMove` values, not normalized moves.

Register the class in `lib.rs`:

```rust
pub mod move_projector;
// ...
move_projector::register(m)?;
```

Add the exact stub:

```python
class MoveProjector:
    def __init__(self, move_token_to_id: dict[str, int]) -> None: ...
    def project(self, board: Board) -> tuple[list[int], list[Move], list[str], int]: ...
```

- [ ] **Step 4: Rebuild and run projector tests**

Run:

```bash
uv sync
.venv/bin/pytest -q native/imba_chess_native/tests/test_imba_chess_native.py \
  -k move_projector
```

Expected: all projector tests pass.

- [ ] **Step 5: Run the complete native package suite**

Run:

```bash
.venv/bin/pytest -q native/imba_chess_native/tests/test_imba_chess_native.py
cargo test --manifest-path native/imba_chess_native/Cargo.toml
```

Expected: both pass without warnings introduced by the new module.

- [ ] **Step 6: Commit the projector**

```bash
git add native/imba_chess_native
git commit -m "perf: project legal moves in Rust"
```

### Task 3: Prove projection exactness and the 2x performance gate

**Files:**
- Create: `tests/test_native_projection.py`
- Create: `scripts/bench_native_move_projector.py`
- Modify: `docs/superpowers/specs/2026-08-21-imba-chess-native-design.md` (append measured Results)

**Interfaces:**
- Consumes: old `cozy_chess.Board`, new `imba_chess_native.Board`, `MoveProjector`.
- Produces: exact differential evidence and a pass/fail benchmark decision; no production caller changes.

- [ ] **Step 1: Add exact old/new projection differential tests**

Construct old and native boards from the same FEN. Build the projector from move-only vocabulary entries by filtering `pad_token`, `start_token`, and `unk_token`. Compare native output against `_legal_moves_ids_ucis` using IDs, UCI strings, total count, and move fields/raw strings rather than object equality.

Use:

```python
_PROJECTOR_RANDOM_BOARDS = _random_boards(200, seed=911)
_PROJECTOR_RANDOM_POSITIONS = _PROJECTOR_RANDOM_BOARDS[
    :: max(1, len(_PROJECTOR_RANDOM_BOARDS) // 200)
][:200]


@pytest.mark.parametrize(
    "board",
    [chess.Board(fen) for fen in EDGE_FENS] + _PROJECTOR_RANDOM_POSITIONS,
)
def test_native_projector_matches_python_oracle(board):
    vocab = MoveVocab.build_static()
    old_board = cozy_chess.Board.from_fen(board.fen())
    new_board = native_cc.Board.from_fen(board.fen())
    move_tokens = {
        token: token_id
        for token, token_id in vocab.token_to_id.items()
        if token not in {
            vocab.config.pad_token,
            vocab.config.start_token,
            vocab.config.unk_token,
        }
    }

    expected_ids, expected_moves, expected_ucis, expected_total = (
        _legal_moves_ids_ucis(old_board, vocab)
    )
    ids, moves, ucis, total = native_cc.MoveProjector(move_tokens).project(new_board)

    assert total == expected_total
    assert ids == expected_ids
    assert ucis == expected_ucis
    assert [_move_fields(move) for move in moves] == [
        _move_fields(move) for move in expected_moves
    ]
```

Add restricted-vocabulary and two-simultaneous-vocabulary cases.

- [ ] **Step 2: Run the differential tests**

Run:

```bash
.venv/bin/pytest -q tests/test_native_projection.py
```

Expected: exact pass across all fixtures. Any mismatch blocks benchmarking until explained and fixed in Rust.

- [ ] **Step 3: Add an interleaved benchmark harness**

Create `scripts/bench_native_move_projector.py` by reusing the same 80 opening positions as `bench_project_decompose.py`. Build old and new boards and both projector/oracle state before timing. Alternate old/native order across at least 40 repetitions, verify exact output before timing, and print medians, ranges, per-node microseconds, and speedup.

The timed functions must each perform move generation, mapping, output-list construction, and canonical sorting. They must not reuse projected board output from a previous iteration.

- [ ] **Step 4: Run the isolated gate**

Run:

```bash
.venv/bin/python scripts/bench_native_move_projector.py
```

Expected acceptance:

```text
exact outputs: PASS
python median: approximately 11.12 us/node
native median: <= 5.56 us/node
speedup: >= 2.00x
```

If native exceeds 5.56 microseconds/node, stop the plan here. Do not execute Tasks 4-5. Record the measured failure in the spec and report that production remains unchanged.

- [ ] **Step 5: Generate the pre-cutover label baseline after the speed gate passes**

Run the current production binding before import migration:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/generate_search_rollouts.py \
  --config config/imba_chess_exit_seeded_rollout.toml \
  --checkpoint artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt \
  --output-path /tmp/pre_native_projection.parquet \
  --max-games 20 --search-budget 2048 --concurrent-games 8 \
  --dtype float32 --sample-seed 42 --profile --profile-every-games 20 \
  --flush-every-games 0
```

Record rows, waves, evaluations, and timing buckets from stdout.

- [ ] **Step 6: Append Results and commit the validated experimental path**

Append the exact differential count, benchmark medians, speedup, and gate decision under a `## Stage 1 Results` section in the design spec.

```bash
git add native/imba_chess_native tests/test_native_projection.py \
  scripts/bench_native_move_projector.py \
  docs/superpowers/specs/2026-08-21-imba-chess-native-design.md
git commit -m "test: validate native move projection"
```

### Task 4: Clean application cutover to the owned binding

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/imba_chess/data/board_state.py`
- Modify: `src/imba_chess/eval/cozy_bridge.py`
- Modify: `src/imba_chess/eval/position_evaluator.py`
- Modify: `src/imba_chess/eval/actor_worker.py`
- Modify: `src/imba_chess/eval/search.py`
- Modify: all tests/scripts importing `cozy_chess`
- Modify: `tests/test_native_projection.py`
- Modify: `scripts/bench_native_move_projector.py`
- Remove: `tests/test_native_binding_parity.py`
- Remove obsolete production memo/projection tests or rewrite them around projector caching.

**Interfaces:**
- Consumes: accepted `MoveProjector` and native Board/Move parity.
- Produces: `cozy_bridge.project_legal_moves(board, move_vocab) -> (ids, moves, ucis, total)` as the sole production projection path; no external binding dependency.

- [ ] **Step 1: Write failing shared-adapter tests before production changes**

Add tests for special-token filtering, weak per-vocab cache isolation, canonical alignment, and restricted vocabularies:

```python
def test_project_legal_moves_filters_special_tokens_and_caches_per_vocab():
    board = native_cc.Board()
    first = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=True))
    second = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=False))

    first_result = cozy_bridge.project_legal_moves(board, first)
    second_result = cozy_bridge.project_legal_moves(board, second)

    assert first_result[0] == [first.token_to_id["e2e4"]]
    assert second_result[0] == [second.token_to_id["e2e4"]]
    assert first_result[2] == second_result[2] == ["e2e4"]
```

Run the new tests before adding the function.

Expected: FAIL with `AttributeError: module ...cozy_bridge has no attribute 'project_legal_moves'`.

- [ ] **Step 2: Implement the torch-free projector cache**

In `cozy_bridge.py`, import `MoveVocab`, add a typed `WeakKeyDictionary`, filter special tokens once, and create one native projector per vocab:

```python
_PROJECTORS: "WeakKeyDictionary[MoveVocab, cc.MoveProjector]" = WeakKeyDictionary()


def project_legal_moves(cozy_board: cc.Board, move_vocab: MoveVocab):
    projector = _PROJECTORS.get(move_vocab)
    if projector is None:
        special = {
            move_vocab.config.pad_token,
            move_vocab.config.start_token,
            move_vocab.config.unk_token,
        }
        projector = cc.MoveProjector(
            {
                token: token_id
                for token, token_id in move_vocab.token_to_id.items()
                if token not in special
            }
        )
        _PROJECTORS[move_vocab] = projector
    return projector.project(cozy_board)
```

Keep the independent Python oracle in test code, not in production.

- [ ] **Step 3: Migrate every binding import in one pass**

Replace every production/test/script import of `cozy_chess` with `imba_chess_native`. Re-run a repository search and require zero production imports of the old module.

Update root dependencies:

- remove `"cozy-chess-py==0.1.1"`;
- retain `"imba-chess-native"` and its path source;
- run `uv lock && uv sync`.

Delete `tests/test_native_binding_parity.py` only now, because the old module is intentionally no longer installed. Permanent native-package tests and python-chess/application differential tests remain.

- [ ] **Step 4: Replace both production projection implementations**

In `position_evaluator.py`, call `cozy_bridge.project_legal_moves` from wave consumption and `_project_legal_logits_cozy`. Remove `_CASTLE_RAW_TO_UCI`, `_MOVE_ID_MEMO`, `_cozy_move_id_and_uci`, and the Python loop/sort implementation.

In `actor_worker.py`, replace `_legal_vocab_projection` calls with the shared bridge function, then delete the duplicate function.

Rewrite `tests/test_cozy_move_id_cache.py` to assert native projector cache isolation/reuse through `cozy_bridge.project_legal_moves`; do not retain tests for removed memo internals.

Rewrite `tests/test_native_projection.py` so its permanent oracle operates on
`imba_chess_native.Board` without importing the removed binding. Keep the
oracle independent of `MoveProjector`:

```python
def _python_project(board, move_vocab):
    triples = []
    legal_moves = list(board.generate_moves())
    for move in legal_moves:
        uci = str(move)
        is_castle = (
            board.piece_on(move.from_square) == native_cc.Piece.King
            and board.color_on(move.to_square) == board.side_to_move()
        )
        if is_castle:
            target_file = (
                "g" if int(move.to_square) % 8 > int(move.from_square) % 8 else "c"
            )
            uci = uci[:2] + target_file + uci[3:]
        move_id = move_vocab.token_to_id.get(uci)
        if move_id is not None:
            triples.append((uci, int(move_id), move))
    triples.sort(key=lambda item: item[0])
    return (
        [item[1] for item in triples],
        [item[2] for item in triples],
        [item[0] for item in triples],
        len(legal_moves),
    )
```

Update `scripts/bench_native_move_projector.py` at the same time: after the
old distribution is removed, its Python arm must use a local equivalent of
this oracle on native boards rather than importing `_legal_moves_ids_ucis`.
Preserve the pre-cutover 11.12 microseconds/node baseline in the printed
comparison and continue timing fresh projection work on every iteration.

- [ ] **Step 5: Run targeted cutover tests**

Run:

```bash
.venv/bin/pytest -q \
  native/imba_chess_native/tests/test_imba_chess_native.py \
  tests/test_native_projection.py \
  tests/test_cozy_bridge.py \
  tests/test_cozy_differential.py \
  tests/test_cozy_move_id_cache.py \
  tests/test_actor_worker.py \
  tests/test_actor_server.py \
  tests/test_batched_projection.py \
  tests/test_search.py \
  tests/test_search_stepwise.py
```

Expected: all pass with no weakened tolerances.

- [ ] **Step 6: Commit the clean cutover**

```bash
git add pyproject.toml uv.lock native/imba_chess_native \
  src tests scripts
git commit -m "perf: use owned native chess projection"
```

Before committing, inspect the staged path list and remove unrelated user experiment files from the index.

### Task 5: Final label, performance, documentation, and regression gates

**Files:**
- Modify: `docs/GENERATION_PERF_HANDOFF.md`
- Modify: `docs/superpowers/specs/2026-08-21-imba-chess-native-design.md` Results
- Test: full repository and real rollout generation

**Interfaces:**
- Consumes: clean native cutover from Task 4 and `/tmp/pre_native_projection.parquet` from Task 3.
- Produces: final evidence and documented next bottleneck.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
.venv/bin/pytest -q
```

Expected: zero failures. Existing third-party deprecation warnings may remain; no new native warnings are accepted.

- [ ] **Step 2: Generate the post-cutover rollout**

After confirming the GPU is idle and receiving the required GPU-run go-ahead, run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/generate_search_rollouts.py \
  --config config/imba_chess_exit_seeded_rollout.toml \
  --checkpoint artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt \
  --output-path /tmp/post_native_projection.parquet \
  --max-games 20 --search-budget 2048 --concurrent-games 8 \
  --dtype float32 --sample-seed 42 --profile --profile-every-games 20 \
  --flush-every-games 0
```

- [ ] **Step 3: Compare labels and trajectory exactly**

Create a temporary comparator that loads both parquet files, sorts by `game_id, ply`, and checks:

```python
assert reference_table.equals(candidate_table)
assert reference["best_arm_move_uci"].equals(candidate["best_arm_move_uci"])
assert max(abs(reference["best_arm_backed_value"] - candidate["best_arm_backed_value"])) == 0.0
assert all(
    np.array_equal(left, right, equal_nan=True)
    for left, right in zip(reference["arm_evals_spent"], candidate["arm_evals_spent"])
)
assert all(
    np.array_equal(left, right, equal_nan=True)
    for left, right in zip(reference["arm_log_prior"], candidate["arm_log_prior"])
)
```

Also require identical reported waves and evaluations. Any difference blocks completion and triggers root-cause analysis; do not relax the gate.

- [ ] **Step 4: Re-run the isolated benchmark after cutover**

Run:

```bash
.venv/bin/python scripts/bench_native_move_projector.py
```

Expected: exact output and at least 2x speedup still hold in the final dependency environment.

- [ ] **Step 5: Run changed-file diagnostics and native checks**

Run:

```bash
cargo fmt --check --manifest-path native/imba_chess_native/Cargo.toml
cargo clippy --manifest-path native/imba_chess_native/Cargo.toml -- -D warnings
ruff check native/imba_chess_native/tests src/imba_chess tests scripts/bench_native_move_projector.py
```

Expected: all pass. Preserve existing test-module E402 exceptions only where `pytest.importorskip` requires them.

- [ ] **Step 6: Document measured results and build requirements**

Update the generation handoff and design Results with:

- native package ownership and pinned Rust dependencies;
- old/new per-node medians and speedup;
- rollout timing buckets;
- exact label/work gate;
- full-suite result;
- whether forcing detection remains the next target.

Do not claim an end-to-end speedup below the machine's established noise floor.

- [ ] **Step 7: Request final code review and commit documentation**

Request a read-only reviewer focused on unsafe type assumptions, castling normalization, projector cache isolation, packaging reproducibility, and test-oracle independence. Fix all Critical/Important findings, rerun affected tests, then commit:

```bash
git add docs/superpowers/specs/2026-08-21-imba-chess-native-design.md \
  docs/GENERATION_PERF_HANDOFF.md
git commit -m "docs: record native projection results"
```

- [ ] **Step 8: Push only after verification**

Confirm the branch contains only intended commits and the unrelated main-workspace experiments remain untouched. Push the feature branch or follow the user's explicit integration instruction; never force-push.
