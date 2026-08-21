# In-repo native chess binding and move projection

Date: 2026-08-21
Status: approved design; implementation not started

## Problem

Rollout generation remains CPU-bound after the shared K/V arena cutover. The two largest host buckets are search bookkeeping and legal-move projection. The legal projection path runs once for every evaluated search node and currently crosses the Python/PyO3 boundary repeatedly to generate moves, normalize castling UCIs, map moves into the model vocabulary, and sort the aligned outputs.

The current isolated baseline over representative positions is 16.87 microseconds per node:

| Phase | Microseconds/node |
| --- | ---: |
| Move generation | 1.15 |
| Move-to-vocab/UCI mapping | 7.16 |
| Canonical sorting | 2.81 |
| Tensor conversion | 3.84 |
| Marginal softmax | 1.91 |

Move generation, mapping, and sorting therefore cost 11.12 microseconds per evaluated node before tensor projection. A separate forcing-move scan costs 19.76 microseconds on nodes where it runs.

A second PyO3 extension cannot directly unwrap the private Rust `cozy_chess::Board` held by the installed `cozy-chess-py` extension. Calling that object through Python from another extension preserves the crossings this work is intended to remove. Converting through FEN adds formatting/parsing and creates incompatible move types. The native package must own the Board and Move types used by the application.

## Goals

1. Add an in-repository PyO3/maturin package, `imba_chess_native`, that depends directly on the Rust `cozy-chess` crate.
2. Provide parity for the `cozy-chess-py` API currently consumed by imba-chess.
3. Add a reusable native `MoveProjector` that performs legal move generation, castling-normalized UCI mapping, vocabulary lookup, and canonical sorting in one Python call.
4. Share one torch-free projection entry point between rollout generation, G=1 evaluation, and actor workers.
5. Preserve rollout labels and search trajectories exactly.
6. Adopt the new binding only if the isolated projection is at least 2x faster than the current Python implementation.
7. Establish a long-term native home for later forcing, terminal, and search-bookkeeping optimizations without committing to those ports now.

## Non-goals

- Porting Torch model execution, CUDA batching, scheduling, dataset streaming, parquet output, or training to Rust.
- Porting sequential halving, Gumbel sampling, or tree bookkeeping in Stage 1.
- Computing forcing/check metadata for every projected node.
- Modifying or depending on an unreleased upstream `cozy-chess-py` change.
- Maintaining two production Board types after cutover.
- Changing move ordering, vocabulary semantics, search work allocation, or numeric model calculations.

## Package layout

The repository gains a standalone maturin package:

```text
native/imba_chess_native/
├── Cargo.toml
├── pyproject.toml
├── LICENSE
├── imba_chess_native.pyi
├── src/
│   ├── lib.rs
│   ├── board.rs
│   ├── board_builder.rs
│   ├── bitboard.rs
│   ├── castle_rights.rs
│   ├── chess_move.rs
│   ├── enums.rs
│   ├── functions.rs
│   ├── move_projector.rs
│   └── piece_moves.rs
└── tests/
```

The binding starts from the small MIT-licensed API shape of `cosy-chess-py` 0.1.1 and retains its license attribution. Its Rust dependencies are pinned initially to the versions already used by that release:

```toml
cozy-chess = "0.3.4"
pyo3 = "0.23"
```

The Python distribution and import module are both named `imba_chess_native`. This makes ownership explicit and avoids shadowing the third-party `cozy_chess` module during the side-by-side validation period.

The root project adds a path dependency only after the native package builds independently:

```toml
[project]
dependencies = [
  "imba-chess-native",
]

[tool.uv.sources]
imba-chess-native = { path = "native/imba_chess_native" }
```

`cozy-chess-py` remains installed during parity and performance validation. It is removed in the final clean cutover, together with every production import of `cozy_chess`.

## Binding parity

Stage 1 implements the existing wrapper surface used by the repository:

- `Board`, `BoardBuilder`, `Move`, `BitBoard`, and `PieceMoves`;
- `Color`, `Piece`, `File`, `Rank`, `Square`, and `GameStatus`;
- `CastleRights`;
- move-generation and attack-table free functions;
- copying, hashing, equality, formatting, FEN, status, and board query methods.

The initial implementation may copy the complete small wrapper surface rather than maintaining an application-specific subset. This avoids a second compatibility convention, keeps tests simple, and leaves one coherent Board API for later native work.

All production imports migrate in one cutover from:

```python
import cozy_chess as cc
```

to:

```python
import imba_chess_native as cc
```

The affected production modules are currently `board_state.py`, `actor_worker.py`, `cozy_bridge.py`, `position_evaluator.py`, and `search.py`.

## MoveProjector API

Python constructs one projector for each `MoveVocab`:

```python
projector = cc.MoveProjector(move_vocab.token_to_id)
ids, moves, ucis, total_legal = projector.project(board)
```

The constructor accepts a mapping from canonical standard UCI strings to integer vocabulary IDs. It validates that keys parse as moves and values fit the integer range returned to Python.

The projector stores a Rust lookup indexed by a compact move key. Each entry contains:

- vocabulary ID;
- canonical UCI sort key;
- a reusable Python string reference for the canonical UCI.

`project(board)`:

1. generates all legal moves from the owned Rust board;
2. detects castling represented by cozy-chess as king-to-own-rook;
3. normalizes castling to standard king-to-g/c UCI for lookup and output while retaining the raw cozy move for `Board.play`;
4. looks up the normalized move in the projector;
5. drops unmapped moves;
6. sorts mapped results lexicographically by canonical UCI;
7. returns aligned `list[int]`, `list[Move]`, `list[str]`, and the unfiltered legal-move count.

The method returns separate aligned lists rather than a list of tuples so Python callers do not need an unzip loop. UCI Python objects are reused from projector state instead of reconstructed per node.

The projector owns only immutable mapping state after construction and is safe to call repeatedly under the GIL. Stage 1 does not release the GIL because it creates Python result objects; internal pure-Rust generation and sorting remain coarse-grained within one call.

## Castling and move identity

The Rust move used to play a child remains cozy-chess's native representation. For castling this can be king-to-own-rook. Only the lookup/output key is normalized to standard UCI.

Normalization follows the board, not a hardcoded four-move table:

- the moving piece is a king;
- the destination contains a same-color rook;
- the normalized destination file is `g` for short castling and `c` for long castling;
- the original king source square and rank are preserved.

This supports the existing standard-chess behavior and does not make Stage 1 depend on a standard starting king file. Differential tests pin current standard positions and explicit Chess960 castling cases.

## Python integration

A torch-free bridge owns projector caching and the shared projection function:

```python
def project_legal_moves(
    board: cc.Board,
    move_vocab: MoveVocab,
) -> tuple[list[int], list[cc.Move], list[str], int]:
    ...
```

A `WeakKeyDictionary` keyed by `MoveVocab` stores one native projector per vocabulary instance. This preserves the existing no-cross-vocabulary-leak invariant.

Both production callers use this function:

- `CachedPositionEvaluator.consume_decode_result` for rollout and G=1 evaluation;
- actor worker root/wave projection for G>1 Stockfish evaluation.

After validation, production removes:

- `_MOVE_ID_MEMO`;
- `_cozy_move_id_and_uci`;
- `_legal_moves_ids_ucis`'s Python move loop and sort;
- actor worker's duplicated `_legal_vocab_projection` implementation.

The pure-Python mapping remains as a test-only oracle, with expectations derived independently from python-chess/current behavior.

## Development and cutover sequence

### Phase A: native package parity

1. Add the maturin crate and license attribution.
2. Port the current wrapper API without changing imba-chess imports.
3. Run native-package unit tests and imba differential tests against both bindings.
4. Keep the installed third-party binding as production.

### Phase B: native projector

1. Write failing projector tests in the native package.
2. Implement constructor validation, compact lookup, castling normalization, projection, and sorting.
3. Add the torch-free Python cache/adapter without switching production callers.
4. Benchmark the projector and Python oracle in the same process.

### Phase C: guarded application cutover

Proceed only if move generation + mapping + sorting improves by at least 2x and exact differential tests pass.

1. Migrate every production and test import to `imba_chess_native`.
2. Switch rollout and actor projection to the shared native adapter.
3. Remove old production memo/projection code.
4. Remove `cozy-chess-py` from project dependencies.
5. Run the complete correctness and performance gates.

If the native projector misses the performance threshold, do not migrate application imports. Remove or retain the experimental native package only by an explicit decision based on whether later native stages remain justified.

## Correctness gates

### Binding parity

- Port the wrapper's existing unit tests.
- Differentially compare old and new bindings on existing edge FENs and at least 200 random reachable positions.
- Check legal move sets, FEN, status, hashes, piece/color bitboards, play/copy behavior, and move string/identity semantics.

### Projection differential

For every test board, compare native output against an independent Python oracle:

- total legal count;
- mapped count;
- IDs element-for-element;
- move `(from_square, to_square, promotion)` fields and raw move strings
  element-for-element (objects from different extension modules are not
  compared by Python identity/equality);
- canonical UCI strings element-for-element;
- canonical ordering and alignment among IDs, moves, and UCIs.

Coverage includes:

- standard castling;
- Chess960 castling normalization;
- en passant;
- promotions and underpromotions;
- checks and check evasions;
- positions with no mapped moves using a deliberately restricted vocabulary;
- two different vocabularies alive concurrently;
- repeated calls proving immutable projector state.

### Application gates

- Actor worker projection tests against the independent oracle.
- Existing cozy bridge and board-state differential suites.
- Full project test suite.
- Twenty-game, budget-2048, G=8 rollout parquet comparison:
  - complete table bit-identical;
  - best move/value exact;
  - `arm_log_prior` exact;
  - `arm_evals_spent` exact;
  - waves and evaluations unchanged.
- Existing fixed-composition Stockfish move probe.

A new 200-game Stockfish run is not required for a bit-identical Stage 1 cutover unless the fixed move probe or numeric gates differ.

## Performance gates

Use the existing representative 80-board projection benchmark and run old/new variants interleaved in one process.

Primary acceptance:

- native move generation + mapping + sorting no slower than 5.56 microseconds/node, a 2x improvement over the current 11.12 microseconds/node;
- output construction included in the timed native call;
- no benchmark variant may reuse already-projected per-board output across iterations.

Secondary evidence:

- end-to-end projection total;
- Python/C-level call count reduction;
- rollout `decode_project` bucket share;
- synchronized full rollout wall time, interpreted under the existing machine noise rules.

The isolated benchmark decides whether Stage C starts. End-to-end wall time is corroboration, not a substitute for the isolated gate.

## Later native stages

Later stages require separate designs and measurements.

### Native forcing indices

Add one board call used only on opponent nodes. It may compute promotion/capture/check forcing indices aligned to already projected canonical moves. Do not compute checks for every node during Stage 1.

### Native child transition and terminal metadata

Move board copy/play, repetition-hash update, capture/zeroing metadata, and terminal/material detection behind one coarse call. Preserve python-chess claim semantics through differential tests.

### Native search bookkeeping

Only if search bookkeeping remains dominant, add a Rust search-session state machine that yields decode waves and consumes model evaluations. Keep model execution and batching in Python. Python supplies root/Gumbel ordering initially so RNG and label trajectories do not change.

## Risks and mitigations

### Build portability

A path maturin dependency requires a Rust toolchain for editable/source installs. The repository will add wheel-building CI before treating the package as broadly distributable. Existing Python versions and Linux/macOS targets remain explicit test targets.

### Binding drift

Owning the wrapper creates maintenance responsibility. Pin `cozy-chess` and update it only through differential tests. Keep the wrapper thin; application-specific behavior belongs in coarse classes such as `MoveProjector`, not scattered Board methods.

### Accidental partial cutover

Both Board types may coexist only during validation. Stage C migrates every import and removes the old dependency in one commit. No compatibility aliases or mixed Board conversion helpers are retained.

### Benchmark-only success

The projector is accepted only after exact rollout labels and unchanged work counts. A microbenchmark speedup cannot justify label drift.

## Success criteria

Stage 1 is complete when:

1. `imba_chess_native` builds reproducibly from the repository.
2. Binding parity and projection differential tests pass.
3. Native projection is at least 2x faster than the Python movegen/mapping/sort baseline.
4. All production imports use the owned binding.
5. The third-party binding and old production projection code are removed.
6. The full suite passes.
7. The 20-game rollout table and search trajectory are bit-identical.
8. Performance evidence and the new ownership/build constraints are recorded in the generation handoff.
