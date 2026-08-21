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
special_tokens = {
    move_vocab.config.pad_token,
    move_vocab.config.start_token,
    move_vocab.config.unk_token,
}
move_tokens = {
    token: token_id
    for token, token_id in move_vocab.token_to_id.items()
    if token not in special_tokens
}
projector = cc.MoveProjector(move_tokens)
ids, moves, ucis, total_legal = projector.project(board)
```

The Python adapter filters the vocabulary's pad/start/unknown tokens once at
projector construction. The constructor requires every supplied key to parse as
a canonical UCI move and every value to fit the integer range returned to
Python; malformed move entries fail loudly rather than being skipped.

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

A new 200-game Stockfish run is not required for a bit-identical Stage 1
cutover. The established post-arena baseline is 0.5950 at SF2200; rerun it only
if actor projection, numeric, or rollout trajectory gates differ.

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

## Stage 1 Results

Measured 2026-08-21 on the feature worktree (`feat/imba-chess-native`), CPython
3.12.12, machine verified idle (GPU 0% util, no browser) per the generation
handoff's measurement rules.

### Projection differential (exact)

`tests/test_native_projection.py`: **415 passed**, no divergence.

- 206 positions per vocabulary — 6 `EDGE_FENS` plus 200 positions strided
  across 200 random reachable games (`_random_boards(200, seed=911)`) — run
  against the full static vocabulary (1,970 tokens) and against a restricted
  1-in-7 slice whose ids deliberately do not match the static vocab's.
- Compared on ids, UCI strings, total legal count, and move fields, not object
  equality: the two bindings wrap incompatible Rust `Move` types.
- Guard cases: the restricted vocabulary is shown to actually drop moves (never
  vacuously equal); two projectors interleaved on the same boards each keep
  matching their own oracle; and the full vocabulary drops nothing
  (`len(ids) == total` everywhere), since a silent drop would starve the search
  of a candidate move.

Native package suite (`native/imba_chess_native/tests/`): **105 passed**,
including 9 `MoveProjector` cases. `cargo fmt --check`, `cargo check`, and
`cargo test` clean.

### Speed gate (PASS)

`scripts/bench_native_move_projector.py`, 80 opening positions imported from
`bench_project_decompose.py` (mean 31.3 legal moves/node), 60 repetitions per
side with alternating order, exact-output check before timing:

| path   | median us/node | min  | max   |
|--------|----------------|------|-------|
| python | 11.63          | 11.27| 16.56 |
| native | 2.90           | 2.74 | 3.90  |

**speedup 4.02x**; gate `native <= 5.56 us/node` -> **PASS**. Two independent
re-runs gave native 2.99 and 3.06 us/node (4.15x, 4.06x), so the result is
reproducible and not a single-sample artifact. The Python side's 11.63 us/node
is consistent with the 11.12 us/node baseline of record; this harness times the
real `_legal_moves_ids_ucis` call rather than the decomposed phases.

Expected end-to-end effect, stated honestly: at 346,091 search evaluations per
20-game run, removing 8.2 us/node saves ~2.8s of a 41.5s run, i.e. **~6.9%**.
That is well below this machine's ~+/-20% wall-time noise floor and must not be
claimed as an end-to-end win from a wall-clock comparison. The isolated,
interleaved number above is the evidence.

### Pre-cutover label baseline

20 games, `--search-budget 2048 --concurrent-games 8 --dtype float32
--sample-seed 42`, checkpoint `best_hr10_checkpoint_23_hr10=0.9564.pt`,
config `config/imba_chess_exit_seeded_rollout.toml`. Output at
`/tmp/pre_native_projection.parquet`, preserved at
`artifacts/rollouts/pre_native_projection.parquet` since Task 5 compares
against it after cutover.

- 169 rollout rows from 20 games; 268 search waves; 346,091 search evaluations
- generation wall time 44s (2.23 s/game); instrumented total 41.5s

| bucket             | time  | share |
|--------------------|-------|-------|
| search_gpu         | 13.1s | 31.6% |
| search_bookkeeping | 12.9s | 31.2% |
| decode_project     | 10.1s | 24.3% |
| decode_prep        |  4.3s | 10.5% |
| root_eval          |  1.0s |  2.3% |
| batch_build        |  0.0s |  0.1% |
| ply_bookkeeping    |  0.0s |  0.1% |

`decode_project` is 29.2 us/eval, of which movegen + mapping + sort is the
~11.1 us the projector replaces; the remainder is tensor construction and
log_softmax, which the projector does not touch.

> **Superseded -- do not use this run as the label gate's reference.** It
> predates the change threading the `[dataset]` shuffle settings into
> generation, so it streamed a different set of games (169 rows here vs 209 on
> the identical command afterwards). Both arms were regenerated on a common
> stream; see "Label and work equality gate" below. The stale parquet has been
> deleted rather than left to be picked up by mistake.

Unrelated defect observed during this run and left unfixed (out of Task 3's
scope): the generator completed all 20 games, wrote its parquet and progress
sidecar, and printed its final report, then never exited -- it sat for 11
minutes holding 4 GB of GPU memory until interrupted.
`_main_with_hard_exit_on_crash` installs its `os._exit` escape only on the
exception path, so a *successful* run still goes through CPython's normal
shutdown and can block forever joining a non-daemon thread -- precisely the
failure that wrapper's own docstring describes.

### Gate decision (Stage 1 projection)

Accepted: exact on every differential fixture and 4.02x on the isolated
benchmark against a 2x requirement. Tasks 4-5 (production cutover) unblocked.

## Stage 1 Results: post-cutover (2026-08-21)

Production now calls `cozy_bridge.project_legal_moves` exclusively;
`cozy-chess-py` and both Python projection implementations are gone.

### Label and work equality gate (PASS)

The pre-cutover baseline recorded above was **discarded before this gate ran**:
it predates the change that threads the `[dataset]` shuffle settings into
generation, so it streamed a different set of games (169 rows vs 209 on the
same command). Both arms were regenerated on the same stream -- the
pre-cutover arm from merge commit `91e1200` in a temporary worktree with
`cozy-chess-py` reinstalled, the post-cutover arm from the cutover commit.

Identical command on both: 20 games, budget 2048, 8 concurrent, float32,
`--sample-seed 42`.

| | pre-cutover | post-cutover |
|---|---|---|
| rows / games | 209 / 20 | 209 / 20 |
| search waves | 324 | 324 |
| search evaluations | 428,016 | 428,016 |
| instrumented total | 54.5s | 50.9s |
| `decode_project` | 13.5s (24.7%) | 9.1s (17.8%) |
| `search_gpu` | 18.7s (34.4%) | 18.7s (36.7%) |
| `search_bookkeeping` | 14.7s (26.9%) | 15.9s (31.3%) |
| `decode_prep` | 6.5s (11.9%) | 6.2s (12.1%) |

All 19 parquet columns compare **bit-identical** after sorting by
`(game_id, ply)`, including every array column: `best_arm_move_uci`,
`best_arm_backed_value` (max |delta| exactly 0.0), `arm_evals_spent`,
`arm_log_prior`, `arm_move_uci`, `arm_backed_value`, and the rest. Wave and
evaluation counts match exactly, so the search did the same work, not merely
similar work.

`search_gpu` is the drift control and did not move (18.7s both arms), which is
what makes the `decode_project` delta readable.

### Speed gate, final environment (PASS)

| path | median us/node | min | max |
|---|---|---|---|
| python (reconstructed retired path) | 12.18 | 11.80 | 13.08 |
| native (via `project_legal_moves`) | 3.17 | 2.98 | 3.67 |

**3.85x**, gate `<= 5.56 us/node` -> PASS. Re-runs: 3.16, 3.34, 3.18 us/node
(3.84x, 3.73x, 3.82x). Slightly slower than the 2.90 measured pre-cutover
because this times the production entry point, including its per-call
projector-cache lookup, rather than `MoveProjector.project` directly -- the
more honest number of the two.

### On the end-to-end number

Total wall time moved 54.5s -> 50.9s, **1.07x**. That is below this machine's
~+/-20% noise floor and is **not** claimed as a result. What is measurable is
the targeted bucket: `decode_project` fell 13.5s -> 9.1s on identical work,
i.e. 4.4s over 428,016 evaluations = **10.3 us/node**, against 9.0 us/node
predicted from the isolated benchmark. Prediction and observation agree.

The remaining `decode_project` cost is tensor construction and log_softmax,
which the projector does not touch. `search_bookkeeping` (15.9s, 31.3%) is now
the largest Python bucket and the next target.

### Review

One finding, fixed in `15dcbc7`. On a Chess960 board whose king does not start
on the e-file, a plain king step and a castle can normalize onto the same
vocabulary token -- king on b1 with a rook on a1 makes both `b1a1` (long
castle) and the step `b1->c1` map to `"b1c1"`. `project()` returned both, with
an identical id and UCI and no way to distinguish them; a consumer keying a
dict by UCI silently lost a legal move. It now raises. Unreachable through this
application (standard chess only), but real in the shipped binding, which
offers Chess960 as a first-class feature.

The guard costs nothing measurable: a collision is exactly one entry reached
twice, so pointer equality decides it (3.24-3.46 us/node with the guard vs
3.16-3.34 without). No other correctness findings.
