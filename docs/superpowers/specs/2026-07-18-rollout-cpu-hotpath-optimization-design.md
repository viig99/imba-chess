# Rollout CPU hot-path optimization: performance report & staged plan (2026-07-18)

Decision ladder agreed with the user: **profile → can a Rust/C++ library fix it →
cheap Python improvements → Rust/C++ for what remains.** This document records
the evidence for each rung and the resulting plan. All numbers below were
measured fresh on 2026-07-18 on the local machine (RTX 3070 Ti laptop).

## 1. Profile — what is slow (re-measured, confirms 2026-07-15 investigation)

`generate_search_rollouts.py --profile`, 20 games / 169 positions / 1,371
search waves / 346,093 search evals, `config/imba_chess_exit_full.toml`
(search_budget=2048), checkpoint_23, total 107.9s:

| Bucket | 2026-07-18 | 2026-07-15 |
|---|---|---|
| search_bookkeeping (Python CPU: tree mgmt + chess calls) | **46.2%** | 43.5% |
| search_gpu (forward_decode waves) | 36.6% | 38.5% |
| root_eval (root forward) | 17.2% | 17.9% |
| batch_build / ply_bookkeeping | ~0% | ~0% |

Stable across three days and two measurement sessions. Inside the bookkeeping
bucket (cProfile, 2026-07-15): `board.gives_check()` ~15% of total wall time
(1M+ calls), `board.outcome()` ~10% (243k calls), `_search_copy`+`push` ~4%.

Scope notes:
- `scripts/train.py` and all loss-eval scripts never import python-chess —
  training is already decoupled. Data-pipeline chess work (PGN parse, per-ply
  replay, encode) measures ~0% here and is out of scope until shown binding.
- Amdahl ceiling: eliminating ALL Python chess cost caps rollout speedup at
  ~1.85x. The 12–35x coverage gap additionally needs cross-game batched search
  (the lc0-shaped architecture); this work is the complementary CPU half and
  makes that future work cleaner, not a substitute for it.

## 2. Can a Rust/C++ library fix it? — Yes, measured

### Library survey

| Library | Verdict |
|---|---|
| **cozy-chess-py** (PyO3 binding of cozy-chess, MIT-core) | **Only ready-made Python binding; benchmarked below.** Binding is young (v0.1.1, 2 releases, single maintainer) but thin, wheeled (cp38–cp313), typed, 96 tests. Gaps: no `gives_check` (use copy+play+`checkers()`), `status()` lacks repetition/50-move/insufficient-material, castling encoded king-takes-rook (`e1h1` not `e1g1`), no SAN. |
| shakmaty (+ pgn-reader) | Best rules coverage, actively maintained (python-chess author), but no Python bindings — would require our own PyO3 crate. GPL-3. pgn-reader solves a data-pipeline cost we don't measurably have. |
| Pleco | Stockfish-derived, fast, but dormant since 2019, no bindings. Pass. |
| rschess | Explicitly feature-rich *at the cost of performance*. Pass. |
| rust_move_gen | Movegen-only, minimal adoption, no bindings. Pass. |
| lc0 src/search | Not extractable as a library (coupled to lc0's NN backends). Its lesson is architectural: batched C++ search feeding batched NN evals is the endgame for the 12–35x gap. |

### Microbenchmark: python-chess 1.11.2 vs cozy-chess-py 0.1.1

2,087 real positions (every ply of 30 training-split games via
`LichessDataset.stream`), 63,251 legal moves, best of 3 runs, Python 3.13.
Script: `bench_chess_libs.py` (session scratchpad; ops mirror `search.py`'s
per-node work).

| Operation (per item) | python-chess | cozy-chess-py | Speedup |
|---|---|---|---|
| gives_check per move (cc: copy+play+`checkers()`) | 2,995 ns | 239 ns | **12.5x** |
| node expansion per move (copy(stack=False)+push vs copy+play) | 2,881 ns | 196 ns | **14.7x** |
| legal movegen per position | 22,216 ns | 1,324 ns | **16.8x** |
| terminal check per position (`outcome(claim_draw=False)` vs `status()`) | 4,313 ns | 53 ns | 81x (semantics differ — see gaps) |
| FEN per position (`board.fen()` / `Board.from_fen()`) | 16,176 ns | 404 ns | — |
| hybrid per node: py `fen()` → cc board → movegen + all-moves gives_check | — | 24,985 ns | vs ~91,000 ns pure-py equivalent |

**Correctness:** cozy gives-check agreed with python-chess on all ~9k moves of
a 300-position differential sample (0 mismatches, castling translated).

**Two findings that change the design:**
1. **Per-move FFI overhead is NOT the bottleneck** (239 ns/call including
   crossing + Rust board copy). The earlier fear (per-call overhead killing the
   win, as in the reverted KV-cache optimization) does not apply at PyO3 costs.
2. **The expensive handoff is python-chess's own `board.fen()` (16 µs)** — 65%
   of the hybrid per-node cost. Any hybrid design must hand positions over via
   raw bitboard ints (`board.pawns`, `board.occupied_co`, … are plain int
   attribute reads) or, better, keep the search tree's boards native cozy from
   the root down (convert once per search call).

**Projected end-to-end effect:** at these ratios the bookkeeping bucket's
chess-call share (~29% of total) compresses by ~10x, taking total rollout time
down ~1.35–1.6x. That is most of the Amdahl ceiling for the CPU half.

## 3. Cheap Python improvements — smaller than the library win

Candidates examined against `search.py` as it exists today:
- `_is_forcing` short-circuit ordering (promotion → is_capture → gives_check):
  **already implemented.**
- `outcome()` repetition-scan guard (`halfmove_clock >= 7`): **already
  implemented** (~20x on that path, per code comment).
- Partial-stack `_search_copy`: **already implemented** (~150x per comment).
- Remaining candidate: per-node check-mask precompute (direct-check destination
  masks from enemy-king square + discovered-check origin mask), making
  gives_check two bitmask tests per move. Realistic gain: maybe 3–6x on the
  gives_check share in pure Python — well short of cozy's 12.5x, at similar
  implementation-plus-differential-test effort to the library route, with more
  hand-rolled edge cases (en passant discovery, castling, promotion) to get
  wrong ourselves.

Conclusion: the easy Python wins are already harvested; the remaining Python
option duplicates, at comparable effort and higher correctness risk, roughly
half of what the measured library gives. The ladder therefore proceeds to a
staged library adoption rather than a Python round.

## 4. Staged plan (incremental, differential-tested, revertible)

Constraint honored throughout: python-chess stays the interface currency and
the correctness oracle; cozy is an internal acceleration detail behind
`search.py`'s existing seams, switchable by flag, one stage at a time.

- **Stage 0 — differential harness first.** A test module generating random +
  adversarial (position, move) pairs — en-passant discovered checks, castling
  both sides, promotions with check, pins — asserting cozy-derived
  gives_check / legal-move sets / terminal statuses match python-chess.
  Includes the castling-UCI translation (`e1g1` ↔ `e1h1`). This harness gates
  every later stage.
- **Stage 1 — forcing-move floor on cozy.** At each expanded node, build one
  cozy board (from raw bitboards or maintained incrementally) and evaluate
  `_is_forcing` via cozy for all candidate replies. Attacks the ~15%
  gives_check share. Behind a flag during validation only; fixed-seed
  rollout-equivalence run + full test suite + `--profile` re-measurement
  before/after.
- **Stage 2 — node expansion + terminal check on cozy (dual-board).**
  Search-tree nodes carry a cozy board alongside the python-chess board
  (root converted once per search call; children via cozy copy+play at
  ~196ns); python-chess remains at the interface and is what the evaluator/
  encoder consume. Terminal detection = cozy `status()` for checkmate/
  stalemate + cheap cozy pre-filter for insufficient material, with
  python-chess's own `outcome(claim_draw=True)` kept as the rare slow path
  behind the existing `halfmove_clock >= 7` guard — exact oracle semantics
  by construction, no hand-rolled repetition tracking. Attacks the ~10%
  outcome share + the gives-check conversion overhead from Stage 1.
  (Evaluator-side movegen `_project_legal_logits` and a cozy-native
  `BoardStateEncoder` are deliberately NOT in this stage — they touch
  `position_evaluator.py`/`board_state.py` and belong to Stage 3's
  go/no-go.)
- **Stage 3 (only if profile still says so) — own thin PyO3 crate.** If binding
  maturity becomes a liability or we want `expand_node`-style coarse calls /
  a cozy-native `BoardStateEncoder`, vendor a small maturin crate in-repo over
  cozy-chess (MIT). Not needed to start.

**No leftover flags or dead code.** Flags exist only inside a stage as its
validation mechanism. A stage's definition of done is the *cutover*: once its
equivalence gates pass, the old python-chess code path and the flag are
deleted in that same stage's final commit — cozy becomes the only
implementation of that piece, and `main` never carries two implementations
across stage boundaries. Rollback before cutover is "turn the flag off";
rollback after cutover is `git revert` of a small, self-contained commit.
The Stage 0 differential harness is not dead code and stays permanently: it
is test-only, and python-chess remains a dependency regardless (public
interface currency, PGN parsing in the data pipeline, oracle in tests).

Risks & mitigations: binding youth (pin exact version; Stage 0 harness is the
real safety net; Stage 3 is the exit); castling encoding (translate at the
boundary, covered by harness); `status()` semantic gap (draw logic stays ours,
same as today's guard); Python 3.13 wheels exist (repo venv is 3.13; no cp314).

## 5. Benchmark provenance

- Rollout profile: `--profile` run 2026-07-18, output in session scratchpad
  (`profile_rollout.parquet` + task log); same flags as nightly except
  `--max-games 20`.
- Microbenchmark venv: uv, Python 3.13, `python-chess==1.11.2` +
  `cozy-chess-py==0.1.1` wheels.
- FEN corpus: first 30 games of the training split via `LichessDataset.stream`
  (same config as rollout generation), one FEN per ply, 2,087 positions.

## 6. Results (2026-07-18, branch cozy-chess-hotpaths)

Stage 1 and Stage 2 of the §4 staged plan are implemented and cut over
(commits `f8df0e9` and `e31504d`). Stage 0 (differential harness,
`tests/test_cozy_differential.py`) gates both. Each stage's A/B gate below is
a fixed-seed 20-game rollout (`config/imba_chess_exit_full.toml`,
`search_budget=2048`, `best_hr10_checkpoint_23_hr10=0.9564.pt`), run
sequentially with the GPU idle beforehand, `--profile` on. Per this repo's
run-to-run wall-clock variance (~±10% between separate sessions — compare
this section's 107.9s baseline in §1 against Task 3's 120.9s py-side run of
the *same* code path), the paired numbers **within** each task's own A/B are
the trustworthy comparison; absolute wall time across different sessions is
not.

HEAD (`e31504d`) is byte-identical to the code that produced Task 5's B-side
profile run below, so that run stands as the final post-Stage-2 measurement
— no re-profile was needed for this section. Final full suite check:
`.venv/bin/pytest -q` → **171 passed**, 81 warnings, matching every stage's
expected count.

### Stage 1 A/B (Task 3): forcing-move floor on cozy

py (`IMBA_SEARCH_FORCING=py`):
```
wrote 169 rollout rows from 20 games (skipped 0) to /tmp/tmp.JfJnGfs7qG/py.parquet
timing after 20 games / 169 positions (1371 search waves, 346093 search evals, total 120.9s):
  search_bookkeeping (heap/tree mgmt, CPU): 55.4s (45.8%)
  search_gpu (search forward_decode waves, GPU): 44.3s (36.6%)
  root_eval (root forward, GPU): 21.1s (17.5%)
  batch_build (tensor construction, sampled plies): 0.0s (0.0%)
  ply_bookkeeping (chess+encode, every ply): 0.0s (0.0%)
```

cozy (`IMBA_SEARCH_FORCING=cozy`):
```
wrote 169 rollout rows from 20 games (skipped 0) to /tmp/tmp.JfJnGfs7qG/cozy.parquet
timing after 20 games / 169 positions (1371 search waves, 346093 search evals, total 106.8s):
  search_bookkeeping (heap/tree mgmt, CPU): 42.6s (39.9%)
  search_gpu (search forward_decode waves, GPU): 41.3s (38.6%)
  root_eval (root forward, GPU): 22.9s (21.4%)
  batch_build (tensor construction, sampled plies): 0.0s (0.0%)
  ply_bookkeeping (chess+encode, every ply): 0.0s (0.0%)
```

Both runs: identical wave/eval/position counts (1371 waves, 346093 evals, 169
positions from 20 games, 0 skipped) — the two implementations chose the same
moves throughout the search tree, not just at the leaves. `search_bookkeeping`
45.8% → 39.9%; total wall time 120.9s → 106.8s (−11.7%).

Parquet equivalence: `pd.testing.assert_frame_equal(a, b)` raised nothing —
`EQUAL: 169 rows`, full frame equality including nested list columns.

### Stage 2 A/B (Task 5): dual cozy/python-chess boards threaded through the search tree

A-side (pre-change HEAD `3c0abd5`):
```
wrote 169 rollout rows from 20 games (skipped 0) to task5_a.parquet
timing after 20 games / 169 positions (1371 search waves, 346093 search evals, total 100.2s):
  search_gpu (search forward_decode waves, GPU): 42.2s (42.1%)
  search_bookkeeping (heap/tree mgmt, CPU): 39.6s (39.5%)
  root_eval (root forward, GPU): 18.4s (18.4%)
  batch_build (tensor construction, sampled plies): 0.0s (0.0%)
  ply_bookkeeping (chess+encode, every ply): 0.0s (0.0%)
```

B-side (working tree, Stage 2 changes applied — this is also the final,
post-cutover measurement, since HEAD is byte-identical to this run):
```
wrote 169 rollout rows from 20 games (skipped 0) to task5_b.parquet
timing after 20 games / 169 positions (1371 search waves, 346093 search evals, total 92.0s):
  search_gpu (search forward_decode waves, GPU): 41.4s (45.1%)
  search_bookkeeping (heap/tree mgmt, CPU): 32.2s (35.1%)
  root_eval (root forward, GPU): 18.2s (19.8%)
  batch_build (tensor construction, sampled plies): 0.0s (0.0%)
  ply_bookkeeping (chess+encode, every ply): 0.0s (0.0%)
```

Total wall time 100.2s → 92.0s (−8.2%). `search_bookkeeping` 39.6s → 32.2s
(−18.7% absolute, 39.5% → 35.1% of total) — a smaller relative drop than
Stage 1's, because Stage 2 both removes cost (python-chess `outcome()` calls
replaced by `terminal_value_fast`) and adds new cost (a cozy `copy.copy()` +
`.play()` at every tree-expansion site, not just where `_forcing_index_set`
needed a board). Net effect still a clear win; see Task 5's report for the
full accounting.

Parquet equivalence: `a.shape == b.shape == (169, 19)`;
`pd.testing.assert_frame_equal(a, b)` raised nothing — byte-for-byte
identical including nested list columns (moves/policy/value arrays). Same
config, checkpoint, and `--max-games 20` on both sides.

### Overall speedup and final bucket shares

Chaining the two tasks' paired runs (both driven from the same 20-game
fixed-seed workload, same checkpoint, same config): **120.9s (py, Task 3
A-side) → 92.0s (cozy, Task 5 B-side / final HEAD) ≈ 1.31x faster
end-to-end.** `search_bookkeeping`'s share of total wall time moved
45.8% → 39.9% → 35.1% across the two stages, its absolute time dropping
55.4s → 42.6s → 32.2s (−42% absolute across both stages combined).

Final-state bucket shares (Task 5 B-side, = current HEAD):
`search_gpu` 45.1%, `search_bookkeeping` 35.1%, `root_eval` 19.8%.

### Stage 3 go/no-go decision: **NO-GO**

The spec's own criterion (§4, Stage 3): "own thin PyO3 crate... only if
profile still says so" — i.e. justified only if `search_bookkeeping` remains
the *largest* bucket after Stage 1+2. It no longer is. At the final measured
state, `search_gpu` (45.1%) is larger than `search_bookkeeping` (35.1%);
GPU-side forward_decode waves, not Python/cozy CPU bookkeeping, are now the
single biggest cost. Stage 3's remaining candidates (evaluator-side movegen
in `_project_legal_logits`, a cozy-native `BoardStateEncoder`) would be
chasing a bucket that is no longer dominant — applying the criterion
honestly, there is no case for building the additional PyO3 crate right now.

This matches the spec's own Amdahl note from §1: eliminating *all* Python
chess cost caps rollout speedup at ~1.85x, and the 12–35x throughput gap
against the KL-coverage target needs cross-game batched search (the
lc0-shaped architecture) regardless of how far the CPU-bookkeeping share is
driven down. Stage 1+2's ~1.31x is real, equivalence-gated progress toward
that 1.85x ceiling, but the next big lever — for both the remaining Amdahl
headroom and the now-larger `search_gpu` bucket — is cross-game batched
search, not a deeper cozy port. Revisit Stage 3 only if a future change
(e.g. batched search reducing `search_gpu`'s share) makes
`search_bookkeeping` the largest bucket again.
