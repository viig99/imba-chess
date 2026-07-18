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
  gives_check share. Behind a flag; fixed-seed rollout-equivalence run + full
  test suite + `--profile` re-measurement before/after.
- **Stage 2 — node expansion + movegen + terminal check on cozy.** Search-tree
  nodes carry cozy boards (converted once per search root); python-chess
  remains at the interface (root board in, `chess.Move` out). Terminal
  detection = cozy `status()` + existing halfmove guard + Zobrist-history
  repetition check (cozy `hash()`), differentially tested against
  `terminal_value_for_color`. Attacks the ~10% outcome share + copy/push +
  movegen. This is the bulk of the win.
- **Stage 3 (only if profile still says so) — own thin PyO3 crate.** If binding
  maturity becomes a liability or we want `expand_node`-style coarse calls /
  a cozy-native `BoardStateEncoder`, vendor a small maturin crate in-repo over
  cozy-chess (MIT). Not needed to start.

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
