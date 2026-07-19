# Stage 3: cozy-native search tree (design, 2026-07-18)

## Problem (profile-backed)

After cross-game batching (see `2026-07-18-cross-game-batched-search-design.md`
Results), rollouts are CPU-bound again: search_bookkeeping 55-65% of wall
time at the production config (fp32, G=8, 3.1s/game local). A fresh cProfile
(10 games, G=8 fp32, 2026-07-18) decomposes that bucket:

1. **python-chess legal movegen + UCI mapping in `_project_legal_logits`
   (~15-18%)** — `list(board.legal_moves)` on the py board for every
   evaluated node (155k nodes → 4.76M movegen generator calls, 7.5M
   `move.uci()` string builds for vocab mapping).
2. **The python-chess half of `_dual_push` (~11%)** — 487k py copy+push per
   tree edge. The py boards exist in the tree ONLY to feed item 1 and the
   encoder; chess rules already run on the cozy twins.
3. **Translation churn (~4-5%)** — `py_move_to_cozy` 2.6M calls/edge.
4. The original hot spots are now small: cozy gives_check ~3.5%, terminal
   ~4% (py `outcome()` fallback fires only ~5.6k times/10 games).

Stage 3 removes python-chess from the search hot path entirely: the tree
becomes cozy-only, deleting items 1+2+3 (~30-35% of remaining CPU).
Projected: ~2.3-2.5 s/game locally; same relative gain on the CPU-starved
remote box. `eval_vs_stockfish` inherits everything via shared `search.py` /
`CachedPositionEvaluator`.

Approach decision (user-approved): full cozy-native tree with native
repetition handling — over (b) partial (movegen+encoder only, keeps the 11%
dual-board cost and the permanent dual-board invariant) and (c) a custom
Rust `expand_node` crate (unnecessary: the remaining cost is Python-side
movegen/copies/strings, which the existing binding removes).

## Step 0 (prerequisite): canonical legal-move order

python-chess and cozy generate legal moves in different orders, and order is
load-bearing: Gumbel root sampling draws noise per index in list order, and
prior-sort ties break by index. Naively swapping movegen would silently
reassign rng draws — no byte-identity, and a hard-to-review behavior change.

**Step 0 sorts legal moves by UCI string inside `_project_legal_logits`**
(and anywhere else a legal-move list is built for search), BEFORE any cozy
migration. Effects:
- One-time, distribution-identical relabeling of which Gumbel draw lands on
  which move (resampling, not a policy change). Gated statistically (arm-level
  distribution checks), like the fp32 adoption.
- After Step 0, sorted-by-UCI is trivially identical from either movegen, so
  every subsequent Stage-3 step is gated **byte-identically** against a fresh
  post-Step-0 20-game baseline parquet.

Implications assessed (user-reviewed): trained checkpoints unaffected (model
consumes board tokens + static vocab ids, never list positions); existing
rollout parquets remain valid and mixable (schema stores uci strings/values/
log-priors; Phase-1b softmax-over-arms target is order-invariant); coverage
bookkeeping keys on game ids. Deterministic eval play changes only through
exact prior ties (essentially never); old ladder runs stay comparable within
noise but are not bitwise-reproducible from new code (git history serves
that need).

## Design

### 1. Encoder

`BoardStateEncoder.encode_cozy(cozy_board) -> BoardState`: piece ids scanned
from `int(board.colors(c) & board.pieces(p))` raw bitboards (same scan
pattern as the tuned py path); castle id from `castle_rights(color)`; ep file
from `en_passant()`; halfmove/fullmove buckets from `halfmove_clock`/
`fullmove_number` getters (all verified present on cozy Board). The py
`encode()` stays untouched — the data pipeline (PGN world) keeps it; both are
live paths. Gate: `encode(py) == encode_cozy(cozy)` over the FEN corpus AND
along played game lines where the cozy board was reached via `play()` rather
than conversion — the played-line variant exists specifically to catch
en-passant semantics drift between the libraries. Survey item for the plan:
confirm which ep mode (`fen`/`xfen`/`legal`) the production config uses and
match its semantics exactly.

### 2. Evaluator

`_project_legal_logits` goes cozy: `generate_moves()` + `cozy_move_to_uci`
+ vocab dict, list sorted by UCI (Step 0's canonical order).
`PositionEval.legal_moves` becomes `list[cc.Move]` internally;
`CachedPositionEvaluator.evaluate` receives cozy boards and encodes via
`encode_cozy`; `extend` consumes the uci via `cozy_move_to_uci`. The PUBLIC
search API is unchanged: callers pass a python-chess board + root legal-move
list and get an index back — root py moves map 1:1 to cozy moves by index
(one `py_move_to_cozy` pass per search call, ~30 moves).

### 3. Tree and terminal (the fiddly part)

`_TreeNode` carries a cozy board plus a parent-linked Zobrist hash chain
(plain ints via `board.hash()`, chain restarted at irreversible moves,
detectable by `halfmove_clock` reset). The search root seeds the chain with
the real game line's hashes since its last irreversible move (the game
coroutine maintains this; eval_vs_stockfish's driver likewise from its game
board — survey item).

Terminal detection goes fully native, replacing `terminal_value_fast`'s
python-chess fallback:
- Checkmate/stalemate: cozy `status()` (proven in Stage 2).
- Insufficient material: implemented from cozy bitboards replicating
  python-chess's exact `is_insufficient_material()` rule; differential-gated.
- Fifty-move claim: `halfmove_clock >= 100` (checked after status(), matching
  python-chess's checkmate-takes-precedence ordering).
- Threefold claim: count of current hash in (game-line seed + path chain)
  >= 2, PLUS python-chess's "claimable one reversible ply early via a move
  that reaches the third occurrence" rule — probing child hashes via cozy
  copy+play (~200ns each) at `halfmove_clock >= 7` nodes only (the existing
  guard bound). This replicates the exact semantics the current code gets
  from `outcome(claim_draw=halfmove>=7)`; the differential harness (replayed
  shuffle-heavy games + adversarial repetition fixtures, python-chess as
  oracle) gates it BEFORE search.py is touched.

### 4. Deletions (no-dead-code rule)

`_dual_push` collapses to a single cozy copy+play (the py half and
`_search_copy`'s tree usage are deleted); per-edge `py_move_to_cozy` churn
goes (root-only remains); `terminal_value_fast`'s py-board parameter and
fallback go. Survivors, deliberately: `board_to_cozy` (root conversion +
oracle tests), py `encode()` (data pipeline), the full differential harness.
`IMBA_DUAL_PUSH_VERIFY` is renamed/retargeted to verify the cozy board vs a
py-chess replay oracle in debug mode (or deleted if the harness makes it
redundant — plan decides, no orphaned flag either way).

### 5. Gates

- **Step 0**: statistical gate (arm-level distribution equivalence, move-set
  identity per position) + fresh 20-game fp32 G=8 AND G=1 baseline parquets.
- **Each subsequent step**: differential-harness extensions first (encoder
  equivalence incl. played lines; insufficient material; repetition/claim
  semantics), then the **G=1 byte-identical rollout gate vs the Step-0
  baseline** — the machinery that has caught every real bug in both prior
  projects.
- **Final**: 20-game fp32 G=8 profile re-measurement + cProfile decomposition
  (expect items 1-3 gone); full suite; `eval_vs_stockfish` smoke (few games)
  to confirm the shared-code path runs.

## Out of scope

- Remote 5090 G-sweep / shards×G retune (separate session; biggest remaining
  throughput jump).
- Layer-3 acceptance run (train on new-pipeline corpus vs old, SF-ladder
  parity) — unchanged standing item before remote fleet production use.
- Eval-driver batching; kernel-level GPU work (post-batching shapes make it
  possible, but GPU is no longer the bottleneck).

## Results (2026-07-18, branch cozy-native-tree)

### Gates
- **Step 0** (canonical UCI move order): validated two ways — deterministic-mode
  A/B showed **100% identical arm sets** (2686/2686; ordering changed
  scheduling, never consideration) with 98.2% move agreement; Gumbel-mode
  agreement 91.1% (expected resampling band). Learning recorded: the plan's
  shared-arm value-delta prediction was wrong — halving budget allocation
  depends on the full arm set, so agreement + arm-set identity are the
  meaningful gates, not value deltas. New baselines minted.
- **Task 4** (cozy evaluator): G=1 rollout **byte-identical** vs Step-0 baseline.
- **Task 5** (tree cutover): G=1 AND G=8 rollouts **byte-identical** vs Step-0
  baselines; eval_vs_stockfish functional smoke healthy (games playing and
  winning; the compile-on Inductor failure predates this work — nightly
  already runs `--no-compile` and tracks it as separate follow-up).
- Differential-harness additions along the way: encoder equivalence incl.
  played lines and all three ep modes; exact insufficient-material
  transcription; native repetition/fifty-move claims oracle-gated on 800
  replayed games (25 draw-claims exercised) plus curated phantom-ep and
  capturable-ep fixtures; mate-attribution tests at all four color_is_stm
  sites. Suite 198 → 212.

### Key semantic findings (recorded for posterity)
- cozy `hash()` includes phantom (uncapturable) ep flags; python-chess's
  transposition key does not → repetition counting uses a canonical
  `repetition_hash` (native `hash_without_ep()` when ep is phantom).
- cozy `status()` auto-draws at halfmove 100 with checkmate precedence (the
  spec's explicit fifty-move branch was dead code); the one-ply-early fifty
  claim needs `generate_moves()` truthiness, not `status()==Ongoing`.
- `board_to_cozy` previously dropped raw ep on capturer-less double pushes —
  fixed (conversion now matches cozy native play semantics).

### Throughput (20-game set, fp32 G=8 production config)
| Stage | total | s/game |
|---|---|---|
| Pre-Stage-3 (clean, post-batching) | 62.4s | 3.1 |
| **Post-Stage-3** | **37.2s** | **1.86** |

**1.68x from Stage 3 alone.** Bookkeeping bucket 34.9s → 12.1s. Bucket shares
now: root_eval 39.4%, search_bookkeeping 32.6%, search_gpu 27.8%. cProfile
(10 games): 100.1s → 68.2s profiled; python-chess movegen (was ~15s), py
push (~5s), `uci()` strings (~2.6s), per-edge translation (~3.2s) all gone;
no single remaining CPU item exceeds ~5% — the hot path is genuinely thin.

Cumulative project arc (same 20-game workload, local 3070 Ti): ~5.9 s/game
(2026-07-15 baseline) → 1.86 s/game ≈ **3.2x**, at fp32 fidelity throughout
the final pipeline. Single-process local rate now ~1,935 games/hr.

### What's next (unchanged priorities)
- Remote 5090 session: G-sweep + shards×G retune — with CPU cost now thin,
  large-G on the 32GB card should finally approach the coverage target.
- Layer-3 acceptance run before production training on the new pipeline.
- No Stage-4 CPU work is currently justified: the profile has no dominant
  single item left; next levers are GPU-side (root_eval 39.4% — batched
  further by larger G) and operational (remote scale-out).
