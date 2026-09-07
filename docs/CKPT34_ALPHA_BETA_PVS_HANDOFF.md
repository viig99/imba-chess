# ckpt34 alpha-beta / PVS handoff

## Status and stage boundaries

Implemented the evaluation-only architecture alongside halving. No policy defaults,
training code, or rollout policy selectors were changed by this work.

| Stage | Commit | Change | Recorded gate |
|---|---|---|---|
| 1 | `26aa5c1` | Iterative alpha-beta, budget aborts, persistent nodes, evaluation integration | 29 core + 122 regression tests |
| 2 | `73ae249` | Independently selectable floating-point PVS | 104 tests + 4 KV/protocol integration tests |
| 3A | `d6009ed` | Per-turn exact-context score cache | 116 tests |
| 3B | `063ae18` | Guarded LMR, verification and selective bounds | 138 tests |

Gate logs are in [validation/ckpt34-ab-pvs](validation/ckpt34-ab-pvs).
Final validation recorded 1,001 passing regression tests, 21 integration tests,
68 core tests, 108 core/shared-adapter regression tests, and 80 final
root-output/integration regression tests (these sets overlap).
Later integration/diagnostic tests exercise all variants together. These are CPU
correctness gates, **not chess-strength screens**. Screens have not run: `nvidia-smi` cannot communicate with the driver and
PyTorch reports `cuda_available=False`. The exact ckpt34 was found in
`artifacts/checkpoints_v4` (not the older `artifacts/checkpoints` directory).
It uses `config/imba_chess_v4.toml`: model dimension 1024, 16 heads, 8 layers,
attention dimension 64. No other checkpoint was substituted. A real ckpt34 CPU compatibility smoke
loaded that configuration and exercised all five new-policy variants at budget 32,
depth 2, fp32, without compilation. Its JSON is recorded alongside the test logs;
this only verifies checkpoint/decoder compatibility, not playing strength.

The working tree already contained changes to README, four TOML configs,
`config.py`, halving search, and their tests. Those changes remain outside these
commits. Campaign snapshots include the working-tree patch because those settings
matter to reproducibility. The fixed experiment CLI overrides the relevant legacy
control settings explicitly.

## Public configuration and drivers

Select `value_search_alphabeta` or `value_search_pvs` with
`--model-move-policy`. Both use `--search-budget` and `--search-max-depth`.

| CLI | TOML `[eval_vs_stockfish]` | Default |
|---|---|---|
| `--[no-]search-iterative-deepening` | `search_iterative_deepening` | true |
| `--search-score-cache off/context` | `search_score_cache` | off |
| `--[no-]search-lmr` | `search_lmr` | false |

CLI values override TOML, including explicit false values. LMR requires PVS.
Context score reuse requires one of the two new policies. Tactical coverage and
nonzero quiescence remain halving-only; incompatible settings fail before model
loading/match startup. Missing value heads fail at checkpoint loading.

The torch-free controller is `src/imba_chess/eval/alphabeta.py`:
`search_stepwise` yields existing `EvalRequest` objects; `select_value_search`
is its synchronous driver. Serial selection, scheduler decode waves, and actor
workers all dispatch to this core. A search requests one new position at a time.
Independent games still batch in the existing server; no sibling speculation or
second GPU server was introduced.

For compatibility with existing evaluation plumbing, the internal argument and
worker wire key named `halving_config` carry either `HalvingConfig` or
`AlphaBetaConfig`; the policy selects the deserializer. This is an internal name,
not a claim that halving controls affect alpha-beta.

## Scores, depths, complete results

Model values remain floating-point `P(win)-P(loss)` in [-1,1]. Non-finite values,
out-of-range values, malformed move arrays, and incomplete native legal coverage
raise position-specific errors. Policy orders moves but never changes backed-up
scores. A completed iteration's best move comes first at each node, followed by
descending log policy probability with UCI ties. Equal scores retain the first
searched move.

Checkmate at root-relative ply p scores `-(10000-p)` for the side to move. Draws
score zero. Mates dominate model estimates, shorter wins are preferred, and forced
losses are delayed. Root mating moves are inspected before any budgeted decode;
proving an immediate mate completes depth 1 and stops with `terminal_position`.
Native child terminal/draw-claim handling and the existing root terminal oracle
are retained, including repetition history. There is no new quiescence: checked
nonterminal horizon positions receive their static neural value.

For the new policies depth 1 evaluates the position after our move; depth 2 includes
an opponent reply. Selectable ceilings are 1..128. Halving keeps its convention of
counting below the candidate root move: halving depth 8 and new depth 9 have the
same nominal maximum root distance. Configured ceilings remain ceilings, not a
promise of completion under budget. The campaign passes depth explicitly.

`SearchReport` contains chosen root index (None for terminal root), score, completed
and attempted depths, legal UCI PV, stop reason, counters, and a selective marker.
Iterative deepening publishes only completed iterations. Budget abort returns the
last completed result. With no completed iteration, including interrupted fixed
depth, the highest-policy root move is returned with score None and empty PV.
No interrupted backed-up score is published.

Stop reasons: `depth_ceiling`, `evaluation_budget`, `terminal_position`, and
`no_completed_iteration_fallback`. The fallback counter counts move decisions,
not recursive abort frames.

## Neural budget, KV, and context safety

The budget counts actual new non-root decode rows. A needed intermediate position
must be evaluated to construct its KV state before any child can decode, even when
its value is not a leaf. Every such row counts. Root prefill, terminal checks, and
raw-evaluation hits are excluded. Requests are checked against budget before being
yielded; aborts use an exception path rather than an invented score.

Children are interned by parent identity and legal move within a move decision.
They retain native board, repetition history, terminal result, opaque evaluator
handle, raw `PositionEval`, and completed-iteration ordering hints. Each canonical
node decodes at most once, across iterative passes and PVS re-searches. Equal board
hashes on different paths do not identify a node. The controller is released at
move completion; actor turn release and the existing persistent game-prefix
lifecycle remain in place.

PVS searches the first child with its full window, later children with a scout
window using `math.nextafter(alpha, math.inf)`, then re-searches improvements strictly
inside the full window. Re-search visits are counted separately from fresh decodes.
With LMR off, PVS and alpha-beta agree at equal completed depth and leaf rules;
raw-evaluation budgets can lead them to complete different depths.

The separate LRU score cache has a 65,536-entry per-turn limit. Keys contain stable
node identity, exact remaining depth, and search profile (including PV status when
LMR is enabled). Exact scores or sufficient fail-high/fail-low bounds may cut off.
Bounds are classified against the original call window. Exact entries retain a
complete PV. Different depths do not substitute for one another. Interrupted nodes
are never stored; completed descendants can be reused. Board-hash repetition is
only a diagnostic. This cache is not a board-only transposition table and does not
replace policy ordering; no high transposition-hit-rate claim is made.

## Guarded reductions

Only PVS non-PV nodes, away from root and out of check, at remaining depth >=3,
may reduce the fourth or later ordered move. Captures (including en passant),
promotions, and checking moves are excluded. The child is searched at depth-2
instead of depth-1. Any result exceeding alpha is verified at normal child depth;
ordinary wider-window PVS re-search then follows when needed.

Unverified reductions mark their result selective. The marker propagates
conservatively through dependent backups, including the root report. Such cached
entries can guide ordering but never produce score cutoffs. Fully verified,
unaffected results remain usable. LMR deliberately changes leaf coverage;
full-depth minimax equivalence is not claimed with LMR enabled.

No null-move pruning, futility pruning, aspiration windows, board-only score reuse,
reduction tables, or additional pruning was added.

## Diagnostics and completion accounting

Per-move reports persist in `search_reports` with game index and ply. Counters include
new neural evaluations, raw-evaluation hits, recursive visits/by remaining depth,
completed/attempted depth maxima, fallbacks, alpha-beta cutoffs, PVS scouts and full
re-searches, context-cache probes/hits/cutoffs, reductions, full-depth verifications,
selective results, board-hash repeats, and controller elapsed time.

`game_records` retain game index, result, completion flag, color, and plies.
Aggregate games and incomplete games remain explicit; failures still abort the
existing match runner rather than disappearing from scores. The campaign wrapper
persists exit code/status and logs even when a match fails. A failed run is not a
strength result. Max-plies games remain incomplete rather than being recoded draws.

`inference_stats` record actual execution boundaries. Scheduler keys use `root_*`
and `decode_*`; actor keys retain `root_*`, `incremental_root_*`, and `wave_*`.
Batch histograms use `*_batch_size_N` (number of actual calls with N rows).
Actor incremental roots have dedicated counters, as they reuse game-prefix state.
Root requests/prefills are separate from budgeted `wave_rows`/`decode_rows`.
Counters sum and `max_*` depths aggregate by maximum. Outer selection timing includes
root inference; controller timing measures only the search. Timings are host elapsed
times, not isolated CUDA-kernel timings.

Root top-m, own expansion, opponent reply cap, halving rounds, and lambda are
reported as `not applicable` in new-policy result configuration. No root-prior
penalty or branching cap is used by either new policy.

## Reproduce validation

From the repository root with the native extension installed in `.venv`:

```bash
.venv/bin/pytest tests/test_alphabeta.py tests/test_alphabeta_integration.py -q
.venv/bin/pytest tests/test_search.py tests/test_search_stepwise.py tests/test_search_tactical.py tests/test_actor_server.py tests/test_actor_worker.py tests/test_eval_vs_stockfish.py tests/test_eval_vs_stockfish_hard_exit.py tests/test_config.py tests/test_batch_scheduler.py tests/test_native_terminal.py tests/test_cozy_differential.py -q
```

The core tests cover exhaustive minimax at depths 1..4, varied/tied ordering,
adjacent floats, fail-low/high, scout re-search, all request-budget boundaries,
shallow refutation, legal PVs, no duplicate decode, context isolation, cache bounds,
depth/profile mismatch, LRU eviction, interruption, LMR exclusions/verification,
selective cache safety, mate/draw claims, repetition, castling, promotion, en passant,
checked static horizons, and missing vocabulary coverage. Integration tests cover
real CPU model KV, serial/scheduled agreement, worker protocol, two-process actor
cleanup, settings precedence, unsupported settings, and persisted diagnostics.

## Reproduce the screening campaign

Prepare without starting matches (a fresh output directory is required):

```bash
.venv/bin/python scripts/run_ckpt34_ab_pvs_campaign.py --output artifacts/eval/ckpt34-ab-pvs-prepared
```

After restoring CUDA, run in a **new** directory:

```bash
.venv/bin/python scripts/run_ckpt34_ab_pvs_campaign.py --checkpoint artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt --output artifacts/eval/ckpt34-ab-pvs-screen --run
```

The wrapper records checkpoint/Stockfish SHA256, source revision, source archive,
working-tree patch/status, config snapshot/hash, Python/PyTorch/platform/GPU,
exact commands, per-run logs, JSON results, exit codes, and completion status.
The source archive is committed HEAD; apply the accompanying working-tree patch
to reproduce local tracked changes. The archived config includes the inherited
settings; explicit CLI flags fix experiment conditions. Preserve the full output
directory alongside the checkpoint and binary identified by the hashes.

Each variant gets a two-game startup smoke, then a 100-game seed-42 screen:

1. Fresh halving control: budget 2048, top-m 16, own expansion 3, replies 4,
   lambda .05, depth 8, coverage off, Q=0.
2. Alpha-beta: budget 2048, depth 9, iterative deepening, cache off.
3. PVS: same, cache off and LMR off.
4. PVS + context cache, LMR off.
5. PVS + LMR, cache off.
6. PVS + context cache + LMR.

All use SF2400, 40,000-node/5-second limit, one Stockfish thread, 128 MiB hash,
fp32, compilation enabled, four concurrent games, alternating model colors,
standard start, zero random opening plies, seed 42, and a 512-ply cap. These
explicit conditions must stay fixed within comparisons. No 750-game confirmation
or automatic default adoption is scheduled. The wrapper stops if a smoke/screen
fails or leaves incomplete games; investigate rather than silently dropping them.

Compare 1 vs 2, 2 vs 3, 3 vs 4, 3 vs 5, then 6 after individual ablations.
Equal neural-evaluation budget is the strength objective, not equal thinking time.
Report score with game completion, completed-depth distributions, budget utilization,
and secondary latency/throughput. A slower backend is not automatically rejected.
One hundred games are screening evidence only; identify promising candidates and
propose fresh-seed confirmation separately. **No strength conclusion or adoption
recommendation is justified by the CPU tests alone.**
