# ckpt34: tactical coverage and bounded quiescence experiment

Date: 2026-09-05. Implementation update: the two independent switches, bounded
search behavior, CLI/TOML wiring, serial/actor diagnostics and regression tests
are now implemented. Match experiments below have **not** been run as part of
this implementation. This document remains the experiment protocol; provenance
manifests and match analysis are work for the actual experiment run.

Validation: the full repository suite passed (2,238 tests). After a final
terminal-root diagnostic-counter correction, the focused search/config/eval/
actor suite passed (162 tests). Nine pre-change trace fingerprints verify
legacy decisions, arm values and evaluator request ordering for off/0.
Four-way behavioral, CLI/TOML override, actor-protocol and JSON-output tests
pass. These checks do not establish playing strength or GPU match throughput.

Baseline update: the lambda/reply-count sweep continued after this document
was written. The 0.10/replies=2 settings below reproduce the historical anchor;
for a new four-way comparison, first freeze the configuration selected from the
completed sweep and apply the same lambda/reply overrides to every variant.
Do not mix those parameter changes with tactical/quiescence attribution.

## Objective and scope

Determine whether ckpt34 plays stronger with better tactical move coverage and a short quiescence extension, holding weights, value scale, policy penalty, and total neural search budget fixed. Implement the smallest experiment that answers this question. Do not retrain, change the value target, increase model size, replace sequential halving, or tune unrelated search parameters.

This document is intended to be executable by a fresh coding agent, including gpt-5.6-terra at high reasoning effort. Read the referenced implementation before editing. Finish implementation and correctness checks before running the match stages. A negative or inconclusive result is a valid completed experiment; do not keep tuning until something wins.

The workspace already has user changes. At handoff creation, modified files include README.md, config/imba_chess_v4.toml, two supervision handoffs, lichess_dataset.py and its tests; filter_annotated_corpus.py is deleted and .python-version is untracked. Preserve those changes. Create a new experiment config only if needed; do not rewrite the production v4 config.

## 1. Fixed baseline and verified facts

- Repository: `/home/vigi99/CodeDir/imba-chess`.
- HEAD at inspection: `fe8a23c2ec5b40278c3e406e7a723a573ac24e15`, with the dirty worktree described above. HEAD alone is not a complete experiment snapshot.
- Checkpoint: `artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt`.
- Checkpoint SHA256: `5844b09fdde268f5fd2aba363603c43e9c1020d776c2d5294a17c2f912962826`.
- Model config: `config/imba_chess_v4.toml`; use this to load the checkpoint, with explicit search overrides below.
- Historical result: `artifacts/eval/v4_ckpt34_sf2200_lam010_r2_750.json`.
- Historical W/D/L: 421/170/159 in 750 completed games; score 0.6746667 against strength-limited SF2200.
- Historical lambda was **0.10**, although the current TOML says **0.05**. Always pass `--value-rerank-lambda 0.10`.
- Opponent: `/usr/bin/stockfish`, one thread, hash 64 MB, UCI_LimitStrength=true, UCI_Elo=2200, 40000 nodes with a 5-second safety ceiling.
- Our search: budget 2048, top_m=16, rounds=0 (automatic), refutation_top_r=2, expand_top=3, max_depth=8, no Gumbel root sampling.
- Inference: CUDA, float32, compile enabled; six concurrent games for match runs; seed 42; max_plies=512; opening_random_plies=0.

Important corrections to the initial discussion:

1. `_TreeNode.depth=0` is the board **after** the root candidate move. `max_depth=8` can therefore evaluate positions nine plies from the original decision position. Preserve this convention. With two extra quiescence plies, the deepest node depth is 10, or eleven plies from the decision position.
2. Adding a move to the frontier does not guarantee it gets evaluated before the budget expires or its arm is eliminated. Measure generated and evaluated tactical children separately.
3. ckpt34 already contains two 64-dimensional board-square attention blocks. No board-encoder change belongs in this experiment.
4. Actor mode (`concurrent_games > 1`) is not bitwise deterministic and does not currently support PGN/debug tracing. A fixed seed does not make games paired across variants.
5. `ladder_elos="2200"` creates a strength-limited segment even though the TOML's standalone `stockfish_limit_strength` is false. Verify resolved JSON options, not that isolated config field.

## 2. Implementation map

| File / symbol | Work |
|---|---|
| `src/imba_chess/eval/search.py`: `HalvingConfig`, `_TreeNode`, `_Arm`, `_push_children`, `_backed_stm`, `_halving_stepwise` | Two opt-in controls, coverage selection, quiescence mode/backup, counters. Keep this module torch-free. |
| `src/imba_chess/eval/cozy_bridge.py`: `is_capture_cozy` | Reuse for quiescence capture classification; handles en passant and cozy's castling representation. |
| `src/imba_chess/config.py`: `EvalVsStockfishConfig` | Add defaults for the two controls. |
| `scripts/eval_vs_stockfish.py` | CLI parsing, config resolution, validation, HalvingConfig construction, segment and aggregate run_config serialization, summary metrics. |
| `src/imba_chess/eval/actor_worker.py` | Preserve search-row statistics currently discarded as `_rows`; accumulate and return compact metrics. |
| `tests/test_search.py`, `tests/test_search_stepwise.py` | Behavioral search and budget tests, sync/generator equivalence. |
| `tests/test_config.py`, `tests/test_eval_vs_stockfish.py`, `tests/test_actor_worker.py` | Defaults, flag propagation, actor serialization and metrics. |

`_build_worker_config()` already serializes `asdict(halving_config)`; workers reconstruct with `search.HalvingConfig(**dict)`. Verify this path rather than creating a separate actor-only search implementation. Trace how `_EvalSummaryFragment` is folded into the parent summary before adding metrics. Both execution paths must report the same metric definitions.

Do not alter `PositionEval` or the native projection ABI for this experiment. Existing `legal_forcing`, `move.promotion`, `bool(board.checkers())`, and `is_capture_cozy(board, move)` provide the needed information. No per-node conversion to python-chess/FEN in the production search loop.

Rollout generation must retain legacy defaults. These are evaluation experiments; exposing experimental rollout controls or changing the rollout-store schema is outside scope.

## 3. New switches (implement these exact names)

| CLI | Eval TOML field | HalvingConfig field | Default |
|---|---|---|---|
| `--search-tactical-coverage` / `--no-search-tactical-coverage` | `search_tactical_coverage` | `tactical_coverage` | false |
| `--search-quiescence-plies N` | `search_quiescence_plies` | `quiescence_plies` | 0 |

Use BooleanOptionalAction with CLI default None, then resolve against the config. Validate nonnegative quiescence_plies. Initial experiments use only 0 or 2. Preserve all existing search ordering, backup, defaults, and budget behavior when both controls are disabled. Additive diagnostic fields may differ, but legacy fields and decisions must agree.

Write both resolved controls into every segment and aggregate `run_config.search`, alongside the existing parameters. Persist checkpoint hash, Stockfish binary hash/version, actual concurrency, and an experiment/source identifier in the results or an accompanying manifest. Record the working-tree diff/config snapshot so a later run is reproducible; do not stage or commit unrelated changes.

## 4. Tactical coverage semantics

When `tactical_coverage=false`, preserve existing selection below the normal horizon.

When true:

1. At the original root, retain current top_m plus forcing moves. If the root is in check, additionally include **every mapped legal evasion**, including quiet king moves and blocks. Keep existing picks first, append missing moves in stable order, and avoid duplicates.
2. Below the root and before the normal horizon, if in check, enqueue every mapped legal evasion for either player.
3. Otherwise, on our turns use top expand_top **plus every capture/check/promotion**. Opponent turns keep top refutation_top_r plus every capture/check/promotion.
4. Give forcing children and all check evasions the parent's path priority (zero additional log-prior penalty). Other selected quiet children retain their normal path-log-prior increment. This extends the existing opponent forcing-floor treatment symmetrically. Keep heap counter tie-breaking deterministic.
5. Root arm scores still use `backed_value + 0.10 * root_log_prior`. Coverage does not remove the root prior penalty or guarantee a low-prior arm survives halving.

This is one coverage package, including eligibility and frontier priority. Do not claim a win isolates either component separately. Quiet non-evasion moves outside the top-k still remain a limitation.

## 5. Bounded quiescence semantics

This is a **budgeted quiescence extension inside the existing halving tree**, not a port of Stockfish's alpha-beta qsearch.

Let `D=config.max_depth`, `Q=config.quiescence_plies`:

- Q=0: run exactly the legacy algorithm.
- Q>0: nodes with depth < D use normal search selection, subject to the coverage flag.
- Nodes at depth >= D are quiescence nodes. At depth D through D+Q-1, enqueue only captures and promotions when not in check; when in check, enqueue every mapped legal evasion regardless of coverage flag.
- At depth D+Q stop expanding, even in check. This is a hard bound; count unresolved in-check cutoffs instead of silently extending again.
- Do not add ordinary non-capturing checks to quiescence. A capture/promotion that gives check is included; its evasions are included at the next level if bounds allow.
- Use `move.promotion is not None or cozy_bridge.is_capture_cozy(board, move)` for the non-check candidate set. `legal_forcing` alone is too broad because it also includes quiet checks.
- Every quiescence child inherits the parent's path priority. Use the same heap, rounds, batched EvalRequests, and budget accounting as normal nodes; no unbudgeted recursive evaluator calls or extra per-leaf allowance.
- Continue to use `cc.push_and_classify` and the existing hash-history stack. Exact checkmate/draw results override learned values and cost no neural evaluation.

### Backup: stand pat is essential

Add explicit per-node metadata for quiescence/stand-pat eligibility, set when the node is evaluated, **including at the depth-D boundary**. Do not infer it from whether the node has children.

For a nonterminal node with static side-to-move value `v`, collect `child_values = [-backup(child) for scored children]`:

```text
if terminal:                         return exact terminal value
if quiescence and not in check:       return max([v] + child_values)
if child_values:                      return max(child_values)
otherwise:                           return v
```

The non-check quiescence case is the stand-pat approximation: the player is not forced to make an unfavorable capture. Do not apply it to normal interior nodes, and do not allow it in check. Like conventional stand pat, it can be wrong in zugzwang; include quiet/endgame controls in diagnostics.

If in check and no child was scored because depth, budget, or arm allocation ended, use the existing static-value fallback and count it as unresolved. Do not return mate, draw, +/-infinity, or pretend that all evasions were searched. Terminal detection remains authoritative.

Terminal children count as scored children even if no evaluator call occurred. Set quiescence mode before any early return from child generation, so backup remains correct when every candidate capture is terminal or no capture exists.

### Budget and partial-search rules

All nonterminal evaluations, including quiescence descendants, debit the existing global budget. For each move, `sum(arm.evals_spent) <= budget`; the root policy/prefill evaluation is outside this legacy budget and must be reported separately when discussing time.

Do not reserve extra evaluations for quiescence in the first experiment. Count how often it is actually reached. If Q receives no work, the experiment did not test its intended mechanism; report that before proposing a separate scheduling experiment. Wider coverage can reduce reached depth under a fixed budget, which is a measured tradeoff rather than automatically a bug.

Keep the legacy partial-tree backup and halving elimination behavior outside quiescence. No transposition merging: this evaluator depends on history, so identical FENs are not sufficient cache keys.

## 6. Diagnostics required before expensive matches

Keep per-arm legacy fields and add compact counters. Aggregate across **all arms**, including eliminated arms, rather than reporting only the winner:

- actual neural evaluations; quiescence descendant evaluations (depth > D);
- normal-horizon nodes evaluated (depth == D);
- coverage-added children generated and actually evaluated (exclude candidates legacy selection would already include);
- check-evasion children generated and evaluated;
- maximum node depth and maximum quiescence extension reached;
- nonterminal in-check frontier leaves still unresolved when the search ends, with depth-cap versus other truncation distinguished;
- end-to-end model move-selection time, including root inference and search, excluding Stockfish's turn; mean and a clearly defined tail statistic if retained.

Measure final unresolved leaves once at search completion, not every recursive backup call. Ensure immediate-mate and zero-evaluation paths produce valid zero counters. Sum counts, take maxima for maximum depth, and pool timing totals by model turns when merging worker summaries.

Actor workers currently discard search rows. Forward only compact summaries to the parent, not every search tree. Use the same aggregation helper for the serial path where practical. Persist metrics even when debug_trace_games=0.

Interpret actor move latency as latency **including contention** at the stated concurrency. Also record total match wall time, completed games, and model turns. Do not call equal neural counts equal time or compare per-game runtime without accounting for changed game lengths.

## 7. Correctness gates

Use the existing material/fake evaluators and real legal boards to make small deterministic tests. These are algorithm tests, not claims about ckpt34's chess strength. A test may call the internal expansion/backup helpers to isolate a case, but also include end-to-end selected-move tests. Validate all supplied FENs and move sequences with python-chess.

Required cases:

1. Disabled controls reproduce legacy selected move, legacy row fields, and evaluator request order on fixed fixtures. Capture expected output before changing the algorithm; do not compare two new-code runs and call that a legacy regression gate.
2. A low-prior forcing move on our deeper turn is excluded by legacy top-three selection and considered with coverage enabled. With adequate budget, demonstrate the changed backed value/decision.
3. A quiet evasion outside top-k is included in check at the original root and on both sides below it. Check evasions need not themselves give check or capture.
4. The existing low-prior opponent refutation test still passes.
5. An exchange crossing a shallow test horizon is corrected with Q=2. Use D=1 or 2 for a compact fixture; do not require a contrived nine-ply line just to test the mechanism.
6. A bad optional capture cannot lower a non-check quiescence node below its static value; a favorable capture can improve it.
7. In check, static value cannot compete with an actually scored evasion. If every scored evasion loses, back up that loss; if none are scored, exercise and count the fallback.
8. Non-capturing checks are excluded in non-check quiescence; captures, en passant and all legal promotion choices are included; cozy castling is not a capture.
9. Depth D+Q is a hard ceiling, including repeated checks; max reported extension is Q. Q=0 preserves the original depth convention.
10. Tiny/zero budgets, many forcing root arms, partially evaluated siblings and early arm elimination never overspend. Terminal children consume zero evaluations and back up with the correct sign.
11. Existing mate, stalemate, repetition, fifty-move and insufficient-material behavior remains intact through the shared native terminal path.
12. Sync and stepwise search agree for all four flag combinations; actor config round-trips the new fields and preserves diagnostic counters. Check cached decode with a longer search suffix and a late-game prefix.

Run the focused suite first, using the existing environment:

```bash
cd /home/vigi99/CodeDir/imba-chess
.venv/bin/python -m pytest -q \
  tests/test_search.py tests/test_search_stepwise.py tests/test_config.py \
  tests/test_eval_vs_stockfish.py tests/test_actor_worker.py \
  tests/test_native_terminal.py tests/test_cozy_differential.py \
  tests/test_prefix_decode.py tests/test_grouped_decode.py
```

Then run `.venv/bin/python -m pytest -q` once after implementation is stable. Report unavailable-environment skips separately from passes. Do not install a new torch version or rebuild the native module merely for these Python changes.

## 8. Match commands

The shared command and the two experimental switches below are now available.
Complete correctness checks before running the expensive match stages.

Do not use `eval_best_checkpoint.sh`: it auto-selects a checkpoint and supplies `--no-compile`. Pin the exact checkpoint and compilation explicitly.

In one Bash session, define:

```bash
cd /home/vigi99/CodeDir/imba-chess
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN_DIR=$(mktemp -d /home/vigi99/CodeDir/imba-chess/artifacts/eval/ckpt34_tactical_XXXXXX)
echo "$RUN_DIR"
sha256sum artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt
sha256sum /usr/bin/stockfish

run_variant() {
  local tag=$1 games=$2 concurrency=$3
  shift 3
  .venv/bin/python scripts/eval_vs_stockfish.py \
    --config config/imba_chess_v4.toml \
    --checkpoint artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt \
    --model-move-policy value_search_halving \
    --value-rerank-lambda 0.10 --value-rerank-top-k 16 \
    --search-budget 2048 --search-top-m 16 --halving-rounds 0 \
    --search-refutation-top-r 2 --search-expand-top 3 --search-max-depth 8 \
    --stockfish-path /usr/bin/stockfish \
    --stockfish-time-sec 5 --stockfish-nodes 40000 \
    --stockfish-threads 1 --stockfish-hash-mb 64 \
    --ladder-elos 2200 --ladder-games-per-segment "$games" \
    --no-include-full-strength-segment \
    --device cuda --dtype float32 --compile \
    --concurrent-games "$concurrency" --seed 42 --max-plies 512 \
    --opening-random-plies 0 --debug-trace-games 0 --no-save-games \
    --output-json "$RUN_DIR/${tag}.json" \
    "$@"
}
```

Use a unique tag for every call: the evaluator can overwrite an existing output path. Run matches sequentially on the same idle host, not concurrently with training or other variants. If six actors OOM, use four for **every** compared run and record the change; do not reduce only the candidate's concurrency.

### Stage 0: historical comparison and smoke

Optional pre-implementation reference run: `run_variant legacy_before 2 1`. This is a plumbing check, not a strength estimate. Save deterministic fake-evaluator reference traces separately as required above.

After implementation, run four two-game serial smokes:

```bash
run_variant smoke_a 2 1 --no-search-tactical-coverage --search-quiescence-plies 0
run_variant smoke_b 2 1 --search-tactical-coverage --search-quiescence-plies 0
run_variant smoke_c 2 1 --no-search-tactical-coverage --search-quiescence-plies 2
run_variant smoke_d 2 1 --search-tactical-coverage --search-quiescence-plies 2
run_variant smoke_actor_d 2 2 --search-tactical-coverage --search-quiescence-plies 2
```

Check JSON contains resolved controls and diagnostics, every selected move was legal, games complete, budget limits hold, and Q work occurs on at least some positions. A targeted deterministic exchange fixture must exercise Q even if the smoke games do not. For a serial traced run, override with `--save-games --debug-trace-games 2 --save-games-dir "$RUN_DIR/trace_d"`; do not enable tracing for actor runs.

### Stage 1: four-way screen, 400 games total

| Variant | Tactical coverage | Extra Q plies | Comparison |
|---|---|---|---|
| A | off | 0 | Fresh legacy baseline |
| B | on | 0 | Coverage package alone |
| C | off | 2 | Quiescence alone |
| D | on | 2 | Combined |

```bash
run_variant screen_a 100 6 --no-search-tactical-coverage --search-quiescence-plies 0
run_variant screen_b 100 6 --search-tactical-coverage --search-quiescence-plies 0
run_variant screen_c 100 6 --no-search-tactical-coverage --search-quiescence-plies 2
run_variant screen_d 100 6 --search-tactical-coverage --search-quiescence-plies 2
```

Use the screen to identify broken mechanisms, large regressions, and one candidate for confirmation. It is too small to establish modest Elo gains. Select the highest-scoring nonbaseline candidate without correctness issues, inspect its coverage/Q usage and latency, then freeze the candidate. Do not pool screening games into its confirmation estimate. If all variants lose or fail to exercise the intended mechanism, report that result and inspect diagnostics before spending on confirmation.

### Stage 2: independent confirmation, 1500 games total

Repeat A and the frozen candidate with 750 games each and a fresh seed (43). Example below assumes D won the screen; substitute B or C's flags if appropriate:

```bash
run_variant confirm_a 750 6 --seed 43 --no-search-tactical-coverage --search-quiescence-plies 0
run_variant confirm_d 750 6 --seed 43 --search-tactical-coverage --search-quiescence-plies 2
```

A fresh baseline is essential. Historical 0.6747 is context, not a control against a potentially different Stockfish binary or execution environment. Do not restart a run merely because its early score looks bad.

### Stage 3: time-cost check if confirmation is promising

The primary experiment compares equal neural budgets. Report actual average evaluations, reached depth, model-turn latency, and game throughput beside strength. If the candidate is slower, it is not yet a demonstrated improvement at equal time.

For a bounded follow-up, measure candidate budgets 1024, 1536, and 2048 on the same fixed, replayed position prefixes after warmup; choose the largest budget whose mean model-selection latency does not exceed baseline budget 2048. Freeze that budget before another match comparison. Existing `--search-budget` can override the shared function's value. Replay complete histories through the existing model evaluator, not isolated FENs: the network uses history.

This is an approximate time-matched test, not a hard per-move clock. Do not introduce a new time-control/search scheduler in the first implementation. If fixed-prefix timing tooling is needed, add a small dedicated script with documented input prefixes and output timings; otherwise report equal-budget results and the unresolved time tradeoff honestly.

## 9. Analysis and stopping rule

Report a table with variant, W/D/L, completed/incomplete games, score, score difference from fresh A, uncertainty, mean actual evaluations, Q usage, maximum depth, and latency. Report results by color too. Nonzero incomplete games invalidate a clean strength comparison; diagnose the cause rather than silently dropping them.

For completed games, assign scores X in {1, 0.5, 0}:

```text
n = W + D + L
s = (W + 0.5*D) / n
sample_variance = (W + 0.25*D - n*s*s) / (n - 1)
SE(s) = sqrt(sample_variance / n)
SE(candidate - baseline) = sqrt(SE(candidate)^2 + SE(baseline)^2)
approximate 95% interval = difference +/- 1.96 * SE(difference)
```

These intervals assume independent games and are approximate: repeated starting positions and actor nondeterminism limit interpretation. Do not use a paired test just because seeds match. Prefer raw score differences to claiming an absolute human Elo. If the confirmation interval crosses zero, call the result inconclusive. A positive interval supports improvement under this specific SF2200/equal-budget protocol, not universal superiority.

For a convincing positive result, subsequently validate on diverse, fixed opening prefixes with color reversal and against another opponent strength. The current harness only exposes uniform random opening plies; that is not an implemented paired opening suite. Building that suite is a separate follow-up and not required to answer the initial bounded experiment.

Do not change production defaults automatically from the 100-game screen. Finish with the tested patch, exact commands/artifact paths, correctness results, confirmation outcome (or why it was not run), runtime tradeoff, and one recommendation: retain baseline, retain candidate as experimental, or recommend adoption supported by the measured evidence.

## 10. Deliverables checklist

- [x] Two flags work in serial and actor evaluation; disabled defaults preserve legacy behavior.
- [x] Coverage and quiescence semantics match sections 4–5, including stand pat, evasions, depth convention and budget bounds.
- [x] Meaningful correctness tests pass; environment limitations are recorded.
- [x] Metrics persist in JSON for both execution paths and explain whether the added mechanisms ran.
- [ ] Checkpoint and engine provenance recorded; historical result/config discrepancy handled.
- [ ] Smokes and four-way screen completed on available hardware; one frozen candidate confirmed if warranted.
- [ ] A results note, preferably `docs/CKPT34_TACTICAL_SEARCH_RESULTS.md`, records evidence and limitations without implying unrun tests succeeded.
- [ ] Production v4 weights, training recipe, and unrelated user edits preserved.

## Background references

- [NNUE guide](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md): motivation for evaluation/search co-design.
- [Stockfish search.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp): mature search and quiescence reference. This experiment deliberately specifies a smaller budgeted extension compatible with our existing evaluator and scheduler.
- Local historical match JSON and search code listed above are the authoritative baseline for this experiment; older planning documents may describe superseded implementations.
