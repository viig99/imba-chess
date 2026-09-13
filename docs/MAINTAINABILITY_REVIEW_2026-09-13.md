# Maintainability review — 2026-09-13

Reviewed after implementation commit `d35a7096b278df88da58a30d82709a656ff346d4` was pushed to `origin/main`. This review proposes decisions; it removes no additional code, configs or historical evidence.

The strongest next cleanup is the stale experiment tooling and old implementation plans. Most large runtime modules still serve either current self-play, supervised training, or the established halving/Stockfish comparison. Removing them based only on size would remove useful capabilities.

Method: tracked-file inventory, source/CLI inspection, import and symbol references across `src`, `scripts`, `tests`, config existence checks, and comparison with documented current workflows. I did not run old launchers: several can start/stop jobs or write experiments. “No current consumer” refers to checked-in code; external scripts and manual use may exist. Counts include comments and blank lines and describe scope, not promised deletions.

## Inventory

| Tracked area | Files | Lines |
|---|---:|---:|
| `src/imba_chess` | 50 | 11,792 |
| `scripts` | 40 | 10,646 |
| `docs` (excluding root Markdown) | 53 | 20,681 |
| `config` | 8 | 722 |

The preceding test cleanup already replaced the circular terminal oracle and redundant hard-exit suites, preserved the native parity corpus without collection-time generation, removed opaque scheduling hashes, and separated extended checks. The default CPU suite passed 2,280 tests in 20.35 seconds; do not count these savings again in the proposals below.

## Decisions worth making

| Priority | Area | Evidence and scope | Recommendation / decision needed |
|---|---|---|---|
| 1 | Stale experiment launchers and probes | **13 scripts / 1,458 lines** reference deleted ExIt configs in executable defaults or argv. One paired benchmark explicitly refuses to reconstruct its removed baseline. Exact list below. | Remove from active `scripts/`, retaining Git history and a short result index. If a measurement is still valuable, port its scenario into the current explicit-input benchmark first. |
| 1 | Old implementation plans/specs | `docs/superpowers/plans`: **12 files / 12,329 lines**; `specs`: **12 / 2,870**. Several include patch recipes for features already implemented or retired. Five plans alone account for 8,540 lines. | Keep current architecture, decisions and measured results in short maintained documents. Retire implementation recipes to Git history, or move to a clearly historical directory with an index if discoverability matters. Moving alone changes navigation, not total size. |
| 1 | Contradictory current docs | README limitations still say games are sequential and training has no legal masking; those statements need stage-1/evaluation/stage-2 scope. Its history describes removed distillation alongside current operation. Readiness report is ~101 KB of cumulative chronology. | Shorten README to current entrypoints/protocols and split readiness into current status plus dated experiment records. Preserve settings, failed experiments and measured outcomes; remove repeated plans and obsolete runnable instructions. |
| 2 | Old offline halving-rollout subsystem | `generate_search_rollouts.py` **772 lines**, `data/rollout_store.py` **127**, and three direct test modules **638**: **1,537 total**. Generator says training no longer consumes these files. Source confirms neither current trainer uses `load_rollout_lookup`; only exports/tests reference that loader. Generator still grows `all_rows` and rewrites the entire output. | Decide whether offline halving-target analysis remains a planned experiment. If no, retire generator, schema/exports, direct tests and dependent launchers together. If yes, label it an analysis tool and retain one explicit-input command; do not route nightly stage-2 learning through it. Keep shared search/evaluator/scheduler code used elsewhere. |
| 2 | Older microbenchmarks | Six native/encoding/projection/suffix probes total **1,067 lines**. Several remain useful for native-boundary work. `bench_wave_suffixes.py` explicitly measures a former paths representation, with optional FBGEMM, rather than current arena storage. | Preserve independent native-boundary probes as experiments. Move obsolete representation benchmarks out of normal tooling or retire them with a result note. Do not delete all microbenchmarks merely because end-to-end benchmarking exists. |
| 2 | Parallel checkpoint-match implementation | `match_two_checkpoints.py`: **386 lines**; separately owns opening sampling, two actors, game handling and summary output. It uses deterministic halving and live training-corpus openings. Stage-2 evaluation uses prepared held-out seeds, Gumbel, persisted progress and stricter completion rules. | Consolidate only after deciding whether halving-vs-halving matches remain useful. Prefer a reusable match protocol with selectable search, but do not silently substitute Gumbel for a historical halving ruler. This is overlapping functionality, not an identical duplicate. |
| 3 | Stockfish evaluation controller | `eval_vs_stockfish.py`: **2,322 lines**, including synchronous and stepwise move selection (120/142 lines), single-process games and actor orchestration, CLI, persistence and reporting. Both execution routes are selected by current CLI options. | Reduce duplication behind shared evaluation/result interfaces. Keep the established Stockfish protocol operational during migration. Removing the actor route would currently remove concurrent halving evaluation. |
| 3 | Dataset/cache estimator | `estimate_lichess_cache.py`: **1,292 lines**, with remote/local discovery, sampling, uncertainty and reporting. README still recommends it; collection/training does not call it. | Decide whether corpus-sizing tooling is still used. Keep as an operational utility or trim report/options if not; lack of an import from training does not make a CLI dead. Lower priority than provably stale launchers. |
| 3 | Config recipes | Eight files / 722 lines, including v3/v4, Stockfish fine-tuning, original self-play, two-hour laptop pilot, faster laptop and 5090. Current self-play config IDs hash both settings and base-config bytes; existing checkpoints validate IDs on resume. | Keep v4, fast laptop and 5090 visible. Mark original/pilot configs as compatibility recipes if desired. Do not mutate or discard configs needed by resumable runs or older checkpoints merely to deduplicate constants. Decide older checkpoint retention first. |

## Stale scripts: concrete first removal set

These files are outside the current `run_self_play.py` → collector/trainer/evaluation path. Each references an absent `config/imba_chess_exit_*.toml` in runtime arguments/defaults:

- `bench_decode_wave.py` (118 lines)
- `bench_python_opts_paired.sh` (149): additionally checks that removed `_cozy_move_id_and_uci` exists and exits when it does not.
- `bench_root_eval.py` (174): default config is missing; its root benchmark also describes the old block-mask workload. A reusable root benchmark could be retained after rebasing to current inputs.
- `generate_rollouts_remote.sh` (51)
- `lr_probe.sh` (75)
- `nightly_alpha01_train_and_eval.sh` (42)
- `probe_bookkeeping_phases.py` (154)
- `probe_project_phases.py` (141)
- `probe_push_children.py` (158)
- `profile_torch_waves.py` (127)
- `rollout_budget_sweep.sh` (92)
- `rollout_equivalence_gate.sh` (101)
- `rollout_nightly_start.sh` (76)

The missing-config scan found **15** files, but two need different treatment:

- `calibrate_stockfish_nodes.py:49` sets its actual default to missing `config/imba_chess_exit_full.toml`, and main loads it. **Repair the default or require an explicit config**; calibration remains useful and has independent tests. This is a retained-tool defect, not a reason to delete calibration.
- `materialize_corpus.py:18` has an obsolete config in its docstring example. Its actual CLI accepts explicit inputs, and seed preparation still uses materialization. **Fix the example and keep the tool.**

`lr_probe_summarize.py` and `label_gate_diff.py` are companion utilities: review them with the workflows they serve. Their file-format analysis can still work independently, so they are not included in the 13-script broken-default set.

The separate native microbenchmark group is `bench_encode_cozy_micro.py`, `bench_move_id_micro.py`, `bench_native_move_projector.py`, `bench_project_decompose.py`, `bench_search_bookkeeping_decompose.py`, and `bench_wave_suffixes.py`.

## Preserve deliberately

- **Gumbel and halving controllers.** Gumbel is the stage-2 algorithm; halving is the existing measured comparison engine. Rerank/depth-two policies remain wired into the Stockfish CLI. Retire those only if their baseline role is explicitly dropped.
- **Full-forward, cached, grouped, eager and compiled decoder paths.** They form the independent numerical/reference chain and support different query patterns. Whole-decoder compilation is the CUDA self-play default. Experimental SDPA remains bounded to the benchmark path and preserves the measured negative/inconclusive result; moving it to an experiment module is possible, wholesale deletion is not necessary now.
- **Actor server/worker/protocol: 1,989 lines.** `eval_vs_stockfish.py` uses this subsystem for concurrency >1. It is unused by synchronous Gumbel collection but is not globally dead. Existing cache semantics are also a reference for future persistent actual-game roots.
- **Native Python reference functions.** They are the parity oracle for Rust and are now checked against independent python-chess outcomes. Deleting them would weaken correctness even when production invokes Rust.
- **Game renderer.** `game_animation.py` is used by the Stockfish evaluator when game saving is enabled; single-process saving is an active feature.
- **Checkpoint-compatible model heads and stage-1 data/loss behavior.** Stage-2 disabling moves-left supervision does not make that head dead for supervised training or old checkpoints.
- **Licenses and reference fixtures.** mctx provenance/license, fixed-noise fixtures and native chess cases preserve legal attribution and independent correctness evidence.
- **Historical negative results.** Alpha-beta/PVS runtime and its dedicated tests are already removed. Its handoff and 14 validation artifacts (1,145 lines) are evidence, not active code. Archive or summarize them if desired; do not recreate the removed implementation from the document.

## Proposed sequence after your decision

1. Approve retiring the 13 stale scripts; repair the retained calibration default and materialization example in the same focused cleanup.
2. Decide whether offline halving-target generation and halving checkpoint matches remain on the experiment list; remove or consolidate their complete dependency groups accordingly.
3. Approve documentation consolidation: a short current guide and an indexed historical record, with obsolete implementation plans recoverable from Git.
4. Keep configs and checkpoint-compatible runtime paths until their consumers/protocols are explicitly retired. Treat Stockfish-controller consolidation as a separate behavior-preserving refactor with real process/evaluation checks.

These changes would improve maintainability and discoverability. Except for the test improvements already measured, this audit makes no GPU-throughput or runtime-speed claim.
