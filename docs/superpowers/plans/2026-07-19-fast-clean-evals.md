# Fast Clean Evals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batched, node-limited, fp32 SF-ladder evals per `docs/superpowers/specs/2026-07-19-fast-clean-evals-design.md`.

**Architecture:** `eval_vs_stockfish.py`'s `_run_segment` game loop becomes a coroutine driven by the existing `BatchScheduler`, reusing the rollout merged executors for `root_eval`/`decode_wave` plus a new `sf_move` kind whose executor fans out to G slot-owned Stockfish engines via a thread pool. A calibration probe fixes the node budget; controller-run gates and the re-anchor runs finish the protocol switch.

**Tech Stack:** Python 3.13, python-chess `chess.engine`, existing scheduler/executors, pytest.

## Global Constraints

- Fail fast, no silent errors (repo policy): engine/executor failures kill the run loudly. No catch-and-continue anywhere.
- Rollout pipeline untouched (`generate_search_rollouts.py`, its executors' behavior — `merged_executors.py` may gain code but existing functions stay byte-identical; rollout suite green).
- The eval script's single-game observable behavior at `--concurrent-games 1` must be search-decision-identical to today (SF game outcomes are stochastic; the move-probe gate is the decision-level judge).
- `search.py`/`batch_scheduler.py` stay torch-free. Suite green at every commit (baseline 212).
- GPU runs: controller-run only (standing user authorization covers the calibration probe and re-anchor runs). Implementer subagents never run GPU scripts.
- Commits end `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

### Task 1: Stockfish node-calibration probe script

**Files:**
- Create: `scripts/calibrate_stockfish_nodes.py`
- Test: `tests/test_calibrate_stockfish_nodes.py` (pure-function tests only)

A small script that replays the model-vs-SF loop of `_run_segment` for N games
(default 8) at the CURRENT time-based settings, recording Stockfish's reported
`nodes` for every engine move (`engine.play(..., info=chess.engine.INFO_ALL)`
→ `result.info.get("nodes")`; SURVEY: confirm the info key on this
python-chess version and that `SimpleEngine.play` accepts `info=`; if nodes
are absent from play-info, fall back to `engine.analyse` probing on the same
positions — record which path was used). Reuses `load_hstu_checkpoint`,
`_select_model_move`, `_SequenceHistory` imports from the eval script's
modules; CLI: `--config`, `--checkpoint`, `--games`, `--stockfish-elo`,
`--output-json`. Output JSON: per-move nodes list, count, median, p25/p75,
and the recommended `stockfish_nodes` (median rounded to 2 significant
figures). Unit tests cover the stats/rounding helpers only (no engine, no
GPU). Commit: `feat: stockfish node-budget calibration probe`.

### Task 2: sf_move executor + engine pool in merged_executors-adjacent module

**Files:**
- Create: `src/imba_chess/eval/engine_pool.py`
- Test: `tests/test_engine_pool.py`

**Interfaces (produced):**
```python
class EnginePool:
    def __init__(self, *, spawn: Callable[[], Any], size: int) -> None: ...
    # spawn() -> engine handle (chess.engine.SimpleEngine in prod; fake in tests)
    def engine_for_slot(self, slot_index: int) -> Any: ...
    def close(self) -> None:  # quit() all engines; raise the FIRST error after closing all


def make_sf_move_executor(*, pool_threads: int) -> Callable[[list[Any]], list[Any]]:
    # payload: (engine, board, limit_dict_or_Limit) -> result: chess.engine PlayResult
    # Fans out concurrently via ThreadPoolExecutor(max_workers=pool_threads);
    # any engine exception propagates (fail fast) after all futures resolve.
```
Tests use fake engine objects (callable `play` with sleep + recorded calls)
to assert: concurrent fan-out (wall time ~1 think not G), result order
matches payload order, an exception from one engine propagates and the
executor does NOT swallow it, `EnginePool.close` quits all engines even when
one `quit` raises (then re-raises). No torch, no chess.engine subprocess in
tests. Commit: `feat: engine pool + concurrent sf_move executor`.

### Task 3: batched eval driver

**Files:**
- Modify: `scripts/eval_vs_stockfish.py` (`_run_segment` → coroutine + scheduler wiring; new `--concurrent-games` flag, default 1; dtype stays config-driven — fp32 adoption happens in Task 5's config change, not hardcoded)
- Test: `tests/test_eval_vs_stockfish.py` (existing) + new CPU tests with fake executors/engines

The game coroutine mirrors the rollout one: per model turn, `yield
WorkRequest("root_eval", ...)` then drive `search._halving_stepwise`
forwarding `EvalRequest`s as `WorkRequest("decode_wave", (evaluator,
batch))`; per engine turn, `yield WorkRequest("sf_move", (engine, board_copy,
limit))`; opening-random plies and all summary/debug/trace bookkeeping stay
inside the coroutine (traces already carry game indices; interleaving at G>1
is acceptable). Rerank/d2 policies: route through their stepwise generators
identically (all three policies must work batched — they share
`_expand_root_candidates_stepwise`). Engine ownership: one engine per slot
from `EnginePool`, passed to the coroutine at spawn; a slot's next game
reuses its engine (matches today's one-engine-for-all-games behavior at G=1).
Scheduler wiring mirrors the rollout `main()`: game factory yields
color-alternating games in index order; `on_game_done` merges the per-game
summary fragment into the segment `EvalSummary` (stream order keeps
deterministic aggregation); errors: none handled — any exception kills the
run (fail-fast policy; the rollout script's hard-exit wrapper pattern is
reused for the eval script's `main`). At `--concurrent-games 1` the request
sequence per game must equal today's call sequence (single-item executor
calls; the move-probe gate verifies decisions). CPU tests: drive 2 fake
games (scripted PositionEvals + fake sf results) through the scheduler,
assert summary aggregation, color alternation, engine reuse per slot, and
that an engine exception aborts the run. Commit: `feat: batched eval driver
(--concurrent-games) with sf_move scheduling`.

### Task 4 (controller, GPU): gates

1. Move-probe gate: extend the scratchpad probe methodology — fixed 98-FEN
   set, eval config, fp32, run once through a G=1-driven eval evaluator path
   and once through G=8 batched (probe harness drives the scheduler with
   pseudo-games that evaluate the fixed positions); byte-identical chosen
   moves required. (Implemented as a controller script in scratchpad; if it
   fits naturally as a permanent test with a tiny CPU model, promote it —
   implementer/controller judgment.)
2. G=1 structural gate: 4-game smoke at current time-based settings — runs,
   sane aggregates, no crash.
3. G=8 smoke: 8 games node-limited (use a provisional 60k nodes if Task 1's
   calibration hasn't run yet; final number from calibration), confirm
   speedup and stability, `nvidia-smi` VRAM check.

### Task 5 (controller, GPU): calibration + re-anchor + adoption

1. Run Task 1's probe (~8 games, idle machine) → median nodes → set
   `stockfish_nodes` in `config/imba_chess_exit_full.toml` `[eval_vs_stockfish]`
   (plus `dtype = "float32"` and eval `concurrent_games` default 8 — config
   keys per existing config-dataclass conventions; survey `config.py`).
2. Re-anchor run A: 200 games, node-limited, fp32, G=8, opening_random_plies=1.
3. Re-anchor run B: same with opening_random_plies=0 + duplicate-game-rate
   report (unique move-sequence count; if <90% unique, flag diversity
   concern and keep 1 as default).
4. Adopt defaults per outcomes; append Results to the spec; update memory
   status; commit docs+config. Report back any score landing outside
   0.52±0.07 (calibration-gate escalation) as a deviation.
