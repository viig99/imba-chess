# Cross-Game Batched Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run G rollout games concurrently in one process, merging their search-eval batches into large GPU calls, per `docs/superpowers/specs/2026-07-18-cross-game-batched-search-design.md`.

**Architecture:** `search.py`'s halving search becomes an internal generator (yields eval requests, receives results) with the existing sync API as thin wrappers; a new torch-free `batch_scheduler.py` runs G game coroutines in deterministic lockstep ticks; `hstu_model.py` gains a grouped-prefix decode so merged waves keep per-game prefix sharing; `generate_search_rollouts.py` gains `--concurrent-games`. Layered regression gates; G=1 must stay byte-identical to today.

**Tech Stack:** Python 3.13, PyTorch (eager), pytest. No new dependencies.

## Global Constraints

- `search.py` stays torch-free; `batch_scheduler.py` must ALSO be torch-free (it moves opaque request/result objects; executors are injected).
- Public signatures of `select_value_search_halving` / `select_value_search_d2` / `select_value_rerank` unchanged; `eval_vs_stockfish.py` and `tests/test_search.py` untouched and passing.
- G=1 through the scheduler must issue exactly today's per-game batches in today's order (Layer 1 byte-identical gate depends on this).
- No leftover flags/scaffolding: the sync wrappers, single-prefix `forward_decode`, and grouped decode are all permanently live paths (wrappers serve eval_vs_stockfish; single-prefix serves G=1 and eval; grouped serves G>1) — nothing else survives past its task.
- Test command: `.venv/bin/pytest` from repo root (baseline 173 passing). CPU-only; safe to run anytime.
- **GPU runs (any `generate_search_rollouts.py` invocation) are Tasks 5-6 ONLY and require explicit user go-ahead — the GPU may be in use for gaming. Do not launch them from earlier tasks.**
- Commit messages end with: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (implementing model).

## Baseline numbers (for Task 5/6 comparison)

20-game fixed-seed profile (2026-07-18, post-cozy, commit e31504d): total 92.0s; search_gpu 45.1%, search_bookkeeping 35.1%, root_eval ~19.8%; 1,371 waves / 346,093 evals; eager parquet at `--sample-seed 42` is deterministic (byte-identical across runs).

---

### Task 1: Stepwise generator core in search.py

**Files:**
- Modify: `src/imba_chess/eval/search.py`
- Test: `tests/test_search.py` (must pass UNCHANGED), new `tests/test_search_stepwise.py`

**Interfaces:**
- Produces: `EvalRequest` (NamedTuple: `batch: list[tuple[Any, chess.Board]]`); generator functions `_halving_stepwise(...)` and `_expand_root_candidates_stepwise(...)` yielding `EvalRequest` and returning their current return values via `StopIteration.value`; `_drive(gen, evaluator)` helper. Public sync functions become wrappers. Task 4 consumes the generators; nothing else changes.

- [ ] **Step 1: Write the failing generator-equivalence test**

```python
# tests/test_search_stepwise.py
"""The stepwise generator core must be call-for-call identical to the sync API.

_RecordingEvaluator wraps a real evaluator and logs every evaluate() batch
(handles + board FENs). Driving the generator by hand must produce the same
chosen move, same rows, and the same sequence of evaluate() batches as the
sync wrapper — proving the wrapper/generator refactor changed nothing.
"""

import random

import chess
import pytest

from imba_chess.eval import search
from tests.test_search import _ArmValueEvaluator, _MaterialEvaluator


class _RecordingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.calls: list[list[str]] = []

    def extend(self, handle, board_before, move):
        return self.inner.extend(handle, board_before, move)

    def evaluate(self, batch):
        self.calls.append([board.fen() for _, board in batch])
        return self.inner.evaluate(batch)


def _drive_by_hand(gen, evaluator):
    try:
        request = next(gen)
        while True:
            request = gen.send(evaluator.evaluate(request.batch))
    except StopIteration as stop:
        return stop.value


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
])
def test_halving_generator_matches_sync_wrapper(fen):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_log_priors = [-1.0 - 0.01 * i for i in range(len(legal_moves))]
    config = search.HalvingConfig(budget=64, top_m=8, max_depth=3)

    sync_eval = _RecordingEvaluator(_MaterialEvaluator())
    sync_result = search.select_value_search_halving(
        evaluator=sync_eval, root_handle=None, board=board,
        legal_moves=legal_moves, legal_log_priors=legal_log_priors,
        config=config, rng=random.Random(7),
    )

    gen_eval = _RecordingEvaluator(_MaterialEvaluator())
    gen = search._halving_stepwise(
        root_handle=None, board=board, legal_moves=legal_moves,
        legal_log_priors=legal_log_priors, config=config, rng=random.Random(7),
        extend=gen_eval.extend,
    )
    gen_result = _drive_by_hand(gen, gen_eval)

    assert gen_result == sync_result
    assert gen_eval.calls == sync_eval.calls


def test_d2_and_rerank_wrappers_unchanged_behavior():
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    priors = [-1.0] * len(legal_moves)
    evaluator = _RecordingEvaluator(_ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6}))
    idx, rows = search.select_value_search_d2(
        evaluator=evaluator, root_handle=None, board=board,
        legal_moves=legal_moves, legal_log_priors=priors, top_k=4, lam=0.05,
    )
    assert legal_moves[idx].uci() == "e2e4"
    assert evaluator.calls  # evaluator was exercised through the wrapper
```

Note: `_halving_stepwise` cannot receive the evaluator (it must not call
`evaluate` itself) but it still needs `extend` for handle derivation —
`extend` is per-node CPU bookkeeping, not a GPU call, so the generator takes
it as an injected callable. Check `_ArmValueEvaluator`'s exact constructor in
`tests/test_search.py` and adapt the d2 test's expected winner if its
semantics differ from the assumption above (fix the test fixture, not the
production code).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_search_stepwise.py -v`
Expected: FAIL with `AttributeError: ... no attribute '_halving_stepwise'`

- [ ] **Step 3: Implement the generator core**

Mechanical transformation of the existing bodies — every
`evaluator.evaluate(X)` becomes `yield EvalRequest(batch=X)` receiving the
result, every `evaluator.extend(...)` becomes `extend(...)`:

```python
class EvalRequest(NamedTuple):
    batch: list[tuple[Any, chess.Board]]


def _drive(gen, evaluator: PositionEvaluator):
    """Run a stepwise search generator to completion synchronously."""
    try:
        request = next(gen)
        while True:
            request = gen.send(evaluator.evaluate(request.batch))
    except StopIteration as stop:
        return stop.value
```

- `_expand_root_candidates` body → `_expand_root_candidates_stepwise(*, extend, root_handle, board, cozy_root, legal_moves, legal_log_priors, top_k)` — same logic, its one `evaluator.evaluate(batch)` becomes `position_evals = yield EvalRequest(batch=batch)`; `evaluator.extend` becomes `extend`. Returns the same `(candidates, mate_index)`.
- `select_value_search_halving` body → `_halving_stepwise(*, extend, root_handle, board, legal_moves, legal_log_priors, config, rng)` — the wave loop's `evals = evaluator.evaluate([...])` becomes `evals = yield EvalRequest(batch=[...])`. Returns `(best.local_idx, rows)`.
- Wrappers (exact public signatures preserved):

```python
def select_value_search_halving(*, evaluator, root_handle, board, legal_moves,
                                legal_log_priors, config, rng=None):
    return _drive(
        _halving_stepwise(
            extend=evaluator.extend, root_handle=root_handle, board=board,
            legal_moves=legal_moves, legal_log_priors=legal_log_priors,
            config=config, rng=rng,
        ),
        evaluator,
    )
```
  and analogously `select_value_search_d2` / `select_value_rerank` call
  `_drive(...)` around bodies that use `yield from _expand_root_candidates_stepwise(...)`
  (their own remaining `evaluate` calls also become yields; d2 has one more
  for the board2 batch). The cozy dual-board threading from the previous
  project is untouched — it is plain code inside the generator bodies.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `.venv/bin/pytest tests/test_search_stepwise.py tests/test_search.py -v` then `.venv/bin/pytest -q`
Expected: all pass; `tests/test_search.py` unchanged (verify `git diff --stat` shows no edit to it).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/search.py tests/test_search_stepwise.py
git commit -m "refactor: stepwise generator core for search strategies, sync API as thin wrappers

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Torch-free batch scheduler

**Files:**
- Create: `src/imba_chess/eval/batch_scheduler.py`
- Test: `tests/test_batch_scheduler.py`

**Interfaces:**
- Produces:

```python
class WorkRequest(NamedTuple):
    kind: str          # e.g. "root_eval" | "decode_wave" — opaque to the scheduler
    payload: Any       # opaque; executor understands it

# A game coroutine: Generator[WorkRequest, Any, list[Any]]
#   yields WorkRequest, receives that request's result via send(),
#   returns its finished rows (list) via StopIteration.value.

class BatchScheduler:
    def __init__(self, *, game_factory: Iterator[tuple[str, Generator]],
                 executors: dict[str, Callable[[list[Any]], list[Any]]],
                 concurrent_games: int,
                 on_game_done: Callable[[str, list[Any] | None], None],
                 on_game_error: Callable[[str, BaseException], None]) -> None: ...
    def run(self) -> None: ...
```

  `game_factory` yields `(game_id, coroutine)` in dataset-stream order.
  `executors[kind]` receives the tick's merged `list[payload]` and returns a
  same-length, same-order `list[result]`. `on_game_done` is called **in
  stream order** (hold-back buffer): the scheduler tracks each slot's stream
  index and buffers finished games until all earlier-started games finish.
  On a coroutine exception: `on_game_error(game_id, exc)` then
  `on_game_done(game_id, None)` in stream order (so ordering never stalls).
- Consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batch_scheduler.py
"""Scheduler semantics, tested with plain-Python fake games — no torch, no chess.

Fake game coroutines yield WorkRequest("echo", payload) a scripted number of
times; the fake executor returns [p * 10 for p in payloads] and logs each
tick's merged payload list, letting every scheduler property be asserted
deterministically.
"""

from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest


def _fake_game(game_id, num_requests, log):
    rows = []
    for i in range(num_requests):
        result = yield WorkRequest(kind="echo", payload=(game_id, i))
        rows.append(result)
    log.append(f"done:{game_id}")
    return rows


def _run(games, concurrent, tick_log):
    done_order = []

    def executor(payloads):
        tick_log.append(sorted(payloads))
        return [p for p in payloads]

    scheduler = BatchScheduler(
        game_factory=iter(games),
        executors={"echo": executor},
        concurrent_games=concurrent,
        on_game_done=lambda gid, rows: done_order.append((gid, rows)),
        on_game_error=lambda gid, exc: done_order.append((gid, repr(exc))),
    )
    scheduler.run()
    return done_order


def test_merges_requests_across_games_per_tick():
    log = []
    ticks = []
    games = [(f"g{i}", _fake_game(f"g{i}", 3, log)) for i in range(4)]
    _run(games, concurrent=4, tick_log=ticks)
    # First tick carries one request from each of the 4 games.
    assert ticks[0] == sorted([("g0", 0), ("g1", 0), ("g2", 0), ("g3", 0)])


def test_emission_is_stream_order_even_when_completion_is_not():
    log = []
    ticks = []
    # g0 needs 5 requests, g1 needs 1 — g1 finishes first but must emit second... 
    # (stream order: g0 started first)
    games = [("g0", _fake_game("g0", 5, log)), ("g1", _fake_game("g1", 1, log))]
    done = _run(games, concurrent=2, tick_log=ticks)
    assert [gid for gid, _ in done] == ["g0", "g1"]


def test_slot_refill_keeps_concurrency():
    log = []
    ticks = []
    games = [(f"g{i}", _fake_game(f"g{i}", 2, log)) for i in range(5)]
    _run(games, concurrent=2, tick_log=ticks)
    assert len({gid for tick in ticks for gid, _ in tick}) == 5  # all games ran


def test_error_isolation_drops_game_and_continues():
    def _bad_game(gid):
        yield WorkRequest(kind="echo", payload=(gid, 0))
        raise ValueError("boom")

    log = []
    ticks = []
    games = [("bad", _bad_game("bad")), ("ok", _fake_game("ok", 2, log))]
    done = _run(games, concurrent=2, tick_log=ticks)
    assert done[0][0] == "bad" and "boom" in done[0][1]
    assert done[1] == ("ok", [("ok", 0), ("ok", 1)])


def test_determinism_same_inputs_same_ticks():
    def run_once():
        log, ticks = [], []
        games = [(f"g{i}", _fake_game(f"g{i}", 3, log)) for i in range(3)]
        _run(games, concurrent=3, tick_log=ticks)
        return ticks

    assert run_once() == run_once()
```

Note on the error test: the Interfaces contract governs — errors surface via
`on_game_error(gid, exc)` and the game still reaches `on_game_done(gid, None)`
in stream order. Adapt the sketch's assertions to that contract exactly.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_batch_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.eval.batch_scheduler'`

- [ ] **Step 3: Implement BatchScheduler**

Single-threaded tick loop; no torch import:

```python
# src/imba_chess/eval/batch_scheduler.py
"""Deterministic G-game lockstep scheduler.

Runs G game coroutines concurrently: each tick, every live slot advances to
its next WorkRequest; requests are grouped by kind and executed as one merged
call per kind; results are scattered back. Completed games are reported in
dataset-stream order via a hold-back buffer so downstream flush/resume
semantics (--flush-every-games, progress sidecar, --skip-games) are
unchanged. Torch-free by design: payloads/results are opaque."""
```

Core loop shape (implementer writes the full class):

```python
def run(self) -> None:
    slots = {}   # slot_id -> _Slot(stream_idx, game_id, gen, pending_request)
    self._fill_slots(slots)
    while slots:
        # Phase 1: advance every slot lacking a pending request.
        for slot in list(slots.values()):
            if slot.pending is None:
                self._advance(slot, slots, send_value=None, first=True)
        # Phase 2: group pending requests by kind (stable slot order).
        by_kind: dict[str, list] = defaultdict(list)
        for slot_id in sorted(slots):
            slot = slots[slot_id]
            if slot.pending is not None:
                by_kind[slot.pending.kind].append((slot_id, slot.pending))
        if not by_kind:
            break
        # Phase 3: one merged executor call per kind; scatter results.
        for kind, entries in by_kind.items():
            results = self._executors[kind]([req.payload for _, req in entries])
            assert len(results) == len(entries)
            for (slot_id, _), result in zip(entries, results):
                self._advance(slots[slot_id], slots, send_value=result)
        self._fill_slots(slots)
```

`_advance` wraps `next(gen)` / `gen.send(...)` in try/except: `StopIteration`
→ record rows, free slot, run the hold-back emitter; any other exception →
`on_game_error`, record `None` rows, free slot, hold-back emitter. The
hold-back emitter walks a `stream_idx -> (game_id, rows)` dict emitting while
the next-expected index is present.

- [ ] **Step 4: Run tests + full suite**

Run: `.venv/bin/pytest tests/test_batch_scheduler.py -q` then `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/batch_scheduler.py tests/test_batch_scheduler.py
git commit -m "feat: torch-free deterministic G-game batch scheduler

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Grouped-prefix forward_decode

**Files:**
- Modify: `src/imba_chess/model/hstu_model.py` (and `hstu_attention.py` only if the decode attention lives there — survey first)
- Test: `tests/test_grouped_decode.py`

**Interfaces:**
- Produces: `model.forward_decode_grouped(*, new_token_batch, positions, group_index, prefix_kv_grouped, prefix_lens, suffix_kv, suffix_positions, suffix_mask) -> same dict as forward_decode`, where `group_index` is `[B]` long (row → game group g), `prefix_kv_grouped` is per-layer `(k, v)` with shapes `[G, H, maxP, d]` (games padded on the token dim), `prefix_lens` is `[G]`. All other args exactly as `forward_decode` (suffix machinery is already per-row). Single-prefix `forward_decode` is untouched.
- Consumed by Task 4's merged decode executor.

- [ ] **Step 1: Survey the existing decode attention**

Read `forward_decode` in `src/imba_chess/model/hstu_model.py` end-to-end and
record in your report: where prefix attention happens, tensor shapes, mask
conventions, and how `positions` feeds relative attention bias. The grouped
variant must reproduce these exactly.

- [ ] **Step 2: Write the failing equivalence test**

```python
# tests/test_grouped_decode.py
"""forward_decode_grouped == per-game forward_decode, fp32 CPU, tight tolerance.

Builds a small random HSTUChessModel (fixed seed), fabricates G=3 games with
different prefix lengths (via real prefix forwards), then decodes a merged
batch of rows (mixed depths, suffixes built the same way CachedPositionEvaluator
builds them) both ways: per-game forward_decode calls vs one
forward_decode_grouped call. Outputs must match to 1e-5.
"""
```
The test constructs the model exactly as `tests/test_prefix_decode.py` does
(reuse its helpers/fixtures — survey that file first and mirror its setup:
it already validates single-prefix decode against full forwards, so it shows
the canonical way to build prefixes, suffixes, and token batches in tests).
Assert per-row: `torch.testing.assert_close(grouped_out["logits"], seq_out_logits, atol=1e-5, rtol=1e-5)` and same for `value_logits` and each layer's returned `kv`.
Include: a group with maxP padding (different prefix lens), a row at depth 0
and one at depth ≥2, and G=1 grouped vs plain forward_decode (degenerate case).

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_grouped_decode.py -v`
Expected: FAIL with `AttributeError: ... 'forward_decode_grouped'`

- [ ] **Step 4: Implement forward_decode_grouped**

The only new math is prefix attention with a per-row group lookup. Pattern
(adapt names/shapes to what Step 1 found):

```python
# q: [B, H, 1, d] (new token per row)
# prefix_k/v (this layer): [G, H, maxP, d]; prefix_lens: [G]
q_g = q[group_index_sorted...]  # keep rows in given order; gather per row:
scores_p = torch.einsum("bhd,ghpd->bhp", q.squeeze(2), prefix_k[group_index])
# ^ index_select on dim 0 with group_index materializes [B, H, maxP, d] — DO NOT.
```
**Memory rule:** never index `prefix_k[group_index]` (that materializes per-row
prefix copies — the exact blowup this design avoids). Instead sort/bucket rows
by group and run one `torch.baddbmm`/`einsum` per group over that group's row
block, or use `torch.nn.functional.scaled_dot_product_attention` per group on
`[1, H, B_g, d] x [1, H, P_g, d]` views — G small calls inside ONE Python
function is acceptable (the per-wave Python/dispatch overhead is paid once;
kernel count per layer is G, vs per-game calls paying the full stack G times).
Mask padded prefix tokens with `-inf` before softmax-equivalent (match the
existing decode's silu/normalization convention EXACTLY — read, don't assume).
Then combine with the existing suffix + self attention contributions the same
way `forward_decode` already does.

- [ ] **Step 5: Run tests + full suite**

Run: `.venv/bin/pytest tests/test_grouped_decode.py tests/test_prefix_decode.py -q` then `.venv/bin/pytest -q`
Expected: all pass, tolerance 1e-5 met.

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/model/ tests/test_grouped_decode.py
git commit -m "feat: grouped-prefix forward_decode for cross-game merged waves

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Game coroutine, merged executors, --concurrent-games

**Files:**
- Modify: `src/imba_chess/eval/position_evaluator.py` (split `CachedPositionEvaluator.evaluate` into request-build / result-consume halves; `evaluate()` keeps calling both — single-game behavior byte-identical)
- Modify: `scripts/generate_search_rollouts.py` (`_process_game` → coroutine; `--concurrent-games`; scheduler wiring; merged executors)
- Test: `tests/test_rollout_coroutine.py` (CPU-only pieces), existing suite

**Interfaces:**
- Consumes: Task 1 generators, Task 2 `BatchScheduler`/`WorkRequest`, Task 3 `forward_decode_grouped`.
- Produces (in `position_evaluator.py`):
  - `CachedPositionEvaluator.build_decode_request(batch) -> _DecodeRequest` (CPU half: encode boards, token tensors, suffix gather, positions — everything before the model call; `_DecodeRequest` also carries `prefix_kv`, `prefix_len`, and the nodes list)
  - `CachedPositionEvaluator.consume_decode_result(request, out) -> list[PositionEval]` (path_kv extension + logits→PositionEval)
  - `evaluate()` = `consume_decode_result(req := build_decode_request(batch), model.forward_decode(...req...))` — refactor, not a fork.
- Produces (in the script): game coroutine yielding `WorkRequest("root_eval", payload)` and `WorkRequest("decode_wave", payload)`; executors: root-eval executor pads sequences to batch max and calls `_forward_model` once (splitting results per game), decode executor stacks per-game prefixes → `forward_decode_grouped`, splits results, and calls each game's `consume_decode_result`. `--concurrent-games` (default 1) always routes through the scheduler.

- [ ] **Step 1: Refactor evaluate() into build/consume halves**

Pure code motion inside `position_evaluator.py`; `evaluate()`'s observable
behavior must be unchanged (existing tests cover it: `tests/test_prefix_decode.py`,
eval-path tests). Run `.venv/bin/pytest -q` — green before proceeding.

- [ ] **Step 2: Restructure _process_game into a coroutine**

`_process_game` currently: replay plies; at each sampled ply build the root
batch, call `_forward_model` (root eval), construct `CachedPositionEvaluator`,
call `select_value_search_halving`. The coroutine version: replace the root
`_forward_model` call with `out = yield WorkRequest("root_eval", root_payload)`,
and drive `search._halving_stepwise` inline, forwarding its `EvalRequest`s as
`WorkRequest("decode_wave", (evaluator, eval_request.batch))`:

```python
gen = search._halving_stepwise(extend=evaluator.extend, ...)
try:
    request = next(gen)
    while True:
        position_evals = yield WorkRequest("decode_wave", (evaluator, request.batch))
        request = gen.send(position_evals)
except StopIteration as stop:
    chosen_index, rows = stop.value
```
All row assembly, timing-stats bookkeeping, and rng derivation stay inside the
coroutine, unchanged. `--profile` buckets keep working: the executors time the
GPU calls (search_gpu / root_eval); the coroutine times its CPU segments.

- [ ] **Step 3: Implement the two executors + scheduler wiring**

- Root-eval executor: payloads are per-game `(batch_dict, meta)`; pad each
  game's token arrays to the tick's max seq len, run ONE `_forward_model`,
  split per game (use `seq_offsets` — the model already handles ragged
  batches via `create_batch_block_mask`, so prefer concatenation over padding
  if the existing batch format is ragged-native — Step 1 of Task 3 surveyed
  this; follow what the format actually supports and record which you used).
- Decode executor: group payloads by game; when the tick has ONE game, call
  the game evaluator's existing single-prefix path (`forward_decode`) so G=1
  is byte-identical; when >1, build `prefix_kv_grouped`/`prefix_lens`/
  `group_index` from each game's `_DecodeRequest` and call
  `forward_decode_grouped`, then split rows back per game and call each
  game's `consume_decode_result`.
- `main()`: build `game_factory` from `lichess_dataset.stream()` (wrapping
  each game dict into a coroutine), executors dict, and
  `BatchScheduler(concurrent_games=args.concurrent_games)`; `on_game_done`
  appends rows + flush/sidecar exactly where the current per-game loop does.

- [ ] **Step 4: CPU-testable coverage**

`tests/test_rollout_coroutine.py`: drive one game coroutine by hand with fake
executors (scripted `PositionEval`s / root outputs, no model, no GPU) over a
short fixed game record; assert it yields root_eval before decode_waves, and
produces rows with the same schema/fields as `_arm_rows_to_dicts` expects.
Keep it small — the real gates are Task 5's GPU runs.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/pytest -q` — all pass. **Do NOT run generate_search_rollouts.py (GPU).**

```bash
git add src/imba_chess/eval/position_evaluator.py scripts/generate_search_rollouts.py tests/test_rollout_coroutine.py
git commit -m "feat: G-game rollout coroutines + merged executors + --concurrent-games

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: GPU gates — ⛔ STOP: requires explicit user go-ahead (GPU may be in use)

**Files:** none committed except gate outputs recorded in the Task-6 docs.

- [ ] **Step 0: ASK THE USER before any run in this task.**

- [ ] **Step 1: Layer 1 (byte-identical, G=1)** — standard 20-game profile command (see cozy plan's Global Constraints for the exact invocation) with `--concurrent-games 1` → parquet must equal the pre-branch parquet (`pd.testing.assert_frame_equal`). If not identical: STOP, debug (the G=1 path must issue today's exact batches; the scheduler/coroutine refactor has a bug), do not proceed to G>1.
- [ ] **Step 2: Layer 2 (statistical)** — same 20 games at `--concurrent-games 4` and `16`: best-arm move agreement ≥99% vs G=1; p99 |Δ backed_value| ≤ 1e-3 (comparison script pattern: scratchpad `compare_compile.py` from 2026-07-18). Record `--profile` blocks at G∈{1,4,16}.
- [ ] **Step 3: G-sweep** — G∈{8,16,24,32} short runs (5-10 games each), record games/hr and peak VRAM (`nvidia-smi`) to pick local default G.
- [ ] **Step 4:** if any gate fails, report findings and stop for user decision.

---

### Task 6: Results docs + Layer-3 plan

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-cross-game-batched-search-design.md` (append Results section: gate outcomes, per-G profiles, chosen default G, projected remote G)
- Modify: memory `imba-chess-elo-goal.md` status line (not committed)

- [ ] **Step 1:** Append results + the Layer-3 acceptance procedure as a follow-up item (short localcoverage training run on batched rollouts + eval_vs_stockfish vs sequential twin — scheduled as a nightly, not run inline).
- [ ] **Step 2:** `.venv/bin/pytest -q` final check; commit docs.

```bash
git add docs
git commit -m "docs: cross-game batched search results and gate outcomes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
