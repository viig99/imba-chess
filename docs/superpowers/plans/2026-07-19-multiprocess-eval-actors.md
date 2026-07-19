# Multiprocess Eval Actors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process-per-game eval actors with a main-process GPU inference server, per `docs/superpowers/specs/2026-07-19-multiprocess-eval-actors-design.md`.

**Architecture:** Torch-free worker processes run the whole game loop (search generators, cozy, encoder, own Stockfish) exchanging plain-data messages over pipes with the main process, which owns the model and an ID-keyed KV store and merges requests across workers into the existing grouped GPU calls. `--concurrent-games 1` keeps the in-process reference driver; `>1` routes to actors; the in-process G>1 eval path is deleted at cutover.

**Tech Stack:** Python 3.13 multiprocessing (spawn), existing merged_executors/grouped decode, pytest.

## Global Constraints

- Workers must be torch-free: `src/imba_chess/eval/actor_worker.py` (and anything it imports at runtime in the worker) may not import torch. `search.py`/`batch_scheduler.py` stay torch-free. Rollout pipeline byte-untouched (`generate_search_rollouts.py`, `merged_executors.py` existing functions).
- Fail fast, loud: worker crash/pipe EOF → server kills all workers + engines, exits nonzero. No catch-and-continue.
- All payloads across process boundaries: plain Python data (no tensors, no chess objects).
- Canonical UCI-sorted move order preserved end-to-end (server-side projection).
- Suite green at every commit (baseline 254). GPU runs are Task-4-only and REQUIRE FRESH USER GO-AHEAD.
- Commits end `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

### Task 1: actor protocol + torch-free worker

**Files:**
- Create: `src/imba_chess/eval/actor_protocol.py`, `src/imba_chess/eval/actor_worker.py`
- Test: `tests/test_actor_worker.py`

**Interfaces (produced):**
```python
# actor_protocol.py — plain dataclasses, torch-free, chess-free where possible
@dataclass
class RootEvalRequest:  # worker -> server
    worker_id: int; turn_id: int
    batch_arrays: dict[str, list]     # the _SequenceHistory batch dict, tensor fields as plain nested lists + ints
@dataclass
class RootEvalResponse:  # server -> worker
    turn_id: int; value_stm: float
    legal_ucis: list[str]; legal_log_priors: list[float]   # UCI-sorted
@dataclass
class WaveRow:
    node_id: int; parent_id: int | None   # None = child of the turn's root prefix
    prev_move_vocab_id: int
    board_state: dict                      # vars(BoardState): plain ints/lists
@dataclass
class WaveRequest:
    worker_id: int; turn_id: int; rows: list[WaveRow]
@dataclass
class WaveResponse:
    rows: list[tuple[float, list[str], list[float]]]  # (value_stm, legal_ucis, legal_log_priors) per row, request order
@dataclass
class GameDone:  # worker -> server
    worker_id: int; game_idx: int
    summary_fragment: dict                # EvalSummary field increments
@dataclass
class WorkerFinished:
    worker_id: int
```
```python
# actor_worker.py
def run_eval_worker(conn, worker_config: dict) -> None:
    # worker_config: config snapshot fields (search knobs, SF path/options/limit,
    # opening plies, seed, debug flags), assigned game indices, worker_id.
    # Loop: for each assigned game_idx: play the game — SF via own SimpleEngine
    # (direct, synchronous), model turns via conn.send(RootEvalRequest/WaveRequest)
    # + conn.recv(); send GameDone per game; WorkerFinished at end; close engine.
```
The worker's per-game loop is a port of the eval coroutine's logic
(`scripts/eval_vs_stockfish.py` `_play_game` — survey it) minus WorkRequest
yields: model turns drive `search._halving_stepwise`/`_d2_stepwise`/
`_rerank_stepwise` with an evaluator SHIM whose `evaluate(batch)` packages
rows into a `WaveRequest` (minting node ids, tracking parent ids from the
handles) and whose `extend(handle, uci)` mints child handles carrying
(node_id, vocab id) — survey `CachedPositionEvaluator.extend`/`_CachedNode`
for the exact handle semantics being mirrored, and `_project_legal_logits_cozy`
consumers for what PositionEval fields search needs (the shim rebuilds
`PositionEval` from response rows: cozy moves reconstructed worker-side from
ucis via `cc.Move.from_str` after the standard-uci→cozy castling translation
(`py_move_to_cozy` needs a py board — SURVEY: the worker has the cozy board;
reconstruct cozy moves by matching response ucis against
`cozy_move_to_uci(board.generate_moves())` — one dict per node, cheap).
TORCH-FREE CHECK in tests: import `actor_worker` with a meta-path hook that
raises on `import torch` — a permanent test.
CPU tests: fake in-process "server" (a function answering conn messages with
scripted values) drives 2 short games end-to-end; assert summary fragments,
SF engine called via a fake engine object injected through worker_config
(spawn-compat: accept an engine FACTORY callable path or fake flag — design
minimally, document), message ordering (root before waves per turn),
node-id/parent-id chain shape.

- [ ] Steps: survey → tests first → implement → `.venv/bin/pytest -q` green → commit `feat: actor protocol + torch-free eval worker`.

### Task 2: GPU inference server

**Files:**
- Create: `src/imba_chess/eval/actor_server.py`
- Test: `tests/test_actor_server.py`

**Interfaces (produced):**
```python
class ActorInferenceServer:
    def __init__(self, *, model, move_vocab, board_state_encoder, device, dtype): ...
    def register_root(self, worker_id, turn_id, batch_arrays) -> RootEvalResponse:
        # tensorize, run root forward (merged across pending workers where
        # possible — see service loop), store prefix KV under (worker_id, turn_id)
    def service(self, requests: list[RootEvalRequest | WaveRequest]) -> list[...]:
        # group by type; root evals -> ragged merge (reuse merged_executors
        # patterns); waves -> ONE forward_decode_grouped across workers
        # (grouped by worker/turn prefix), storing per-node KV under
        # (worker_id, node_id); returns responses in request order
    def release_turn(self, worker_id, turn_id) -> None:  # frees that turn's KV tree
```
Survey `CachedPositionEvaluator.build_decode_request/consume_decode_result`
and `merged_executors._merge_decode_requests` — the server reimplements their
composition over an ID-keyed store instead of `_CachedNode` object links; the
tensor math paths (grouped decode call, ragged root merge) are REUSED not
reimplemented (import from merged_executors/position_evaluator where the
existing functions fit; refactor-extract shared helpers there ONLY if import
shapes force it — flag any such edit prominently for review since
merged_executors is rollout-shared).
Memory rule: `release_turn` after each game turn (the worker sends its next
root_eval which implies the previous turn is done — explicit release message
vs implicit-on-next-root: pick explicit (in protocol Task 1's GameDone/next
RootEvalRequest — decide, document, test).
CPU tests with the tiny CPU model (reuse `tests/test_grouped_decode.py`'s
model fixtures): a scripted two-worker request sequence must produce
per-row outputs fp32-EXACT (`torch.testing.assert_close` atol/rtol 1e-6)
equal to running the same positions through the in-process
`CachedPositionEvaluator` path; KV store size returns to zero after
release_turn (no leak).

- [ ] Steps: survey → tests → implement → suite green → commit `feat: ID-keyed actor inference server`.

### Task 3: orchestration, routing, cutover deletions

**Files:**
- Modify: `scripts/eval_vs_stockfish.py`
- Test: `tests/test_eval_vs_stockfish.py` (+ new spawn-based integration test, CPU-only)

Wire `--concurrent-games > 1` → actor mode: spawn G workers
(`multiprocessing.get_context("spawn")`), static round-robin game
assignment, serve loop (poll pipes worker-id order; service; forward
GameDone fragments into `EvalSummary` aggregation IN GAME-INDEX ORDER —
hold-back buffer mirroring the scheduler's stream-order rule), fail-fast
supervision (dead worker/pipe EOF/server exception → terminate all children,
hard-exit nonzero), engine lifecycle worker-side. `--concurrent-games 1`
keeps the in-process scheduler driver untouched (reference path).
CUTOVER DELETIONS (grep-clean in report): eval-side G>1 in-process wiring —
the `sf_move` thread-pool executor usage in this script and eval's G>1
merged-executor glue; `engine_pool.py`: SURVEY remaining consumers — if the
worker uses SimpleEngine directly and nothing else imports it, DELETE the
module and its tests (no orphan); if kept for the G=1 path, document why.
CPU integration test: real spawn of 2 workers with a FAKE model server
(tiny CPU model) playing 2 short games vs a fake engine (worker_config fake
flag from Task 1) — asserts end-to-end summaries + clean shutdown + a
worker-kill test asserting the run dies nonzero.

- [ ] Steps: survey → implement → deletions → suite green → commit `feat: actor-mode eval orchestration; delete superseded in-process G>1 path`.

### Task 4 (controller, GPU — ⛔ FRESH USER GO-AHEAD REQUIRED)

1. Fixed-composition move-probe: actor-served decisions vs in-process G=1 on
   the 98-FEN set, fp32 → byte-identical required.
2. Perf: 200 games, G=8, new protocol → success well under 30 min.
3. Score within 0.5775 ± 0.07; results appended to spec; final whole-branch
   review; merge; push.
