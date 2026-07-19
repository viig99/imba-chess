# Multiprocess eval actors (design, 2026-07-19)

Successor to the fast-clean-evals follow-up sketch (see that spec's Results +
follow-up section). Goal: 200-game SF-ladder evals in ~15-20 min at G=8 by
parallelizing per-game CPU across processes and overlapping SF thinks with
GPU work. User-approved 2026-07-19 with standing autonomous execution;
GPU acceptance runs require a fresh user go-ahead (gaming may be underway).

## Architecture

**Workers are torch-free processes.** One worker per concurrent game (G=8
initially). Each worker runs the ENTIRE per-game loop natively and
synchronously: the torch-free stepwise search generators from `search.py`,
cozy-chess ops, `BoardStateEncoder` (plain-int `BoardState` outputs), its own
`chess.engine.SimpleEngine` Stockfish instance (called directly — SF overlap
across workers is automatic), per-game summary fragments, debug traces.
Workers never import torch: all payloads crossing the process boundary are
plain Python data (lists/ints/floats/bytes).

**The main process is a GPU inference server.** It owns the model and ALL KV
state. Per game-turn it maintains the node KV tree currently kept by
`_CachedNode`, re-keyed by `(worker_id, node_id)` integer IDs — workers mint
node ids; the server stores each node's per-layer KV and parent link.
Two request types over per-worker `multiprocessing.Pipe`s:
- `root_eval`: the worker's `_SequenceHistory`-equivalent token arrays
  (plain lists); server tensorizes, merges across workers into the existing
  ragged root batch (`merged_executors` machinery), returns (value, sorted
  legal uci->logit projection inputs, prefix registered under a turn id).
- `decode_wave`: list of (node_id, parent_id, move vocab id, BoardState
  fields); server gathers suffix KV chains by ID, merges across workers into
  ONE `forward_decode_grouped` call, stores each node's new KV, returns
  (value_stm, legal projection) rows.
Vocab projection: server-side (it owns logits); returns per-row sorted
(uci, log_prior) lists — canonical order preserved.

**Collection policy and determinism (decided).** The server services
whatever requests are pending, collecting in worker-id order per servicing
round. Run-level bit-reproducibility at G>1 is NOT guaranteed (batch
composition depends on OS scheduling; fp32 kernel drift on differing shapes
can flip rare near-ties). This is accepted and documented: score-level
reproducibility never existed (SF Elo-randomization), and correctness is
gated by (a) the `--concurrent-games 1` in-process reference path, which
stays byte-deterministic, and (b) a fixed-composition move-probe mode for
the acceptance gate. Chasing bit-determinism across OS process scheduling
would require barriers that reintroduce the stalls this design removes.

**Failure policy (repo rule: fail fast, loud).** A worker crash/EOF on its
pipe, or any server-side exception, kills the entire run: server terminates
all workers and Stockfish children, exits nonzero (hard-exit wrapper).
No skip-and-continue.

**Mode routing and no-dead-code cutover.** `--concurrent-games 1` keeps the
in-process scheduler driver (the byte-deterministic reference/gate path).
`--concurrent-games > 1` routes to actor mode. The in-process G>1 eval
wiring built by fast-clean-evals (the eval-side `sf_move` thread-pool
executor path and eval G>1 merging glue) is DELETED at cutover — actors
supersede it. Shared infra survives untouched: `merged_executors` (rollouts
+ server), `batch_scheduler` (rollouts + eval G=1), `engine_pool`
(worker-side engine lifecycle reuse where it fits, else deleted too —
plan decides, no orphan).

**Process mechanics.** `multiprocessing.get_context("spawn")` (CUDA-safe;
workers import no torch, so spawn is cheap). Workers receive: config
snapshot, game assignments (indices for color/rng derivation — same
per-game seeding as today), SF options/limits. Game assignment is static
round-robin by game index (equal per-worker loads; games are ~equal cost
under node-limited SF; simpler and more deterministic than work-stealing —
revisit only if tail-latency measurements demand it).

## Gates

- CPU: worker loop against a fake in-process server endpoint (scripted
  results) — full games, summaries, traces; server KV-store unit tests with
  the tiny CPU model (request→merge→respond→KV chain correctness vs the
  in-process evaluator's outputs on identical inputs, fp32 exact).
- Suite green throughout; rollout pipeline byte-untouched.
- GPU acceptance (USER GO-AHEAD REQUIRED before running): (a) fixed-
  composition move-probe: actor-served search decisions vs in-process G=1
  on the 98-FEN set, fp32, byte-identical; (b) perf: 200 games at G=8 —
  success is well under 30 min (target 15-20); (c) score within
  0.5775 ± 0.07 band.

## Out of scope

- Rollout pipeline migration (explicitly not — its single-process batching
  is near-optimal for its workload).
- Work-stealing/dynamic assignment; multi-GPU; remote execution.
- Budget 4096 A/B (queued behind this, on the new protocol).
