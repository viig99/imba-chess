# Cross-game batched search for rollout generation (design, 2026-07-18)

## Problem

After the cozy-chess CPU work (see
`2026-07-18-rollout-cpu-hotpath-optimization-design.md` §6), rollout
generation's profile is: search_gpu 45.1%, search_bookkeeping 35.1%,
root_eval ~20%. The search_gpu bucket is 1,371 `forward_decode` calls per 20
games at ~30ms/wave for ~252 positions/wave — while the wave's actual math is
worth ~1ms on the local 3070 Ti. The workload is **overhead-bound, not
FLOP-bound** (confirmed independently by the remote-5090 result: a 3× GPU was
slower per game). Wave sizes cannot grow within one game without changing
search behavior (halving structure dictates them), so the only
behavior-preserving fix is amortizing the fixed per-call tax across games:
run G games concurrently in one process and merge their eval batches.

torch.compile was tried first and is a documented dead end for this workload
(deadlock + recompilation storm on the search's variable shapes — see the
2026-07-15 throughput notes addendum). Batching, by making shapes large and
regular, is also the prerequisite that would make kernel-level work
(compile/CUDA graphs/fusion) viable later.

Remaining gap for context: the 2-3-day KL-coverage target needs ~7,000-21,000
games/hr; current local single-shard is ~915 games/hr post-cozy.

## Scope

Rollout generation only (`scripts/generate_search_rollouts.py`).
`eval_vs_stockfish.py` keeps calling the unchanged synchronous search API.
Eval-side batching is possible later but out of scope.

## Mechanism decision: generator-based stepwise search

Chosen over (a) threads + barrier-batching evaluator (GIL serializes the CPU
35% anyway; batch composition depends on thread timing so even G=1 is not
reproducible; barrier edge cases), (b) asyncio (same restructure, extra
machinery, no benefit), (c) multi-process GPU server (per-wave IPC eats the
overhead we're removing; per-process RAM duplication returns).

The generator design is single-threaded and fully deterministic, keeps ONE
search implementation for both calling styles, and — decisively — driven at
G=1 it issues exactly today's per-game batches, so the restructure itself is
gated **byte-identically** (same machinery as the cozy cutovers). Numeric
drift enters only at the G>1 merge step, isolated and separately gated.

## Architecture

Three pieces:

1. **`src/imba_chess/eval/search.py`** — the halving search body becomes an
   internal generator: everywhere it currently calls
   `evaluator.evaluate(batch)`, it instead `yield`s an
   `EvalRequest(batch=[(handle, board), ...])` and receives the
   `list[PositionEval]` via `send()`. `_expand_root_candidates` (shared with
   d2/rerank) becomes a generator the same way (one yield). A module-level
   `_drive(gen, evaluator)` loop reconstitutes synchronous behavior;
   `select_value_search_halving`, `select_value_search_d2`, and
   `select_value_rerank` keep their exact public signatures as thin wrappers.
   No caller changes anywhere; no dead second path.
2. **`src/imba_chess/eval/batch_scheduler.py`** (new) — the G-game event
   loop. Holds G slots, each an in-flight game coroutine. Each tick: run
   every slot's CPU segment to its next yield; group pending requests by
   kind; one merged GPU call per kind; scatter results; refill completed
   slots from the dataset stream. Knows nothing about chess or search
   internals — it moves requests and results.
3. **`scripts/generate_search_rollouts.py`** — `_process_game` becomes a
   game coroutine; new `--concurrent-games G` flag (default 1). G=1 routes
   through the same scheduler (no separate legacy path).

## Request kinds

- **RootEval** — the full-sequence forward at each sampled ply. Variable
  sequence lengths across games; padded to the batch max per tick.
- **DecodeWave** — search `forward_decode` batches. Waves are already
  heterogeneous in node prefix depth *within* one game today;
  `CachedPositionEvaluator`'s existing KV-gathering handles cross-game
  concatenation identically, just with more rows.

Between yields, a game coroutine does its CPU work exactly as today (ply
replay, encoding, `_SequenceHistory` updates, row assembly).

## Operational semantics

- **Output ordering**: rows are emitted in dataset-stream order via a small
  hold-back buffer for out-of-order game completions — `--flush-every-games`,
  the progress sidecar, and `--skip-games` resume semantics are preserved
  exactly.
- **Failure isolation**: a game coroutine that raises is logged and dropped,
  its slot refilled — same observable effect as today's per-game skip.
- **Sharding**: `--shard-id/--num-shards` composes on top (multi-GPU remote:
  one process per GPU, G games each).

## Memory model and choosing G

Per-game steady cost: prefix KV ~13MB bf16 at max seq (513 tokens × 8 layers
× 12 heads × 64 dim × K+V × 2 bytes). Per-game transient during an active
search: ~50MB (2048 nodes × 1-token KV). With model weights + allocator
overhead, starting points: **G=8-16 local (8GB 3070 Ti), G=64+ remote (32GB
5090)**. G is a CLI/config knob; an empirical G-sweep (throughput vs the
known allocator high-water-mark plateau) is part of the implementation plan.
One process with G games duplicates far less memory than G shard processes —
this also relieves the local RAM ceiling (swap thrash at 4 shards) and the
remote GPU-memory plateau (12-shard cap) documented in the 2026-07-15 notes.

## Determinism and seeding

Per-game RNG stays derived from `(sample_seed, game_id, ply)` — search
decisions never depend on scheduling. The scheduler is single-threaded and
deterministic: same config + seed → same batch compositions → **G>1 runs are
reproducible run-to-run**. They differ from G=1 only through kernel-choice
numerics on different batch shapes (~1e-6/step), which is why the gate below
is layered rather than byte-equality-everywhere.

## Regression gates (layered, agreed 2026-07-18)

- **Layer 0** — existing `tests/test_search.py` passes unchanged through the
  sync wrappers (the restructure is pure control flow).
- **Layer 1 (byte-identical)** — fixed-seed 20-game rollout at G=1 produces a
  parquet identical to pre-change output (same gate machinery as the cozy
  cutovers). Possible because G=1 issues today's exact batches.
- **Layer 2 (statistical)** — G∈{4,16} vs G=1 on the same 20 games:
  best-arm move agreement ≥99%; p99 |Δ backed_value| ≤ 1e-3; plus the
  `--profile` throughput report at each G.
- **Layer 3 (acceptance, slow / nightly)** — a short training run on batched
  rollouts, then eval_vs_stockfish within noise of a twin trained on
  sequential rollouts.

## Expected outcome

At G=16, the per-wave fixed tax (~29ms of the ~30ms) amortizes ~16-fold:
search_gpu should compress toward its math floor and root_eval similarly —
projected **~2.5-3.5x** on top of the current baseline (→ roughly
3,000-4,500 games/hr locally), with larger G on the 5090 finally converting
that GPU's width into throughput. CPU bookkeeping then becomes the dominant
bucket again, which is the trigger to reopen the Stage-3 (cozy-native
tree/encoder) decision from the CPU-hot-path spec — on infrastructure that
is by then ready for it.

## Out of scope / future

- Eval-driver (Stockfish ladder) batching.
- Kernel-level work (torch.compile, CUDA graphs, fused decode) — revisit only
  after batching regularizes shapes and IF the post-batching profile shows
  FLOP-bound waves.
- Incremental root-KV reuse across plies (the reverted `_IncrementalRootCache`
  idea) — batching changes its economics (the extra per-ply calls would ride
  existing merged batches); re-evaluate with profile evidence only.
- Stage 3 of the CPU-hot-path spec (cozy-native encoder / coarse Rust calls)
  — reopens if/when post-batching profiles show CPU dominant.
