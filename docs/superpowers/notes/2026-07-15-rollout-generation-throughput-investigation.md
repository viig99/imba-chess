# Rollout-generation throughput investigation (2026-07-15)

Session summary: measured why nightly rollout generation is far too slow to hit a 10,000-20,000-step KL-coverage target in 2-3 days, tried scaling to a rented RTX 5090 box, tried one optimization that didn't pay off (reverted cleanly), and profiled to find the real bottleneck. No production code changed by the end of this session except an optional `--profile` diagnostic flag; the actual fix (a faster `gives_check`) is designed but not yet built.

## 1. The scale of the gap

Target: 10,000-20,000 steps of KL-signal coverage in 2-3 days. At ~50 games/training-step, that's 500,000-1,000,000 unique covered games.

**Local baseline** (single process, correct `search_budget=2048` config): ~5.9s/game, matching the overnight nightly-cron's own measured rate (4,796-4,800 games per ~8h session, both nights). At that single-process rate: 500k games would take ~35 days. Required rate for 2-3 days: ~7,000-21,000 games/hr, i.e. **12-35x today's single-process baseline** — not a tuning problem, a throughput-architecture problem.

## 2. Shard parallelism (local): real, but bounded by RAM not GPU

`scripts/generate_search_rollouts.py` has no DataLoader at all — a single process streams games one at a time via `LichessDataset.stream()`. The only existing parallelism lever is `--shard-id`/`--num-shards` (file-level dataset sharding, already built).

4 concurrent local shards: **~2.7x throughput** (not ~4x) — 1,887 games/hr aggregate vs. 697/hr single-shard. GPU was *not* the constraint (59% util, VRAM headroom); **system RAM was** — swap hit 6.9/8GB with just 4 shards, matching this project's known OOM failure mode. Sub-linear scaling is very likely swap-thrashing degrading every process together, not GPU contention.

## 3. Remote RTX 5090 (vast.ai): more RAM headroom, but a slower single-shard rate

Rented instance specs, and a real gotcha: **`nproc`/`free -h` inside the container report the HOST's full physical resources (96 cores, 188GB RAM) — misleading for planning.** The actual container budget is cgroup-limited: `vast-capabilities` reports the true numbers (23 cores, 48GB RAM), confirmed directly via `cpu.max` (`2304000 100000` = 23.04 core-equivalents) and `cpuset.cpus.effective`.

**Single-shard rate on the 5090 was ~14-18s/game — 2.5-3x *slower* than local**, despite the far more powerful GPU. This remained only partially explained this session (see §5 — most of it turned out to be architecture-agnostic CPU cost, which a faster GPU can't fix regardless).

**GPU-memory "leak" that wasn't**: running 16 shards, GPU memory climbed from ~5GB to ~29GB over several minutes in a bursty, decelerating pattern (not smooth/linear). Killing 4 shards dropped memory immediately (29GB → 22GB) — proof it was genuine per-process allocator state, not a real leak. Mechanism: PyTorch's CUDA caching allocator never releases freed memory back to the driver; each of N independent shard processes accumulates its own high-water-mark cache as it encounters longer games over time, and with more concurrent shards there's more chance *someone* hits a new personal-longest game at any moment. Settled on 12 shards for a safety margin (~4GB headroom at a ~28.6GB plateau).

**CPU pinning investigated and ruled out**: `htop` showed cores 0-48 pegged at 100% while our 12 shards ran on 49-96. Checking `ps` for what's running there ourselves found nothing — **cores 0-48 belong to another tenant on this shared multi-tenant host**, invisible via Docker's PID namespace but still visible in `/proc/stat`-based host-wide CPU accounting. Nothing to pin to; the Linux scheduler was already routing our processes to the actually-idle cores on its own. The real constraint (cgroup CPU quota) is bandwidth-based and applies regardless of which physical cores are used, so pinning wouldn't have helped even ignoring the multi-tenancy angle.

**Measured remote throughput**: 12 shards → ~2,509 games/hr → **~1,204 steps/day** — still 8-17x short of the 10k-20k/day implied by a 2-3 day target.

## 4. KV-cache root-eval optimization: implemented, tested, reverted (no net win)

**Hypothesis**: `_SequenceHistory.build_batch_for_current_position` + `_forward_model` recomputes the *entire* game sequence from BOS on every sampled ply's root eval — O(seq_len) work repeated every ~8 plies, even though 7 of every 8 plies were already known context. The model already has `forward_decode` (the same primitive `CachedPositionEvaluator` uses for search-tree nodes) that could extend a persistent cache by one token per real ply instead.

**Built**: `_IncrementalRootCache` in `scripts/generate_search_rollouts.py`, extending a running per-layer K/V cache one token per real ply (`step()` called once per ply, sampled or not — every real ply is an actual played continuation, never a discarded search branch, so every step both scores and permanently commits). Correctness verified via a differential test against a full-recompute reference implementation on a 24-ply game (tight `1e-4` tolerance, float32) — passed cleanly, plus the full 154-test suite.

**Measured result: no speedup — 165s vs. the 155s baseline for the same 30-game/shard-0 test.** Root cause: the fix added a **new** GPU decode call on every non-sampled ply (7 out of 8) that didn't exist before at all — trading "1 potentially-expensive call every 8 plies" for "8 cheaper-but-not-free calls every 8 plies." In a workload that's fundamentally latency-bound (established repeatedly this session — small-call overhead dominates over raw compute), more calls of any size can be a net loss even when each does less work. Confirmed by the later profiling (§5): root eval was never more than ~18-21% of total time to begin with, so even a perfect elimination had a low ceiling — and this implementation's added overhead ate more than that ceiling was worth.

**Reverted cleanly** (`_IncrementalRootCache` and its wiring removed, test file restored via `git checkout`) — back to 154 passing tests, zero net change to the script's behavior.

**Lesson, stated plainly**: optimized a cost that was assumed significant without profiling first. Should have measured before building.

## 5. Profiling (the fix for lesson 4): two layers of instrumentation

**Layer 1 — coarse phase breakdown**, added as an opt-in `--profile`/`--profile-every-games` flag on `scripts/generate_search_rollouts.py` (default off, negligible overhead, zero behavior change otherwise). Buckets: `ply_bookkeeping` (chess + board-state encoding, every ply), `batch_build` (tensor construction, sampled plies), `root_eval` (root forward, GPU), `search_gpu` (search's own `forward_decode` wave calls, via a thin `_TimedEvaluator(CachedPositionEvaluator)` subclass that times its own `evaluate()`), `search_bookkeeping` (computed as search-total-time minus `search_gpu` — i.e. `select_value_search_halving`'s own CPU cost, without touching `search.py`, which stays torch-free by design).

Result (20 games / 169 positions, local):

| Bucket | % of total |
|---|---|
| **search_bookkeeping** (CPU: heap/tree mgmt) | **43.5%** |
| search_gpu (GPU: search's forward_decode waves) | 38.5% |
| root_eval (GPU: root forward) | 17.9% |
| batch_build / ply_bookkeeping | ~0% |

**The single biggest bucket is pure Python CPU overhead inside the search itself — bigger than either GPU call.** This explains both open questions from this session: why the 5090's extra power didn't move the needle as much as hoped (a GPU-agnostic ~43.5% of time is CPU-bound), and why §4's optimization had a low ceiling (`root_eval` was only ~18-21%).

**Layer 2 — `cProfile` over a short real run**, to find exactly which call inside `search_bookkeeping` dominates (zero code changes needed — stdlib tool, run externally via `python -m cProfile`). Result:

| Function | Cumulative time | % of total | Calls |
|---|---|---|---|
| `_push_children` (search.py) | 23.2s | 36.6% | 77,823 |
| → `_is_forcing` → `board.gives_check()` | 9.4s | **~15%** | **1,053,993** |
| → `terminal_value_for_color` → `board.outcome()` | 6.2s | ~10% | 243,304 |
| → `_search_copy` | 2.5s | ~4% | 243,304 |

**`board.gives_check()` is the single biggest specific hot spot in the entire pipeline (~15% of all time, 1M+ calls).** python-chess's implementation internally pushes the move, checks for check, then pops it — a full simulate-and-undo per call — and `_push_children`'s forcing-move floor scans essentially every candidate move at every expanded search node this way, not just the ones that end up mattering.

## Where this leaves things (not started)

**Planned next step**: a static (non-simulating) `gives_check` reimplementation — direct check (piece's attack pattern from its destination square) plus discovered check (does vacating the origin square reveal an attacker), with known-fiddly edge cases (en passant discovered check, castling's rook, promotion's piece type). Plan is to differentially test against python-chess's own `gives_check()` as reference oracle across thousands of random + hand-built edge-case (position, move) pairs *before* touching `search.py` at all — `search.py` is shared production code used by both this script and `eval_vs_stockfish.py` (the actual Stockfish-ladder benchmark), so a silent correctness bug here is a much worse failure mode than a performance miss. Expected win: uncertain, plausibly ~half-to-two-thirds of the 15%, not all of it (per-call Python overhead ate the previous optimization's theoretical win — no reason to assume that risk is absent here).

**Also not started**: whether to keep 12 remote shards vs. try pushing higher now that the GPU-memory plateau pattern is understood; `search_budget` reduction as a direct (but quality-costing) throughput lever; the CPU-quota headroom (~11 more core-equivalents available before the 23-core cgroup cap) is real but secondary to the GPU-memory ceiling that's currently binding.
