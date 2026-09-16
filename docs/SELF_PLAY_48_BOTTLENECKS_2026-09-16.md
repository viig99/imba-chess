# Remaining self-play bottlenecks at concurrency 48

The direct history cache remains the only maintained workspace implementation.
This investigation changes no runtime code or production concurrency setting.
It uses 48 as the next optimization baseline, following the
[concurrency comparison](SELF_PLAY_CONCURRENCY_2026-09-16.md).

## What the measurements support

The earlier 100-game run at concurrency 48 produced approximately **88,608 played
search moves/hour and 1,132 games/hour**, with 3.87 GiB peak allocated CUDA memory
and 4.96 GiB peak reserved memory. That is one workload sample, not a sustained
hourly guarantee or proof that 48 is optimal. Timing variation and the interrupted
24 repeat remain unresolved. It does not establish improvement over the user's
historical 95,000 moves/hour under unspecified comparison conditions.

In that 317.91-second run, host scopes were preparation 91.92 seconds, decoder
submission/waits 61.87 seconds, result processing/readback 74.19 seconds, root
evaluation 20.73 seconds, and residual work 69.20 seconds. These scopes are not
exclusive CPU computation or GPU execution times. In particular, blocking readback
can inherit the wait for preceding GPU work.

The new diagnostic used 48 warmup games, 48 baseline games, 48 separately profiled
games, and a short trace of 48 searches. The smaller complete-game workload has a
larger tail: mean leaf batch 13.16, versus 26.85 in the 100-game workload. Do not
transfer its percentages mechanically to an indefinitely replenished collector.

## Padding and history refreshes

Across 456,746 decoder rows in 34,710 batches, 394,062 rows reused their staged
history. The remaining 62,684 rows copied 368.43 GB and zeroed 217.47 GB of history
buffers (decimal units). Disjoint refresh attribution was:

| Cause | Dirty rows | Copied GB | Share of copied bytes |
|---|---:|---:|---:|
| Existing owner moved to another batch row | 56,182 | 323.41 | 87.78% |
| New owner | 3,835 | 25.41 | 6.90% |
| Layout refresh of an otherwise unchanged row | 2,667 | 19.60 | 5.32% |

There were 529 maximum-prefix-width changes and 529 full refreshes. Owner changes
take attribution priority when they coincide with a width change. These are staging
copies, not neural recomputation of valid histories. The attribution sums exactly
match production copied-byte, zeroed-byte, and dirty-row counters.

**Preallocating a larger tensor alone does not address the dominant copying cause.**
Geometric allocation already exists. Fixed row strides and a view through the
longest active history would avoid repacking unchanged rows when that width changes.
Stable owner-to-row placement is also needed to tackle the larger reordering cost.
That requires preserving request/result correspondence and handling inactive slots;
otherwise avoiding copies could add unnecessary decoder rows. A noncontiguous view
must be checked for compiler-inserted copies, different kernels, and numerical
changes. Capacity must still grow beyond any initial 200–300-move reservation.

Weighted over actual decoder queries, padding occupied **33.41% of prefix slots**
and **89.58% of the fixed 32 branch slots**. Mean valid branch depth was 3.33.
Including the new token, 43.15% of total attention slots were padding. This is an
attention-slot count, not a fraction of model FLOPs or wall time: projection and
feed-forward work do not disappear with ragged attention.

The branch gather wrote 957.87 GB, including its padding. A bounded branch-width
experiment could therefore reduce both gathering and padded attention. Logical
width must continue to support depth 32; merely lowering search depth would change
the algorithm. Bucketing widths also introduces shape/launch tradeoffs that need
measurement. Jagged prefix attention is a larger decoder change and would still
need a solution for ownership and packing; it is not a free replacement for the
existing dense views, relative bias, and joint prefix/branch softmax.

## CPU, launches, and synchronization

The complete-game cProfile run took 238.08 seconds, including instrumentation.
History validation took only 0.234 seconds cumulatively. Direct refresh took
15.77 seconds, including 8.83 seconds in its Python body. Ancestor gathering and
mask/position preparation together took 15.32 seconds cumulatively, a 6.43% host
submission upper bound. The blocking tensor `.cpu()` calls accumulated 32.16
seconds; that number includes device waits and is not a transfer-only cost.

The short CUDA trace had 34,820 kernels. Kernel/copy/memset intervals covered
565.92 ms of a 1,761.08 ms device-event span, or 32.14%. Instrumentation itself adds
gaps, so this is not a production utilization estimate. It nevertheless supports
testing whether a bounded CUDA Graph replay can reduce repeated dispatch overhead.
Compiled decoder host scope was 571.49 ms out of the 1,799.31 ms search scope;
result processing was 295.93 ms. These inclusive scopes must not be added to their
child operators. Profiler GPU annotation rows also overlap and must not be summed.

Preparation's exclusive CPU/launch contribution, excluding explicit wait APIs,
was **7.82%**, below the agreed 10% fusion trigger. No preparation fusion was
attempted. No CUDA Graph, asynchronous readback, fixed-stride, or jagged candidate
was implemented in this diagnostic.

## Recommended experiments and regression assessment

1. Run the bounded CUDA Graph experiment against direct caching, with a small
   shape set and explicit memory limits. Launch gaps remain a broad opportunity;
   graph shape coverage and retained memory are the principal constraints.
2. Test stable owner placement together with fixed-stride history storage.
   Preallocation alone targets only a small measured fraction of copying. Preserve
   active-game semantics and inspect the compiled layout before promotion.
3. Measure bounded branch-width buckets. High branch padding affects both gather
   traffic and attention, but slot savings are not a speedup prediction.
4. Investigate output-buffer reuse and overlapped readback after the dispatch
   experiment. The next search wave depends on results, and pinned input storage
   currently relies on the blocking readback for safe reuse.
5. Re-profile before a jagged attention rewrite or additional native bookkeeping.
   Native interior selection still used 5.83 seconds in this profile, but it is
   smaller than the combined staging and dispatch costs.

No new leak or continuing compilation was observed: baseline/profile games and
targets matched bitwise, all transfer counters matched, compilation remained at
two graphs, caches cleared, and retained allocation was 201,884,160 bytes. The
direct-cache promotion previously passed its measured speed/correctness gates.
Increasing concurrency exposes more staging and padding work; the 100-game
comparison showed approximately 51% more copied history bytes at 48 than at 24,
offset by fewer decoder batches. This is a changed cost balance, not evidence of a
new regression caused by removing alternative history implementations.

Every candidate still needs cold and updated-weight checks, full-game correctness,
repeated unprofiled throughput/latency measurements, and bounded memory/retention.
No gain from the proposed experiments is claimed yet.

## Evidence

Artifacts: `artifacts/self_play_validation/profile48_bottlenecks_2026-09-16/`.
`profile_with_padding.py` adds diagnostic-only attribution around the unchanged
workspace. Its SHA-256 is
`ca0aff75cd1939a672b91bff96696dd9d7b1aab979b9e4d1fe08d3f7c851e767`.
`run/metadata.json` archives source/config/checkpoint/seed identities;
`padding_refresh_attribution.json`, `verification.json`, `device_activity.json`,
`fusion_trigger.json`, full cProfile output, Chrome trace, game records, targets,
latencies, and memory/copy/compilation counters are under `run/`.
The wrapper adds Python work only to the cProfile pass. No thermal controls or
thermal telemetry were introduced. Production checkpoint/replay were untouched.
