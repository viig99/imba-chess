# Stable history placement and selective ancestor gather

The combined candidate passed all three alternating 128-game comparisons against
current direct. It is integrated into the existing reusable decode workspace as
one implementation, without a mode flag. Integrated full-game verification and separate profiling passed.

## Matched collector measurements

Actor111, FP32, TF32 off, 128 simulations per played move, top-m 16, depth 32,
48 concurrent games, four CPU threads, and native Gumbel selection were fixed.
Each fresh process warmed on 48 complete games before timing 128 complete games.
All six timed collections played the same 10,386 search moves. GNOME remained
running. Profiling was excluded from these timings.

| Pair and order | Direct seconds | Combined seconds | Direct moves/hour | Combined moves/hour | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1: direct → combined | 370.026 | 338.993 | 101,046 | 110,296 | 9.15% |
| 2: combined → direct | 403.921 | 337.699 | 92,567 | 110,719 | 19.61% |
| 3: direct → combined | 402.275 | 333.759 | 92,945 | 112,026 | 20.53% |

Median paired throughput improvement is **19.61%**; every pair improves.
The combined median is **110,719 played search moves/hour**, or **1,365 complete
games/hour** (range 1,359–1,381). A move here is a played ply with 128 search
simulations, not a leaf evaluation. Rates use played moves divided by synchronized
whole-collector elapsed time, including collection bookkeeping. They exclude
warmup, profiling, and optimizer updates.

| Metric | Direct | Combined |
| --- | ---: | ---: |
| p95 move latency, pair 1 | 1.932 s | 1.681 s |
| p95 move latency, pair 2 | 2.088 s | 1.657 s |
| p95 move latency, pair 3 | 2.086 s | 1.649 s |
| Peak allocated CUDA memory, every run | 3.940 GiB | 3.940 GiB |
| Peak reserved CUDA memory, every run | 4.977 GiB | 4.977 GiB |
| Allocated after collection, every run | 201,884,160 bytes | 201,884,160 bytes |

Median paired p95 latency improved **20.63%**. All runs stayed below the 6 GiB
allocated-memory gate, retained no workspace buffers after collection, and
performed no further compilation for warmed shapes (two compiled graphs).
Complete games, legal ordering, visits, outcomes, and search counters matched;
**targets were bitwise equal in every pair**.

The roughly 110.7k rate is about 16.5% above the user's historical 95k moves/hour,
but that historical comparison is not controlled. The matched-pair improvement
above is the evidence for this change. Desktop/browser activity remains a source
of variation; neither SM isolation nor a universal hourly rate is claimed.

## What changed

Stable placement keeps each surviving history owner in its previous active row
where possible, fills holes, and restores outputs to the scheduler's requested
order. Owner/revision validation, packed contiguous prefix layout, and logical
prefix width stay intact. Width changes still refresh all active histories.

Selective gather reads the maximum occupied ancestor depth from the existing CPU
staging data. It gathers only those columns, initializes new branch storage to
zero, and zeros vacated columns across all capacity rows when depth shrinks.
Inactive rows therefore remain safe when the batch grows. Independent per-layer
buffers and the full 32-column contiguous decoder input layout are preserved.
Decoder arithmetic and returned-node K/V ownership are unchanged.

| Logical traffic over the same 128 complete games | Direct | Combined | Reduction factor |
| --- | ---: | ---: | ---: |
| History copies + padding zeroing | 3,452.792 GB | 458.089 GB | 7.54× |
| Ancestor gather writes + branch zeroing | 2,597.201 GB | 1,153.915 GB | 2.25× |
| Sum of those categories | 6,049.993 GB | 1,612.004 GB | 3.75× |

These counters describe cumulative logical tensor copy/write volume, not measured
physical DRAM bandwidth and not resident VRAM. History dirty rows fell from
284,680 to 37,600. Full history refreshes stayed at 832, decoder batches at 49,910,
and readbacks at 49,910. The decoder still computes over padded history and
32-column branch layouts; this change reduces preparation traffic, not all
attention computation. Thus a 7.54× history-traffic reduction does not imply a
7.54× latency reduction.

## Validation and selection

The combined prototype passed exact branch-buffer/padding comparisons, restored
result ordering, depth 32, owner replacement, batch shrink/growth, zero histories,
cold compilation and single-row tails. Real Actor111 scratch collect → optimizer
update → collect matched direct bitwise before and after the update. Cancellation
and injected decoder exceptions released workspace buffers. The production
checkpoint hash remained unchanged.

During integrated verification, a read-only GPU sample recorded GNOME at 80% SM
and the collector at 9%. This confirms contention at that instant, not continuous
utilization or a complete causal explanation of timing differences. The sample is
archived as `integrated_gpu_sample.txt`; no workload was stopped.

The maintained implementation also completed a fresh 48-game warmup and 128-game
verification against the saved direct reference. Targets and complete games were
bitwise identical; search counters and all transfer counters exactly reproduced
the measured combined prototype. Warmed compilation stayed unchanged, the
single-game tail completed, and cleanup/retained memory matched the reference.
This later unprofiled verification took 914.470 seconds (40,887 moves/hour), with
p95 4.687 seconds under the observed desktop contention. It is not a matched
throughput comparison and is not substituted into the three-pair promotion study.
It shows why the 110.7k hourly estimate must be qualified by available GPU time.
Peak allocated/reserved memory remained 3.940/4.977 GiB.

Integrated CPU/CUDA workspace tests: **14 passed, 1 skipped** (the intentionally
unsupported compiled CPU combination). The integration test compares against the
independent legacy decoder, verifies exact branch padding, and requires no history
recopy for reordered owners at unchanged width. Ruff checks passed.

Standalone experiments completed before the user narrowed scope to the combined
candidate. Stable-only results were +49.34% then −46.28% under highly variable
desktop activity, so it did not pass promotion. Gather-only was −1.32% in its
first pair and stopped. These sequential observations do not isolate interaction
strength or establish that selective gather alone is intrinsically slower.
Bounded CUDA Graphs failed their throughput gate (−0.63%); shortening decoder
branch width failed target tolerance. No alternative runtime modes are retained.

## Remaining bottlenecks

The separate 128-game function profile completed in 502.894 seconds with bitwise
reference targets and identical transfer counters. This is diagnostic elapsed
time, not a throughput measurement. Cumulative host scopes were:

| Scope | Profiled seconds |
| --- | ---: |
| Result processing/readback | 145.09 |
| Decoder calls/submission | 100.11 |
| Preparation, including history and branch work | 86.80 |
| Root evaluation | 29.78 |

These host scopes include asynchronous wait placement and are not exclusive GPU
execution measurements. Blocking `.cpu()` calls alone accumulated 105.65 seconds.
History refresh accumulated 10.74 seconds, ancestor gather 16.93, masks/positions
8.19, stable placement 1.86, and history validation 0.57. The placement bookkeeping
is small in this profile; reduced traffic did not introduce a comparable new host
bottleneck. Attention padding and synchronous result consumption remain costs.

The short CUDA trace attributes **9.83%** of profiled wall time to exclusive
mask/relative-position/ancestor-gather CPU and launch work, excluding explicit
wait APIs and neural execution. This is below the **10% fusion trigger**, so no
compiled preparation alternative was added. It is a short workload sample, not
a precise whole-collection fraction. All warmed collection passes retained two
compiled graphs with unchanged before/after counters; the separate trace workload
ended with three total specializations, outside the throughput measurements.

Bounded exact-shape CUDA Graphs already failed the whole-collector gate despite
improving submission in a microbenchmark. No replay implementation is retained.
The evidence favors investigating wait/readback scheduling and the remaining
padded decoder computation in a future bounded experiment. The current stage
ends with only the passing combined cache implementation.

## Evidence archive

`artifacts/self_play_validation/cuda_graph_stage_2026-09-16/` contains:

- `pairs/combined/summary.json`: three pairs and promotion gate calculations.
- `pairs/combined/pair_*/`: complete targets, games, timing distributions, source,
  config/checkpoint/seed hashes, compiler counters, memory and transfer counters.
- `combination_sources.json`: hashes of the exact experimental harnesses.
- `combined_unit.log`, `combined_updates/summary.json`: combined correctness gates.
- `integrated_verification.json`: exact integrated/prototype counters and reference targets.
- `decision.json`: final selection and commit.
- `integrated_unit.log`, `integrated/`: maintained implementation verification and
  separate function/CUDA profiling.

Checkpoint SHA-256:
`7bdce092f7b265b3f13edb456dd9bda8e335e63568aac345bf29140593fd3f63`.
Production replay, checkpoint, optimizer settings, and the config's default
concurrency were not changed. Reported concurrency 48 is an explicit benchmark
override. Measurements used the existing worktree, including its pre-existing edits; source
hashes archive that context. Unrelated workspace edits are excluded from this
change.
