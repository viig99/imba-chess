# Direct-cache self-play concurrency comparison — 2026-09-16

Matched workload settings: Actor111, FP32, 128 simulations, top-m 16, depth 32,
four CPU threads, native Gumbel selection, and the single direct history-cache
implementation. The run order is **24 → 12 → 48 → 24**. Each process warms on
48 complete games, then times 100 complete games from the same seed manifest.
Compilation, warmup, and diagnostic profiling are outside the measurement. There
are no thermal controls or temperature-based waits. Production replay, checkpoints,
and the default concurrency are unchanged.

**48 was fastest in the completed comparison; 12 was not faster than 24.**
This is exploratory evidence from one completed measurement per setting. The user
reported a slowdown during the supplemental final 24 run, which was deliberately
interrupted during warmup and excluded. Its incomplete-workload assertion in the
log is expected from the interruption, not a decoder correctness failure. No clean
repeated comparison or production default change is claimed.

| Concurrent games | Seconds / 100 games | Games/hour | Played/search moves/hour | Usable positions/hour | p95 seconds | Peak allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 418.06 | 861 | 67,451 | 64,033 | 0.78 | 1.19 / 1.44 |
| 24 | 349.00 | 1,032 | 80,478 | 76,415 | 1.36 | 2.16 / 2.70 |
| 48 | 317.91 | 1,132 | 88,608 | 84,204 | 2.07 | 3.87 / 4.96 |

Against the completed 24 case, 12 produced 16.2% fewer moves/hour and 48 produced
10.1% more. Treat these as observed single-comparison differences, not reliable
long-run gain estimates. Individual-move latency increases with concurrency; the
48 setting is promising for training-data throughput and fits within 8 GiB, but is
not a proven optimum. The configured default remains 24.

Each setting completed the same 100 game IDs. The 24 case played 7,821 moves;
12 and 48 played 7,849 (0.36% more). Rates use actual played moves. Across settings,
99 of 100 trajectory records matched and all outcomes matched; target digests were
not bitwise equal. This is not a concurrency promotion/equivalence test. All runs
kept the warmed graph count at two, cleared caches, and retained exactly 201,884,160
allocated bytes afterward. Checkpoint, source, seed manifest, and all settings other
than concurrency matched. Production checkpoint contents were unchanged.

## What the counters explain

The measured costs show both sides of the batching tradeoff:

| Concurrency | Decoder batches | Mean active rows | History copied / zeroed GB | Preparation host seconds | Decoder host seconds | Result-processing host seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 84,638 | 11.03 | 440.6 / 528.5 | 81.77 | 137.37 | 100.70 |
| 24 | 47,746 | 19.50 | 767.0 / 899.1 | 79.88 | 84.84 | 86.84 |
| 48 | 34,773 | 26.85 | 1157.2 / 1134.9 | 91.92 | 61.87 | 74.19 |

At 12, history-copy traffic is lower, but there are 77% more decoder batches than
at 24. Preparation time stays similar and decoder/result host time increases.
This supports the interpretation that smaller batches lose throughput to additional
calls and less work per call. At 48, logical history copying increases by 51%, yet
batch count falls by 27%; the extra preparation time is outweighed by lower decoder
and result-processing host time in this measurement.

Host scopes include CPU submission and waits and are not exclusive GPU timings.
The counters are logical bytes, not measured physical DRAM traffic. These data
support a batching/copying tradeoff; they do not isolate every kernel or establish
the cause of machine timing variation. Mean batch size is below concurrency because
of scheduling and the shrinking complete-game tail. A longer collection may have a
different balance. More VRAM use alone does not establish better performance.

### Why the earlier 48 result was misleading

The earlier 48-concurrency exploration took 486.648 seconds (57,937 moves/hour).
The fresh run took 317.914 seconds (88,608 moves/hour). These two runs have identical
complete-game signatures, bitwise target digests, search counts, batch histograms,
and transfer counters. The slower run therefore did not execute a larger search
workload or more logical history copies. Its execution-time variation has not been
isolated to a particular hardware or runtime cause. It does not establish that
48 concurrency is intrinsically slower than 24.

### Cache refreshes do not invalidate every game's history

Each game's cached K/V history stays valid during its search. The reusable decoder
input packs active games into a rectangular batch. Unchanged owners in unchanged
rows reuse their input histories. Changed owners or row order refresh affected rows.
Changing the logical maximum prefix width changes the current contiguous layout,
which refreshes all active rows; storage growth also causes a full refresh.

The first 24 run recorded 878 full refreshes and 138,396 dirty-row refreshes.
Even assuming every full refresh touched 24 rows, they account for at most
21,072 / 138,396 = **15.2% of dirty-row events**. This is a count bound, not a bound
on bytes or time. Most dirty rows therefore come from ownership/order changes;
eliminating width changes alone cannot eliminate most row-refresh events.

### Fixed-capacity views and jagged histories

A fixed-capacity buffer with a sliced active view is a plausible smaller experiment
than rewriting attention around jagged histories. Capacity is already overallocated;
the proposed change would preserve fixed strides rather than repacking contiguous
rows whenever the logical width changes. That can avoid width-induced copies, but
short histories still require masking and stable game placement remains important.
The compiled decoder must handle those strides efficiently and pass numerical and
complete-game checks. No performance gain from this proposal has been measured.

Jagged storage can remove rectangular padding, but packing histories into a new
jagged tensor every batch can still copy data. Avoiding both costs requires the
attention path to consume existing variable-length caches efficiently, including
relative-position bias and branch suffixes. PyTorch provides jagged tensors and
attention support, with operation-specific limitations; it is not an automatic
replacement for this decoder. [PyTorch nested-tensor documentation](https://docs.pytorch.org/docs/2.14/nested.html).

Games have different history lengths because they are at different move numbers;
they need not belong to different collection sessions. Neither a jagged rewrite nor
a fixed-stride cache experiment was introduced during this concurrency comparison.

## Evidence and reproduction

Artifacts are under
`artifacts/self_play_validation/concurrency_12_24_48_2026-09-16/`:
`commands.json`, per-process metadata/source hashes, full games/targets/latencies,
copy counters, memory/compilation statistics, `summary.json`, and the comparison
with the earlier 48 run. `a24`, `b12`, `c48`, and `d24` identify the ordered cases.
The interrupted second 24 run has a separate `d24_timing_note.json`; it is excluded
from the results. `verification.json` checks source/settings identity and cleanup.

For each case, use a fresh output directory and the same persisted kernel-tuning
cache. Substitute concurrency 24, 12, 48, then 24, in that order:

```bash
.venv/bin/python scripts/bench_gumbel_pipeline.py \
  --config config/self_play_laptop_pilot.toml \
  --checkpoint artifacts/eval/self_play_actor111_sf2400_2026-09-14/checkpoints/actor111.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play_validation/concurrency-example \
  --concurrency 24 --games 100 --warmup-games 48
```
