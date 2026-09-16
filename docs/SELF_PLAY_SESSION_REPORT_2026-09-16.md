# Self-play optimization session report — 2026-09-16

The compatible optimized CUDA collector now uses revision-validated histories and
direct updates to decoder history buffers. Three alternating comparisons of 128
complete games per variant passed the promotion gates: **7.52% median paired
throughput improvement**, improvement in every pair, and **3.48% lower median paired
p95 move latency**. The comparison starts from the already optimized reusable-buffer
and native-Gumbel collector, not from the original preparation path.

## What changed this session

- Immutable self-play histories use owner/revision/length validation instead of
  inspecting every layer's tensor identity, pointer, and version at each leaf.
  Mutable callers retain fingerprint validation. Explicit replacement invalidates
  old branch handles, and updated weights receive newly computed root histories.
- Decoder history buffers receive original K/V directly, only for changed rows.
  The candidate removes intermediate stacked and compact histories. Batched copies
  and separate padding zeroing preserve exact metadata, layout, and independent
  per-layer storage. Unchanged rows survive compatible batch-size changes.
- Profiling and validation now separate cold starts, warmed collection, function
  profiling, short CUDA traces, and scratch collect/update/collect checks. Artifacts
  include complete games, targets, latency distributions, memory, compilation,
  copy counters, and source/config/checkpoint hashes.
- The reusable decoder workspace now has a single history implementation: direct.
  The current/revision alternatives and their runtime switch have been removed.
  CPU compatibility and mutable-history safety checks are preserved.

## Measured results and practical impact

| Pair | Throughput change | p95 latency change |
| --- | ---: | ---: |
| Current then direct | +21.32% | −16.00% |
| Direct then current | +7.52% | −3.48% |
| Current then direct | +2.11% | −0.07% |
| Median paired change | **+7.52%** | **−3.48%** |

At the validated 24 concurrent games, median observed rates were **1,162 completed
games/hour**, **93,796 played/searched moves/hour**, and **88,817 usable training
positions/hour**. Moves ranged from 86,934 to 94,818/hour. This does not establish
an absolute rate above the previously observed 95,000 moves/hour: the current
reference also ran slower in this session. The paired comparison is the evidence
for improvement. These rates exclude optimizer updates and strength screens.

The history-mode cleanup is for simplicity; no additional speedup is attributed
to removing the alternate implementations.

Each pass played the same 128 games, searched 10,358 positions, and generated 9,782
usable training positions. Direct therefore produces more fresh training data per
collection hour without reducing simulations or changing search decisions. The
median paired gain corresponds to roughly 7% less collection time for the same
workload. It is not a measured improvement in chess strength, training quality, or
complete collect/train/evaluate cycle speed; training and strength screens still
consume time outside collection.

Direct's three observed collector rates ranged from about 82,321 to 89,798 usable
positions/hour (median 88,817). Those are bounded collector measurements with
scratch replay writes, not an overnight end-to-end production forecast. Pairwise
gains are the primary comparison; ratios of separately selected median rates are
a different statistic.

Peak allocated CUDA memory fell from **3.36 GiB to 2.19 GiB**, leaving about
1.17 GiB more headroom. Logical history-copy traffic fell from **5.633 TB to
1.005 TB** per full pass; Python history-copy submissions fell from **403,587 to
16,263**. These counters do not represent physical DRAM traffic or kernel counts.
The 32-game revision-only ablation attributed about 3.8% throughput gain to cheaper
validation; the combined short ablation gained 22.1%. Full-game repeated results
above govern rollout.

All measured moves, legal ordering, visits, outcomes, and search counters matched.
Serialized targets were bitwise equal. Cold processes exercised single-game tails;
the scratch optimizer update changed weights and matched reference targets both
before and afterward. All full passes held the warmed graph count at two, cleared
caches, and retained exactly 201,884,160 allocated bytes after collection. Production
checkpoints and replay were not modified.

## Follow-up concurrency comparison

A subsequent matched 100-game comparison measured **67,451 moves/hour at 12**,
**80,478 at 24**, and **88,608 at 48**. Thus the earlier slow 48 result below did not
repeat: the fresh 48 case had identical work/targets but much lower elapsed time.
The supplemental final 24 repeat was interrupted and excluded after the user
reported a slowdown. These are exploratory single measurements, not robust speedup
estimates; no default was changed. See the
[concurrency report](SELF_PLAY_CONCURRENCY_2026-09-16.md) for rates, latency, memory,
batching/copy costs, the interrupted-run caveat, and the fixed-buffer discussion.

## Requested 48-concurrency exploration

The direct-only collector completed a warmed **100-game** measurement at **48
concurrent games**. Actor111, FP32, 128 simulations, top-m 16, depth 32, four CPU
threads, and seed manifest remained fixed. The 48-game warmup is excluded below.

| Measure | Result |
| --- | ---: |
| Timed collection | 486.648 seconds |
| Completed games | 100 |
| Played/searched moves | 7,849 |
| Usable training positions | 7,436 |
| Completed games/hour | **739.8** |
| Played/search moves/hour | **57,936.6** |
| Usable positions/hour | **55,008.1** |
| p95 move latency | 5.332 seconds |
| Peak allocated CUDA memory | **3.87 GiB** |
| Peak reserved CUDA memory | **4.96 GiB** |

More VRAM was used, but this exploration did not outperform the earlier 24-slot
results. Keep the production default at **24**. This single 100-game run and the
prior repeated 128-game runs are not a matched concurrency experiment; no precise
causal regression percentage is claimed. Preparation occupied 154.37 seconds of
host scopes, decoder launch/execution waits 116.87 seconds, and result processing
73.81 seconds. These include waits and are not exclusive CPU or GPU durations.
Memory capacity alone does not determine collector throughput.

The run exercised a single-game tail, held the warmed compiled-graph count at two,
and released caches, retaining 201,884,160 allocated bytes afterward. Against the
same 100 game IDs in the prior 24-slot result, 99 complete trajectory records were
identical; none of the full per-game target records were bitwise equal. Thus this
is also not correctness/promotion evidence for changing concurrency. Changing batch
shapes changes floating-point execution, and the cross-concurrency discrepancy has
not been isolated further. The direct-only cleanup is independently checked at 24.

Evidence is under
`artifacts/self_play_validation/direct_only_48x100_2026-09-16/`.
`run/baseline/` holds metrics, every move latency, games, and targets;
`cross_concurrency_check.json` records the comparison. `measured_harness.py` preserves
the exact running script. The process had already started with the former cooldown
option before that option was removed; it completed unchanged and its pre-timing
pause is excluded. Future benchmark commands have no thermal wait or telemetry.

## Cleanup and commit boundary

Following promotion, the user chose one maintained history-cache implementation.
Direct was fastest; current and revision were alternate implementations of the same
feature. The cleanup removes their stacked/compact buffers, copy helpers, counters,
mode branches, and `history_cache_mode` plumbing. It also removes a redundant refresh
boolean. Immutable revision validation and mutable-history fingerprinting both
remain inside direct, because mutable callers require mutation detection.

The benchmark and profiler use only the production CUDA implementation.
`scripts/bench_gumbel_pipeline.py` measures throughput; `scripts/profile_gumbel_pipeline.py`
adds function profiling and a CUDA trace. Both share the same harness. The
`--skip-profile` and `--thermal-cooldown` flags and automatic cooldown waits are
removed, along with thermal telemetry. Performance profiling remains available. `--runs` repeats
warmed measurements, `--concurrency` changes the exploration workload, and saved
`--reference-targets`/`--reference-games` support comparisons across Git revisions.
Obsolete in-process variant/pair/resume machinery and its gate-only tests are removed.
`scripts/validate_history_cache.py` now has one purpose: a scratch collect/update/
collect comparison of the optimized workspace against the existing compatibility
decoder. It does not retain either retired history-copy implementation.

Commit `5a23adf` preserves the full promotion implementation and harness. The
simplification is a separate follow-up commit. Historical report commands and
current/revision ablations refer to that earlier commit. CPU decoding, the
non-workspace compatibility decoder, and native/Python selection are separate
capabilities and are outside this removal of duplicate history-cache implementations.

Post-cleanup verification passed **117 focused tests** in an isolated checkout of
the selected source (nine extended cases deselected), plus lint and whitespace
checks. Two complete 32-game CUDA passes at 24 concurrency matched the archived
game trajectories and **bitwise targets**, exercised single-game tails, and cleared
caches. These verify unchanged direct behavior independently of the 48-slot
exploration. The test count decreased because retired modes and their duplicate
parameterizations were removed; direct-path behavioral coverage remains.

The earlier reusable-buffer/native stage reported a 58.3% median gain against its
own older baseline; that is prior-stage evidence, not this session's incremental
gain. The two percentages should not be presented as a measured combined result.
Unrelated training compilation, loss metrics, evaluation experiments, observe-only
screens, and UI/design edits remain outside these commits.

Current measurement command (scratch output directory):

```bash
.venv/bin/python scripts/bench_gumbel_pipeline.py \
  --config config/self_play_laptop_pilot.toml \
  --checkpoint artifacts/eval/self_play_actor111_sf2400_2026-09-14/checkpoints/actor111.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play_validation/direct-exploration \
  --concurrency 48 --games 100 --warmup-games 48
```

## Measurement limits and next step

CUDA worked outside the sandbox; initial failures were sandbox visibility, not
missing GPU hardware. Independent compiler tuning caused current-versus-current
numerical differences, so cold graph compilation used shared persisted tuning
choices. Earlier measurements also exposed thermal drift. The final sequence used
common cool starting conditions outside timed collection, retained all three
pairs, and archived the earlier failed hot sequence. The 2.11–21.32% gain range
shows substantial laptop variability despite these controls.

The short direct trace still contains about 32,000 kernels and substantial host
launch gaps. A bounded CUDA Graph experiment is the next justified investigation,
with its own correctness, updated-weight, memory, and whole-collector gates.
Graphs have not been implemented or credited with a speedup. Asynchronous readback
and further native bookkeeping remain later candidates.

Detailed evidence and reproduction commands are in
[the history-cache report](HISTORY_CACHE_PROFILE_2026-09-16.md). Prior-stage evidence
is in [the reusable-buffer/native report](SELF_PLAY_PROFILE_2026-09-15.md).
