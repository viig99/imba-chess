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
- The production CUDA default selects `direct` only on the compatible optimized
  path. Explicit `current` remains the validated rollback; CPU and legacy behavior
  are preserved.

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

The lower memory use makes 32 concurrent games a reasonable next benchmark, then
48 if useful. Higher concurrency is not yet validated or enabled and may increase
CPU/launch work. Keeping rollback and attribution modes does not execute them or
allocate their history buffers alongside direct; the small mode checks are already
included in the measurements. Deleting supported modes has no demonstrated speedup.

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

## Cleanup audit and commit boundary

The audit covered the history workspace, runtime options, native selection, and
profiling harness. It removed the redundant `refresh_prefix` boolean, whose state
was identical to whether the dirty-row list was nonempty. The focused post-cleanup suite passed 120 tests. An isolated checkout containing
only the selected commit files then passed 138 tests, including self-play tests
(15 extended cases deselected). Lint and staged whitespace checks passed.
No unused collector feature flag was found: the alternatives have active callers or validation uses.

| Retained path | Why it remains |
| --- | --- |
| `history_cache_mode="current"` | Explicit validated rollback and performance reference |
| `history_cache_mode="revision"` | Reproduces the required validation-only attribution ablation |
| `reuse_decode_buffers=False` | CPU/legacy compatibility and prior collector reference |
| Python Gumbel selection | Exact native-versus-Python oracle and explicit fallback |
| Legacy profiler study | Reproduces the preceding reusable-buffer/native promotion evidence |
| Mutable-history fingerprinting | Detects external replacement and ordinary in-place tensor changes |

These paths are not removed merely because the optimized CUDA default bypasses
them. Broader retirement would change supported behavior or evidence reproduction
and should be a separate agreed change with appropriate revalidation. No fused
preparation implementation was added: the full-profile upper bound was 8.03%,
below the 10% trigger.

The commit includes the previously uncommitted reusable-buffer/native-selection
prerequisites, the new history-cache optimization, their tests, profiling tools,
and reports. It excludes unrelated training compilation, loss metrics, evaluation
experiments, observe-only screens, and UI/design files already in the workspace.
The earlier reusable-buffer/native stage reported a 58.3% median gain against its
own older baseline; that is prior-stage evidence, not this session's incremental
gain. The two percentages should not be presented as a measured combined result.

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
