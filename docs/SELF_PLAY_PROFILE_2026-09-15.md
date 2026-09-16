# Gumbel-128 self-play profile — September 15, 2026

The validated CUDA default now uses reusable leaf buffers and fused native
Gumbel selectors. Across three alternating 128-game comparisons, median collector
throughput improved **58.3%** over the existing compiled path, with matching game
decisions and replay targets. Median paired p95 move latency improved **33.5%**;
peak CUDA allocation was **3.37 GiB**. The compiled reference remains available.
The initial profile and the implementation/promotion evidence follow below.

## Overnight reference

The September 13–14 run lasted 28,013.23 seconds (7h 47m), using 128 simulations,
16 root candidates, depth 32, 24 concurrent collection games, FP32 and compiled
neural decoding. It completed 51 collect/train cycles and 11 strength screens.

| Measure | Total | Per elapsed hour |
|---|---:|---:|
| Completed collection games | 4,490 | 577 |
| Training-eligible games | 4,051 | 521 |
| Fresh usable training positions | 321,970 | 41,377 |
| Sampled training exposures | 660,047 | 84,823 |
| Optimizer updates | 1,124 | 144 |

An update averaged 587 supervised positions. Configured reuse is 2.0: exposures
include replay reuse, not new independent outcomes. Update count alone does not
establish undertraining. Replay reuse should be evaluated through strength and
generalization, not chosen to maximize update count.

Collection occupied 20,073.13 seconds (71.7%); recorded optimizer steps occupied
674.50 seconds (2.4%); the remaining 7,265.60 seconds (25.9%) includes screens,
checkpoint/replay handling and unmeasured orchestration. The new profile below
does not attribute that last category. During collection alone the overnight run
produced 805 completed games/hour and 57,743 usable positions/hour.

Sources: `artifacts/self_play/laptop-continuation-2026-09-13/sessions/2026-09-13/`
launch/finished JSON, training and monitor logs; replay manifest split metadata.

## New bounded measurement

Actor111 on the RTX 3070 Ti Laptop, four CPU threads, TF32 disabled. Real collector,
original self-play Gumbel noise, 128 simulations, 16 candidates, depth 32, 24 slots.
Each pass completed the same 32 continuation games, starting from seed-manifest
prefixes. Scratch replay includes actual serialization and flushes. Production
replay/checkpoints were not changed.

| Pass | Wall seconds | Purpose |
|---|---:|---|
| Initial pass | 134.31 | Warm graph shapes and run complete games |
| Warmed unprofiled pass | 166.64 | Host-scope wall-time attribution |
| cProfile pass | 201.78 | Function timings and call counts |

All three game signatures matched exactly: identical game IDs, moves and outcomes.
The warmed baseline introduced no additional compiled graphs (two before/after).
One bounded warmed pass is not a precise sustained-throughput estimate. The initial
pass was faster despite warmup; machine timing variability should be covered with
interleaved repeats before claiming any optimization speedup. cProfile took 21%
longer than the baseline, so its timings must not replace production timings.

Baseline workload:

- 32 games, 2,204 played/search moves, 282,112 simulations.
- 265,685 neural evaluations: 2,204 roots and 263,481 leaves.
- 18,619 terminal hits; 15 depth-cutoff observations.
- 23,755 leaf batches: average 11.09 rows, maximum 24; 11,001 batches had <=4 rows.
- 1,690 root executor calls, sometimes subdivided by the 1,024-token root limit.
- Peak allocated CUDA memory: 1,007,584,768 bytes (0.938 GiB).
- Actual concurrent move latency: median 1.495 seconds, p95 1.799 seconds.
- Aggregate throughput: 13.23 moves/s, 691 games/hour, 46,145 usable positions/hour.

The 32-game tail causes declining occupancy as games finish. These throughput and
batch-size figures are not interchangeable with the longer overnight collector.
166.64/2,204 = 75.61 ms is amortized wall time per move across games, not the latency
experienced by an individual game.

## Non-overlapping host-scope breakdown

| Scope | Seconds | Wall share | Amortized ms / move |
|---|---:|---:|---:|
| Leaf input/KV preparation | 54.11 | 32.5% | 24.55 |
| Decoder call, including host setup/launch | 44.22 | 26.5% | 20.06 |
| Leaf result projection/cache updates/readback | 32.72 | 19.6% | 14.85 |
| Root evaluation executor | 5.09 | 3.1% | 2.31 |
| Other leaf executor work | 3.36 | 2.0% | 1.52 |
| Outside executors: search/game handling/serialization | 27.13 | 16.3% | 12.31 |

These are scopes on the host timeline. They sum to wall time but are not a split
between CPU computation and GPU computation. CUDA launches are asynchronous;
readback can charge waiting for earlier GPU work to the result-processing scope.
The existing counter named `search_gpu` measures the decoder-call host scope, not
GPU kernel execution time. Root inference is eager; the whole neural leaf decoder
is compiled. Its input/cache preparation remains outside the graph.

## Function profile

The following are from the separate 201.78-second cProfile pass. Inclusive rows
overlap and must not be added. Builtin self-time includes time spent in the call,
which may include CUDA API overhead or waits; it is not GPU kernel duration.

| Function / operation | Calls | Self seconds | Inclusive seconds |
|---|---:|---:|---:|
| `build_decode_request` | 263,481 | 4.55 | 32.18 |
| `_padded_chain_indices` | 234,657 | 2.41 | 16.71 |
| `_merge_decode_requests` | 23,755 | 2.39 | 17.49 |
| `_pack_prefixes` | 3,062 | 1.04 | 9.71 |
| `consume_batched_decode_results` | 23,755 | 1.89 | 28.30 |
| `DecoderRunner.__call__` | 23,755 | 2.78 | 52.20 |
| `completed_q` | 1,239,439 | 4.37 | 11.98 |
| `interior_action` | 952,919 | 2.97 | 20.98 |
| Python search `softmax` | 1,220,808 | 4.41 | 7.58 |
| `torch.tensor` | 785,230 | 17.97 | 17.97 |
| Torch padding builtin | 2,014,988 | 17.32 | 17.32 |
| Tensor `.cpu()` | 47,510 | 10.94 | 10.94 |

Each new leaf constructs a decode request; almost every deeper leaf creates chain
indices/masks and gathers its path's cached K/V. Each batch then merges/pads the
per-game data and performs additional decoder workspace preparation. Those small
operations remain expensive even though the neural computation is compiled.

`gumbel_search.py` accounts for 26.21 seconds of summed Python self-time, excluding
its callees. Python search therefore is material. The scheduler module itself
accounts for only 0.21 seconds of self-time; inclusive scheduler timings mostly
contain work it dispatches and must not be interpreted as scheduler overhead.

## What the tensor constructions and padding actually do

These counts are calls over 32 games, not simultaneous tensors, GPU allocations,
or model layers. `torch.tensor` constructs tensor storage from Python data here;
some tensors are CPU-side, others also require a host-to-device transfer. The
785,230 figure does not include every allocation made by `zeros`, indexing,
concatenation, or compiled kernels. Similarly, an `F.pad` call is not necessarily
one GPU kernel. The caching allocator can reuse storage without eliminating the
dispatch, initialization, copying, and bookkeeping work.

A searched move runs up to 128 simulations. Each nonterminal new leaf needs its
board evaluated. This run therefore evaluated 263,481 leaves, organized into
23,755 cross-game batches. Small per-leaf costs multiply quickly.

### Data needed by one leaf

For example, a game may have a cached history of 60 board tokens and the current
simulation may explore a hypothetical branch three boards deep. The new query
must see the cached history, its own branch ancestors, and itself. It must not
see sibling branches or another game's cache. Every one of the eight HSTU layers
has its own K and V for those positions. The caches reuse prior neural work;
reconstructing these K/V by running the full history for each leaf would lose
that benefit.

```mermaid
flowchart LR
    A[Python search chooses a leaf] --> B[Encode board and list ancestor cache rows]
    B --> C[Gather each game's branch K/V]
    C --> D[Merge games and align their lengths]
    D --> E[Extend branch buffers to capacity 32]
    E --> F[Compiled neural decoder]
    F --> G[Store new K/V and read policy/value back]
    G --> A
```

The *information* above is required by the current trained model. Constructing
new tensors and repeatedly repacking that information is an implementation choice.
Dropping history would change the model's inputs; removing redundant packing need
not change them.

### Where `torch.tensor` is used

Caller records and the source identify these main families; counts are rounded
because caller accounting has minor inconsistencies in this threaded profile.
The aggregate 785,230 is the recorded builtin total.

| Purpose | Approximate constructions | Why it exists | Reduction candidate |
|---|---:|---|---|
| Ancestor row indices and lengths, `_padded_chain_indices` | 469,000 | Locate each leaf's ancestors in its game's KV arena | Construct one cross-game index/length table per wave, reuse buffers, gather together |
| Merged board features, positions, prefix lengths | 237,550 | Give the network typed numeric inputs | Pack fields in reusable host/device storage and transfer together |
| Legal-action indices and lengths | 47,500 | Extract and normalize probabilities only over legal moves | Reuse a batched action-index buffer and combine output readback |
| Root-history features and root legal projection | About 31,000 | Initialize a new search from its history | Secondary priority; root executor is only 3.1% of baseline wall time |

The merged input path already defers board-feature tensor creation until games
are batched: eight feature tensors, positions, and prefix lengths give ten tensor
constructions per wave, or 237,550 over 23,755 waves. It would be incorrect to say
all board inputs are still tensorized separately for every leaf. The remaining
per-game chain indices/masks are a stronger target.

Using `as_tensor` on Python lists cannot make their conversion disappear. A
zero-copy CPU view becomes useful if the producer already writes into compatible
contiguous storage. It still does not remove a required CPU-to-GPU transfer.
Likewise, constructing a tiny tensor directly on CUDA does not make the transfer
of Python-owned data free. Packing many inputs together is the relevant change.

### Why there are about two million padding calls

Padding makes variable lengths rectangular. For branch lengths `[1, 3, 2]`,
cross-game packing produces three rows of length 3; a separate validity mask
excludes missing entries. The runner then extends these rows to capacity 32.
Zero values alone are not a sufficient mask: even a zero key can receive softmax
probability and alter normalization.

| Location | Approximate calls | Current reason |
|---|---:|---|
| `_pack_prefixes` | 816,000 | Pad each game's history K/V to the cohort's longest history, per layer |
| `_pack_suffix_layers` | 415,000 | Align gathered branch K/V across games; layers are already stacked for this operation |
| `_merge_decode_requests` | 415,000 | Align branch positions and validity masks |
| `DecoderRunner.__call__` | 369,000 | Extend K/V to the fixed 32-position branch capacity, separately for each layer |

Prefix packing is cached while the ordered group of active game owners is stable.
It ran 3,062 times, not on all 23,755 batches. Nevertheless, changing the active
group invalidates that packed cache. A prefix repack can require two pads per
layer per shorter game: with eight layers this multiplies one repack into many
small calls. The suffix merge already stacks layers to reduce calls; the runner
still loops over layers when extending to capacity.

The fixed capacity is useful for compiled shapes. We can keep it while avoiding
the two-stage allocation/copy path: gather valid ancestors directly into a reused
final-capacity buffer, update lengths/masks, and skip the intermediate padded
tensors. Stable game/cache slots could avoid repacking unchanged prefixes when a
different game finishes. Reused buffers must retain correct masks, game ownership,
layer-specific K/V, and storage lifetimes; returned K/V must not alias scratch
storage that gets overwritten on the next wave.

The measured builtin self-time is 17.97 seconds for `torch.tensor` and 17.32
seconds for padding in the 201.78-second instrumented pass. Together that is about
17.5% of that pass. This does not make all 35.29 seconds removable: useful copies,
input preparation and synchronization remain, and adjacent gathers/packing also
cost time. The broader host-scope table is the better guide to total opportunity.

## Which attention or ragged-tensor API could help?

The actual profiled leaf mode is `compiled`, not `compiled-sdpa`. It compiles the
whole neural decoder and uses the custom one-query attention path; the repository
also has a separate SDPA decoder mode. Collection runs under inference mode.
Compiling the neural decoder does not compile the surrounding Python tree,
request construction, or cache preparation.

### Reusable dense buffers: first experiment

Keep the current attention calculation and FP32 behavior. Batch chain metadata,
gather directly into reusable branch buffers, and reduce prefix repacking. This
targets measured overhead without requiring a new attention backend or a new
model. Some masked capacity remains, but repeated allocation and copying can be
reduced. A batched gather/scatter across a shared arena may be needed; merely
replacing `pad` with equally numerous Python `copy_` calls is not enough.

### Variable-length attention

Varlen attention packs valid tokens with sequence boundaries, removing the need
to pad every sequence to the longest one. The installed PyTorch 2.14 API also
exposes `seqused_k`, paged `block_table`, and a preallocated-output variant; it is
not limited to repacking a flat buffer on every call. See the
[API documentation](https://docs.pytorch.org/docs/2.14/nn.attention.varlen.html).

However, the public signature does not accept our arbitrary learned per-head
relative-position bias or a `score_mod` callback. The documented supported dtypes
are FP16/BF16, whereas this evaluation is FP32. GPU/backend compatibility must also
be checked on this particular laptop. The
[official tutorial](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)
describes packed inputs and dtype/hardware requirements.

Consequently, this is not a drop-in, numerically equivalent replacement. Flat
varlen packing also does not inherently remove the gathering of a leaf's tree
ancestors. Paged storage could address copies, but needs a compatible cache
manager and a solution for the bias. Removing that bias just to use a kernel would
change the trained model.

### FlexAttention / FlexDecoding

FlexAttention offers short-query decoding and programmable score/mask modifiers;
these can express learned relative bias and valid-history/branch membership.
PyTorch also describes paged KV integration. See its
[inference implementation guide](https://pytorch.org/blog/flexattention-for-inference/).

My inference from those capabilities and our code: this is a more natural backend
experiment for preserving our attention semantics. Initially it could operate on
the reusable dense buffers. A larger redesign could let attention read a shared
cache through mappings instead of materializing every branch. Chess search is a
tree: ordinary linear-sequence paging is not automatically an efficient solution
for small, noncontiguous ancestor chains. Page granularity, per-game isolation,
logical relative positions and branch sharing all need explicit design.

Replacing only `_one_query_attention` leaves Python tensor creation and padding
in place. Building an expensive new BlockMask for every leaf could add overhead;
reuse topology/masks where valid and update batched metadata. Benchmark the short
histories and small batches we actually have, not an LLM long-context benchmark.

### FBGEMM jagged-to-dense

FBGEMM represents jagged data with values and offsets and provides conversion
and jagged arithmetic operators. `jagged_to_padded_dense` creates a padded dense
result: it can consolidate many conversion operations, but does not eliminate
the dense output or its masked slots. See the
[format overview](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/overview/jagged-tensor-ops/JaggedTensorOps.html)
and [operator list](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/jagged_tensor_ops.html).

FBGEMM is not installed in this environment. It is a candidate if we first produce
batched values/offsets efficiently. Our ancestor data is selected from separate
tree arenas; conversion alone does not construct those selections. Packing into
FBGEMM format only to densify again could add work. A direct batched gather into
the decoder's final workspace may be simpler and should be the baseline to beat.

## Native/Rust boundary

| Native operation | Calls | Total seconds | Approx. microseconds / call |
|---|---:|---:|---:|
| `MoveProjector.project` | 263,481 | 1.894 | 7.19 |
| `push_and_classify` | 267,421 | 0.498 | 1.86 |
| `encode_board_state` | 263,481 | 0.219 | 0.83 |

These three operations total 2.61 seconds, about 1.3% of the profiled pass. They
measure calls across the Python/native boundary, including their native execution;
this is not an internal Rust flamegraph. Collection replay flushing took 0.326
seconds inclusive. Neither these chess operations nor scratch replay flushing
is the main collector bottleneck in this sample.

## CUDA timeline

A separate, warmed trace contains 24 simultaneous root searches at prefix lengths
sampled across the seed manifest: 3,072 simulations and 3,038 neural evaluations.
There are 128 leaf executor batches. Trace capture took 2.326 seconds inside the
search scope; serialization/analysis afterward is excluded.

- 73,612 GPU kernel executions, summed/union kernel duration about 413.48 ms.
- Including device copies and memsets: union active duration 427.22 ms.
- 11,972 CUDA memcpy operations and 5,716 `cudaStreamSynchronize` calls.
- Mean kernel duration about 5.6 microseconds.
- CPU API time: 206 ms in stream synchronization, 165 ms in memcpy calls,
  156 ms in `cudaLaunchKernel`, plus driver launch work.

Device activity occupies about 18.4% of the instrumented search interval. Profiler
overhead exaggerates gaps: this is not an estimate of production GPU utilization,
nor a claim of a recoverable 5x speedup. The many tiny kernels, copies and syncs
support the diagnosis of launch/preparation overhead. Matrix multiplication,
copy/index/fill kernels, and attention all appear in the GPU trace.

Do not sum `torch.profiler.key_averages()` device times across CPU operators,
kernels and GPU annotation ranges: those contain overlapping/nested events. The
numbers above use actual `kernel`, `gpu_memcpy` and `gpu_memset` trace events and
union their intervals.

## Recommended optimization order

1. Batch chain-index/mask construction and KV gathers across games; reuse tensor
   buffers and fixed-capacity suffix workspaces. Reduce per-leaf tensor allocation
   and per-layer padding rather than only changing the attention kernel.
2. Pack CPU-to-GPU inputs and GPU-to-CPU results into fewer transfers per wave.
   Preserve required sequential-search dependencies and legal-policy projection.
3. Measure steady-state occupancy over a normal collection cycle; test more live
   games with real long-context memory checks. Consider stable cache slots to
   reduce prefix repacking when a cohort changes.
4. Profile and move coarse Python Q completion/action selection/backup operations
   into a native batch routine if still material after cache/transfer improvements.
   Optimizing existing Rust move generation has much less measured upside.
5. Once shapes and buffer addresses are stable, test CUDA Graph replay for launch
   overhead. This is a candidate, not an already measured improvement.

For every candidate, retain actor weights, seeds, Gumbel noise, search budget and
targets; check legal actions, moves, policies, values and outcomes against the
reference, then measure unprofiled interleaved throughput over production-sized
cycles. Faster generation must preserve replay reuse/exposure rules. Root-forward
compilation and faster weight updates have smaller direct shares in these data.

Illustrative Amdahl calculation, not a forecast: halving the measured 32.5% prep
scope alone would yield about 1.19x collector throughput. Halving both prep and
result scopes would yield about 1.35x. At this initial profiling stage, no speedup had been implemented.

## Reproduction and artifacts

Script: `scripts/profile_gumbel_pipeline.py`.
Artifact directory: `artifacts/self_play_validation/gumbel128_profile_2026-09-15/`.

Files include baseline/cProfile metrics, complete game signatures, source and
checkpoint hashes, `cprofile/functions.json`, `cprofile/profile.pstats`,
`torch_events.json`, `trace_analysis.json`, and `trace.json` (about 369 MB, viewable
in Perfetto). The profile completed successfully and left no training job running.

## Reusable decode buffers and native selectors — implementation follow-up

The candidate is controlled by `reuse_decode_buffers` in `load_runtime` and
`InferenceRuntime`. It uses an executor-owned pooled node arena, weak evaluator
ownership, persistent history slots, and compact active rows. Ancestor chains
remain Python integer lists until the cross-game batch is assembled. Each layer
gathers all active games directly into its reusable 32-position K/V buffers;
new node K/V are scattered across games in two stacked writes. History rows are
refreshed on owner/content or logical-width changes. Collection cleanup releases
all slots, buffers and ownership information.

Features, ancestor indices, lengths, positions and legal IDs share one pinned
upload. GPU legal logits and the three value logits share one blocking readback;
that readback also completes the upload before host staging is reused. Legal
ordering and policy/value normalization remain on the CPU.

Preserving the compiled decoder required more than equal tensor values. Shared
input storage and different tensor strides selected different cold-compiled
arithmetic, producing policy-target differences beyond the gate. The final
inputs therefore have independent reusable storage, reference-compatible
strides, and matching inference-tensor metadata. In particular, history packing
in the normal collector produces ordinary tensors; constructing these inside
`inference_mode` caused a post-update mismatch. The workspace preserves the
caller's inference context for these history inputs. Tests cover cold compilation
and independent input storage explicitly. Returned node K/V remain independent
of these mutable buffers.

A final standalone native-first run also exposed a single-row cold-compilation
issue hidden by shared warmups: Dynamo guarded the prefix view's underlying base
shape. The reference's one-row `stack` fast path exposes a logical `[H, P, D]`
base, whereas a view into the reserved buffer exposed its capacity dimensions.
The workspace now resizes tensor metadata within the reserved storage and matches
that base shape/stride exactly. A fresh 32-game native-first run then produced
bit-for-bit identical replay targets. Tests check base metadata and cold
compilation when entering the single-row tail. All nine promotion measurements were repeated after this correction in a fresh
process; the earlier run is retained as diagnostic evidence.

An initial corrected 32-game cache run completed in 89.60 warmed seconds, with
no new graphs after warming the two decoder graphs. Its complete replay targets
were bit-for-bit equal to the compiled reference. This is a component-validation
result; promotion uses the separate alternating 128-game runs below.

The separate 127.95-second function profile measured 23.33 seconds (18.23%) in
exclusive Python `gumbel_search.py` work, exceeding the 10% native-work trigger.
It recorded 30,856 `torch.tensor` calls, zero padding calls, and 23,755 leaf
readbacks, compared with 785,230, 2,014,988 and 47,510 in the original profile.
These are instrumented call counts, not unprofiled timings or GPU kernel counts.
The host preparation scope now includes legal-move projection before upload, so
its boundary differs from the original preparation/projection split.

Two native routines fuse completed-Q calculation with root or interior
selection. Python retains tree management, RNG, final target calculation and
reference selectors. The Rust path preserves double arithmetic, expansion
summation with half-even correction, cached clamped priors, visit penalties,
root eligibility, score clamping and first-index ties. Direct numerical and
selection tests cover 1,200 cases plus deterministic complete searches.

A real scratch Actor111 collect/update/collect check passed after one optimizer
step, including matching post-update reference/candidate replay targets. The
production checkpoint checksum was unchanged. Evidence is in
`artifacts/self_play_validation/reuse_decode_update_cycle_v2/summary.json`.

Promotion measurements are recorded separately in
`artifacts/self_play_validation/reuse_decode_promotion_128_v2/`. The three variants
are `reference` (existing compiled preparation and Python search), `candidate`
(reusable preparation and Python search), and `native` (reusable preparation and
native selectors). Each uses Actor111, FP32, four CPU threads, 24 games in flight,
128 simulations, `top_m=16`, depth 32, identical seeds/game IDs, and unchanged
replay settings. The order reverses between rounds. Warmup uses 32 games; each
measured pass uses 128 complete games. Compiler counters are checked for every
measured pass, so an insufficient warmup fails the gate rather than being counted
as warmed performance.

## Promotion results — validated September 16

All nine measured passes completed 128 games and 10,358 searched positions.
Moves, outcomes, visit counts, legal IDs/order and search counters matched exactly;
stored policy/value/Q targets were bit-for-bit identical across all nine passes
(and passed the existing `1e-6` gate). Both changes independently passed every gate
in `promotion.json`.

| Round | Compiled reference (s) | Cache only (s) | Cache + native (s) |
|---|---:|---:|---:|
| 1 | 581.91 | 410.28 | 369.37 |
| 2 (reverse order) | 591.36 | 410.72 | 371.68 |
| 3 | 587.19 | 408.57 | 370.96 |

| Median measure | Compiled reference | Cache only | Cache + native |
|---|---:|---:|---:|
| Usable positions/hour | 59,972 | 85,831 | 94,930 |
| Games/hour | 785 | 1,123 | 1,242 |
| Move latency p50 (s) | 1.300 | 0.907 | 0.810 |
| Move latency p95 (s) | 1.636 | 1.195 | 1.088 |
| Peak allocated CUDA memory (GiB) | 1.49 | 3.37 | 3.37 |

Cache-only paired throughput gains were 41.8%, 44.0%, and 43.7% (median 43.7%).
Native selection added 11.1%, 10.5%, and 10.1% (median 10.5%) over the cache path.
Each pair improved. Median paired p95 latency changes were -27.2% and -8.5%,
respectively. Combined median paired throughput gain was 58.3%.

Every measured pass started and ended with two compiled graphs, with no new
graphs during measurement. Every workspace was empty after collection; allocated
memory returned to exactly 201,884,160 bytes in all nine passes. The larger arena
raises peak memory relative to the reference while remaining below the 6 GiB gate.

Each 128-game candidate pass used 70,915 packed leaf uploads and 70,915 leaf
readbacks: 2,277.18 MiB uploaded and 226.81 MiB read back. These counters exclude
unchanged root inference. Device-copy counters (including history/compact packing,
decoder input copies, branch gathers and node scatters) are in each pass's
`metrics.json`; they count logical bytes, not physical DRAM traffic.

In the earlier diagnostic run, the harness compared shared game IDs across the
32-game warmup and 128-game measurement and stopped after the first reference
measurement. Collection
size changes tail batches, so this is not a valid numerical comparison even for
the unchanged reference. The harness now groups target comparisons by workload
size, with a regression test. It resumed after verifying model/config/source
hashes and repeating all warmups, retaining the completed first reference result.
No tolerance was relaxed. That run and its original metadata remain archived in
`artifacts/self_play_validation/reuse_decode_promotion_128/`. The final results
above come from the complete fresh repeat, with all nine passes rerun after the
cold-tail metadata correction.

`load_runtime` enables both improvements for the compatible optimized CUDA path.
`reuse_decode_buffers=False` selects the previous compiled preparation and Python
search; `native_gumbel=False` with default buffers isolates the cache-only path.
Explicit legacy decoder/packing ablations and CPU defaults retain their previous
behavior. The low-level `InferenceRuntime` constructor keeps opt-in defaults.

Reproduce the promotion measurements (use a fresh output directory):

```bash
.venv/bin/python scripts/profile_gumbel_pipeline.py \
  --config config/self_play_laptop_pilot.toml \
  --checkpoint artifacts/eval/self_play_actor111_sf2400_2026-09-14/checkpoints/actor111.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play_validation/reuse_decode_promotion_new \
  --games 128 --warmup-games 32 --concurrency 24 --pairs 3 \
  --include-native --skip-profile
```

The release native extension must be rebuilt after pulling these source changes
(`uv pip install --python .venv/bin/python --no-deps --reinstall ./native/imba_chess_native`).

## Final profile and regression evidence

The separate final profile is in
`artifacts/self_play_validation/reuse_decode_native_final_profile_v2/`.
Cold and warmed 32-game native runs both matched the compiled reference's complete
replay targets. The warmed run took 82.76 seconds. Its host scopes were 18.17 seconds
in leaf preparation, 30.25 in decoder execution/launching, 15.53 in result processing
and readback, and 4.89 in root inference. These host scopes include waits and are
not GPU kernel durations.

The 109.03-second function profile attributed 7.45 seconds (6.83%) to exclusive
Python Gumbel work, down from the cache-only profile's 18.23%. Native interior
selection used 2.74 seconds across 952,919 calls. The profile recorded 30,856
`torch.tensor` calls, zero padding calls, and 23,755 leaf readbacks. This profile
is separate from the promotion timings.

The short trace covers 128 leaf batches at 24 concurrent searches. It recorded
30,476 kernels and 0.322 seconds of unioned GPU kernel/copy/set activity across a
0.897-second device span (35.9% active in this instrumented trace). Decoder kernels,
history/input preparation, and launch/readback gaps remain the main costs. The
trace is not a utilization claim for an unprofiled full collector run. No attention
backend, precision, CUDA Graph, or concurrency changes were introduced.

Final regression checks passed: 827 selected Python tests, plus the CUDA decoder
and workspace suite (16 passed, three intentional skips). The promoted-default
scratch collect/update/collect cycle also passed, with both optimizations enabled,
matching updated-weight policy/value/Q targets, and an unchanged production
checkpoint checksum. Its reproducible script and summary are in
`artifacts/self_play_validation/reuse_decode_update_cycle_promoted/`.

## History-cache follow-up (2026-09-16)

Revision validation and direct decoder-history updates passed the subsequent CUDA
promotion gates. The compatible CUDA default now uses direct history caching;
explicit `history_cache_mode="current"` retains this report's optimized reference.
Three alternating 128-game pairs improved throughput by a median 7.52%, with
bitwise-equal targets and lower p95 latency. Preparation fusion missed its trigger.
See [the history-cache report](HISTORY_CACHE_PROFILE_2026-09-16.md) for the full
measurements, thermal and compiler-cache controls, and bounded CUDA Graph rationale.
