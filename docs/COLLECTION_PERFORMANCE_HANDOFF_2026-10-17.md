# Handoff: improve self-play collection throughput

## Objective and working preference

Improve completed self-play collection throughput while keeping search settings,
model behavior, inference precision, and training-target semantics unchanged.
Measure useful completed positions/hour and neural evaluations/second, including
phase drain, rather than optimizing an isolated attention kernel alone.

The user's preferred direction is **reduce Python work and move ownership into
Rust wherever practical**. Keep trees, boards, histories, legal projections,
paths, and search statistics native so they do not repeatedly become Python
objects or cross the Python/Rust boundary. Improve CPU time and allocation cost
first, benchmark the resulting collection pipeline, and only then investigate
GPU bottlenecks in depth if the remaining evidence warrants it. Do not turn this
into another GPU-first attention investigation by default.

Think independently. The opportunities below are starting hypotheses from a
source review, not an exhaustive list or a measured ranking. Read the current
implementation, challenge these assumptions, find additional opportunities, and
choose changes with a clear mechanism and reasonable engineering cost. Avoid a
wholesale rewrite justified only by the language being Rust.

## Context and constraints

- The user reports GPU utilization around 90%. Their working hypothesis is that
  GPU utilization is already healthy and CPU work deserves priority. That is a
  reasonable direction to test, but utilization alone does not establish whether
  GPU execution limits throughput: it records activity, not useful work or the
  contribution of launch gaps, copies, and kernels to the critical path. Do not
  claim either a CPU or GPU bottleneck has been proven.
- The reviewed run used five collector processes and 160 concurrent games,
  approximately 32 games per worker. Recent retained collection reports were
  around 22k neural evaluations/second and 378k usable positions/hour. These are
  historical reference values, not a controlled baseline for future changes.
  Inspect current run arguments and frozen source; TOML defaults can differ from
  CLI overrides, including collection dtype and concurrency.
- Playout-cap support is already enabled in the reviewed configuration. With
  `full_search_probability = 1.0`, undecided positions retain full search; decided
  positions use the cheaper search. Changing that probability, simulations,
  top-m, depth, resignation, or target eligibility is outside this objective.
- Gumbel CUDA training compilation and CUDA FP32 root-forward compilation were
  enabled by default in the preceding work. The batched Gumbel custom decoder
  remains the relevant leaf path. The attention-only benchmark favored it over
  the alternatives at batch 16; this does not rank end-to-end collection costs.
- Preserve numeric behavior, move ordering, tie handling, seeded noise,
  terminal/repetition rules, ownership validation, cancellation, and cache
  invalidation across game completion and weight updates. Storage/layout changes
  have previously affected compiled numerical parity.
- Inspect current processes and repository state. This handoff does not request
  stopping, restarting, or altering a live training run.

## What is already native, and what remains in Python

Rust already implements chess operations, terminal classification, board
encoding, legal-move projection, selection arithmetic, and backup. Python still
owns the tree traversal and connects these primitives for each simulation.

The current path is roughly:

```text
Python game/search advancement
  -> native selection and child creation/classification
  -> Python leaf and cache-handle objects
  -> native encoding/legal projection returning Python objects
  -> Python fills batched staging buffers
  -> GPU ancestor gather, compiled decoder, KV scatter
  -> one blocking result readback per wave
  -> CPU normalization and Python PositionEval objects
  -> native NodeStats initialization and backup
```

Relevant entry points:

- `src/imba_chess/eval/gumbel_search.py`: `_Node`, `gumbel_stepwise`.
- `native/imba_chess_native/src/gumbel.rs`: `NodeStats`, selectors, backup.
- `native/imba_chess_native/src/terminal.rs`: `push_and_classify`.
- `native/imba_chess_native/src/move_projector.rs`: native legal projection.
- `src/imba_chess/data/board_state.py`: `encode_cozy`.
- `src/imba_chess/eval/position_evaluator.py`: cached-node handles and results.
- `src/imba_chess/eval/decode_workspace.py`: `prepare`, `consume`, GPU buffers.
- `src/imba_chess/eval/merged_executors.py`: decode-wave execution.
- `src/imba_chess/eval/batch_scheduler.py`: game advancement and batching.
- `src/imba_chess/self_play/workers.py`: collector process lifecycle and IPC.

## CPU-first opportunities to assess

1. **Native tree ownership and coarser advancement calls.** Python currently
   walks child dictionaries, builds paths and nodes, calls native selection at
   each depth, and constructs a list of native-stat objects for backup. Consider
   a Rust tree arena with integer node IDs, native boards/histories/paths, and an
   `advance_until_evaluation` / `apply_evaluation` interface. A later batched API
   could advance all games in a worker. This removes repeated boundary crossings
   and object conversion while retaining the same search. Assess incremental
   steps rather than assuming the entire migration must land at once.

2. **Native ownership of evaluation inputs and legal projections.** Native
   encoding currently returns piece IDs through Python lists and a BoardState
   wrapper before staging. Legal projection returns Python lists/move wrappers.
   Consider batch encoding/projection directly into preallocated staging arrays,
   retaining legal moves and IDs natively. Repetition history also currently
   travels through Python lists and native vectors when creating children.

3. **Bulk result application without float-list round trips.** After readback,
   CPU tensors become Python lists, PositionEval objects, and another copy of
   priors before native NodeStats initialization. Consider applying flat result
   arrays directly to native nodes in a batch. Preserve the existing numerical
   normalization contract; moving normalization between CPU and GPU is a
   separate change. Keep Python conversion at reporting/training-data boundaries
   where practical.

4. **Native allocation and repeated work.** Selection allocates temporary Q,
   completed-value, logits, and weight vectors. Backup marshals a Python-object
   path and allocates a HashSet to validate it. Native ownership may eliminate
   much of this overhead; scratch-buffer reuse and validation at stable ownership
   boundaries may help too. Preserve validation guarantees and arithmetic order.
   Check whether repeated Python scheduler/handle bookkeeping can also disappear.

## Secondary opportunities, conditional on the remaining bottleneck

- **Worker/batch balance:** compare worker counts at fixed total concurrency
  (for example 2x80, 4x40, 5x32, 8x20 for 160 games). More processes provide CPU
  parallelism but split inference batches and duplicate model/cache resources.
  Reconsider this balance after reducing CPU work. The current leaf path is
  local to each worker; there is no per-leaf coordinator IPC to eliminate.
- **GPU staging submissions:** ancestor gathering issues one gather per K/V per
  layer (16 for eight layers). Feature copies, attention-index/mask preparation,
  and output KV stacking/scattering surround the compiled decoder. Fusion or
  broader tensor compilation could reduce submissions if these are material.
- **Prefix refreshes:** changes to the logical maximum prefix length invalidate
  all staged prefix rows. Stable length buckets could trade some padding for
  fewer refreshes and more stable shapes. Measure refresh bytes/frequency first.
- **Overlap and shared inference:** independent game cohorts could overlap CPU
  preparation with GPU execution using separate buffers and completion events.
  A shared inference service could combine requests across workers, but should
  own GPU caches rather than transfer KVs through IPC. These are larger designs;
  existing processes already provide some overlap. Each game still needs its
  leaf result before selecting its next leaf; speculative search changes are
  outside scope.

Do not propose existing optimizations as new work: the reusable workspace already
has one packed metadata upload and one blocking result readback per wave, caches
unchanged prefixes, gathers only the required ancestor width, and selects legal
logits on the GPU before readback.

## Validation and measurement sequence

1. Establish a lightweight reproducible collection baseline from the current
   source, fixed checkpoint, seeds, precision, search settings, concurrency, and
   completed-game workload. Separate warm-up/compilation from steady collection
   and account for drain. This does not require a lengthy profiling campaign.
2. Independently assess the CPU/ownership design, make justified changes in
   reviewable steps, and run relevant parity, lifecycle, and correctness checks.
   Prefer CPU-work reduction over speculative GPU tuning.
3. Benchmark end-to-end collection again against the matched baseline. Report
   throughput, CPU time, memory, batch sizes, startup/drain effects, and any
   numerical or output differences. Do not infer a speedup from less Python code.
4. If the remaining limit is unclear or gains stall, profile collector workers
   and the GPU timeline to decide whether GPU work is now the next lever. Avoid
   unnecessary synchronization in instrumentation. Existing decode timers use
   host wall time: asynchronous GPU waiting can be charged to result consumption
   rather than the decoder call. Use appropriate GPU timing for that distinction.

Useful counters already exist in DecodeWorkspace for uploads/readbacks, prefix
refresh/copy bytes, ancestor width/gather bytes, and scatter bytes. InferenceRuntime
tracks batch-size distributions. Export these through worker reporting if useful;
they are not all exposed in collection metrics today. The existing
`scripts/profile_self_play.py` samples the parent main thread, not collector
processes, and explicitly warns of GIL sampling bias. It cannot by itself identify
the current multi-worker search bottleneck.

The eventual handoff/result should state what changed, why it should help, matched
before/after measurements, correctness evidence, and what remains uncertain.
