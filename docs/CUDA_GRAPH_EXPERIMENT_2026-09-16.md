# Bounded CUDA Graph experiment and follow-on prototypes

Status: **bounded exact-shape CUDA Graphs rejected**. The subsequent combined
stable-placement and selective-gather candidate passed all three paired throughput
and correctness gates. See [the combined report](SELF_PLAY_COMBINED_CACHE_2026-09-16.md)
for final selection and integration verification. The following sections preserve
the graph experiment and intermediate evidence; intermediate conclusions below are superseded
by that report. GNOME remained running throughout.

## Graph decision

The valid quieter 128-game pair used candidate → reference order, 48-game warmups,
48 concurrent games, and matching configuration/checkpoint/seeds/tuning. Both
variants completed the same 10,386 played search moves with bitwise-identical
targets and exact games/search counters.

| Metric | Direct | Bounded graphs |
|---|---:|---:|
| Collector seconds | 422.102 | 424.788 |
| Played search moves/hour | 88,580 | 88,019 |
| Complete games/hour | 1,092 | 1,085 |
| p95 latency | 2.200 s | 2.245 s |
| Peak allocated VRAM | 3.940 GiB | 4.165 GiB |
| Peak reserved VRAM | 4.977 GiB | 4.260 GiB |
| Allocated after collection | 201,884,160 bytes | 201,884,160 bytes |

Graph throughput was **0.63% lower**, and p95 latency was **2.03% higher**. It fails
the requirement that every pair improve, so further promotion repetitions were
stopped. Keep direct; do not retain graph code or flags in the maintained runtime.
The experimental script and test were removed from `scripts/` and `tests/` and
preserved only in the artifact archive.

An earlier pair showed 48.7% higher graph throughput but crossed a large desktop
activity change: a reference warmup sample reported GNOME at 77% SM, while later
candidate/reference warmup samples reported no GNOME SM value. Candidate pace
changed markedly mid-pass. That pair is excluded from promotion timing and must
not be credited as a code improvement. GNOME was left running throughout.

The revised cache achieved about 85.2% replay coverage in the first 128-game pass
(42,524 replays / 49,910 calls), with 868 captures and at most eight retained
graphs. Correctness, cold tails, updated weights, cleanup, memory, and compilation
passed. High replay coverage and a faster fixed-input microbenchmark did **not**
translate into a passing whole-collector result under the matched conditions.

Ordinary, unsynchronized host scopes in the valid pair provide supporting
attribution: decode submission/capture fell from 76.85 to 54.44 seconds, while
result processing/readback rose from 111.90 to 147.17 seconds. These scopes include
asynchronous wait placement and are not exclusive GPU timings; their movement
must not be described as faster decoder execution. Direct preparation remained
121.61 seconds of the 422.10-second collector pass, motivating the next placement
experiment.

## Bounded graph candidate

The archived graph harness wraps the reusable workspace's compiled decoder.
The first measured version retained four exact-shape graphs and allowed 128
captures per collection. The revised candidate retains eight graphs and allows
1024 captures, admitting shapes after eight calls. It retains views of existing
feature/history/branch storage instead of copying those large inputs for replay.
Only relative-position and mask tensors are copied into static graph inputs.
The implementation follows PyTorch's side-stream warmup and stable-address
[CUDA Graph requirements](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

The experiment is restricted to immediate workspace consumption: graph outputs
are borrowed until the next replay. The compatibility decoder, whose returned K/V
may be retained by callers, is not wrapped. Cache clearing synchronizes and releases
graphs before workspace storage is released, including cancellation and errors.

Admission reserves 768 MiB of headroom below 6 GiB allocated, rejects captures with
excessive allocation growth, and disables further admission on a memory rejection.
These are admission safeguards; measured peak allocation is still a promotion
gate. The revised candidate identifies buffer allocation cohorts through the workspace
prefix tensor and keys on logical shapes, replacing the original per-tensor
inspection. Offline simulation of the full 48-game shape sequence predicts
87.32% replay coverage with eight entries, versus 17.70% with the original budget;
this is a coverage estimate, not a speedup measurement.

## Completed evidence

| Check | Result |
|---|---|
| Two complete cold games and two warmed games | Games and targets matched between passes; about 92% replay coverage |
| Focused CUDA test | Passed changed borrowed/copied inputs, eviction, capture budget, clearing, and changed weights |
| 48 complete cold games against archived direct reference | Exact game records, bitwise targets, and identical search/transfer counters |
| Cold 48 peak allocated memory | 4,349,792,256 bytes, approximately 4.05 GiB |
| Allocation after cold collection | 201,884,160 bytes, matching direct |
| Compilation | Two compiled graphs; no additional specialization in the cold 48 pass versus the existing direct decoder's two shapes |
| Cancellation during subsequent warmed repetition | All retained CUDA graphs released; incomplete pass excluded |

The 48-game cold pass took 758.06 seconds, including cold startup. Its result scope
accumulated 488.65 seconds, versus 47.18 seconds in the earlier direct diagnostic.
This is **not a valid measured regression**: subsequent checks established severe
competing graphics activity. The warmed repetition was stopped to investigate,
and the fresh reference timing run was also stopped once contention was confirmed.

Across the complete cold pass and partial warmed repetition together, the wrapper
made 36,772 calls, captured 159 graphs, replayed 7,851 calls, and retained at most
four graphs. Thus aggregate replay coverage was only 21.35%; these counters must
not be presented as the complete cold pass's standalone coverage. The largest
individual capture allocation increase was 12,103,168 bytes. No graphs remained
after cleanup. Exact-shape variability is a limitation of this policy; this is not
evidence that every bounded CUDA Graph design would fail.

An alternating direct → graph → graph → direct diagnostic used identical real
inputs with prefix shape `[48, 16, 102, 64]`. Each block warmed for ten calls and
measured 100 calls with CUDA events and a synchronized wall clock. Outputs matched
bitwise. Wall milliseconds per decode were:

| Mode | First block | Second block |
|---|---:|---:|
| Direct | 14.34 | 15.81 |
| Graph replay | 13.17 | 12.90 |

This supports continuing to investigate replay, but is not a collector speedup or
a hardware-isolated benchmark. It uses repeated identical inputs and had
the same unresolved graphics contention. Low collection replay coverage, capture
costs, and host-side key checks still need resolution.

## Timing contamination

Read-only `nvidia-smi pmon` samples showed `gnome-shell` (PID 3781) using 62%, then
72%, 82%, and 92% of GPU SM capacity while the reference ran. After all our CUDA
checks finished, it still used 82%, 82%, and 83% in successive samples. The desktop
process was left untouched. Earlier host samples also showed 19–21% I/O wait.
These observations establish contention at sampling time; they do not establish
its start time or explain every earlier benchmark difference.

The fresh direct two-game warmup took 131.78 seconds, reinforcing that slowdown
was not confined to graph replay. Do not compare this stage's elapsed times against
the earlier 100-game 88,608-moves/hour sample as if conditions were matched. No new
hourly throughput claim or default promotion is justified.

## Follow-on correctness prototypes

Two isolated scratch prototypes were prepared while graph tests ran. They are
archived under the artifact directory below, not installed in production:

- `bench_stable_rows.py` keeps existing owners in their previous active batch rows
  where possible, fills remaining slots, then restores results to caller order.
  It changes neither prefix strides nor decoder shapes. Real CPU and compiled CUDA
  decoder comparisons passed depth 32, zero history, repeated reordering, owner
  replacement, batch shrink/growth, single-row execution, and cleanup. In the
  deliberately reordered test, dirty rows fell from 112 to 9, copied bytes from
  242,432 to 18,688, and zeroed bytes from 297,472 to 20,224. These synthetic savings
  are not full-collection savings or a throughput prediction.
- `bench_branch_width.py` selects logical branch widths from 1/2/4/8/16/32 based
  on active depths, retaining search depth 32 and independent contiguous layer
  buffers. Existing mixed-depth workspace checks passed on CPU and compiled CUDA,
  including decoder tolerance `1e-5`, result tolerance `1e-6`, cold single-row
  compilation, K/V lifetime, cleanup, and changed weights. Its subsequent complete-game target check failed, as described below.

The initial 48-game comparison against the archive was confounded by different
compilation warmups: the new experiments warmed on two games, while the archive
warmed on 48. Stable-row and layout-preserving gather produced bitwise-identical
48-game targets to each other. Their differences from the archive therefore do
not establish a stable-row correctness failure. The fresh direct control with the same two-game warmup confirmed bitwise equality
of all 48-game targets, exact game records, and identical search counters for both
prototypes. All three target hashes are
`360495265c3dc8fe4e2e7d4ed8de466c7020f7bace3796325a5bc6ced80493be`.
No promotion decision uses the mismatched comparison.

On this matched 48-game workload, history copies were 368,352,100,352 bytes for
direct and 79,983,017,984 for stable placement; history zeroing was
217,455,853,568 and 36,952,276,992 bytes respectively. Selective gather reduced
ancestor writes from 957,867,884,544 to 306,657,755,136 bytes, adding
120,959,533,056 bytes of explicit zeroing. All three peaked at 4,118,503,424
allocated bytes (3.84 GiB), returned to 201,884,160 bytes after collection,
and retained the same two compiled graphs. These are traffic/correctness
comparisons, not paired throughput evidence.

Shortening the branch width did fail the target tolerance against a matching
two-game direct warmup (a policy difference of about 2.2e-6). A follow-up keeps
the full 32-column decoder layout and gathers only active ancestor columns,
zeroing newly exposed padding. Full-buffer equality checks passed on CPU and
CUDA; the matched complete-game control also passed bitwise.

Fixed-stride histories
and jagged attention were not implemented. The branch prototype tests width
buckets independently of stable placement and graphs.

## Stable-placement 128-game attribution (first pair; provisional timing)

Both variants completed the same 128 games and 10,386 played search moves, with
bitwise-identical targets and exact search counters. Peak allocated memory was
4,230,578,688 bytes and peak reserved memory 5,343,543,296 bytes for both.

| Counter | Direct | Stable placement |
|---|---:|---:|
| History copied bytes | 1,604,099,899,392 | 252,677,062,656 |
| History padding zeroed bytes | 1,848,691,785,728 | 205,411,516,416 |
| Dirty history rows | 284,680 | 37,600 |
| History copy API submissions | 13,825 | 13,998 |
| Full prefix refreshes | 832 | 832 |
| Ancestor gather bytes | 2,597,201,117,184 | 2,597,201,117,184 |
| Decoder batches | 49,910 | 49,910 |

History copying fell 6.35x; copying plus zeroing fell 7.54x. These are logical
transfer-volume counters, not resident VRAM savings or measured physical DRAM
bandwidth. Other preparation operations, decoder arithmetic, and search/readback
remain. There were 5,099 single-row decoder batches, where cross-game row
stabilization has little to save.

Observed preparation host time fell from 148.26 to 65.14 seconds, total collector
time from 945.73 to 633.28 seconds, and p95 from 5.290 to 3.805 seconds. The user
reported Firefox activity, and substantial timing variation was observed; these
first-pair wall-time differences must not be presented as an established code gain.
Repeats remain pending. Eight-game completion intervals are archived in
`pairs/stable/pair_0_completion_intervals.json`; completion order matched exactly.

## Selection policy

Keep one maintained runtime solution. These prototypes are experiments, not new
production modes. Select using exact correctness, complete-game throughput,
p95 latency, peak allocated/reserved memory, retained allocation, compilation,
and implementation complexity. Reject failing candidates and archive their
measurements; do not keep redundant implementations behind flags. Compatible
optimizations may form the single selected pipeline only after that exact
combination passes its gates.

## Final selection

The combined stable-placement and selective-gather candidate passed three
alternating 128-game pairs against direct: +9.15%, +19.61%, and +20.53%
throughput, with bitwise targets and identical search counters. Median paired p95
latency improved 20.63%; allocated/reserved peaks were unchanged at 3.940/4.977
GiB. It replaces the old workspace behavior without a runtime selection flag.

Stable-only failed its second pair (+49.34%, then −46.28%) amid pronounced desktop
variation. Gather-only failed its first pair (−1.32%). Their individual correctness
checks passed. These observations do not quantify a causal interaction effect;
the combined matched comparisons are the promotion evidence. No further
standalone or graph variants are being tested or maintained.

See [the combined report](SELF_PLAY_COMBINED_CACHE_2026-09-16.md) for the
implementation, final verification, caveats, and artifact paths.

## Artifacts and reproduction

Root: `artifacts/self_play_validation/cuda_graph_stage_2026-09-16/`.
It contains `smoke/`, `cold48/`, `micro/`, the interrupted `fresh_reference/`,
`contention.json`, logs, scratch prototype/check scripts, `matched_correctness.json`, and `unit_v2.log`.
Each collection archives model/config/seed/source hashes and targets. The exact
measured graph script is `measured_graph_harness.py`, SHA-256
`830e8dbb2ce0c9b2d869e02fa1a098d06c271a58d6ce4d238f7d296cf083296f`.

The graph benchmark accepts the existing benchmark arguments, for example:

```bash
PYTHONPATH=scripts .venv/bin/python artifacts/self_play_validation/cuda_graph_stage_2026-09-16/graph_harness_v2.py \
  --config config/self_play_laptop_pilot.toml \
  --checkpoint artifacts/eval/self_play_actor111_sf2400_2026-09-14/checkpoints/actor111.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play_validation/graph-example \
  --concurrency 48 --games 48 --warmup-games 48
```

Use the shared persisted kernel-tuning cache from the history-cache study for
cross-process numerical comparisons. Production code never imports this harness.
