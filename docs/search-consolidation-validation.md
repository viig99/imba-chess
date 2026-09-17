# Search inference consolidation

Original revision: `67aee959c729d8bca738928a84a163033138cefc`.
Validation date: 2026-09-17.

**Stage 1 is implemented and adopted. Stage 2 was prototyped and is not adopted.**
The user clarified two acceptance criteria during implementation:

- Halving uses its existing Stockfish concurrency: 4 in recent runs, 6 in the
  supported config. Gumbel's 24/48-game settings are not halving requirements.
- Prefer simpler code over approximately 5% performance differences. Added
  complexity needs a substantial benefit. This supersedes the original strict
  3% no-regression rule; correctness tolerances were not changed.

## Production changes

Collection, checkpoint matches, Stockfish evaluation, calibration and search
benchmarks now share `eval/inference_runtime.py`. Production inference requires
CUDA FP32 with TF32 disabled. The algorithm determines execution:

- Gumbel: compiled tensor decoder, reusable workspace and unconditional native
  selection/backup. Collection and screening retain their exploration protocol;
  standalone evaluation explicitly passes zero noise.
- Value-search halving: grouped cached decoder with batched input, suffix and
  projection preparation. Multi-leaf requests, budgets, tactical coverage,
  quiescence and tie-breaking remain intact. Stockfish engine calls remain concurrent.

The separate model-search actor backend, greedy/rerank/depth-two policies,
experimental SDPA decoders, production execution switches and duplicate Gumbel
entrypoints are removed. Ordinary model-component SDPA is preserved. Common
commands expose exactly `gumbel` and `value_search_halving`; halving remains the
default. `search_lambda` replaces `value_rerank_lambda` with its value preserved.
Gumbel simulations and halving neural-evaluation budgets remain distinct.

Evaluation metadata records algorithm, active budget/unit, exploration, FP32,
TF32 policy and runtime revision. Incompatible screening resumes are rejected.
Exact-byte aliases for the four shipped evaluation-only config migrations retain
existing self-play training identities. Checkpoint, replay and optimizer formats
are unchanged; unrelated config edits still invalidate resume.

## Correctness checks

- Production default suite: **2,254 passed**, 19 deselected.
- Compiled CUDA decoder/workspace checks: **3 passed** for Stage 1.
- Gumbel original/candidate comparisons: short and long histories; budgets 128
  and 512; concurrency 1, 24 and 48. Moves, legal IDs, visits and counters match
  exactly; values and training targets meet the existing `1e-6` result tolerance.
- Halving original actor/candidate comparisons: concurrency 1, 4 and 6, budget
  2048, five warmed comparisons plus initial passes. Selected moves, counters
  and candidate rows match; numerical rows meet `atol=rtol=1e-5`.
- Real-checkpoint CUDA collect/train/collect: four completed games, an actual
  training step, then four more completed games. Original/candidate trajectories,
  outcomes, identities and targets match. Updated model tensors and checkpoint
  schema match. These are short mate continuations for correctness, not a
  representative collection-throughput benchmark.
- A further four-game collection comparison from real training prefixes completed
  all games (265 searched positions). Moves, outcomes, counters and training
  targets matched exactly, including every recorded floating-point target.
- Both algorithms completed common Stockfish CLI games at concurrency 1 and 4.
  Legal coverage was 100%. Checkpoint-match drivers passed paired-color smoke
  games at concurrency 1 and 4 with separate original and trained model instances.
- Protocol tests cover stale/foreign owners, refresh, cancellation, result
  mapping and engine cleanup after partial startup failure.

## Stage 1 performance evidence

Checkpoint SHA256:
`578ceb9e6da103ad6e40d14f69799d4230f1398043b2de158ed21ffba6074577`.
Long-position manifest SHA256:
`1fa5d57c0a3597f74a26bfd7d499a38d5b053d36c5d557019a56af7c87228c89`.
Hardware: RTX 3070 Ti Laptop, 8,220,180,480 bytes VRAM; PyTorch 2.14.0+cu130.
Benchmark settings: FP32, TF32 off, four intra-op threads and one inter-op thread,
CUDA synchronization at measurement boundaries, profiling disabled.

Ten paired fixed-position halving measurements after the desktop was left idle:

| Concurrent searches | Median grouped/actor wall-time ratio |
| --- | ---: |
| 1 | 0.554 |
| 4 | 0.846 |
| 6 | 0.824 |

These compare the execution composition, exclude actor history prefill, and share
allocator state. They are not independent cold-start or memory comparisons.

Five complete Stockfish pairs at concurrency 4 used the recent production
protocol: halving budget 2048, depth 8, lambda 0.05, SF2400, 40,000 nodes,
five-second ceiling, one engine thread and 64 MiB hash. Median grouped/actor
**model-turn throughput was 0.9655**, about 3.5% lower. Individual paired ratios
were 1.0089, 0.9991, 0.9366, 0.9655 and 0.9325. Limited-strength Stockfish produced
different game lengths, so games/hour alone does not compare equal search work.
This modest observed tradeoff is accepted for the simpler implementation under
the user's revised preference; no universal performance equivalence is claimed.

Earlier runs were affected by GNOME GPU activity and are excluded as performance
adoption evidence. An additional Stockfish repeat was stopped when that contention
returned. Earlier halving 24-game OOM probes are historical explorations, not
adoption requirements. Neither path was required to support that concurrency.

Five paired warmed Gumbel measurements also completed for every combination of
short/long histories, budgets 128/512 and concurrency 1/24/48. Median Stage 1 /
original throughput ratios ranged from **0.999 to 1.014** across the 12 workloads.
No material warmed Gumbel regression was observed. All 245 sampled GNOME SM
readings were idle. Raw timings, memory peaks, compiler counters and search outputs
are under `validation/gumbel-performance/`.

The four-game collection comparison took 30.15 seconds originally and 29.68
seconds with Stage 1 after warmup. Peak allocated/reserved VRAM was identical
(394,758,144 / 452,984,832 bytes). This is one complete-game pair, supporting
correctness but not establishing a repeatable collection-throughput improvement.

## Stage 2 trial — not adopted

The isolated prototype extends the existing workspace and tensor decoder. It
separates owner histories from leaf rows, pads queries by owner without copying
history K/V per leaf, preserves result mappings and supports configured suffix
depths beyond 32. The one-query Gumbel path remains a shape specialization.

The new CPU parity check passed. Four compiled CUDA checks passed, including
existing Gumbel checks and multi-leaf cases with mixed depths up to 40, siblings,
changing query counts, game replacement and singleton tails. Real-checkpoint
halving at concurrency 4 matched Stage 1 moves and backed-up rows within the
unchanged tolerances.

Five sequential warmed pairs on the fixed long-history workload, with normal
workspace-buffer reuse and no observed GNOME SM activity:

| Metric | Stage 1 grouped | Stage 2 compiled multi-leaf |
| --- | ---: | ---: |
| Median time, four searches | 0.4905 s | 0.4595 s |
| Median paired throughput gain | baseline | **6.0%** |
| Median peak allocated VRAM | 2,682,851,328 B | 2,144,845,824 B |
| Median peak reserved VRAM | 4,081,057,792 B | 2,413,821,952 B |
| Peak host RSS | 1,610,661,888 B | 1,704,013,824 B |
| Initial search trial | 0.817 s | 7.857 s |

Compiler disk caches were already populated for the paired initial trials; these
are not pristine cold-start measurements. The earlier first compiled probe took
18.54 seconds. The repeated workload stayed at one compiled graph after warmup.
A separate five-pair run clearing workspace buffers between searches showed only
2.7% median throughput improvement.

**The gain is too small to justify the additional complexity under the user's
preference.** Production retains the validated Stage 1 grouped halving decoder.
The prototype is archived only; there is no experimental runtime flag or silent
fallback. Stage 2 complete-game and broader adoption gates were not pursued once
its search-throughput benefit proved insufficient. No Stage 2 end-to-end game
speedup is claimed.

## Artifacts and limits

All raw data are under `artifacts/search_consolidation/`:

- `original/source.tar`: original tracked source; original CUDA/CPU logs and
  Gumbel/halving reference outputs are alongside it.
- `stage1-baseline/`: validated Stage 1 source/patch and test log captured before
  the Stage 2 trial. `stage1-candidate/` holds integration outputs and final tests.
- `stage1-final/`: final production source, patch and manifest after formatting
  and cleanup; the preceding baseline archive remains unchanged.
- `preflight/halving-idle/`, `preflight/stockfish-idle-c4/`: fixed-position and
  complete-game halving measurements. The additional cancelled run has an explicit
  exclusion record.
- `validation/`: correctness summaries and validation-only harnesses. Actor
  harnesses require the archived original source and its original config.
- `stage2-candidate/`: unadopted source/patch, manifest, CUDA log and paired raw
  measurements. `paired-c4-reuse/` is the production-style reuse comparison;
  per-process GPU-monitor logs accompany both paired runs.

Historical source and result snapshots were preserved. The checks above do not
establish universal latency equivalence, a complete cold-start matrix, or long-run
Gumbel collection throughput. Stage 2 memory improvements were measured but were
not substituted for the requested substantial search-throughput benefit.
