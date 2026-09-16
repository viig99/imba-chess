# History-cache preparation: CUDA validation

The combined `direct` mode **passed all promotion gates** and is now the default
for the compatible optimized CUDA runtime. Explicit `history_cache_mode="current"`
retains the prior optimized path; CPU and legacy behavior are unchanged. The reference for this stage is reusable decoder buffers plus native
Gumbel selection, as promoted in [the previous report](SELF_PLAY_PROFILE_2026-09-15.md).

## Implementation

- `CachedPositionEvaluator` has a history revision and an opt-in immutable-prefix
  contract. Self-play opts in because each search receives a newly computed root
  history. Immutable callers must not change tensors through any alias during a
  revision. Container replacement and logical-length assignment route through
  `replace_history`, which invalidates old and in-flight branch handles.
- `revision` uses owner/revision/length validation for immutable histories.
  Mutable callers and `current` retain identity/pointer/version fingerprints,
  including ordinary-tensor in-place edits and external list element replacement.
- `direct` includes revision validation, removes stacked and compact histories,
  and copies original history tensors into the independent decoder layer buffers.
  Dirty rows use one foreach copy submission and a separate foreach zero submission
  for padding. Unchanged rows receive no history copies. Logical-width changes and
  storage replacement refresh all active rows; batch changes within capacity retain
  valid rows, including rows temporarily outside the active batch.
- Independent storage, contiguous layouts, inference metadata, geometric capacity,
  and the single-row logical base shape are preserved. Branch gather/scatter and
  decoder arithmetic remain unchanged. Executor errors clear reusable storage;
  collection cleanup also runs on cancellation and completion.
- Internal options are `current`, `revision`, and `direct`. `fused` is deliberately
  unavailable until the cache promotion and fusion trigger pass. Explicit
  `reuse_decode_buffers=False` continues to select the legacy preparation path.

Counters cover fast/fallback validation, dirty rows, full refreshes, logical copied
and zeroed bytes, and history copy/zero submissions. Copy submissions count Python
copy operations (a foreach call is one submission), not CUDA kernels. Copy bytes
include intermediate staging for the reference. Allocation-time zero initialization
is outside the refresh-byte counters. Counters persist through cleanup for reporting.

## Correctness coverage

The selected regression suite passed 151 tests (15 extended tests deselected).
The final workspace check passed 21 tests. Outside the sandbox, the extended CUDA
workspace/decoder suite passed 10 tests with five intentional CPU/compiler skips.
The final workspace/harness/runtime-option suite passed 47 tests (nine extended
cases deselected), including promoted-default and explicit rollback routing.
Ruff and `git diff --check` passed.

Coverage includes exact metadata/history K/V, decoder tolerance 1e-5,
projected/target tolerance 1e-6, immutable revision checks, mutable fingerprint
fallback, stale/foreign handles, row replacement/reordering, batch and prefix
size changes, zero-length histories, padding exclusion, arena growth/reuse,
cross-game isolation, returned-K/V lifetime, errors, cancellation, and updated
weights. Promotion tests reject incomplete workloads, non-improving pairs,
latency/memory regressions, continuing graph compilation, and retained cache state.

## CUDA evidence and measurement method

CUDA is available on the host (RTX 3070 Ti Laptop GPU, PyTorch 2.14.0+cu130).
The initial failure came from hidden device nodes inside the sandbox. All GPU
validation below ran outside that sandbox. The original sandbox observations remain
archived for provenance; they are not evidence of unavailable host CUDA.

The main evidence directory is
`artifacts/self_play_validation/history_cache_stage_cuda_2026-09-16_v2/`.
Actor111, FP32, 128 simulations, top-m 16, depth 32, 24 concurrent games, four CPU
threads, seeds/game IDs, replay settings, and optimizer settings were held fixed.
Native Gumbel selection and reusable decoder buffers are enabled in every mode.

Cold checks use separate processes with fresh graph compilation and shared
persistent kernel tuning decisions. Fully independent tuning initially caused
current-versus-current targets to differ by up to 3.16e-5; the diagnostic is
preserved. Loading a dynamic graph first compiled by the smaller update workload
also changed current-versus-current results. The study therefore disables FX graph
cache loading in every history-study process while retaining kernel tuning caches;
the update check has its own cache directory in the final runner. These are
benchmark startup controls, not changes to decoder arithmetic or tolerances.

The cold current/revision/direct checks each completed 32 games, exercised a
single-game tail, and cleared their caches. All candidate trajectories, legal
ordering, visits, outcomes, search counters, and serialized targets matched the
reference; targets were bitwise equal. The scratch collect → update → collect check
also matched bitwise both before and after one optimizer step. The maximum checked
head-weight change was 1.00136e-5. The production checkpoint SHA-256 remained
`7bdce092f7b265b3f13edb456dd9bda8e335e63568aac345bf29140593fd3f63`.

## Separate 32-game ablations

These warmed, unprofiled passes are attribution evidence, not the promotion test.
All three matched the cold reference bitwise.

| Mode | Collector seconds | Gain versus current |
| --- | ---: | ---: |
| current | 111.757 | — |
| revision | 107.646 | 3.8% |
| direct | 91.540 | 22.1% |

The corresponding function profiles took 141.794, 135.912, and 117.164 seconds.
History-stamp cumulative time fell from 6.167 seconds to 0.149 seconds. In direct,
`_refresh_direct` took 5.830 seconds cumulative (2.747 seconds exclusive Python).
Host decoder scopes include launching and waits; they are not GPU execution times.

For the 32-game workload, direct reduced logical history-copy traffic from
629.9 GB to 148.1 GB and history copy submissions from 63,013 to 2,949. Peak
allocated memory fell from 2.113 GB to 1.454 GB. Counter bytes describe logical
copies, including reference intermediates; a foreach submission can launch several
kernels and should not be interpreted as one CUDA kernel.

## Fusion decision and next experiment

Fusion was not attempted. In the full direct function profile, mask/relative-position
preparation and ancestor gathering consumed at most 9.413 seconds of 117.164 seconds
(8.03%). This is already below the 10% trigger before removing waits. A short trace
reported 11.57% exclusive preparation CPU/launch work, but that short window does
not override the full-collector upper bound.

The direct short trace contained 31,982 kernels. Unioned device activity occupied
0.318 seconds of a 1.042-second device span (30.5%); the decoder host scope occupied
about 46% of the traced interval. Instrumentation changes timing, so these are not
unprofiled utilization figures. The many small launches and host gaps support
prioritizing a bounded CUDA Graph feasibility experiment next, with the same
correctness, updated-weight, memory, and whole-collector gates. They do not establish
that Graphs will improve throughput. Asynchronous readback and further native
bookkeeping remain later candidates.

## Full 128-game promotion measurements

All three alternating pairs completed 128 games per variant (768 measured games).
The median paired throughput improvement was **7.52%**, every pair improved, and
the median paired p95 change was **−3.48%**. All promotion gates passed.

| Pair / order | Current seconds | Direct seconds | Throughput gain | p95 change |
| --- | ---: | ---: | ---: | ---: |
| 1: current → direct | 481.005 | 396.489 | 21.32% | -16.00% |
| 2: direct → current | 421.647 | 392.162 | 7.52% | -3.48% |
| 3: current → direct | 436.807 | 427.777 | 2.11% | -0.07% |

Every pass matched exact complete-game trajectories and bitwise targets. Each
performed 1,325,824 simulations and 1,246,189 neural evaluations. Direct peaked at
2.19 GiB allocated versus 3.36 GiB for current. All passes retained exactly
201,884,160 allocated bytes after collection, cleared caches, and kept the warmed
compiled-graph count at two. Single-game tails were exercised in every pass.

Across a full 128-game pass, direct reduced logical history copies from 5.633 TB
to 1.005 TB and copy submissions from 403,587 to 16,263. Both modes refreshed
183,309 dirty rows. These reductions do not translate directly into speedup:
per-layer row views and copies still require host work, and the full workload has
more history churn than the shorter ablation. Revision alone was measured for
attribution and is not separately promoted.

An earlier uncontrolled pass is preserved under `promotion/`: reference 366.138 s,
direct 460.196 s (20.4% lower throughput). During the following direct pass,
NVML reported software thermal slowdown and CPU package temperature reached 96°C.
That sequence was stopped and restarted with the same workload and explicit
between-pass cooling. The failed result is diagnostic evidence, not discarded
successful-pair selection. The restarted sequence uses all three pairs regardless
of their outcomes.

The optional `--thermal-cooldown` waits outside timed collection for GPU ≤60°C,
CPU ≤70°C, no thermal-slowdown flags, and at least 60 seconds of rest. Start/end
telemetry and cooldown samples are archived. No clock, power, fan, concurrency,
or CPU-thread setting is changed. This gives comparable starts, not a guarantee
that long passes stay cool. The six passes began at GPU 55–59°C and CPU 56–60°C;
end GPU readings were 81–83°C with no thermal-slowdown flags at those snapshots.
Within-pass thermal behavior was not continuously sampled. Performance varied
substantially, so the 7.52% result applies to this fixed laptop workload and protocol.
Profiling runs remain separate from timed promotion.

## Reproduction and archived evidence

Run outside a sandbox that hides CUDA, using a fresh output directory:

```bash
.venv/bin/python scripts/validate_history_cache.py \
  --config config/self_play_laptop_pilot.toml \
  --checkpoint artifacts/eval/self_play_actor111_sf2400_2026-09-14/checkpoints/actor111.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play_validation/history_cache_stage_cuda
```

The runner performs cold checks, a scratch update cycle, warmed 32-game ablations
with function profiles and short traces, then three alternating 128-game pairs.
It records eligibility without silently changing production defaults. Explicit
`history_cache_mode="current"` is the rollback; `reuse_decode_buffers=False`
retains the legacy path. Low-level runtime/workspace options remain internal.

Targets, game trajectories, full move-latency distributions, throughput, peak and
retained allocation, compilation counters, copy counters, traces, and source/config/
checkpoint hashes are archived under the evidence directory. `cold_*`, `update/`,
and `profile_*_fixed/` contain the passing prerequisites and ablations. Earlier
uncached and graph-cache-contamination diagnostics remain separately archived.
Production replay and checkpoints and unrelated GPU jobs were not modified.
