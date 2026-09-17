# GPU overlap and native node statistics — 2026-09-16

Selected for production: direct CPU metadata staging plus native-owned node statistics and one backup call per simulation, retaining synchronous readback and the reference scheduler. The user explicitly authorized mainlining after the passing short comparison, superseding the original requirement for three full pairs for this simpler candidate. The 30.8% result is one short comparison, not a repeatability claim. The asynchronous combination and two-group pipeline are rejected; additional Rust scratch reuse is not pursued.

## Fixed workload and reproducibility

Actor111, FP32, TF32 disabled, 128 simulations, top_m=16, depth 32, 48 concurrent games, four CPU threads. Each throughput process warms on 48 complete games and measures complete games with identical checkpoint, seeds and game IDs. The promotion workload uses three alternating pairs of 128 games, ordered reference/candidate, candidate/reference, reference/candidate. GNOME and user applications remain running. These rates measure the collector only; they are not optimizer-inclusive training throughput. Played search moves/hour includes searched moves, not only replay positions retained for training.

The reference hot-path files are exactly `2d64a11`. Pre-existing unrelated workspace changes were copied identically into all frozen variants; the source manifests identify the full tested trees, rather than claiming an entirely clean checkout. Those user edits remain untouched.

CUDA validation ran on the local host outside the filesystem sandbox, which hides `/dev/nvidia*`. Hardware is the RTX 3070 Ti Laptop GPU, with PyTorch 2.14.0+cu130. The production checkpoint, replay and concurrency configuration were not modified.

Local measurement archive: `artifacts/self_play_validation/overlap_2026-09-16/`. `environment.json` hashes the raw config, model config, vocabulary, seeds and checkpoint. Every variant has a full Python source manifest. Each run's `execution.json` identifies its source snapshot, native binary and harness hashes; `harness_versions/` preserves the exact scripts. Targets, game records, timing distributions, batch histograms, compiler counters and traces are retained. Source patches and `snapshot.py` reconstruct rejected experiments without production flags. The original native extension is archived under `reference_native/`.

## Archived asynchronous candidate behavior

The candidate preserves each scheduler tick's request snapshot, batch composition, per-game sequencing, canonical legal order and result delivery order. Executors expose internal submit/finish operations while retaining synchronous call compatibility. Independent root/decode groups from the same tick are submitted before consuming their results.

A pending result retains requests, GPU source storage, pinned host destinations, completion events and generation until consumption or cancellation finishes. Root legal IDs upload together and each root keeps its original GPU normalization arithmetic and shape; packed priors/WDL read back once per root execution group. Decode logits read back on a shared transfer stream before independent K/V scatter; decode normalization remains on CPU. Explicit producer/completion events protect every host read and buffer reuse. Cleanup drains owned GPU work before releasing buffers or changing weights.

Direct leaf preparation writes the existing metadata layout, preserving owner/revision validation, history placement and selective ancestor gather. Native-owned per-node priors, probabilities, visits, sums and means eliminate repeated list extraction on selection. One native call backs up a completed simulation in the original reverse-path order. Python retains tree ownership, chess transitions, RNG and coroutine scheduling. Final target construction exports node statistics once. Decoder arithmetic, attention padding, model precision and history K/V residency are unchanged.

## Diagnostics and alternatives

Short CPU/CUDA traces are attribution evidence, never throughput evidence. The reference trace recorded 2.253 ms of actual D2H copies and no D2H/kernel overlap. The combined trace recorded 1.716 ms of D2H, with 1.012 ms overlapping kernels. CUDA-event spans can include dispatch gaps; trace memcpy durations are used for actual copy attribution. The earlier 106 seconds inside `.cpu()` included waiting for preceding GPU work and is not a recoverable-speedup estimate.

In the combined diagnostic, 27.73% of traced wall time contained CPU work while the collector had no CUDA activity, excluding explicit runtime waits. This is collector inactivity, not global GPU idleness: desktop applications remained running. It exceeded the 10% trigger for testing two stable groups of 24 games.

The bounded two-group prototype used separate workspaces, one compute stream, a shared transfer stream and at most two outstanding groups. Small-model scheduling, cancellation/recovery, refill, uneven games and tails passed. Actor111 complete-game targets failed the fixed tolerance: policy probability 0.7061101650055054 became 0.7061082748712082. The pipeline is rejected and remains only in the archive. The original failure and all raw outputs are retained; tolerance was not relaxed.

A further native scratch-buffer prototype made isolated selector calls about 21% faster than the first native-owned version. No incremental whole-collector improvement has been demonstrated. The user prefers simplicity and a timely decision, so additional scratch-buffer experiments are cancelled; the refinement remains archived and is not a mainline candidate.

## Correctness and lifecycle checks

The combined candidate suite passed 110 tests, with one intentional unsupported compiled-CPU skip. Coverage includes exact transfer/staging contents; delayed completion and host/source lifetime; buffer reuse and growth; stale owner/revision/model results; history replacement and cross-game isolation; mixed root/decode work; result ordering; cancellation and exceptions followed by another collection; and cold/warmed shapes and single-game tails. Decoder tolerance remains 1e-5 and projected/target tolerance 1e-6.

Two lifecycle defects found during development were reproduced and fixed before the final candidate was frozen: closing an old workspace pending result could unlock a newer one, and executor cleanup could leave an outer pending closure retaining payload owners. Cleanup now closes the owning result and identity-checks releases. The interrupted early screen and reproductions are preserved under `rejected_stale_pending/`, `stale_pending_reproduction.log` and `pending_owner_reproduction.log`.

Actor111 scratch collect → optimizer update → collect matched reference moves and targets bitwise before and after a real update. A repeated post-update collection retained exactly 595,898,880 allocated CUDA bytes in both cycles; pre-update cleanup retained 201,884,160 bytes. The optimizer changed weights by up to 1.0013580322265625e-5, and the checkpoint hash remained unchanged. These are correctness checks, not training-throughput measurements.

## Component screens

Each screen used 48 warmup and 48 measured complete games. Overlap: 140.552 s; preparation: 142.173 s; native statistics: 141.064 s; combined: 134.815 s. All four matched reference complete-game decisions, counters and targets bitwise. The combined candidate was more than 2% faster than each individual component in this screen and was selected for full paired testing.

The earlier reference screen took 488.195 s. That large difference from later screens is retained as raw desktop-contended evidence but is not a promotion claim. Only the fresh alternating pairs determine the speedup. Combined screen peak allocated memory was 4,113,172,480 bytes, reserved 5,215,617,024 bytes and pinned host capacity 159,040 bytes. After collection, allocated CUDA memory returned to 201,884,160 bytes. Two decoder graphs were warmed and no additional graphs compiled during measurement. Batch distributions and copy counts are in the per-run metrics.

## Alternating pairs: asynchronous combination rejected

All three pairs matched complete-game moves, outcomes, counters and serialized targets bitwise. The combined asynchronous candidate failed promotion: paired median throughput gain was 3.50% (below 5%), the second pair regressed 46.36%, and paired median p95 latency regressed 33.55% (above the 5% limit).

| Pair / order | Reference seconds | Candidate seconds | Throughput change | p95 change |
|---|---:|---:|---:|---:|
| 1: reference → candidate | 434.699 | 419.013 | +3.74% | +33.55% |
| 2: candidate → reference | 494.590 | 922.033 | −46.36% | +63.03% |
| 3: reference → candidate | 721.929 | 697.543 | +3.50% | −11.44% |

Each measured run completed 128 games and 10,386 searched moves, after 48 complete warmup games. Across runs, median collector rates were 931.7 reference versus 660.6 candidate games/hour, and 75,597 versus 53,602 played search moves/hour. These aggregate medians differ from the median paired relative gain because conditions varied substantially; neither view supports promotion. Candidate peak allocated memory was 4,231,612,416 bytes in every pair; reserved memory was 5,360,320,512 bytes. Cleanup returned to 201,884,160 allocated bytes. Every warmed run retained two decoder graphs without further compilation. Batch histograms and inference counts matched reference exactly: 49,910 decode calls with mean batch size 24.814. Logical result-copy count fell from 70,682 to 58,112 (reference root copies inferred from its two per-root result reads). Pinned host capacity rose from 120,320 to 159,040 bytes. Actual-copy overlap is documented separately by the diagnostic traces; copy count alone is not speedup evidence.

There was substantial variation in both implementations: reference warmups took 162.7, 206.1 and 464.6 seconds; candidate warmups took 355.7, 263.5 and 473.0 seconds. Read-only GPU process snapshots showed the benchmark and GNOME, with no competing CUDA compute process. A slow-reference snapshot showed 1785 MHz core and 7001 MHz memory clocks. The cause of the variability was not isolated. It is not justified to attribute the full timing difference to code or to GNOME. All unfavorable observations remain in the archive, and no profiled timing is used for promotion.

## Simpler follow-up and final verification

After the asynchronous candidate failed, the sole follow-up isolates direct CPU staging plus native-owned statistics while retaining the reference scheduler and synchronous readback. Simple Python cleanup moves imports out of the hot loop, holds native statistics directly in backup paths, and allocates unused Python statistic arrays only for the compatibility path.

The expanded six-screen queue was stopped before any scratch screen began. Its already-running reference child was preserved. One bookkeeping-only screen followed that reference, each with 48 warmup and 48 measured complete games. The user subsequently instructed “mainline it” after reviewing the results and the limitation of a single short comparison. No further variants or full throughput campaign were run.

Bookkeeping-only passed 97 CUDA regression tests with one unsupported compiled-CPU skip. The archived scratch refinement passed 105 with the same skip. The simpler candidate completed 48 games / 3,839 searched positions in 217.659 seconds versus reference 284.664 seconds: 30.78% greater collector throughput, 793.9 versus 607.0 games/hour, and 63,495.5 versus 48,549.8 played search moves/hour. p95 move latency improved 13.59%. Complete-game decisions/counters and serialized targets matched bitwise in warmup and measurement. Peak allocated CUDA memory was 4,116,768,768 bytes (3.83 GiB), reserved 5,205,131,264 bytes (4.85 GiB); cleanup returned to 201,884,160 bytes. Warmup took 380.641 versus 495.562 seconds and is excluded from throughput. Raw metrics: `bookkeeping_screens/bounded_comparison.json`.

Integration keeps only the two selected Python hot-path changes and the basic Rust statistics implementation. Python ASTs match the measured candidate after the native namespace migration. The native extension was rebuilt; superseded list-based public selectors were removed and their tests migrated. Integrated CPU/CUDA regression tests passed 99 tests, with one unsupported compiled-CPU skip. The lightweight monitoring test also passed. Actual Actor111 collect → optimizer update → collect passed in the integrated production implementation: all three cycles matched reference games and targets bitwise. Both post-update cycles retained exactly 595,898,880 allocated CUDA bytes; the checkpoint was unchanged. The scratch optimizer performed one step with maximum weight change 1.0013580322265625e-5. The integration harness initially mixed native Board types from two extension builds; isolating only the old selector functions corrected the harness before these checks ran. No production-code workaround was required.

## Overnight continuation

The user redirected remaining work toward nightly self-play and simplicity. No further scratch-buffer experiment or automatic full promotion campaign is scheduled. The existing actor111 continuation retains its optimizer, RNG, sampler, replay, configuration identity, learning settings and 24-game execution override.

Nightly operation is set to 22:00–08:00 America/Toronto through the existing September timer. Screens become due after three hours, at the next completed training phase; an existing unfinished screen is resumed. The interval restarts after each screen finishes and at process startup, so this targets roughly two or three screens per uninterrupted night rather than guaranteeing a count. The existing observation-only policy and deferred confirmations remain.

Profiling is disabled. A standard-library-only monitor reads appended normal metrics once per minute and reports session games/hour, searched positions/hour and optimizer steps/hour, including training and evaluation time. Counts update at normal collection/step reporting boundaries; there are no profiler hooks, CUDA diagnostic events or replay scans.

Periodic training recovery saves use 3,600 seconds. Phase boundaries still save; this is not a guarantee of only one disk write per hour. Retention keeps two recent recovery checkpoints plus the current actor and fixed baseline needed for resume/evaluation. Replay garbage collection pins shards referenced by all retained checkpoints. Existing session-end snapshots are not deleted. Default CLI behavior stays at 60-second recovery saves, one recovery checkpoint and actor-count screening unless overridden.

## Artifact cleanup

Removed 103 disposable cache/build paths and compressed 74 large trace/target files losslessly, verifying decompressed SHA-256 before deleting each uncompressed original. Reclaimed 3,065,482,750 bytes (2.85 GiB). The cleanup manifest is `cleanup.json`; older readers can decompress the `.json.gz` evidence before use. Source/config manifests, timing results, adverse observations, active checkpoints/replay and historical model checkpoints remain.

Nightly command uses the existing continuation with `--resume --concurrent-games 24 --screen-seconds 10800 --checkpoint-seconds 3600 --keep-recovery-checkpoints 2 --observe-only-screen --defer-confirmation`, with the wrapper setting the absolute 08:00 Toronto deadline. The user authorized an immediate start tonight and restoring the existing 22:00 September timer. The service has a 12-hour outer bound to accommodate the early start; the runner’s absolute deadline remains 08:00.
