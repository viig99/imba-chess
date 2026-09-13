# Self-play readiness — current status, updated 2026-09-13

This is the consolidated current status. Full original implementation/run chronology and exact historical benchmark commands remain in [the pre-consolidation report](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md). [The experiment index](EXPERIMENT_HISTORY.md) links retained decisions and retired tooling.

## What runs

| Area | Implemented and verified | Remaining gate |
|---|---|---|
| Search | Torch-free Gumbel controller; reference sequential halving/completed Q; exact budgets and terminal backups; all-legal soft policies | Strength under matched search settings |
| Collection | Same frozen actor for both colors; prepared human prefixes; one pending leaf/game; full history and context guards | Longer unattended workload stability |
| Replay | Immutable Parquet shards, bounded pending buffers/window, deduplication, atomic manifest and recovery; read-only readers do not publish | Sustained long-run memory and retention checks |
| Learning | Full-model soft legal-policy + outcome-WDL CE, prefix masking, StableAdamW, exact resume of optimizer/RNG/sampler/phase | Demonstrated improvement and calibrated draw prediction |
| Runtime | Batched one-query attention, cached packed prefixes, batched preparation/readback, whole-decoder compilation | Longer compiled soak; 5090 pilot |
| Evaluation | Restartable matched held-out pairs, independent best checkpoint, rollback-stop; permanent capped games raise protocol failure | Complete promotion confirmation before claiming a new best |
| Operations | Run locks, signal handling, atomic publication, bounded drain/hard deadline, morning supervisor and TensorBoard monitor | External scheduling remains disabled |

The offline halving-target generator was retired on 2026-09-13. It is separate from the preserved halving search and Stockfish evaluator. Model A/B halving matches are also retained.

## Algorithm and learning contract

- Gumbel reference: mctx revision `88f92056a420c2673bed282f5a0c00211f126e78`. [Fixtures](../tests/fixtures/gumbel/search.json), [generator](../tests/fixtures/gumbel/generate_reference.py), [license/attribution](licenses/mctx-NOTICE.md). No JAX/mctx runtime dependency.
- Initial search settings: 128 simulations, 16 root candidates, depth 32. Completed-Q constants: maxvisit initialization 50, scale 0.1, epsilon 1e-8. Root noise is fixed per search and excluded from target logits.
- Values use node side-to-move perspective, with sign reversal per edge. Terminal/depth-limited revisits consume simulations; terminal leaves need no neural evaluation. No board-only transpositions or extra halving forcing rules are added to Gumbel.
- Human takeover plies are sampled from eligible 20–120 positions. Stable source-game hashing reserves monitoring trajectories. Human continuations/results/annotations do not become stage-2 labels.
- Checkmate, stalemate, insufficient material and available repetition/fifty-move claims terminate games. Administrative limits/deadlines/errors produce unfinished records without labels. Guard `root_tokens + depth <= max_position_embeddings`; game ceiling 512 plies.
- Whole trajectories train with complete prefix context. Only continuation pre-move tokens receive targets. Policy CE normalizes over legal moves; outcome WDL uses the side-to-move perspective. No Elo weighting, hard winning-move label or moves-left loss.
- Initial learning recipe: LR 1e-5, value coefficient 1, decay .01, clipping 1, two supervised-position exposures per fresh training position. Actor refresh follows a complete collect/train phase; optimizer updates occur per batch.
- Checkpoint initialization loads weights; resume restores optimizer, RNG, sampler, replay references and phase progress. Collection slot overrides are logged separately and preserve configuration identity. See [config guide](CONFIG_GUIDE.md).

## Measured laptop performance

RTX 3070 Ti Laptop GPU (8 GB), FP32, TF32 disabled, four Torch CPU threads, ckpt34, fixed human prefixes/RNG, G=24, 128 simulations/16 candidates/depth 32. Each final trial completed 32 games: 2,775 searched positions, 2,767 training-eligible positions, 338,353 neural evaluations and 355,200 simulations. Outcomes: 27 mates, four repetition draws, one insufficient-material draw; no unfinished games/errors/depth cutoffs. Includes slot refill, collection, replay publication and late-game tail; excludes model loading. Profiling/tests were separate from final throughput trials.

| Decoder | Trial seconds | Median eligible positions/hour | Peak allocated VRAM |
|---|---:|---:|---:|
| Current eager | 272.55 / 276.00 | 36,319 | .966 GB |
| Full neural decoder compiled | 226.64 / 219.13 | 44,705 | .966 GB |
| Shared-buffer SDPA | 269.48 (one trial) | 36,965 | 1.113 GB |
| Compiled + shared-buffer SDPA | 234.46 / 205.07 | 45,530 | 1.113 GB |

Full compilation improves throughput 23.1%, reducing whole-collection median elapsed time about 18.7%. This does not assert an identical reduction for every move's latency. It is now the CUDA default; `--decoder-mode current` selects eager. CPU probes use eager. Compilation covers embeddings, all eight HSTU blocks, final norm and policy/value heads. It produced two graphs (multi-game and singleton), zero graph breaks and no continuing history-shape recompilation. First-call compilation exists; disk caches were warmed before final trials, so those are not fresh-cache cold-start numbers.

SDPA used PyTorch's FP32 **memory-efficient CUTLASS backend**, not FlashAttention or math fallback. Packed prefixes and reserved branch/leaf slots share one allocation. Prefixes are not recopied on each query while the ordered owner cohort remains fixed; changed cohorts repack. Branch suffixes still update, unequal histories still require masks, and attention still reads K/V. Both eager and compiled SDPA were tested. Its 1.85% incremental median gain over compiled-only, with much greater variation, failed the >=10% adoption gate. It remains available only in benchmark/profiler modes.

Cached packed prefixes were enabled in every final variant, including compiled-only. The table does not isolate their contribution. Earlier batching/preparation improvements produced about 4.6× versus the original runtime on a different 24-game workload; do not multiply ratios across different workloads. An eager production soak completed 64 games / 5,278 training positions in 521.48 seconds, with no unfinished games, about 2,880 MiB peak NVML process memory and 1.61 GB host RSS. This is not an eight-hour compiled soak.

Original trace: 25,444 launches in 16 decode waves; initial batching reduced that to 5,236. Later profiling justified batched suffix preparation and asynchronous input copies. Root executor host time was ~2.5–3% of collection; persistent actual-game-root caching did not meet the 10% trigger. Work outside root/decode executors was ~10–12%; a new native tree boundary was not adopted. Rust remains an option if measured controller overhead reaches 20% after GPU improvements.

Artifacts: `artifacts/self_play_validation/speed_campaign/` and `decoder_campaign/`. The latter contains `analysis.json`, hardware/config/checkpoint hashes, source snapshots, profiler traces, compiler counters, exact `command_*.txt`, and `final_sweep.py`. Checkpoint SHA256: `5844b09fdde268f5fd2aba363603c43e9c1020d776c2d5294a17c2f912962826`.

## Correctness evidence and numerical limits

The decoder campaign reconstructed 440 completed games / 38,586 searched positions. Played trajectories, outcomes and root visit vectors matched their eager references. This does not prove identical interior search decisions. Maximum final compiled-only target probability difference was 4.38e-5. Compiled-SDPA had a 0.001105 outlier: identical root priors/WDL/visits but different backed-up Q for one action. Root-only diagnostics cannot prove its cause. The artifact `largest_sdpa_target_difference.json` retains it; no deeper tracing was done because that mode also failed the performance gate.

Before maintenance reductions, full CPU checks passed 2,335 tests; final decoder checks passed 23 on CUDA. The first maintenance cleanup passed 2,280 default tests in 20.35 seconds, four extended CPU checks, 23 CUDA checks and 107 native binding tests. [The audit](TEST_MAINTENANCE_AUDIT_2026-09-12.md) explains removed redundancy and retained coverage. Offline-generator-only tests are now retired; the shared executor's adversarial device-placement test was preserved in the decoder suite. Latest cleanup passed 2,267 default tests in 20.84 seconds; executable Stockfish/model-match behavior and all eight config bytes are preserved. The maintenance review records details.

Full-forward → cached → grouped → optimized parity, mixed history/depth, immutable prefix/KV ownership, changed weights between phases, exact fake-evaluator search behavior, legal targets, independent chess terminal outcomes, replay recovery, finite gradients, tiny overfit and resume remain covered. Extended compiler/device/random checks run explicitly with `pytest -m extended` or `-m ''`.

## Learning evidence and evaluation

Earlier ckpt34 versus trained actor 000002 checks each completed 50 games from 25 held-out prefixes, colors swapped, against Stockfish 18 at 40,000 nodes/move, one thread, 64 MB hash, full strength. Both scored 14%; paired difference 0 points, 95% interval -6 to +6 points. **No improvement or regression was demonstrated.** This differs from the historical strength-limited SF2400 halving ruler (ckpt34 65.93% across 750 games).

The compiled G=24 overnight resume on September 13 started at 00:25 EDT and halted at approximately 02:53 EDT after the configured regression screen. It published **1,069 new completed games / 93,100 continuation positions / 84,515 training positions** (cumulative 1,221 / 105,549 / 95,888). The failed candidate `actor-000015.pt` remains saved; collection actor/best reverted to `actor-000000.pt`. The latest screen scored **35.5%, paired 95% CI 28.5–43%, 100 games / 50 pairs** against best, triggering `rollback_stop`. The preceding screen scored 54%, CI 47–61.5%, which did not establish improvement. Training is halted for investigation, not silently restarted. At the 06:00 check the overnight supervisor was alive and waiting for the scheduled 08:00 candidate/baseline/Stockfish evaluations. State, screens and logs: `artifacts/self_play/laptop-pilot/`, including `overnight-2026-09-13-compiled/progress.json`. No cause for the regression has yet been established.

Nightly screens use 100 games from 50 held-out prefix pairs. Promote best only after a 500-game confirmation has a paired 95% lower bound above 50%. Screen upper bound below 45% against best triggers rollback and stops automatic learning. Capped evaluation games are protocol failures, not endlessly resumable interruptions; no outcome is fabricated. Morning supervisor snapshots the latest published actor and baseline, then runs model-pair and both Stockfish comparisons with explicit budgets.

Monitor completed eligible positions/hour, unfinished reasons, policy CE/entropy/KL, outcome CE/Brier, predicted/observed draw rates, gradient norms, replay age/reuse and matched strength. Runtime gains and lower training losses do not establish chess improvement. Current run state/logs are under `artifacts/self_play/`; do not infer completion from a stopped terminal alone.

## Regression diagnosis — September 13

The original waiting morning supervisor was cancelled and replaced with a bounded diagnosis queue. Checkpoints 0/14/15, optimizer/sampler state, every retained replay shard, configurations and original screens/logs were copied and SHA256-indexed before experiments. Artifacts and exact executable commands: `artifacts/self_play_diagnosis/2026-09-13/` (`snapshot-manifest.json`, `campaign.py`, `progress.json`, `summary.json`). The original production run remains halted.

Fresh Gumbel screens used the same 50 monitoring source games for both candidates, excluding all source games appearing in earlier saved evaluation identities. All 100 games completed per candidate. Actor14 scored **46%, paired 95% CI 37–54.5%** versus ckpt34; actor15 scored **37.5%, CI 30–45%**, corroborating the earlier regression signal on different openings. Actor15 minus actor14 was -8.5 points, paired CI -21.5 to +5 points: the last phase alone is not proven to account for the decline.

Independent reconstruction passed for **665 retained games / 57,304 positions**, including python-chess legality, termination before every supervised move, final results, side-to-move WDL, prefix masking, previous-move alignment and the actual 513-token context limit. All 57,304 saved search policies exactly matched an independent completed-Q/softmax reconstruction; simulation, candidate and depth bounds passed. Older evicted trajectories are outside this audit. Replaying the final 20 optimizer updates from the preserved step-305 state produced actor15's step-325 weights **bit-for-bit**, matching exposures (195,583), queue and sampler RNG. No production weights/replay were changed by this check.

Fixed monitoring set: 64 trajectories / 5,178 positions. Outcomes describe the recorded continuations, not optimal-play values. Metrics are position-weighted:

| Metric | ckpt34 | Actor14 | Actor15 |
|---|---:|---:|---:|
| Search-policy CE | 1.6741 | 1.6475 | 1.6487 |
| Outcome WDL CE | 3.8931 | 1.3623 | 1.2450 |
| WDL Brier (lower better) | .4746 | .5428 | .5160 |
| WDL accuracy | 67.61% | 61.22% | 62.42% |
| Predicted draw probability | ~0% | 15.82% | 14.15% |
| Actual draw-position fraction | 17.46% | 17.46% | 17.46% |
| Model policy entropy | 1.4769 | 1.6177 | 1.6247 |

Actor15's Brier increase has a source-game bootstrap interval spanning zero; do not call it conclusively worse calibration. Its accuracy difference was -5.20 points (exploratory paired interval -10.05 to -.20). Training loss improvement does not establish playing-strength improvement. Two fixed gradient probes found actor15 shared-backbone value/policy gradient-norm ratios around 5 on a decisive trajectory and 11 on a draw. This motivates testing value-loss weighting; it does not establish the regression's cause.

The next queued controlled experiment forks the identical final-phase weights, optimizer and sampler/replay, changing only value weight 1.0 → 0.1. It is a diagnostic counterfactual, not an automatic production restart or a strength claim. Matched 100-game halving/SF2400 screens (budget 2048, roots 16, refutations 4, own expansions 3, depth 8) are running before that experiment; final results remain pending.

Telemetry now includes `model_policy_entropy` computed without gradients. The historical `policy_entropy` remains the search-target entropy for compatibility. Existing hand-calculated loss coverage checks both distributions, padded legality and exact policy gradients.

## Laptop learning settings checked before restart

A disposable actor-000002/replay training sweep measured 1,024-token batches at 8,669 supervised positions / 15 steps per trial in 15.88 / 10.47 / 9.12 seconds, peak allocated VRAM 2.725 GB. At 2,048 tokens the current training backward path ran out of CUDA memory; 4,096 was not attempted. Retain 1,024 for the resumed run. Full inference-decoder compilation is separate from training attention/backward. Keep 4,096 fresh positions per phase, reuse=2, LR=1e-5 and the existing optimizer/sampler state. These are measured hardware-compatible starting settings, not proven optimal learning hyperparameters. Artifacts: `artifacts/self_play_validation/nightly_tuning_2026-09-13/`.

## Operation and reproduction

- New laptop runs: `config/self_play_laptop_fast.toml` (24 slots). Existing pilot resume: retain `self_play_laptop_pilot.toml`, optionally use the logged execution-only `--concurrent-games 24` override. Learning settings are unchanged by that override.
- `run_self_play.py --until <ISO timestamp with timezone>` uses an absolute deadline. Reserve/drain follow the saved config. For the pilot: stop starting new collect/train phases 15 minutes before deadline; drain up to 10 minutes; hard stop at deadline.
- `run_self_play_overnight.py` resumes the run and queues three morning evaluations. Its progress and per-evaluation results are restartable. Each evaluation is bounded; incomplete results remain incomplete.
- Benchmark: explicit config, checkpoint, seed manifest and fresh output directory are required. Use `--component training --replay <path>` for disposable training measurements; readers never publish replay state. Detailed flags are in `--help`.
- No remote 5090 result or automatic external nightly schedule is established. Retain ~15% VRAM headroom in the actual soak workload and adopt optimizations only with repeated end-to-end gains or measured memory relief.

## Stockfish evaluator consolidation — September 13

Synchronous move selection now drives the same stepwise controller used by the single-process scheduler. Legal projection, value-head/config guards, search dispatch and debug formatting have one implementation. Synchronous greedy evaluation still skips KV output. The halving algorithm, Stockfish settings, CLI defaults, actor worker/server route, and stage-2 morning evaluation code are unchanged. The script shrank by 80 lines; larger actor/reporting consolidation remains a separate decision.

Validation: `.venv/bin/python /tmp/verify_stockfish_refactor.py` compared the saved pre-refactor source against both current execution modes for greedy, value rerank, depth-two and halving policies. Selected moves, complete debug dictionaries (including top-k logits/search statistics) and inference counts matched exactly on deterministic fixtures. Existing parity tests now check complete debug dictionaries. `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q` passed 2,267 tests, 17 extended deselected, in 24.06 seconds. Reproduction script/source copies, differential results and CUDA/Stockfish smoke logs are under `artifacts/self_play_validation/stockfish_refactor_2026-09-13/`. Smoke checks use ckpt34, halving budget 32/top-m 4, two games capped at eight plies, and concurrency 1 then 2. Capped smoke games remain incomplete and supply no strength evidence. Specify both `--games 2` and `--ladder-games-per-segment 2`; the retained v4 ladder has its own game count.
