# Current roadmap

## Implemented

- Supervised HSTU chess training with placement-aware history tokens and engine-annotated value targets.
- Native legal projection, board encoding and history-aware terminal handling.
- Established halving/Stockfish evaluation, concurrent actor inference and model-pair matches.
- Gumbel search, completed-continuation replay, policy/outcome training and restartable paired evaluation.
- Single-GPU collect/train phases, compiled CUDA decoding, bounded overnight supervision and monitoring.

## Next gates

1. Resume bounded laptop learning with measured inference settings; monitor completion, draw calibration, replay reuse and gradient stability.
2. Complete morning matched model-pair and fixed-protocol Stockfish comparisons. Separate strength evidence from throughput gains.
3. Tune training batches using actual replay and measured VRAM/headroom; changing LR/reuse/actor-refresh thresholds remains a controlled learning experiment.
4. Run the remote 5090 pilot and one-hour soak before scheduling unattended remote nights. No 5090 throughput forecast is established yet.
5. Refactor duplicated Stockfish move-selection/reporting only with preservation of the established protocol and real process/decoder checks.
6. Consider persistent actual-game-root caches or native Gumbel bookkeeping only when profiles meet the thresholds documented in the readiness report.

Offline halving-target generation, alpha-beta/PVS runtime and distilled auxiliary-value training are retired. Halving remains a comparison engine. Shared-buffer SDPA and native microbenchmarks remain explicit experiments. Opponent populations, replay reanalysis, distributed actors and within-tree parallelism are deferred.

See [readiness](docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md), [config compatibility](docs/CONFIG_GUIDE.md), and [historical decisions](docs/EXPERIMENT_HISTORY.md). The original PPO/GRPO proposal is retained in Git history; stage 2 currently implements Gumbel policy improvement with supervised search-policy and outcome targets.
