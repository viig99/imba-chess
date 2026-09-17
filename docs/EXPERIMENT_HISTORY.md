# Historical experiments and retired tooling

Current usage is in [README](../README.md), [the self-play readiness report](SELF_PLAY_READINESS_REVIEW_2026-09-11.md), and [the config guide](CONFIG_GUIDE.md). Old implementation recipes and offline halving-target generation were retired on 2026-09-13 after user approval. Their source remains recoverable at the immutable links below; old launch commands are not supported by current code.

## Decisions and measured evidence retained

| Topic | Current decision / evidence |
|---|---|
| Offline halving-target generation | Retired permanently. Stage-2 learns from completed self-play continuations; human continuations and old rollout rows are not training targets. |
| Halving vs Stockfish | Kept as the established comparison protocol in `eval_vs_stockfish.py`; its engine/search defaults were not changed by this cleanup. |
| Model A vs model B | `match_two_checkpoints.py` is retained. It uses halving and does not launch Stockfish; stage-2 paired Gumbel screens use `eval_self_play.py`. |
| Alpha-beta/PVS | Runtime already retired. [Measured outcomes and limits](CKPT34_ALPHA_BETA_PVS_HANDOFF.md) and its validation artifacts remain. |
| Distilled auxiliary values / ExIt | Historical outcomes remain in [value tuning review](superpowers/notes/2026-07-12-value-tuning-and-exit-phase1a-review.md) and [Stockfish supervision history](STOCKFISH_SUPERVISION_HANDOFF.md). |
| Native board, projection and history | Implemented; preserve Python/Rust parity and independent chess-rule tests. Native microbenchmarks remain available. |
| Batched eager and full compiled decoding | Adopted based on completed-position throughput. [Current evidence](SELF_PLAY_READINESS_REVIEW_2026-09-11.md). Shared-buffer SDPA remains an experiment; it did not establish an additional >=10% gain. |
| Deferred performance work | Persistent actual-game roots, native Gumbel controller and additional parallelism require profile evidence; removing old plans does not prohibit those experiments. |

## Full measurement history

- [Pre-consolidation README and evaluation history](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/README.md).
- [Full self-play implementation, run and benchmark chronology](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md).
- [Original roadmap, including superseded PPO/GRPO proposals](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/PLAN.md).

## Retired implementation recipes

- [2026-07-03-eval-game-animation](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-03-eval-game-animation.md).
- [2026-07-04-mcts-lite-search](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-04-mcts-lite-search.md).
- [2026-07-04-prefix-cache-decode](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-04-prefix-cache-decode.md).
- [2026-07-05-value-net-distillation](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-05-value-net-distillation.md).
- [2026-07-10-expert-iteration-value-distillation](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-10-expert-iteration-value-distillation.md).
- [2026-07-14-phase1b-policy-distillation-implementation](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-14-phase1b-policy-distillation-implementation.md).
- [2026-07-18-cozy-chess-hotpath-adoption](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-18-cozy-chess-hotpath-adoption.md).
- [2026-07-18-cozy-native-tree](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-18-cozy-native-tree.md).
- [2026-07-18-cross-game-batched-search](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-18-cross-game-batched-search.md).
- [2026-07-19-fast-clean-evals](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-19-fast-clean-evals.md).
- [2026-07-19-multiprocess-eval-actors](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-07-19-multiprocess-eval-actors.md).
- [2026-08-21-imba-chess-native](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/plans/2026-08-21-imba-chess-native.md).
- [2026-07-03-eval-game-animation-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-03-eval-game-animation-design.md).
- [2026-07-04-mcts-lite-search-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-04-mcts-lite-search-design.md).
- [2026-07-04-prefix-cache-decode-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-04-prefix-cache-decode-design.md).
- [2026-07-05-value-net-distillation-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-05-value-net-distillation-design.md).
- [2026-07-07-expert-iteration-distillation-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-07-expert-iteration-distillation-design.md).
- [2026-07-13-phase1b-policy-distillation-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-13-phase1b-policy-distillation-design.md).
- [2026-07-18-cozy-native-tree-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-18-cozy-native-tree-design.md).
- [2026-07-18-cross-game-batched-search-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-18-cross-game-batched-search-design.md).
- [2026-07-18-rollout-cpu-hotpath-optimization-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-18-rollout-cpu-hotpath-optimization-design.md).
- [2026-07-19-fast-clean-evals-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-19-fast-clean-evals-design.md).
- [2026-07-19-multiprocess-eval-actors-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-07-19-multiprocess-eval-actors-design.md).
- [2026-08-21-imba-chess-native-design](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/docs/superpowers/specs/2026-08-21-imba-chess-native-design.md).

## Retired scripts and offline storage

- [scripts/bench_decode_wave.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/bench_decode_wave.py).
- [scripts/bench_python_opts_paired.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/bench_python_opts_paired.sh).
- [scripts/bench_root_eval.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/bench_root_eval.py).
- [scripts/generate_rollouts_remote.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/generate_rollouts_remote.sh).
- [scripts/lr_probe.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/lr_probe.sh).
- [scripts/nightly_alpha01_train_and_eval.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/nightly_alpha01_train_and_eval.sh).
- [scripts/probe_bookkeeping_phases.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/probe_bookkeeping_phases.py).
- [scripts/probe_project_phases.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/probe_project_phases.py).
- [scripts/probe_push_children.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/probe_push_children.py).
- [scripts/profile_torch_waves.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/profile_torch_waves.py).
- [scripts/rollout_budget_sweep.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/rollout_budget_sweep.sh).
- [scripts/rollout_equivalence_gate.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/rollout_equivalence_gate.sh).
- [scripts/rollout_nightly_start.sh](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/rollout_nightly_start.sh).
- [scripts/generate_search_rollouts.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/generate_search_rollouts.py).
- [scripts/label_gate_diff.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/scripts/label_gate_diff.py).
- [src/imba_chess/data/rollout_store.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/src/imba_chess/data/rollout_store.py).
- [tests/test_generate_search_rollouts.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/tests/test_generate_search_rollouts.py).
- [tests/test_rollout_coroutine.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/tests/test_rollout_coroutine.py).
- [tests/test_rollout_store.py](https://github.com/viig99/imba-chess/blob/3d2d3d53fefbd4b587a82eb9e35772045d4eea50/tests/test_rollout_store.py).

Useful native microbenchmarks, the corpus estimator, calibration, both evaluation commands, and all config recipes are retained.

## Additional tooling retired — 2026-09-16

The following scripts were removed after review; their last retained versions are linked below. `build_static_move_vocab.py` remains available.

- [scripts/bench_move_id_micro.py](https://github.com/viig99/imba-chess/blob/35502f836577720fa88db110ecca813223d8231b/scripts/bench_move_id_micro.py): target function was removed by native projection.
- [scripts/bench_wave_suffixes.py](https://github.com/viig99/imba-chess/blob/35502f836577720fa88db110ecca813223d8231b/scripts/bench_wave_suffixes.py): benchmarks the retired suffix-path representation.
- [sweep.sh](https://github.com/viig99/imba-chess/blob/35502f836577720fa88db110ecca813223d8231b/sweep.sh): fixed-checkpoint depth-two experiment launcher.
- [eval_best_checkpoint.sh](https://github.com/viig99/imba-chess/blob/35502f836577720fa88db110ecca813223d8231b/eval_best_checkpoint.sh): legacy auto-selection and uncompiled evaluation wrapper.
- [scripts/lr_probe_summarize.py](https://github.com/viig99/imba-chess/blob/35502f836577720fa88db110ecca813223d8231b/scripts/lr_probe_summarize.py): historical result reader for the retired LR-probe workflow.

Removed the test-only `MoveVocab.build_from_games` API; its callers now build their small vocabularies directly. Consolidated legal-move-set checks into the bidirectional translation test with shared positions, preserving both prior random samples and the extended sweep. Calibration tests retain singleton percentiles, interpolation, invalid inputs, rounding ties and final recommendations while dropping redundant median/identity checks.
