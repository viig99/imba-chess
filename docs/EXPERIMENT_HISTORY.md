# Historical experiments and retired tooling

Current usage is in [README](../README.md), [the self-play readiness report](SELF_PLAY_READINESS_REVIEW_2026-09-11.md), and [the config guide](CONFIG_GUIDE.md). Old implementation recipes and offline halving-target generation were retired on 2026-09-13 after user approval. Their source remains recoverable at the immutable links below; old launch commands are not supported by current code.

## Decisions and measured evidence retained

| Topic | Current decision / evidence |
|---|---|
| Offline halving-target generation | Retired permanently. Stage-2 learns from completed self-play continuations; human continuations and old rollout rows are not training targets. |
| Halving vs Stockfish | Kept as the established comparison protocol in `eval_vs_stockfish.py`; its engine/search defaults were not changed by this cleanup. |
| Model A vs model B | `match_two_checkpoints.py` is retained. It uses halving and does not launch Stockfish; stage-2 paired Gumbel screens use `eval_self_play.py`. |
| Alpha-beta/PVS | Runtime already retired. [Measured outcomes and limits](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/CKPT34_ALPHA_BETA_PVS_HANDOFF.md) and its validation artifacts remain. |
| Distilled auxiliary values / ExIt | Historical outcomes remain in [value tuning review](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/superpowers/notes/2026-07-12-value-tuning-and-exit-phase1a-review.md) and [Stockfish supervision history](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/STOCKFISH_SUPERVISION_HANDOFF.md). |
| Native board, projection and history | Implemented; preserve Python/Rust parity and independent chess-rule tests. Unused cozy-chess binding surface (ray/move helpers, `PieceMoves`, Chess960 constructors, null moves) was removed on 2026-09-29. |
| Batched eager and full compiled decoding | Adopted based on completed-position throughput. [Current evidence](SELF_PLAY_READINESS_REVIEW_2026-09-11.md). Shared-buffer SDPA did not establish an additional >=10% gain. |
| Halving tactical coverage and quiescence | Retired 2026-09-29. The 100-game SF2400 screen scored coverage 0.470, coverage + two quiescence plies 0.405, and quiescence alone 0.615 with little quiescence work (17k evaluations); neither was adopted. The retained refutation floor (all forcing replies at opponent nodes, forcing root moves) is unchanged. [Experiment spec]({BASE}docs/CKPT34_TACTICAL_SEARCH_HANDOFF.md). |
| Prepared-seed self-play collection | Retired 2026-09-29. Every stage-2 run streams starts from the training corpus; seed manifests supply held-out monitor openings only. |
| Run identity compatibility | Retired 2026-09-29. Config identity hashes every setting and the raw base-config bytes with no aliases, so runs created earlier resume only from their own source snapshot. |
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

## Cleanup — 2026-09-29

Retired after review. Kept: from-scratch HSTU training, Gumbel self-play and evaluation, halving evaluation, streamed starts, and the live self-play value investigation documents. Removed code, configs and reports are linked at their last version.

### Code and configs

- [config/imba_chess.toml](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/config/imba_chess.toml).
- [config/imba_chess_sf_finetune_low_lr.toml](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/config/imba_chess_sf_finetune_low_lr.toml).
- [config/imba_chess_v3.toml](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/config/imba_chess_v3.toml).
- [config/self_play.toml](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/config/self_play.toml).
- [config/self_play_laptop_fast.toml](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/config/self_play_laptop_fast.toml).
- [config/self_play_laptop_pilot.toml](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/config/self_play_laptop_pilot.toml).
- [native/imba_chess_native/src/functions.rs](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/native/imba_chess_native/src/functions.rs).
- [native/imba_chess_native/src/piece_moves.rs](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/native/imba_chess_native/src/piece_moves.rs).
- [scripts/audit_blunders.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_blunders.py).
- [scripts/audit_branch_continuations.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_branch_continuations.py).
- [scripts/audit_frozen_value_branch.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_frozen_value_branch.py).
- [scripts/audit_frozen_value_probes.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_frozen_value_probes.py).
- [scripts/audit_history_ablation.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_history_ablation.py).
- [scripts/audit_policy_regression.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_policy_regression.py).
- [scripts/audit_search_improvement.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_search_improvement.py).
- [scripts/audit_search_scales.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_search_scales.py).
- [scripts/audit_search_settings.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_search_settings.py).
- [scripts/audit_tactical_recognition.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_tactical_recognition.py).
- [scripts/audit_value_probe_scale.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_value_probe_scale.py).
- [scripts/audit_value_readouts.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/audit_value_readouts.py).
- [scripts/bench_encode_cozy_micro.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_encode_cozy_micro.py).
- [scripts/bench_gumbel_pipeline.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_gumbel_pipeline.py).
- [scripts/bench_native_move_projector.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_native_move_projector.py).
- [scripts/bench_project_decompose.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_project_decompose.py).
- [scripts/bench_search.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_search.py).
- [scripts/bench_search_bookkeeping_decompose.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_search_bookkeeping_decompose.py).
- [scripts/bench_self_play.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_self_play.py).
- [scripts/bench_self_play_input.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_self_play_input.py).
- [scripts/bench_training_compile.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/bench_training_compile.py).
- [scripts/compare_gumbel_eval_noise.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/compare_gumbel_eval_noise.py).
- [scripts/estimate_lichess_cache.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/estimate_lichess_cache.py).
- [scripts/generate_self_play.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/generate_self_play.py).
- [scripts/preview_dataset.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/preview_dataset.py).
- [scripts/profile_gumbel_pipeline.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/profile_gumbel_pipeline.py).
- [scripts/profile_self_play.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/profile_self_play.py).
- [scripts/report_frozen_value_probes.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/report_frozen_value_probes.py).
- [scripts/report_tactical_recognition.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/report_tactical_recognition.py).
- [scripts/run_self_play_overnight.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/run_self_play_overnight.py).
- [scripts/summarize_branch_continuations.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/summarize_branch_continuations.py).
- [scripts/test_event_dataloader.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/test_event_dataloader.py).
- [scripts/verify_policy_regression.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/scripts/verify_policy_regression.py).
- [src/imba_chess/self_play/benchmarks.py](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/src/imba_chess/self_play/benchmarks.py).

### Reports and handoffs

The ckpt34 alpha-beta/PVS validation artifacts are at [docs/validation/ckpt34-ab-pvs](https://github.com/viig99/imba-chess/tree/0997951a18984ebdf16a5e1097a9d28382075e00/docs/validation/ckpt34-ab-pvs).

- [BEAM_SEARCH_PLAN.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/BEAM_SEARCH_PLAN.md).
- [STOCKFISH_EVAL_PLAN.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/STOCKFISH_EVAL_PLAN.md).
- [VALUE_HEAD_OPTIONS.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/VALUE_HEAD_OPTIONS.md).
- [docs/ATTENTION_HISTORY_STUDY_PLAN.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/ATTENTION_HISTORY_STUDY_PLAN.md).
- [docs/ATTENTION_WINDOW_GQA_ESTIMATES.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/ATTENTION_WINDOW_GQA_ESTIMATES.md).
- [docs/CKPT34_ALPHA_BETA_PVS_HANDOFF.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/CKPT34_ALPHA_BETA_PVS_HANDOFF.md).
- [docs/CKPT34_TACTICAL_SEARCH_HANDOFF.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/CKPT34_TACTICAL_SEARCH_HANDOFF.md).
- [docs/CUDA_GRAPH_EXPERIMENT_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/CUDA_GRAPH_EXPERIMENT_2026-09-16.md).
- [docs/GENERATION_PERF_HANDOFF.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/GENERATION_PERF_HANDOFF.md).
- [docs/HISTORY_CACHE_PROFILE_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/HISTORY_CACHE_PROFILE_2026-09-16.md).
- [docs/MAINTAINABILITY_REVIEW_2026-09-13.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/MAINTAINABILITY_REVIEW_2026-09-13.md).
- [docs/POLICY_REGRESSION_AUDIT.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/POLICY_REGRESSION_AUDIT.md).
- [docs/POLICY_REGRESSION_RESULTS_2026-09-20.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/POLICY_REGRESSION_RESULTS_2026-09-20.md).
- [docs/SEARCH_IMPROVEMENT_AUDIT.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SEARCH_IMPROVEMENT_AUDIT.md).
- [docs/SEARCH_SCALE_SCREEN_2026-09-19.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SEARCH_SCALE_SCREEN_2026-09-19.md).
- [docs/SELF_PLAY_48_BOTTLENECKS_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_48_BOTTLENECKS_2026-09-16.md).
- [docs/SELF_PLAY_COMBINED_CACHE_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_COMBINED_CACHE_2026-09-16.md).
- [docs/SELF_PLAY_CONCURRENCY_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_CONCURRENCY_2026-09-16.md).
- [docs/SELF_PLAY_INPUT_STREAM_2026-09-17.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_INPUT_STREAM_2026-09-17.md).
- [docs/SELF_PLAY_OVERLAP_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_OVERLAP_2026-09-16.md).
- [docs/SELF_PLAY_PROFILE_2026-09-15.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_PROFILE_2026-09-15.md).
- [docs/SELF_PLAY_SESSION_REPORT_2026-09-16.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_SESSION_REPORT_2026-09-16.md).
- [docs/SELF_PLAY_STREAMING_ASSESSMENT_2026-09-18.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/SELF_PLAY_STREAMING_ASSESSMENT_2026-09-18.md).
- [docs/STAGE2_COMPILE_COMPARISON_2026-09-13.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/STAGE2_COMPILE_COMPARISON_2026-09-13.md).
- [docs/STOCKFISH_SUPERVISION_HANDOFF.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/STOCKFISH_SUPERVISION_HANDOFF.md).
- [docs/TEST_MAINTENANCE_AUDIT_2026-09-12.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/TEST_MAINTENANCE_AUDIT_2026-09-12.md).
- [docs/loss_audit_2026-09.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/loss_audit_2026-09.md).
- [docs/search-consolidation-validation.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/search-consolidation-validation.md).
- [docs/superpowers/notes/2026-07-06-eval-log-archive.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/superpowers/notes/2026-07-06-eval-log-archive.md).
- [docs/superpowers/notes/2026-07-12-alpha-go-mcts-transcript.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/superpowers/notes/2026-07-12-alpha-go-mcts-transcript.md).
- [docs/superpowers/notes/2026-07-12-value-tuning-and-exit-phase1a-review.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/superpowers/notes/2026-07-12-value-tuning-and-exit-phase1a-review.md).
- [docs/superpowers/notes/2026-07-15-rollout-generation-throughput-investigation.md](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/superpowers/notes/2026-07-15-rollout-generation-throughput-investigation.md).
