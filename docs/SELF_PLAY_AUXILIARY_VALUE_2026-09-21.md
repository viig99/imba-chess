# Auxiliary search-value experiment from flattened update 53,250

This run adds a training-only W/D/L readout to the shared private value features.
The existing outcome-trained readout still supplies every neural value used by
self-play and evaluation search. Terminal nodes continue to use game rules.
This is an experiment, not an established fix for the observed value regression.

## Default update (2026-09-24)

New self-play configurations default to `auxiliary_value_weight = 1.0`, with
`value_weight = 1.0` and `auxiliary_value_lambda = 0.95`. This requires an enabled
auxiliary model head and recorded search W/D/L. Set the auxiliary weight explicitly
to `0.0` for models without that head or policy-only controls.
The historical run below used `0.25`; its artifacts remain unchanged. Changing
the loss weight requires a new run rather than resuming its saved learner state.

## Model and objective

The existing 1024 → 512 projection, two residual value blocks, and final LayerNorm
are shared by two separate 512 → 3 outputs. The original output retains its final
outcome target. The new output has 1,539 parameters and is initialized with zero
weights/bias, so it initially predicts uniform loss/draw/win. All pre-existing
checkpoint tensors are preserved exactly. The extra output is skipped in eval
mode and in both continuation decode paths.

The historical run’s self-play loss was:

`policy_CE + outcome_CE + 0.25 * auxiliary_CE`

The auxiliary target is computed backward over each complete continuation:

`y_t = 0.05 * search_wdl_t + 0.95 * swap(y_(t+1))`

`y_T` is the actual terminal result. `swap` exchanges loss and win, leaving draw
unchanged, because successive chess plies alternate players. Targets are fixed
labels with no gradient through search. Prefix positions provide context but
receive no self-play loss. Smoothing happens before batching, so packing cannot
change a target. Lambda 0.95 gives an untruncated mean offset of 19 plies; it does
not discount a distant terminal result toward a draw.

`search_wdl` is a new, explicit measurement: the mean leaf W/D/L backed up to the
root player over the exact search simulations. Terminal visits use one-hot rule
outcomes; evaluated leaves use the main head, including repeated depth-cutoff
visits. Its win-minus-loss equals the visit-weighted mean root action Q. This
includes exploratory visits, so it is not a claim of optimal-play probabilities.
`root_wdl` remains the raw network prediction and is never substituted for missing
search targets. Scalar-only historical evaluators leave `search_wdl` unavailable;
auxiliary training rejects such replay rather than inventing draw probabilities.

This implements one smoothed auxiliary horizon inspired by KataGo's multi-horizon
auxiliary objectives, not an exact reproduction of KataGo's search or targets.
The original head's draw-blind supervised initialization is preserved; the new
objective does not by itself establish that this limitation is resolved.

## Frozen initialization and run

Artifact root: `artifacts/self_play/flatten-53250-auxiliary-2026-09-21/`.

- Source: `artifacts/checkpoints_v4_flatten_ckpt34_lr/last_checkpoint_53250.pt`.
- Source SHA256: `d426e61ec8bfaef906ddfe54cf869026794bfa5de0ad51c292e4ec2a1e715bd9`.
- Augmented, weights-only initialization: `initial.pt`; hashes and tensor checks
  are recorded in `initialization.json`. The supervised source is untouched.
- `model.toml` enables `model.enable_auxiliary_value_head`; strict checkpoint
  loading remains unchanged. Load auxiliary checkpoints using this model config.
- `config.toml`: 200 simulations, scale 0.1, top-m 16, depth 32; noisy training
  collection and noiseless monitoring; FP32 with TF32 disabled.
- 32 concurrent games, 1024 root-batch tokens, 1024 learner tokens, fresh 4096
  positions per phase, 50,000-position replay window, reuse 2.
- Fresh self-play StableAdamW at constant LR 1e-4, weight decay .01, separate
  value/backbone gradient clipping at 1. No gradient-accumulation change.
- Seed 42 and `artifacts/corpus/v4_self_play_seeds_4096.json`, with its existing
  train/monitor split. Fresh replay; no prior experiment's games are consumed.
- Continuous collection/training with 100-game, 50-pair monitoring every six
  iterations against the unchanged initial actor. Observe-only monitoring;
  no automatic promotion or rollback based on these small screens.
- `launch.json` contains PID, exact command/environment, and resume command.
  `run/state.json`, `run/metrics.jsonl`, replay shards and recovery checkpoints
  are maintained by the existing orchestrator. SIGTERM requests graceful drain.

The previous matched control began at update 51,750, whereas this user-requested
run starts at 53,250. Any causal claim about the auxiliary objective needs a
matched 53,250 baseline and adequate paired evaluation, not a comparison of those
two different starting checkpoints. These monitor openings have been used before;
promotion would require a separate confirmation.

## Verification

Tests cover exact simulation budgets and unchanged search actions/targets,
both players' perspectives, terminal win/draw backups, hand-calculated smoothed
targets and cross-entropy, prefix masking and batch offsets, rejection of missing
targets, auxiliary gradients through shared features, independent main readout,
exact training resume, and changed-config rejection. Legacy disabled-run
identities remain stable.

`gpu-verification.json` records the full-size CUDA check: original versus augmented
initial search is bit-exact for White and Black roots at 200 simulations; two
actual near-mate games complete through production collection; auxiliary training
updates the new parameters; save/reload gives bit-exact search. Smoke replay and
its trained checkpoint are separate from the real run, which starts pristine.
