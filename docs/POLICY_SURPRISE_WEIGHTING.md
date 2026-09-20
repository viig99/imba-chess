# Per-game policy surprise weighting

The feature is off by default. Enable it in `[learning]` with:

```toml
policy_surprise_enabled = true
policy_surprise_fraction = 0.5
policy_surprise_cap = 3.0
```

`dataset.policy_weights` computes KL(target || stored actor prior) over the
unpadded legal moves of each completed continuation. Stored actor logs and target
mass are re-normalized in double precision; KL at or below `1e-12` is treated as
roundoff, preventing false surprise when the target equals the actor policy.
Zero target probabilities
contribute zero. Positive base policy weights with available priors participate
in the game mean and normalization; missing priors retain multiplier 1. All-zero
surprise also retains multiplier 1. The cap applies before normalization.
`policy_training_weight` is an optional nonnegative per-target replay field,
defaulting to 1; zero excludes that target from policy training and normalization
while retaining its terminal WDL supervision. This does not generate cheap-search
positions, duplicate samples, change replay sampling, or reweight the value head.

`policy_loss` and `policy_target_kl` retain historical unweighted meanings.
`weighted_policy_loss` enters the total objective. The loss divides weighted CE
by the sum of effective weights (base times surprise multiplier); an empty
policy objective returns differentiable zero. Weights are detached.

Training scalar metrics include `eligible_surprise_mean`,
`eligible_surprise_p95`, `missing_actor_prior_fraction`, `policy_weight_mean`,
`policy_weight_max`, `policy_weight_clipping_fraction`, and `policy_weight_ess`.
Mean/max describe the final surprise multiplier among base-eligible positions;
ESS uses effective base-times-surprise weights. Clipping coverage is among
eligible positions with priors. Quantiles describe each training batch, but
weights are computed before batching and never depend on batch neighbors.
The existing scalar/TensorBoard export includes these keys.

Absent settings and explicit default disabled settings preserve the old config
identity. Legacy checkpoint learning settings are filled with defaults before
comparison; changes to any weighting setting reject resume.

## Replay audit and prepared comparison

A read-only audit of the local 128-simulation baseline replay on 2026-09-19 found
31 training games / 2,136 positions, all with actor priors. Surprise mean was
0.66854 and p95 1.96880. Final weight mean was 1.0, p95 1.90821, and maximum
3.01616. Pre-normalization clipping affected 0.3277%; ESS was 1,759.20
(82.36% of positions). This small benchmark replay is a pilot, not a representative
audit of the current remote replay.

Reproduce on another replay without inference or replay mutation:

```sh
.venv/bin/python scripts/audit_policy_surprise.py --replay REPLAY_DIRECTORY
```

The prepared pilot is at `artifacts/policy_surprise/ckpt34-frozen-pilot/`.
It contains copied replay shards, copied ckpt34 weights, copied monitoring seeds,
input hashes, audit JSON, matched configs, and `run.sh`. It uses fresh identical
StableAdamW state, constant LR 1e-4, value scale 1.0, seed 42, 1,024-token
microbatches, and a 4,272-position exposure budget. Whole-game batch rounding may
exceed the budget; the script verifies identical actual exposures, steps,
sampler state, and reuse counts before evaluation. Only surprise enablement
differs between training arms. No optimizer history is restored from ckpt34.

The generated script runs the existing paired evaluator with 512 simulations and
50 color-swapped opening pairs per comparison: each trained model versus ckpt34,
and weighted versus control. Training and strength evaluation have **not been
launched** as part of preparing this pilot. The running remote experiment is
untouched. Execute on an idle device with:

```sh
bash artifacts/policy_surprise/ckpt34-frozen-pilot/run.sh
```

Prepare a larger comparison using `scripts/prepare_policy_surprise_comparison.py`
with `--replay`, `--checkpoint`, `--config`, `--seeds`, and a fresh `--output`.
Its default exposure budget is twice the frozen training position count; use
`--exposures` to choose another matched budget. `--games` must be even.
The copied source replay must remain stable during preparation (immutable shards
and a single captured published manifest are used).

Review paired results and incomplete-game counts before drawing strength
conclusions. Lower weighted loss alone is not evidence of stronger play. Keep
online weighting disabled until the controlled comparison has been reviewed.

## Detached-value self-play default

Self-play defaults to `learning.value_weight = 1.0` and
`learning.detach_value_features = true`. The value head learns from detached
backbone features, so value loss updates the head but not the shared backbone.
Policy loss still updates the backbone and policy head. The two parameter groups
are clipped independently at `learning.grad_clip`; metrics report
`policy_gradient_norm` and `value_gradient_norm` before clipping. The historical
`gradient_norm` key reports the policy/backbone norm in this mode.

Detachment leaves forward predictions unchanged. Stage-1 training is unaffected.
Value predictions can change through both head updates and policy-driven backbone
updates, and therefore still influence subsequent self-play search targets.

Set `detach_value_features = false` for legacy joint training and global clipping.
Set `value_weight = 0` for the frozen-value-head ablation: the head is excluded
from optimizer groups, weight decay, and moments. Older checkpoints without the
detachment field are interpreted as non-detached when checking resume settings;
resume them using an explicit `detach_value_features = false` configuration.
Changing this mode requires a fresh experiment rather than silently resuming.
