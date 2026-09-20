# Flattened-board training and checkpoint promotion

The permanent model uses square attention → per-square LayerNorm → flatten
64×64 features → Linear(4096, model_dim). There is no mean/flatten configuration
switch or alternate forward path. The v4 model has 52,388,996 unique parameters.

`model/checkpoint.py` loads compatible raw, compiled or distributed model weights
strictly. There is no runtime architecture conversion. Original mean-pooled
checkpoints are incompatible; the verified, function-equivalent flattened copy
of ckpt34 is retained at `artifacts/flatten-board-ckpt34/initial.pt` for comparisons.
Use native flattened checkpoints for future training, resume and evaluation.

## Training

New models automatically use flattening. A laptop warm start from ckpt34 is:

```bash
.venv/bin/python -m scripts.train \
  --config config/imba_chess_v4_laptop.toml \
  --init-weights artifacts/flatten-board-ckpt34/initial.pt \
  --max-steps 10000
```

The laptop config uses ckpt34's saved learning rate, 0.0004151649149149149,
1280-token batches (reduced from 1536 for GPU memory headroom), BF16 autocast, FP32 parameters, and the original supervised
objectives. Optimizer state is fresh; this is not exact continuation of ckpt34's
optimizer or decaying schedule. The configured rate is constant. Shared features
receive both policy and value gradients in supervised training.

The active run was launched before code simplification and already has the same
flattened architecture in memory. Its results are in
`artifacts/checkpoints_v4_flatten_ckpt34_lr`. Its launch config is preserved as
provenance; future resumes should use `config/imba_chess_v4_laptop.toml` or the
compatible `resume-config.toml` beside its checkpoints, since the retired pooling
switch is no longer a configuration key. Resume native checkpoints with
`--resume`, omitting `--init-weights`. Streaming data cursors are not restored.

The initial 1e-6 smoke/continuation branch is archived separately and is not the
active candidate. Its small validation improvement did not establish a strength
improvement. All migration utilities/configs from that transition are archived
under `artifacts/flatten-board-ckpt34/transition-source`; there is no separate
migration CLI in the active codebase.

## SF2400 gate before checkpoint promotion

Keep ckpt34 as the default self-play/experiment starting checkpoint until the
trained candidate has completed the comparison. Use the existing evaluator;
no new search or promotion framework is needed.

The adopted historical recipe is recorded in
`artifacts/eval/ckpt34_overnight_20260905/confirm_r4_b2048_q0_750.json`:
750 completed games; 383 wins / 223 draws / 144 losses; score 0.659333.
The exact protocol is pinned in `config/eval_flatten_sf2400.toml`:

- Value-search halving: budget 2048, top-m 16, automatic rounds, lambda .05,
  four opponent replies, three own continuations, depth 8, tactical coverage
  disabled, quiescence 0.
- Stockfish strength limited to Elo 2400, one thread, 64 MiB hash, 40,000 nodes
  with the original five-second safety ceiling.
- Seed 1042, four concurrent games, no random opening plies, 512-ply safety cap;
  CUDA FP32 with TF32 disabled.

Run both the frozen, equivalent flattened ckpt34 copy and a frozen candidate through the **same current runtime**
and engine binary. The historical runtime used a different compilation path, so
its 65.93% score is a reference, not a substitute for a fresh matched baseline.
Example (replace CANDIDATE.pt with the chosen completed-training checkpoint):

```bash
.venv/bin/python -m scripts.eval_vs_stockfish \
  --config config/eval_flatten_sf2400.toml \
  --checkpoint artifacts/flatten-board-ckpt34/initial.pt \
  --output-json artifacts/eval/flatten-sf2400/baseline.json

.venv/bin/python -m scripts.eval_vs_stockfish \
  --config config/eval_flatten_sf2400.toml \
  --checkpoint CANDIDATE.pt \
  --output-json artifacts/eval/flatten-sf2400/candidate.json
```

Freeze candidate/config/source/engine hashes and keep per-game results. Require
complete games, compare score and uncertainty, and distinguish inconclusive
results from evidence of non-regression. No positive regression tolerance has
been authorized; an overlapping interval alone is not a pass. Do not promote on
training loss, a partial match, or an unpaired historical point estimate.

After the gate passes, make the frozen winning checkpoint the documented default
for self-play and experiments, recording its hash and evaluation evidence. Remove
any remaining disposable transition artifacts only then; retain source checkpoint,
provenance and results needed to reproduce the comparison. That promotion and final
artifact cleanup are pending; neither is implied by the architecture change.
