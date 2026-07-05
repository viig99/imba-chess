# Standalone Value Network (Stockfish Distillation) — Design

## Purpose

The SF1800 budget-scaling curve flattened (0.465 @ 256 → 0.560 @ 512/d6 →
0.595 @ 1024/d6): search compute no longer converts into strength, which
means the value oracle — a head trained on noisy whole-game outcomes — is
the binding constraint. This builds a **separate, position-only value
network** trained on `Lichess/chess-position-evaluations` (388M
Stockfish-annotated FENs, CC0, HF parquet) and plugs it into search as a
config-gated replacement/blend for the model's value head.

Why separate rather than a shared head (decided over warm-start finetune,
from-scratch joint training, and head-only finetune):

- Perfect data-shape match: the eval DB is history-free FENs, and a
  position-only net has zero train/inference history mismatch (chess value
  is Markov in the state token; repetition is already handled exactly by
  the search's terminal rules).
- Training is fully decoupled from the policy runs: no mixing ratios, no
  forgetting risk, plain flat-batch supervised learning.
- The net becomes reusable infrastructure: v4, v5, and a future ExIt loop
  all consume the same value oracle; either side retrains independently.
- De-risked by precedent (DeepMind searchless_chess: position-only value
  transformers at modest scale play strong chess).

The shared-trunk mixed-stream option remains open later; nothing here
forecloses it.

## Part 1: Model — `src/imba_chess/model/value_net.py`

Maximal reuse; no sequence machinery, no jagged batching, no position
embeddings.

```
ValueNetConfig (frozen dataclass):
    dim: int = 256          # square-token width
    num_heads: int = 4
    num_layers: int = 6
    halfmove_vocab_size: int = 128
    fullmove_vocab_size: int = 128
```

`ValueNet(nn.Module)`:

- `piece_square_embedding = nn.Embedding(13 * 64, dim)` — same joint
  (piece, square) scheme as `HSTUChessModel._embed_board` (pair id =
  `piece_id * 64 + square`).
- Scalar state features — turn (2), castle (16), ep file (9), halfmove
  bucket, fullmove bucket — each an `nn.Embedding(*, dim)` with the same
  id-clamping the big model uses; their sum is **broadcast-added to all 64
  square tokens** before the encoder so side-to-move/castling can interact
  with square content.
- Body: **`BoardSquareEncoder` imported from `hstu_model` and reused
  unchanged** — its constructor is already fully parameterized:
  `BoardSquareEncoder(dim=cfg.dim, num_heads=cfg.num_heads,
  num_layers=cfg.num_layers, out_dim=cfg.dim)`, pooling to `[B, dim]`.
- Head: `Linear(dim, dim // 2) → SiLU → Linear(dim // 2, 3)` — WDL logits,
  same shape/order convention as the existing value head (index 0 = loss,
  1 = draw, 2 = win, side-to-move POV).
- `forward(batch: dict) -> logits [B, 3]` consuming exactly the id-tensor
  keys the eval script already builds for decode waves: `piece_ids
  [B, 64]`, `turn_id`, `castle_id`, `ep_file_id`, `halfmove_bucket_id`,
  `fullmove_bucket_id` (all `[B]`). `prev_move_id`/`seq_token_id` are
  ignored — the net is position-only.

Default size ≈ 5M parameters.

## Part 2: Data — `src/imba_chess/data/position_eval_dataset.py`

Streams `Lichess/chess-position-evaluations` with the same
`load_dataset(path="parquet", streaming=True)` + file-level worker
sharding idioms as `lichess_dataset.py`.

Per row (`fen`, `cp`, `mate`, `depth`, ...):

1. Filter: `depth >= depth_min` (config, default 12). Rows with neither
   cp nor mate are dropped. No dedup (repeat evaluations of popular
   positions act as mild importance weighting; accepted for v1).
2. Parse `chess.Board(fen)`; encode via the existing `BoardStateEncoder`
   → the same id fields the model consumes.
3. Build the soft WDL target, **side-to-move POV**:
   - Lichess `cp`/`mate` are **White-POV**; flip sign when the FEN's side
     to move is Black. This conversion gets a dedicated unit test (classic
     silent-bug site).
   - cp rows: 3-class probabilities from the fitted calibration curve
     (below).
   - mate rows: near-saturated target for the mating side — win-side mass
     0.995, remaining 0.005 split evenly across the other two classes,
     from the side-to-move POV; never routed through the cp curve.
4. Emit `{piece_ids, turn_id, castle_id, ep_file_id, halfmove_bucket_id,
   fullmove_bucket_id, wdl_target [3]}` — flat tensors, standard
   fixed-size batches, no packing.

Holdout split: deterministic FEN-hash (e.g. `hash(fen) % 1000 < 5` → val).

### cp→WDL conversion — Stockfish's own win-rate model, hardcoded

A pure function `cp_to_wdl(cp, fullmove) -> (p_loss, p_draw, p_win)` using
the published Stockfish `win_rate_model` polynomial (constants vendored
with a comment naming the SF version they came from): `p_win` from the
polynomial at `cp`, `p_loss` from the same polynomial at `-cp` (symmetry),
`p_draw = 1 − p_win − p_loss`. No annotation run, no fitted artifact.

Rationale: the search consumes values ordinally, so any monotone curve
preserves arm ranking — calibration precision only matters where net values
meet exact terminal values and in draw-vs-slightly-worse comparisons.
Fitting against our own game outcomes was rejected: it would calibrate cp
to *human* conversion rates, re-importing exactly the label noise this
project removes. SF's model is calibrated on engine self-play ("value
under strong play"), which is the intended semantics. Empirical
re-calibration is a future refinement only if eval evidence shows
miscalibration at the seams.

## Part 3: Training — `scripts/train_value_net.py`

Lean standalone script (plain loop, no Ignite):

- StableAdamW + OneCycleLR + bf16 autocast + grad clip — same optimizer
  family and settings style as the main trainer.
- Loss: soft-label cross-entropy (`-(target * log_softmax(logits)).sum()`)
  against the 3-vector targets.
- Periodic held-out validation (soft-CE + hard accuracy vs argmax bucket);
  best checkpoint by val soft-CE, last by cadence; TensorBoard logging.
- Config section `[value_net]` in `config/imba_chess.toml`: model dims,
  `depth_min`, batch size, lr, steps, workers, checkpoint dir
  (`artifacts/value_net/`).

## Part 4: Inference integration

`CachedPositionEvaluator` gains an optional value net. The wave's
`new_token_batch` tensors are already exactly what `ValueNet.forward`
consumes, so the cost is one extra small batched forward per wave (64-token
inputs; a fraction of one decode step). Blending:

```
value_stm = (1 - alpha) * value_from_model_head + alpha * value_net_scalar
# value_net_scalar = p(win) - p(loss) from the net's softmax, as today
```

Config/CLI (existing config-with-override pattern):

- `value_net_checkpoint` (optional path, default unset) — **when unset,
  behavior is byte-identical to today**; the net is only loaded/used when
  provided.
- `value_net_alpha` (default 1.0) — pure value net when a checkpoint is
  given; `0.0`/`1.0` are the pure endpoints so no separate switch exists.

Both recorded in the output JSON `run_config`. All policies (rerank, d2,
halving) get the blend for free through the evaluator; the search module
(`src/imba_chess/eval/search.py`) does not change. Terminal positions keep
their exact values (the blend applies only to value-head evaluations).

## Testing

- ValueNet forward shapes + determinism on fixed seeds (no torch model
  needed beyond the tiny net itself).
- POV flip: mirrored FENs (same position, colors swapped) must produce
  sign-flipped cp→target conversions; a Black-to-move mate-in-1 row maps
  to the correct saturated side.
- cp→WDL: monotone in cp; probabilities sum to 1; symmetric
  (`cp_to_wdl(cp)` reversed equals `cp_to_wdl(-cp)`); mate rows bypass the
  curve.
- Dataset streaming: a scripted parquet fixture through filter/encode/
  target path.
- Evaluator blend: with a stub value net, `alpha=0` reproduces the
  model-head values exactly; `alpha=1` reproduces the net's; `0.5` is the
  mean. No-checkpoint path byte-identical (existing eval tests unchanged).

## Acceptance protocol

Same SF1800 eval as the budget curve: 100 games, seed 42, halving
1024/depth 6, `--value-net-checkpoint ... --value-net-alpha 1.0` vs the
0.595 baseline; one follow-up point at `alpha 0.5`. Secondary metric:
held-out val soft-CE of the net itself during training.

## Out of scope (v1)

- Mate-distance auxiliary head (exact moves-left labels) — noted future.
- Depth-weighted loss (depth filter only).
- Empirically fitted cp→WDL calibration (hardcoded SF win-rate model only).
- Syzygy tablebase integration (separate, orthogonal quick win).
- Any change to the big model, its training run, or the search module.
- Shared-trunk mixed-stream training (kept open as a later option).
