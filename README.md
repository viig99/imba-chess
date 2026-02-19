# imba-chess

`imba-chess` is a research codebase for pretraining chess sequence models from large-scale, high-Elo Lichess games.

Inspiration:
- https://github.com/noamdwc/grpo_chess
- https://github.com/google-deepmind/searchless_chess

## What is implemented

- Streaming dataset pipeline over `Lichess/standard-chess-games` (Hugging Face).
- Temporal month-window splits for `train` / `val` / `test`.
- Avg-Elo filtering (`(WhiteElo + BlackElo) / 2 >= min_avg_elo`) with optional stricter test filter (`test_min_avg_elo`).
- PGN parsing into per-ply records with board-state tokens.
- Static UCI move vocabulary.
- BOS + event sequence construction for next-move prediction.
- 1D jagged token batches with max-token packing.
- HSTU-style model for next-move prediction.
- Ignite-based training loop (StableAdamW + OneCycleLR, mixed precision, periodic fast val/test + periodic full val, TensorBoard logging, best/last checkpointing).

## Data and training flow

`HF parquet stream -> game parse -> event sequence -> jagged batch -> model -> CE loss`

Each game becomes:
- one BOS token
- one token per ply (state features + previous move + target move)

## Loss and Target Logic

Training uses token-level cross-entropy with two important choices:

- BOS is excluded from loss by construction: BOS target is set to `ignore_index` (`-100`).
- Loss is masked and averaged over valid targets only.
- Label smoothing is configurable via `[model].label_smoothing`.
- Elo-weighted loss is configurable via `[model].elo_weight_min_elo`, `[model].elo_weight_max_elo`, `[model].elo_loss_weight_alpha`, `[model].elo_loss_weight_strength`.

For valid token `i`:

- `ce_i = CE(logits_i, target_i, label_smoothing)`
- `norm_i = clamp((played_by_elo_i - min_elo) / (max_elo - min_elo), 0, 1)`
- `w_i = 1 + strength * (norm_i ^ alpha)`
- `loss = sum_i(w_i * ce_i) / sum_i(w_i)`

Design rationale:

- Label smoothing accounts for non-uniqueness of strong moves.
- Elo weighting biases optimization toward higher-skill move decisions.
- Weight normalization keeps gradient scale stable when weighting is enabled.

## Evaluation Logic

Evaluation is split into fast periodic checks and full periodic checks:

- `fast_val`: runs every `[training].eval_every_steps` over first `[training].fast_val_max_games`.
- `fast_test`: runs every `[training].eval_every_steps` over first `[training].fast_test_max_games`.
- `full_val`: runs every `[training].full_val_every_epochs` over `[dataset].val_max_games`.
- `full_test`: run in `--eval-only` mode over `[dataset].test_max_games`.

Metrics reported:

- `loss_ce`, `ppl`
- `top1_acc`, `top3_acc`, `top5_acc`, `hr@10` (`top10_acc`)
- `mrr`
- `token_count`, `game_count`

Checkpointing:

- Best checkpoints are selected by `hr@10` from `full_val`.
- Last checkpoints are saved periodically by step cadence.

Resume behavior:

- On `--resume`, trainer restores model/optimizer/scheduler/scaler/trainer state.
- Immediate `fast_val` and `fast_test` are run once after resume for quick health check.

## Configuration

All runtime settings are in `config/imba_chess.toml`.

Main sections:
- `[dataset]` source, month windows, max games for val/test, optional stricter test Elo, cache, sequence truncation
- `[board_state]` board-state encoding buckets/options
- `[vocab]` static move vocab location
- `[dataloader]` max tokens per jagged batch, workers
- `[model]` HSTU dimensions/layers/head settings + label smoothing + Elo loss weighting
- `[training]` optimizer/scheduler/eval cadence/checkpointing/device/precision (including fast test cadence)

## Quickstart

```bash
uv sync --python 3.13
source .venv/bin/activate
```

Build or load static move vocab:

```bash
python scripts/build_static_move_vocab.py
```

Preview parsed dataset samples:

```bash
python scripts/preview_dataset.py
```

Inspect jagged dataloader batches:

```bash
python scripts/test_event_dataloader.py
```

Run HSTU forward/backward benchmark:

```bash
python scripts/test_hstu_forward.py --device cuda --dtype bfloat16 --compile
```

Start training:

```bash
python scripts/train.py --device cuda --dtype bfloat16 --compile
```

Resume training:

```bash
python scripts/train.py --resume artifacts/checkpoints/last_*.pt
```

Eval only:

```bash
python scripts/train.py --eval-only --resume artifacts/checkpoints/best_hr10_*.pt --eval-split both
```

Run tests:

```bash
uv run --python .venv/bin/python --with pytest pytest -q
```

## Current limitations

- Training is currently single-process in this repo flow (no end-to-end DDP launcher yet).
- No legal-move masking in the prediction head yet (full-vocab classification).
- Streaming order is temporal by month window (newest month first), not global random shuffle.

## References

- `PLAN.md` for roadmap.
- `EVAL_SPEC.md` for evaluation design.
- `TRAINING_EVENT_SCHEMA.md` for input schema details.
- `FEN_TO_BOARD_STATE.md` for board-state encoding details.
