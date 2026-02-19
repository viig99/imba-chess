# imba-chess

`imba-chess` is a research codebase for pretraining chess sequence models from large-scale, high-Elo Lichess games.

Inspiration:
- https://github.com/noamdwc/grpo_chess
- https://github.com/google-deepmind/searchless_chess

## What is implemented

- Streaming dataset pipeline over `Lichess/standard-chess-games` (Hugging Face).
- Temporal month-window splits for `train` / `val` / `test`.
- Avg-Elo filtering (`(WhiteElo + BlackElo) / 2 >= min_avg_elo`).
- PGN parsing into per-ply records with board-state tokens.
- Static UCI move vocabulary.
- BOS + event sequence construction for next-move prediction.
- 1D jagged token batches with max-token packing.
- HSTU-style model for next-move prediction.
- Ignite-based training loop (StableAdamW + OneCycleLR, mixed precision, periodic fast/full validation, TensorBoard logging, best/last checkpointing).

## Data and training flow

`HF parquet stream -> game parse -> event sequence -> jagged batch -> model -> CE loss`

Each game becomes:
- one BOS token
- one token per ply (state features + previous move + target move)

## Configuration

All runtime settings are in `config/imba_chess.toml`.

Main sections:
- `[dataset]` source, month windows, max games for val/test, cache, sequence truncation
- `[board_state]` board-state encoding buckets/options
- `[vocab]` static move vocab location
- `[dataloader]` max tokens per jagged batch, workers
- `[model]` HSTU dimensions/layers/head settings
- `[training]` optimizer/scheduler/eval cadence/checkpointing/device/precision

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
