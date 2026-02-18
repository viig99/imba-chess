# imba-chess

`imba-chess` is a research codebase for training chess sequence models from large-scale high-Elo game data, with a focus on practical pretraining pipelines and later RL fine-tuning.

## Why this repo exists

The goal is to build a clean, reproducible path from:

1. streaming raw Lichess games,
2. converting them into model-ready event sequences,
3. training sequential transformers for next-move prediction,
4. and extending toward GRPO-style post-training.

Inspiration:
- https://github.com/noamdwc/grpo_chess
- https://github.com/google-deepmind/searchless_chess

## Current scope

Today, this repo focuses on the data and input pipeline (not full model training yet):

- Streams `Lichess/standard-chess-games` directly from Hugging Face.
- Filters games by average Elo threshold (default: `>= 2000`, configurable).
- Parses PGN and builds per-ply board state features.
- Builds static UCI move vocabulary.
- Creates BOS+ply event sequences for next-move prediction.
- Packs games into 1D jagged batches (`seq_lens`/`seq_offsets`) with a configurable max-token budget.
- Supports PyTorch iterable loading with rank/worker sharding.

## High-level data flow

`Lichess row -> parsed game -> per-ply board/event features -> jagged batch tensors`

Each game contributes:
- one BOS step,
- then one step per ply with state features and target move ID.

## Configuration

Runtime knobs are centralized in:

- `config/imba_chess.toml`

Sections:
- `[dataset]` dataset source + filters + streaming behavior
- `[board_state]` board-state tokenization settings
- `[vocab]` move-vocab path and options
- `[dataloader]` max tokens per batch, workers, pin memory, distributed fields

## Quickstart

```bash
uv sync --python 3.13
source .venv/bin/activate
```

Preview parsed games:

```bash
python scripts/preview_dataset.py
```

Build static move vocab:

```bash
python scripts/build_static_move_vocab.py
```

Inspect jagged event dataloader output:

```bash
python scripts/test_event_dataloader.py
```

Run tests:

```bash
uv run --python .venv/bin/python --with pytest pytest -q
```

## Project status

The dataset + event pipeline is working and covered by tests.  
Model training modules (transformer architecture, trainer loop, RL stage) are planned next.
