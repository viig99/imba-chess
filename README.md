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
- Head-to-head engine evaluation script (`scripts/eval_vs_stockfish.py`) for model vs Stockfish matches.

## Data and training flow

`HF parquet stream -> game parse -> event sequence -> jagged batch -> model -> CE loss`

Each game becomes:
- one BOS token
- one token per ply (state features + previous move + target move)
- one per-game value label `game_result_white` in `{+1, 0, -1}`

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

### Optional value head (WDL)

When `[model].enable_value_head = true`, training adds a 3-class value head:

- classes are from side-to-move perspective: `loss / draw / win`
- labels are derived from per-game `game_result_white` and per-token `turn_id`
- value loss uses progress weighting toward later plies (`progress ^ [model].value_weight_alpha`)
- total loss becomes:
  - `total_loss = policy_loss + [model].value_loss_weight * value_loss`

Training logs now include:

- `total_loss`
- `policy_loss`
- `value_loss` (logged when value head is enabled)

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
- `[model]` HSTU dimensions/layers/head settings + label smoothing + Elo loss weighting + optional value head knobs
- `[training]` optimizer/scheduler/eval cadence/checkpointing/device/precision (including fast test cadence)
- `[eval_vs_stockfish]` defaults for engine path/limits, ladder settings, decoding policy, and debug controls

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

Estimate high-Elo corpus and cache footprint:

```bash
python scripts/estimate_lichess_cache.py \
  --all-months \
  --min-avg-elo 2000 \
  --sample-parquets 16 \
  --sample-rows-per-parquet 200000 \
  --cache-dir artifacts/hf_cache \
  --output-json artifacts/eval/elo2000_cache_estimate.json
```

Budget planning on configured TOML time windows (`train/val/test`) with Elo + time recommendations:

```bash
python scripts/estimate_lichess_cache.py \
  --split all \
  --target-free-gib 40 \
  --sample-parquets 16 \
  --sample-rows-per-parquet 200000 \
  --output-json artifacts/eval/budget40gib_estimate.json
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

Enable value head training (example):

```toml
[model]
enable_value_head = true
value_loss_weight = 0.15
value_weight_alpha = 1.5
value_label_smoothing = 0.0
```

Resume training:

```bash
python scripts/train.py --resume artifacts/checkpoints/last_*.pt
```

Eval only:

```bash
python scripts/train.py --eval-only --resume artifacts/checkpoints/best_hr10_*.pt --eval-split both
```

Model vs Stockfish eval:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --games 1000 \
  --stockfish-path /usr/bin/stockfish \
  --stockfish-time-sec 0.05 \
  --device cuda --dtype bfloat16 \
  --output-json artifacts/eval/stockfish_eval.json
```

`scripts/eval_vs_stockfish.py` can also read defaults from `[eval_vs_stockfish]` in `config/imba_chess.toml`.
CLI flags override TOML values when provided.

`model_move_policy` modes:

- `greedy`: pick highest-logit legal move.
- `sample`: sample from legal moves with temperature/top-k/top-p.
- `value_rerank`: rerank top-K policy legal moves using one-ply value lookahead.
- `value_search_d2`: run policy-pruned depth-2 value search (our move, opponent best reply).

Value-search knobs (`[eval_vs_stockfish]` or CLI):

- `value_rerank_top_k` (default `8`)
- `value_rerank_lambda` (default `0.35`)

Important: `value_rerank` and `value_search_d2` require a checkpoint trained with value head and a runtime model config with `[model].enable_value_head = true`.

Example:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/last_*.pt \
  --model-move-policy value_rerank \
  --value-rerank-top-k 8 \
  --value-rerank-lambda 0.35
```

Depth-2 value search example:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/last_*.pt \
  --model-move-policy value_search_d2 \
  --value-rerank-top-k 8 \
  --value-rerank-lambda 0.35
```

Stockfish Elo-limited mode:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --games 200 \
  --stockfish-limit-strength --stockfish-elo 2400
```

Segmented ladder eval (phase 2):

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --ladder-elos 1600,1800,2000,2200,2400,2600,2800 \
  --ladder-games-per-segment 200 \
  --include-full-strength-segment \
  --stockfish-time-sec 0.05 \
  --output-json artifacts/eval/stockfish_ladder.json
```

The script reports:
- total/completed/incomplete games
- wins/draws/losses (+ color split)
- average plies/full moves per game
- score rate on completed games and on all games
- in ladder mode: per-segment results and an aggregate summary across all segments

Run tests:

```bash
uv run --python .venv/bin/python --with pytest pytest -q
```

## Current limitations

- Training is currently single-process in this repo flow (no end-to-end DDP launcher yet).
- No legal-move masking in the prediction head yet (full-vocab classification).
- `value_rerank` is one-ply value lookahead.
- `value_search_d2` is depth-2 and substantially slower than greedy/sample/value_rerank.
- Streaming order is temporal by month window (newest month first); training can optionally shuffle month-level parquet file order on process start via `[dataset].shuffle_train_month_files_on_start`.

## References

- `PLAN.md` for roadmap.
- `EVAL_SPEC.md` for evaluation design.
- `TRAINING_EVENT_SCHEMA.md` for input schema details.
- `FEN_TO_BOARD_STATE.md` for board-state encoding details.
