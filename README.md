# imba-chess

`imba-chess` is a research codebase for pretraining chess sequence models from large-scale, high-Elo Lichess games, and for playing them against Stockfish with value-guided move selection at inference time.

Inspiration:
- https://github.com/noamdwc/grpo_chess
- https://github.com/google-deepmind/searchless_chess

## What is implemented

- Streaming dataset pipeline over `Lichess/standard-chess-games` (Hugging Face).
- Temporal month-window splits for `train` / `val` / `test`.
- Avg-Elo filtering (`(WhiteElo + BlackElo) / 2 >= min_avg_elo`) with optional stricter test filter (`test_min_avg_elo`).
- PGN parsing into per-move records with board-state tokens.
- Static UCI move vocabulary.
- BOS + event sequence construction for next-move prediction.
- 1D jagged token batches with max-token packing.
- HSTU-style transformer with two heads: next-move classification and win/draw/loss prediction.
- Ignite-based training loop (StableAdamW + OneCycleLR, mixed precision, periodic fast val/test + periodic full val, TensorBoard logging, best/last checkpointing).
- Head-to-head engine evaluation (`scripts/eval_vs_stockfish.py`) with value-guided lookahead search at inference.

## Data and training flow

`HF parquet stream -> game parse -> event sequence -> jagged batch -> model -> loss`

Each game becomes:
- one BOS token
- one token per move: the board state before the move (piece placement, turn, castling rights, en passant, clocks) + the previous move id, with the played move as the classification target
- one per-game outcome label `game_result_white` in `{+1, 0, -1}`

## Training objectives

One transformer trunk, two linear heads, trained jointly:

```
total_loss = policy_loss + [model].value_loss_weight * value_loss
```

### Policy head: next-move classification

Token-level cross-entropy against the move the human actually played (full move-vocab softmax):

- BOS is excluded from loss by construction (target set to `ignore_index = -100`).
- Label smoothing (`[model].label_smoothing`) accounts for positions where several moves are equally good.
- Each token is weighted by the Elo of the player who made that move, so stronger players' moves pull the gradient harder:
  - `norm_i = clamp((played_by_elo_i - min_elo) / (max_elo - min_elo), 0, 1)`
  - `w_i = 1 + strength * (norm_i ^ alpha)`
  - `policy_loss = sum_i(w_i * ce_i) / sum_i(w_i)`

This is pure imitation learning: no reward signal, no self-play.

### Value head: win/draw/loss classification

When `[model].enable_value_head = true`, a 3-class head is trained to predict the final result of the game from every position, from the perspective of the player about to move:

- The label for every position in a game is that game's final outcome (`game_result_white`, flipped by `turn_id`). The head therefore learns "among training games that passed through positions like this, how often did the side to move end up winning?"
- The target itself is not discounted, but the per-token loss is weighted by game progress (`progress ^ [model].value_weight_alpha`, `progress` in `[0, 1]`): the final outcome is a noisy label for early positions and a clean one for late positions, so early positions contribute little gradient and the last positions contribute full gradient.
- 3-class classification is deliberate (rather than a scalar regression head): win/draw/loss outcomes are genuinely 3-modal — a scalar `0.0` cannot distinguish "certain draw" from "unclear, 50/50 win-or-lose" — and cross-entropy on categories optimizes better than MSE on a bounded scalar. A scalar is recovered at inference as `v = p(win) - p(loss)` in `[-1, 1]`.

Known limitation: game outcomes are high-variance Monte-Carlo labels (a winning position that the player later threw away gets labeled "loss"). Replacing them with engine-annotated position evaluations is the planned upgrade.

Training logs include `total_loss`, `policy_loss`, and `value_loss`.

## Evaluation during training

- `fast_val` / `fast_test`: every `[training].eval_every_steps` over the first `fast_val_max_games` / `fast_test_max_games`.
- `full_val`: every `[training].full_val_every_epochs` over `[dataset].val_max_games`.
- `full_test`: in `--eval-only` mode over `[dataset].test_max_games`.

Metrics: `loss_ce`, `ppl`, `top1/top3/top5_acc`, `hr@10`, `mrr`, `token_count`, `game_count`.

Best checkpoints are selected by `hr@10` from `full_val`; last checkpoints are saved by step cadence. On `--resume`, model/optimizer/scheduler/scaler/trainer state are restored and an immediate `fast_val`/`fast_test` health check runs.

## Playing against Stockfish

`scripts/eval_vs_stockfish.py` plays full games against Stockfish over UCI, either at full strength or Elo-limited, single-segment or as a ladder across several Elo levels. Defaults come from `[eval_vs_stockfish]` in `config/imba_chess.toml`; CLI flags override.

`model_move_policy` modes:

- `greedy`: play the highest-logit legal move.
- `sample`: sample from legal moves with temperature/top-k/top-p.
- `value_rerank`: propose top-K moves with the policy head, grade each by the value head after the move, pick the best grade.
- `value_search_d2`: same, but each proposal is stress-tested against the opponent's best response before grading (see below).

`value_rerank` and `value_search_d2` require a checkpoint trained with the value head enabled.

### How value-guided move selection works

The policy head alone is autocomplete: every move is a single forward pass and nothing ever checks the consequences, so a human-looking move that loses material to a tactic gets played anyway. The search adds the most basic form of thinking ahead: **"if I play this, what is the worst thing my opponent can do to me right after?"**

Per model turn, `value_search_d2` runs three batched forward passes:

1. **Propose (1 sequence).** Encode the real game history, take the policy logits at the last token, mask to legal moves, `log_softmax`. The top `value_rerank_top_k` moves are our candidates.
2. **Opponent responses (≤ K sequences, 1 batch).** For each candidate, simulate playing it (copy the board, append the move token to a copy of the history) and run the model once over all candidates to get the opponent's move distribution in each hypothetical position. Opponent responses considered per position: their policy top-K **plus every capture, check, and promotion** — the refutation of a bad move is often a move the human-imitation policy ranks low, so probability-based pruning alone would hide exactly what we are testing for.
3. **Grade (~K × (K + forcing) sequences, chunked batches).** Apply each response, evaluate all resulting positions with the value head, and collapse each to a scalar `v = p(win) - p(loss)` from our perspective.

Each candidate is then scored pessimistically — assume the opponent picks their best response — with the policy prior as a tiebreaker:

```
grade(move)  = min over responses of v(position after move, response)
score(move)  = grade(move) + lambda * log_prob(move)      # lambda = value_rerank_lambda
play argmax(score)
```

The value head decides; the policy log-prob (default `lambda = 0.1`) breaks near-ties toward moves strong humans actually play, which also guards against value-head noise.

Special cases bypass the network:

- Game-over positions (checkmate, stalemate, claimable draws by repetition or the 50-move rule) are scored with the exact result (+1 / 0 / −1) instead of the value head — final positions never occur as training inputs, so the head's output there is undefined.
- If a candidate move immediately wins the game, it is played without further search.
- Child boards keep the move stack so repetition draws are actually detected in simulated lines.

Batched evaluations are chunked to at most 4096 tokens per forward (`_SEARCH_EVAL_MAX_TOKENS_PER_CHUNK`): the non-compiled attention fallback materializes O(T²) tensors, and one merged batch of ~300 sequences OOMs on an 8 GB GPU.

Cost: 3 model calls and ~K² positions per turn instead of 1 call — roughly 30–50s per game instead of ~2s, buying back the consequence-checking that pure imitation lacks.

`value_rerank` is the depth-1 version of the same idea (grade positions immediately after our move, no opponent response), with the same value-dominant scoring.

### Usage

Basic match (policy defaults from TOML):

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --games 1000 \
  --output-json artifacts/eval/stockfish_eval.json
```

Value search against Elo-limited Stockfish:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --model-move-policy value_search_d2 \
  --value-rerank-top-k 16 \
  --value-rerank-lambda 0.1 \
  --stockfish-limit-strength --stockfish-elo 1400 \
  --games 100
```

Ladder eval across several Stockfish levels:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --ladder-elos 1400,1600,1800,2000,2200 \
  --ladder-games-per-segment 200 \
  --include-full-strength-segment \
  --output-json artifacts/eval/stockfish_ladder.json
```

The script reports wins/draws/losses (with color split), completed/incomplete games, average game length, score rate, legal-move vocab coverage, and per-segment plus aggregate summaries in ladder mode.

### Results vs Stockfish 1400 (sweep in progress)

Setup: checkpoint `best_hr10_checkpoint_5` (hr@10 = 0.9208), Stockfish `UCI_Elo` 1400 at 0.05s/move, 100 games per configuration, seed 42, colors alternating. Score = (wins + 0.5 × draws) / games; ±~0.05 standard error at 100 games.

| Move selection | λ | W / D / L | Score rate |
|---|---|---|---|
| `value_rerank`, old policy-dominant scoring (best of λ sweep) | 0.35 | 12 / 22 / 66 | 0.230 |
| `value_search_d2`, value-dominant scoring, K=16 | 0.05 | 20 / 41 / 39 | **0.405** |
| `value_search_d2`, value-dominant scoring, K=16 | 0.10 | *running* | — |
| `value_search_d2`, value-dominant scoring, K=16 | 0.20 | *queued* | — |

The jump from 0.23 to 0.405 (~+140 Elo vs the same opponent) is inference-only: same checkpoint, fixed search scoring (value-first with policy log-prob tiebreak), exact terminal handling, and forcing-move opponent replies. Draw share roughly doubled — the search stops losing many previously lost games; converting draws to wins is the next frontier (value head endgame quality).

## Configuration

All runtime settings are in `config/imba_chess.toml`:

- `[dataset]` source, month windows, max games for val/test, Elo filters, cache, sequence truncation
- `[board_state]` board-state encoding buckets/options
- `[vocab]` static move vocab location
- `[dataloader]` max tokens per jagged batch, workers
- `[model]` HSTU dimensions/layers + label smoothing + Elo loss weighting + value head knobs
- `[training]` optimizer/scheduler/eval cadence/checkpointing/device/precision
- `[eval_vs_stockfish]` engine path/limits, ladder settings, move-selection policy and knobs, debug controls

## Quickstart

```bash
uv sync --python 3.13
source .venv/bin/activate

# Build or load static move vocab
python scripts/build_static_move_vocab.py

# Preview parsed dataset samples / inspect jagged batches
python scripts/preview_dataset.py
python scripts/test_event_dataloader.py

# Estimate corpus size / cache footprint for the configured windows
python scripts/estimate_lichess_cache.py --split all --target-free-gib 40

# Train
python scripts/train.py --device cuda --dtype bfloat16 --compile

# Resume / eval-only
python scripts/train.py --resume artifacts/checkpoints/last_*.pt
python scripts/train.py --eval-only --resume artifacts/checkpoints/best_hr10_*.pt --eval-split both

# Tests
uv run --python .venv/bin/python --with pytest pytest -q
```

## Current limitations

- Training is single-process (no end-to-end DDP launcher yet).
- No legal-move masking in the prediction head during training (full-vocab classification); legality is enforced at inference.
- Lookahead is fixed at depth 2 (our move + opponent response); no deeper tree or MCTS yet.
- Value labels are raw game outcomes, not engine evaluations (noisy for early positions).
- No time-control filter on training data: bullet/blitz games pass the Elo filter and carry more tactical mistakes.
- Streaming order is temporal by month window (newest first); month-level file order can be shuffled at process start via `[dataset].shuffle_train_month_files_on_start`.

## References

- `PLAN.md` for roadmap.
- `EVAL_SPEC.md` for evaluation design.
- `TRAINING_EVENT_SCHEMA.md` for input schema details.
- `FEN_TO_BOARD_STATE.md` for board-state encoding details.
- `VALUE_HEAD_OPTIONS.md` for value-head design notes.
