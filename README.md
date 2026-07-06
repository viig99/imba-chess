# imba-chess

`imba-chess` is a research codebase for pretraining chess sequence models from large-scale, high-Elo Lichess games, and for playing them against Stockfish with value-guided move selection at inference time.

Inspiration:
- https://github.com/noamdwc/grpo_chess
- https://github.com/google-deepmind/searchless_chess

## What is implemented

- Streaming dataset pipeline over `Lichess/standard-chess-games` (Hugging Face).
- Temporal month-window splits for `train` / `val` / `test`.
- Avg-Elo filtering (`(WhiteElo + BlackElo) / 2 >= min_avg_elo`) with optional stricter test filter (`test_min_avg_elo`).
- Time-control filtering (`min_time_control_sec`, estimated duration = base + 40 × increment) to drop bullet games full of tactical mistakes.
- PGN parsing into per-move records with board-state tokens.
- Static UCI move vocabulary: all geometrically reachable from→to pairs + promotions (1,970 tokens incl. specials) — provably covers every legal standard-chess move.
- Placement-aware board encoding: a joint (piece, square) embedding table, mean-pooled per position (an additive piece+square scheme collapses to a bag of material under pooling).
- BOS + event sequence construction for next-move prediction.
- 1D jagged token batches with max-token packing.
- HSTU-style transformer with two heads: next-move classification and win/draw/loss prediction.
- Ignite-based training loop (StableAdamW + OneCycleLR, mixed precision, periodic fast val/test + periodic full val, TensorBoard logging, best/last checkpointing).
- Head-to-head engine evaluation (`scripts/eval_vs_stockfish.py`) with pluggable value-guided search at inference (`src/imba_chess/eval/search.py`): depth-2 minimax and budgeted sequential-halving tree search (MCTS-lite).
- Per-game PGN + self-contained HTML replay viewer (board animation, clickable move list) for traced eval games.
- Standalone Stockfish-distilled value network (`src/imba_chess/model/value_net.py` + `scripts/train_value_net.py`), blendable into search at eval time.

## Data and training flow

`HF parquet stream -> game parse -> event sequence -> jagged batch -> model -> loss`

Each game becomes:
- one BOS token
- one token per move: the board state before the move (piece placement, turn, castling rights, en passant, clocks) + the previous move id, with the played move as the classification target
- one per-game outcome label `game_result_white` in `{+1, 0, -1}`

## Training objectives

One transformer trunk, two heads (a linear policy head and a small MLP value head), trained jointly:

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

When `[model].enable_value_head = true`, a 3-class MLP head (`Linear → SiLU → Linear`, private capacity so the policy objective doesn't crowd it out of the shared trunk) is trained to predict the final result of the game from every position, from the perspective of the player about to move:

- The label for every position in a game is that game's final outcome (`game_result_white`, flipped by `turn_id`). The head therefore learns "among training games that passed through positions like this, how often did the side to move end up winning?"
- The target itself is not discounted, but the per-token loss is weighted by game progress (`progress ^ [model].value_weight_alpha`, `progress` in `[0, 1]`): the final outcome is a noisy label for early positions and a clean one for late positions, so early positions contribute little gradient and the last positions contribute full gradient.
- 3-class classification is deliberate (rather than a scalar regression head): win/draw/loss outcomes are genuinely 3-modal — a scalar `0.0` cannot distinguish "certain draw" from "unclear, 50/50 win-or-lose" — and cross-entropy on categories optimizes better than MSE on a bounded scalar. A scalar is recovered at inference as `v = p(win) - p(loss)` in `[-1, 1]`.

Known limitation: game outcomes are high-variance Monte-Carlo labels (a winning position that the player later threw away gets labeled "loss"). The separate value net below addresses this at inference time; the trunk itself still trains on outcome labels.

Training logs include `total_loss`, `policy_loss`, and `value_loss`.

### Standalone value net: Stockfish distillation (separate model)

A second, independent value oracle trained on engine evaluations instead of game outcomes, blended into search at eval time.

`ValueNet` (`src/imba_chess/model/value_net.py`, ~3.5M params) is a **position-only** WDL network: the same joint (piece, square) embedding and `BoardSquareEncoder` body as the big model (256d × 6 layers over the 64 squares), with turn/castling/en-passant features broadcast-added to the square tokens. It sees no game history and no clocks — deliberately, so it exactly matches its training data and has zero train/serve skew.

Training data is `Lichess/chess-position-evaluations` (388M Stockfish-evaluated FENs, CC0, Hugging Face). Each row's centipawn eval becomes a soft win/draw/loss target via Stockfish 17's own `win_rate_model` polynomial (value under strong play — deliberately *not* calibrated to human outcomes); mate-in-N rows get near-saturated targets. Evals are White-POV in the source and flipped to side-to-move POV. A deterministic FEN-hash holdout provides validation.

Three-stage pipeline:

1. **Pretrain the big model** on high-Elo human games (`scripts/train.py`): policy head (imitation) + outcome-value head + moves-left auxiliary.
2. **Train the value net** on engine evals (`python scripts/train_value_net.py`, config in `[value_net]`) — fully decoupled from stage 1; either side retrains without touching the other. Plain supervised learning: flat batches, soft cross-entropy, StableAdamW + OneCycle; best/last checkpoints in `artifacts/value_net/` selected by held-out soft-CE; auto-resumes from `value_net_last.pt` (`--fresh` to opt out).
3. **Blend at search time**: with `--value-net-checkpoint` (or config), every search evaluation becomes `value = (1 − α) · model_value_head + α · value_net`. `--value-net-alpha` sets α (0 = pure model head, 1 = pure net; **0.25 is the measured best** — see Results). Both knobs land in the output JSON's `run_config.value_net`; with no checkpoint configured, eval behavior is unchanged; terminal positions keep their exact values regardless. `sweep_value_net_alpha.sh` loops an α grid with collision-free tags and prints a summary table.

Training recipe (field-tested):

- **Download the data once instead of streaming**: set `HF_TOKEN` (unauthenticated hub requests are rate-limited), `hf download Lichess/chess-position-evaluations --repo-type dataset --local-dir <dir>`, then point `[value_net] dataset_name` at that directory.
- The val slice is built once (a few-minute scan with a progress bar) and cached to `artifacts/value_net/val_slice.pt`; the cache key includes the data/batch config, so changing those triggers one rebuild.
- Sample prep is CPU-bound (~24k rows/s per worker); raise `[value_net] num_workers` if the GPU is starved — useful workers cap at the dataset's file count (20; the loader auto-splits files across workers).
- A ~3.5M net can't saturate a big GPU at batch 1024 (per-step Python overhead dominates) — use large batches. Scaling batch N×: scale `max_lr` by ~√N and divide `train_steps` by N to keep the sample budget fixed. Reference: batch 6144, `max_lr 7e-4`, ~27k samples/s on a 24 GB card ⇒ one ~200M-sample epoch in ~2 h.

## Evaluation during training

Held-out next-move prediction: `fast_val`/`fast_test` on a fixed prefix of games every `eval_every_steps`, `full_val` per epoch (checkpoint selection by `hr@10`), `full_test` in `--eval-only` mode. Metrics: `loss_ce`, `ppl`, `top1/3/5_acc`, `hr@10`, `mrr`. `--resume` restores full trainer state and runs an immediate health check. Config-key details: `docs/superpowers/notes/2026-07-06-eval-log-archive.md`.

## Playing against Stockfish

`scripts/eval_vs_stockfish.py` plays full games against Stockfish over UCI, at full strength or Elo-limited, single-segment or as a ladder. Defaults come from `[eval_vs_stockfish]` in `config/imba_chess.toml`; CLI flags override. Search strategies live in `src/imba_chess/eval/search.py` behind a model-agnostic `PositionEvaluator` interface (unit-testable without a checkpoint). Traced games (first `debug_trace_games` per segment) are saved as PGN + a self-contained HTML replay viewer under `save_games_dir`.

All search evaluations run on a prefix-cache decode path: the once-per-turn root forward doubles as a prefill whose per-layer K/V become a shared cache, and each search position is evaluated as a single new token attending to that cache — O(1) new work per evaluation instead of re-encoding the game history (~12 s/game at budget 512/depth 6, vs ~44 s/game uncached at a quarter of that budget).

Shared rules across all value-guided modes: game-over positions (checkmate, stalemate, claimable draws by repetition/50-move rule) are scored with the exact result instead of the value head (final positions never occur as training inputs, so the head is undefined there); a move that mates on the spot is played immediately without search; simulated boards keep their move stacks so repetition draws are detected inside lines. All modes except `greedy` need a checkpoint trained with the value head enabled.

### `greedy`

Play the highest-logit legal move. Pure policy — a single forward pass, nothing ever checks consequences, so a human-looking move that loses material to a tactic gets played anyway. The baseline everything else is measured against.

### `value_rerank`

Propose the top `value_rerank_top_k` moves by policy prior, evaluate the position after each with the value head, play the best:

```
score(move) = v(position after move) + lambda * log_prob(move)
```

Value-dominant scoring with the policy log-prob (`value_rerank_lambda`) as a near-tie breaker — λ = 0 measurably collapses (the search over-exploits value-head noise; Goodhart), so the prior is a necessary regularizer, not a cosmetic tiebreak.

### `value_search_d2`

`value_rerank` plus one level of adversarial lookahead: *"if I play this, what is the worst thing my opponent can do right after?"* Three batched passes per turn:

1. **Propose** — top-K candidates by policy prior at the root.
2. **Opponent responses** — for each candidate, the opponent's policy top-K **plus every capture, check, and promotion**: the refutation of a bad move is often a move the human-imitation policy ranks low, so probability-based pruning alone would hide exactly what is being tested.
3. **Grade** — evaluate all resulting positions (~K × (K + forcing)); each candidate is scored pessimistically, `grade = min over responses of v`, then `score = grade + λ·log_prob` as above.

### `value_search_halving` (MCTS-lite)

d2 spends its budget uniformly — the obviously losing move gets as much attention as the two moves the decision hinges on. Halving fixes the *allocation*: choosing the root move is treated as best-arm identification, using sequential halving (the root allocation of Gumbel MCTS). Per turn:

1. **Arms** = top `search_top_m` moves by prior, plus any capture/check/promotion outside that set.
2. **Rounds** — `search_budget` value evaluations split evenly across rounds and surviving arms; after each round the worst-scoring half of the arms is eliminated and their unspent budget flows to the survivors. Obvious losers die after a handful of evaluations; the final two candidates get deep trees.
3. **Tree growth (beam by plausibility)** — each arm owns a priority queue of unevaluated positions ordered by cumulative policy log-prob of the line (both sides). Expansion: top `search_expand_top` moves at our nodes; top `search_refutation_top_r` replies **plus every forcing reply** at opponent nodes, with forcing replies inheriting their parent's queue priority (a refutation must compete at the plausibility of the line it refutes, or the beam prunes exactly the move that disproves the arm). Depth caps at `search_max_depth` plies; *where* the tree deepens within the cap is decided entirely by the queue — forced lines go deep, wide quiet positions stay shallow.
4. **Scoring** — negamax backup over the realized tree (terminals exact, frontier leaves on their value estimate) + `λ·log_prob(root move)`. **Value never chooses what to expand** — the queue is ordered by prior alone; value enters only at backup and arm comparison. Ranking the beam by value estimates would retain lines where the opponent cooperatively blunders (max-over-noise selection bias — the λ=0 failure in tree form).

`halving_rounds` is the open-loop/closed-loop dial: `1` = pure prior-guided beam (all allocation upfront), `0` (default) = auto `ceil(log2(#arms))` rounds (full reallocation); comparing the two on one budget attributes the gain between tree depth and the feedback loop. Everything is deterministic (no sampling, ties by insertion order) and the budget is a hard cap on evaluator calls. Design rationale: `BEAM_SEARCH_PLAN.md`; full spec: `docs/superpowers/specs/2026-07-04-mcts-lite-search-design.md`.

### Tuning the halving knobs

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `search_budget` | 256 | Total value evaluations per move — the strength ↔ wall-clock dial (cost ≈ linear) | The biggest lever; raise first. |
| `halving_rounds` | 0 (auto) | How often budget is reallocated by observed value | Keep auto. A/B against `1` (pure beam) at the same budget to check the feedback loop earns its keep. |
| `search_refutation_top_r` | 2 | Opponent replies always expanded besides forcing moves | Raise to 3 if `--debug-trace-games` shows arms scored well whose refutation was never evaluated; costs queue slots everywhere, so pay with budget. |
| `search_expand_top` | 3 | Our-side branching per node | Lower = deeper/narrower; sweep only after budget and rounds settle. |
| `search_max_depth` | 4 | Max plies below each candidate | Keep it **even** — an odd horizon ends on our own move and grades unanswered threats optimistically. 6 needs a bigger budget to be meaningful. |
| `search_top_m` | 16 | Root candidates entering the bandit | Rarely binding (forcing moves added regardless); raising dilutes early-round per-arm budget. |
| `value_rerank_lambda` | 0.05 | Policy-prior weight in scores | Flat across 0.05–0.2, collapses at 0; leave fixed while tuning the rest. |

Recommended order: budget → rounds A/B → refutation floor (only if traces demand it) → depth/branching. One knob per eval run — 100 games has ±0.05 SE.

### Usage

```bash
# Basic match (policy + all knobs from TOML)
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --games 1000 --output-json artifacts/eval/stockfish_eval.json

# Halving search + value-net blend vs Elo-limited Stockfish
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --model-move-policy value_search_halving \
  --search-budget 1024 --search-max-depth 6 \
  --value-net-checkpoint artifacts/value_net/value_net_best.pt --value-net-alpha 0.25 \
  --stockfish-limit-strength --stockfish-elo 2000 --games 100

# Ladder across several Stockfish levels
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --ladder-elos 1400,1600,1800,2000,2200 --ladder-games-per-segment 200 \
  --include-full-strength-segment --output-json artifacts/eval/stockfish_ladder.json
```

Wrappers: `POLICIES="value_search_halving" ELO=1800 TAG=mytag ./eval_best_checkpoint.sh [flags...]` picks the best hr@10 checkpoint, runs 100 games, writes JSON + replays, and skips already-existing outputs; `sweep_value_net_alpha.sh` runs an α grid and prints a summary table. Reports include W/D/L with color split, completed/incomplete games, average game length, score rate, and legal-move vocab coverage (per-segment + aggregate in ladder mode).

## Results vs Stockfish

Protocol: 100 games per configuration, seed 42, colors alternating, Stockfish at 0.05s/move with `UCI_Elo` per rung. Score = (wins + 0.5 × draws) / games; ±~0.05 SE at 100 games. "Net α" = the distilled value net blended at that α; model v3 = 512d × 6L (~10M), v4 = 768d × 8L (~27M, `value_loss_weight` 1.0). Full per-run diaries, color splits, and period interpretations: `docs/superpowers/notes/2026-07-06-eval-log-archive.md`.

| Opponent | Move selection | Model | W / D / L | Score |
|---|---|---|---|---|
| SF1400 | `greedy` | v3 e6 | 7 / 28 / 65 | 0.21 |
| SF1400 | `value_rerank` (pre-v3 ckpt, λ sweep — not comparable) | v2 | — | 0.405 |
| SF1400 | `value_search_d2` (K=16, λ=0.05) | v3 e6 | 22 / 16 / 47 @ 85 | 0.34 |
| SF1400 | halving 256/d4 | v3 e10 | 88 / 7 / 5 | **0.915** |
| SF1800 | halving 256/d4 | v3 e12 | 38 / 17 / 45 | 0.465 |
| SF1800 | halving 512/d6 | v3 e12 | 45 / 22 / 33 | 0.560 |
| SF1800 | halving 1024/d6 | v3 e13 | 48 / 23 / 29 | 0.595 |
| SF1800 | halving 256/d4 | v4 e12 | 51 / 18 / 31 | 0.600 |
| SF1800 | halving 1024/d6 | v4 e12 | 56 / 19 / 25 | 0.655 |
| SF1800 | halving 1024/d6 + net α=0.25 | v4 e12 | 62 / 15 / 23 | **0.695** |
| SF2000 | halving 1024/d6 | v4 e12 | 41 / 22 / 37 | 0.520 |
| SF2000 | halving 1024/d6 + net α=0.25 | v4 e12 | 49 / 29 / 22 | **0.635** |
| SF2000 | halving 1024/d6 + net α=0.5 | v4 e12 | 39 / 28 / 33 | 0.530 |
| SF2000 | halving 1024/d6 + net α=1.0 (pure net) | v4 e12 | 24 / 29 / 47 | 0.385 |
| SF2200 | halving 1024/d6 + net α=0.25 | v4 e12 | 35 / 21 / 44 | 0.455 |
| SF2200 | halving 1024/d6 + net α=0.15 | v4 e12 | 37 / 24 / 39 | 0.490 |

How the components came to be, in order:

1. **`greedy`** established the imitation baseline (0.21 vs SF1400 ≈ ~1170-Elo-equivalent play).
2. **`value_rerank`** added inference-time value scoring; its λ sweep fixed two durable design facts — value-dominant scoring beats policy-dominant by ~+140 Elo, and λ = 0 collapses (Goodhart on value-head noise), so the prior term is load-bearing.
3. **`value_search_d2`** added one level of adversarial lookahead with forcing-move refutations: +0.13 over greedy on the same checkpoint, clearing the pre-registered gate for building a real search.
4. **`value_search_halving`** replaced uniform allocation with sequential halving + prior-ordered tree growth: 0.34 → 0.915 vs SF1400 (~+330 Elo), saturating that rung.
5. **Prefix-cache decode** made bigger budgets affordable (O(1) work per evaluation; ~3.7× faster), enabling the 512/1024 rows.
6. **The v4 trunk** (768d × 8L, doubled value loss weight) lifted the value oracle: 0.465 → 0.600 at matched 256/d4 search, and revived a budget curve that had flattened under v3.
7. **The distilled value net + α blend** added an engine-trained second opinion: α=0.25 is the best measured configuration on both rungs tried (0.695 @ SF1800, 0.635 @ SF2000 ≈ a ~2100-Elo-equivalent system).

Reading the table (light hypotheses, not established facts): the budget curve flattening under v3 and reviving under v4 suggests oracle quality sets how well search compute converts; the α curve peaking low and the pure net losing suggests engine values' "under strong play" semantics mis-rank small edges against fallible opponents. The blend's edge grew with opponent Elo from 1800 to 2000 (+0.04 → +0.115), but the corollary that optimal α rises with opponent strength did **not** survive SF2200, where α=0.15 ≥ α=0.25 (0.490 vs 0.455, within noise) — current best guess is a shallow optimum somewhere in α ∈ [0.1, 0.3], tuned per setup rather than extrapolated. Each of these is one-or-two-datapoint territory — treat as directions to test, not conclusions.

## Configuration

All runtime settings are in `config/imba_chess.toml`:

- `[dataset]` source, month windows, max games for val/test, Elo filters, cache, sequence truncation
- `[board_state]` board-state encoding buckets/options
- `[vocab]` static move vocab location
- `[dataloader]` max tokens per jagged batch, workers
- `[model]` HSTU dimensions/layers + label smoothing + Elo loss weighting + value head knobs
- `[training]` optimizer/scheduler/eval cadence/checkpointing/device/precision
- `[eval_vs_stockfish]` engine path/limits, ladder settings, move-selection policy and knobs, value-net blend, debug controls
- `[value_net]` standalone value net dims, data filters, and trainer settings

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
- Prefix K/V caching is per-turn only: the cache is rebuilt each model turn (no cross-turn reuse) and games are played sequentially (no cross-game batching).
- The big model's value labels are raw game outcomes (noisy); the standalone value net mitigates this at inference, but the trunk itself still trains on outcome labels.
- Checkpoints trained before the placement-aware board encoding / 1,970-token vocab are incompatible with current code (check out an older commit to evaluate them; keep a copy of the old `[model]` block and pass `--config` when evaluating old checkpoints after architecture changes).

## References

- `PLAN.md` for roadmap.
- `EVAL_SPEC.md` for evaluation design.
- `TRAINING_EVENT_SCHEMA.md` for input schema details.
- `FEN_TO_BOARD_STATE.md` for board-state encoding details.
- `VALUE_HEAD_OPTIONS.md` for value-head design notes.
- `docs/superpowers/notes/2026-07-06-eval-log-archive.md` for the full eval diaries, per-color splits, and superseded usage examples.
