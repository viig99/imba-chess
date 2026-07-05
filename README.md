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
- `value_rerank`: propose top-K moves with the policy head, grade each by the value head after the move, pick the best grade.
- `value_search_d2`: same, but each proposal is stress-tested against the opponent's best response before grading (see below).
- `value_search_halving`: budgeted tree search — candidate moves compete for a fixed number of value-head evaluations via sequential halving, growing deeper trees under the promising moves (see below).

All modes except `greedy` require a checkpoint trained with the value head enabled. Search strategies live in `src/imba_chess/eval/search.py` behind a model-agnostic `PositionEvaluator` interface (unit-testable without a checkpoint); the eval script supplies the batched-inference adapter. Traced games (first `debug_trace_games` per segment) are saved as PGN plus a self-contained HTML replay viewer under `save_games_dir` for post-hoc inspection.

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

### How `value_search_halving` works (MCTS-lite)

`value_search_d2` spends its evaluations uniformly: every candidate gets one opponent level, no more, no less — the obviously losing move gets as much attention as the two moves the decision actually hinges on. `value_search_halving` fixes the *allocation*: choosing the root move is treated as a best-arm-identification problem (which arm is best, not how good is each arm), and sequential halving — the root allocation used by Gumbel MCTS — is the canonical algorithm for that.

The mechanics, per model turn:

1. **Arms.** Candidates = top `search_top_m` legal moves by policy prior, plus any capture/check/promotion outside that set. A move that mates on the spot is played immediately, spending nothing.
2. **Rounds.** A fixed budget of `search_budget` value-head evaluations is split evenly across `halving_rounds` rounds, and within a round evenly across surviving arms. After each round the worst-scoring half of the arms is eliminated; their unspent budget flows to the survivors. Obvious losers die after a handful of evaluations; the final two candidates get deep trees.
3. **Tree growth (beam by plausibility).** Each arm owns a priority queue of unevaluated positions, ordered by the cumulative policy log-prob of the moves leading there (both sides). Its round budget is spent popping the most plausible positions, evaluating them in one batched forward per wave, and pushing their continuations: at our nodes the top `search_expand_top` moves by prior; at opponent nodes the top `search_refutation_top_r` replies **plus every forcing reply**. Forcing replies inherit their parent's queue priority rather than their own (usually tiny) prior — a refutation must compete at the plausibility of the line it refutes, or the beam prunes exactly the move that disproves the arm. Depth is capped at `search_max_depth` plies; where the tree deepens within that cap is decided entirely by the queue, so forced lines go deep while wide quiet positions stay shallow.
4. **Scoring.** Arm score = negamax backup over the arm's realized tree (terminal positions exact, frontier leaves stand on their value-head estimate) + `value_rerank_lambda × log_prob(root move)`. **Value never chooses what to expand** — the queue is ordered by prior alone, and value enters only at backup and arm comparison. This is deliberate: ranking the beam by value estimates retains lines where the opponent cooperatively blunders (max-over-noise selection bias, the same failure the λ=0 control exposes).

`halving_rounds` is the open-loop/closed-loop dial: `1` disables elimination entirely (pure prior-guided beam — all allocation decided upfront), `0` (default) auto-selects `ceil(log2(#arms))` rounds (full sequential halving — reallocation after every round). Comparing the two on the same budget attributes the gain between the deeper tree and the feedback loop.

Everything is deterministic (no sampling; ties break by insertion order), the budget is a hard cap on evaluator calls, and the same terminal-exactness rules as d2 apply. See `BEAM_SEARCH_PLAN.md` for the design rationale and `docs/superpowers/specs/2026-07-04-mcts-lite-search-design.md` for the full spec.

### Tuning the halving knobs

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `search_budget` | 256 | Total value-head evaluations per move — the strength ↔ wall-clock dial (cost is roughly linear) | The biggest lever. 256 ≈ d2's per-move cost (fair A/B). Raise to 512+ only after the algorithm has proven itself at 256; each search evaluation re-encodes the full game history, so budget is expensive late-game. |
| `halving_rounds` | 0 (auto) | How often budget is reallocated by observed value | Keep auto. Run `--halving-rounds 1` (pure beam) once at the same budget: if beam ≈ halving, the feedback loop isn't earning its keep and prior-allocation suffices; if halving wins, more rounds concentrate budget where it matters. |
| `search_refutation_top_r` | 2 | Opponent replies always expanded besides forcing moves | Raise to 3 if `--debug-trace-games` shows arms scored well whose refutation was never evaluated ("believed for the wrong reason"). Raising it costs queue slots everywhere, so pay for it with budget. |
| `search_expand_top` | 3 | Our-side branching per expanded node | Lower (2) = deeper, narrower trees; higher = wider, shallower. Only worth sweeping after budget and rounds are settled. |
| `search_max_depth` | 4 | Max plies below each candidate move | Keep it **even** — an odd horizon ends on our own move and grades unanswered threats optimistically. 6 needs a bigger budget to be meaningful. |
| `search_top_m` | 16 | Root candidates entering the bandit | Rarely binding (forcing moves are added regardless). Raising it dilutes early-round per-arm budget; lower it only if round-1 budgets get starved. |
| `value_rerank_lambda` | 0.05 | Policy-prior weight in the arm score | Same role and sweep as d2: flat across 0.05–0.2, collapses at 0 (Goodhart on value-head noise). Leave fixed while tuning the knobs above. |

Recommended order: budget (strength/time trade) → rounds 1-vs-auto A/B (attribution) → refutation floor (only if traces demand it) → depth/branching. Change one knob per eval run — 100 games has ±0.05 standard error, so small simultaneous changes are unreadable.

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

Halving search (defaults from `[eval_vs_stockfish]`; shown with the beam-attribution override):

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --model-move-policy value_search_halving \
  --search-budget 256 \
  --halving-rounds 1 \
  --stockfish-limit-strength --stockfish-elo 1400 \
  --games 100
```

For the standard best-checkpoint A/B there is a wrapper: `POLICIES="value_search_halving" ./eval_best_checkpoint.sh` (picks the best hr@10 checkpoint, runs 100 games vs SF1400 per policy, writes JSON + per-policy game replays, skips already-existing outputs).

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

### Results vs Stockfish 1400 (current run, v3 pipeline)

Setup: `greedy` / `value_search_d2` on checkpoint `best_hr10_checkpoint_6` (hr@10 = 0.9131, epoch 6); `value_search_halving` on the run's eventual best checkpoint, `best_hr10_checkpoint_10` (hr@10 = 0.9304). Stockfish `UCI_Elo` 1400 at 0.05s/move, 100 games per configuration, seed 42, colors alternating. Score = (wins + 0.5 × draws) / games; ±~0.05 standard error at 100 games.

| Move selection | W / D / L | Score rate |
|---|---|---|
| `greedy` | 7 / 28 / 65 | 0.21 |
| `value_search_d2` (K=16, λ=0.05) | 22 / 16 / 47 @ 85 games | 0.34 (final, 100 games) |
| `value_search_halving` (N=256, `halving_rounds=0` auto) | 88 / 7 / 5 | **0.915** |

By color (`value_search_halving`): white 46/2/2 (score 0.94), black 42/5/3 (score 0.89) — the black-side weakness visible under `greedy`/`d2` is essentially gone.

Takeaways:

- The raw policy is markedly stronger than the previous run's (greedy 0.145 → 0.21) — attributable to the v3 changes: placement-aware board encoding, game-level shuffle buffer, corrupt-game rejection.
- Search still multiplies the policy: d2 adds +0.13 over greedy on the same checkpoint, clearing the +0.05 gate that justified building the halving search.
- Halving search adds another large jump over d2 (0.34 → 0.915, ~+330 Elo over the same opponent at the two checkpoints tested) — SF1400 is now essentially saturated as an eval opponent for this policy; future sweeps should move to SF1600+.
- Remaining work: the `halving_rounds=1` beam-vs-halving attribution run (does the gain come from the deeper tree or the reallocation feedback loop?) and a 1600/1800/2000 ladder, to find where the model's actual ceiling is.

### Results vs Stockfish 1400 (earlier v2 checkpoint, historical)

Setup: checkpoint `best_hr10_checkpoint_5` (hr@10 = 0.9208, pre-v3 data pipeline — not directly comparable to the table above), Stockfish `UCI_Elo` 1400 at 0.05s/move, 100 games per configuration, seed 42, colors alternating.

| Move selection | λ | W / D / L | Score rate |
|---|---|---|---|
| `value_search_d2`, value only (no policy prior) | 0.00 | 1 / 22 / 77 | 0.120 |
| `value_search_d2`, value-dominant scoring, K=16 | 0.05 | 20 / 41 / 39 | **0.405** |
| `value_search_d2`, value-dominant scoring, K=16 | 0.10 | 27 / 26 / 47 | 0.400 |
| `value_search_d2`, value-dominant scoring, K=16 | 0.20 | 27 / 24 / 49 | 0.390 |
| `value_rerank`, old policy-dominant scoring (best of λ sweep) | 0.35 | 12 / 22 / 66 | 0.230 |

Takeaways:

- Value-dominant scoring nearly doubles the score rate over the old policy-dominant scoring (0.23 → 0.40, ~+140 Elo vs the same opponent), inference-only: same checkpoint, fixed search scoring, exact terminal handling, forcing-move opponent replies.
- The score is flat across λ ∈ [0.05, 0.2]; larger λ trades draws for decisive games at equal expected score.
- The λ = 0 control collapses (0.12): optimizing purely against the learned value head over-exploits its noise (Goodhart). The policy log-prob prior is a necessary regularizer that keeps candidates human-plausible, not a cosmetic tiebreak.
- Draw share roughly doubled at λ = 0.05 vs the old scoring — the search stops losing many previously lost games; converting draws into wins (value-head endgame quality) is the next frontier.

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
- Search evaluations re-encode the full game history per position (no prefix/KV caching), so per-move search cost grows with game length; caching is the planned next step if `value_search_halving` holds up.
- Value labels are raw game outcomes, not engine evaluations (noisy for early positions).
- Streaming order is temporal by month window (newest first); month-level file order can be shuffled at process start via `[dataset].shuffle_train_month_files_on_start`.
- Checkpoints trained before the placement-aware board encoding / 1,970-token vocab are incompatible with current code (check out an older commit to evaluate them).

## References

- `PLAN.md` for roadmap.
- `EVAL_SPEC.md` for evaluation design.
- `TRAINING_EVENT_SCHEMA.md` for input schema details.
- `FEN_TO_BOARD_STATE.md` for board-state encoding details.
- `VALUE_HEAD_OPTIONS.md` for value-head design notes.
