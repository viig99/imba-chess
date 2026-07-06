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
3. **Grade (~K × (K + forcing) sequences, one decode wave).** Apply each response, evaluate all resulting positions with the value head, and collapse each to a scalar `v = p(win) - p(loss)` from our perspective.

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

Search evaluations use a prefix-cache decode path: the once-per-turn root forward doubles as a prefill whose per-layer K/V become a shared cache, and every search position is then evaluated as a single new token attending to that cache — O(1) new work per evaluation instead of re-encoding the full game history.

Cost: ~K² position evaluations per turn instead of 1 — but with the prefix-cache decode path each evaluation is a single new token against the cached game history, so even the much larger halving budgets run at ~12 s/game (vs ~44 s/game these budgets cost uncached).

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
- Halving search adds another large jump over d2 (0.34 → 0.915, ~+330 Elo over the same opponent at the two checkpoints tested) — SF1400 is saturated as an eval opponent for this policy; the ladder continues at SF1800 below.

### Results vs Stockfish 1800 (budget scaling)

Setup: checkpoint `best_hr10_checkpoint_12` (hr@10 = 0.9349), Stockfish `UCI_Elo` 1800 at 0.05s/move, 100 games per configuration, seed 42, colors alternating. Run on the prefix-cache decode path (~12 s/game at budget 512/depth 6 — vs ~44 s/game the uncached path needed for budget 256/depth 4).

| `value_search_halving` config | W / D / L | Score rate | ≈ Elo vs SF1800 |
|---|---|---|---|
| budget 256, depth 4 (defaults) | 38 / 17 / 45 | 0.465 | −24 |
| budget 512, depth 6 | 45 / 22 / 33 | 0.560 | +42 |
| budget 1024, depth 6 (ckpt_13, hr@10 0.9361) | 48 / 23 / 29 | **0.595** | +67 |

What we understand so far:

- **Search does most of the lifting.** The full progression on v3 checkpoints vs SF1400 is greedy 0.21 → d2 0.34 → halving-256 0.915. A pure imitation policy plays ~1170-Elo-equivalent chess; the same network under budgeted value search plays ~1800+.
- **The budget curve is flattening — the value oracle is becoming the bottleneck.** 256→512(+depth) bought +0.095; 512→1024 bought only +0.035 (well inside noise, and the 1024 run even had a slightly newer checkpoint in its favor). Per the decision rule (stop buying search at the first flat doubling), further budget scaling is no longer the lever: the next Elo lives in **better value labels** — engine-annotated (distilled) targets instead of noisy whole-game outcomes.
- Draws grow with opponent strength (17–23 at 1800 vs 7 at 1400): converting drawn endgames is exactly where outcome-label noise hurts most — consistent with the flattening curve.
- Per-color splits at 1800 fluctuate across runs (e.g. 0.43/0.69 at 512/d6) — 50 games/color is noisy; treat asymmetries as variance until they repeat.
- Both follow-ups happened; see the next section. Older (v3) checkpoints remain evaluable via `--config config/imba_chess_v3.toml` after the architecture change.

### Results: v4 trunk + value-net blend (SF1800)

Setup: v4 checkpoint `best_hr10_checkpoint_12` (hr@10 = 0.9468; 768d × 8-layer trunk, `value_loss_weight` 1.0, from a still-running training job) with the distilled value net (3.5M params, one epoch over ~200M engine-evaluated positions). 100 games per configuration, seed 42, 0.05s/move. α = `value_net_alpha`: 0 = pure model value head, 1 = pure distilled net.

α sweep at budget 256/depth 4 — run with a **mid-training** net checkpoint (~45% trained):

| α | W / D / L | Score rate |
|---|---|---|
| 0 | 51 / 18 / 31 | 0.600 |
| 0.25 | 44 / 30 / 26 | 0.590 |
| 0.5 | 41 / 23 / 36 | 0.525 |
| 1.0 | 15 / 26 / 59 | 0.280 |

Re-run at budget 1024/depth 6 with the **finished** net:

| α | W / D / L | Score rate |
|---|---|---|
| 0 | 56 / 19 / 25 | 0.655 |
| 0.25 | 62 / 15 / 23 | **0.695** |

What this taught us:

- **Trunk scale beat label distillation to the punch.** v4's own value head (bigger trunk, doubled value loss weight) already fixed most of what the distillation was built to fix: at matched search (256/d4), v3 scored 0.465 and v4 scores 0.600 — above v3's best-ever 0.595 that needed 4× the budget.
- **Oracle quality sets the search's exchange rate for compute.** v3's budget curve flattened (+0.035 for the last doubling); v4's is alive again (0.600 @ 256/d4 → 0.655 @ 1024/d6). The flattening was the value head, not the search.
- **An underfit oracle is worse than none.** The mid-training net degraded play monotonically in α (a probe showed it scoring a clean knight-up at +0.44 when its own training target says ≈ +1.0). Never evaluate a half-trained oracle and conclude anything about the method.
- **Light mixing wins; heavy mixing loses.** Pure net (α=1.0) is catastrophic even fully trained in the 256/d4 round: Stockfish's win-rate targets encode *value under near-perfect play*, which rounds small advantages to draws — the wrong semantics against a fallible opponent — and the net sees history-free analysis positions, off-distribution from the search tree. At α=0.25 the model head stays in charge and the net acts as a second opinion: **more wins, not more draws** (62/15/23 vs 56/19/25), and the best score the project has produced (0.695, ≈ +140 Elo vs SF1800).
- This is the λ=0 lesson again from a new angle: offline label accuracy is not in-search usefulness; oracles must be judged by play.

### Results vs Stockfish 2000 (full α grid)

Same setup (v4 `checkpoint_12`, finished net, 1024/depth 6, 100 games, seed 42), next rung of the ladder:

| α | W / D / L | Score rate | ≈ Elo vs SF2000 |
|---|---|---|---|
| 0 | 41 / 22 / 37 | 0.520 | +14 |
| 0.25 | 49 / 29 / 22 | **0.635** | **+96** |
| 0.5 | 39 / 28 / 33 | 0.530 | +21 |
| 1.0 | 24 / 29 / 47 | 0.385 | −81 |

- **The blend's edge grows with opponent strength**: +0.040 over pure model head at SF1800, **+0.115 at SF2000** (~2.3 SE — no longer a noise candidate). A clean humped curve peaking at α=0.25 in both settings.
- The likely mechanism: against stronger opponents there are fewer free wins from blunders, so value accuracy matters more — and the net's "value under strong play" semantics become *more* correct as the opponent approaches the play its labels assume. If this holds, the optimal α rises with opponent Elo.
- Even pure net (α=1.0, finished, 1024/d6) is respectable now at 0.385 — the earlier 0.280 collapse was substantially the underfit checkpoint, not the concept.
- Net position: **the α=0.25 system scores 0.635 vs SF2000** (0.05s/move) — roughly a 2100-Elo-equivalent player on this ladder, built from a 27M-param imitation policy, a 3.5M distilled value net, and a 1024-node search.

### Historical: the λ sweep (v2 checkpoint)

An earlier sweep on a pre-v3 checkpoint (not comparable to the numbers above) established two durable design facts, both baked into the current defaults: **value-dominant scoring beats policy-dominant scoring** by ~+140 Elo inference-only (0.23 → 0.405 vs SF1400), and **λ = 0 collapses** (0.12) — optimizing purely against the learned value head over-exploits its noise (Goodhart), so the policy log-prob prior is a necessary regularizer, not a cosmetic tiebreak. Score was flat across λ ∈ [0.05, 0.2]; `value_rerank_lambda = 0.05` has been the default since.

## Standalone value network (Stockfish distillation)

The big model's value head learns from whole-game outcomes — noisy labels (a
won position later thrown away is labeled "loss", and the label encodes
*human* conversion ability). Once the search's budget-scaling curve flattened
at SF1800, those labels became the binding constraint. The fix is a second,
independent value oracle trained on engine evaluations.

`ValueNet` (`src/imba_chess/model/value_net.py`, ~5M params) is a
**position-only** WDL network: the same joint (piece, square) embedding and
`BoardSquareEncoder` body as the big model, scaled up (256d × 6 layers over
the 64 squares), with turn/castling/en-passant features broadcast-added to
the square tokens. It sees no game history and no clocks — deliberately, so
it exactly matches its training data and has zero train/serve skew.

Training data is `Lichess/chess-position-evaluations` (388M Stockfish-
evaluated FENs, CC0, streamed from Hugging Face). Each row's centipawn eval
becomes a soft win/draw/loss target via Stockfish 17's own `win_rate_model`
polynomial (value under strong play — deliberately *not* calibrated to human
outcomes); mate-in-N rows get near-saturated targets. Evals are White-POV in
the source and flipped to side-to-move POV. A deterministic FEN-hash holdout
provides validation.

### The three-stage pipeline

1. **Pretrain the big model** on high-Elo human games (`scripts/train.py`):
   policy head (imitation) + outcome-value head + moves-left auxiliary.
2. **Train the value net** on engine evals — fully decoupled from stage 1;
   either can be retrained without touching the other:

   ```bash
   python scripts/train_value_net.py            # config from [value_net] in the TOML
   python scripts/train_value_net.py --steps 50000 --device cuda
   ```

   Plain supervised learning: flat batches, soft cross-entropy,
   StableAdamW + OneCycle, best/last checkpoints in `artifacts/value_net/`
   selected by held-out soft-CE (TensorBoard logs alongside).

   Practical recipe (from the first real run):

   - **Download the data once instead of streaming it** — set `HF_TOKEN`
     (unauthenticated hub requests are rate-limited) and run
     `hf download Lichess/chess-position-evaluations --repo-type dataset
     --local-dir <dir>`, then point `[value_net] dataset_name` at that
     directory. "Streaming" becomes local disk reads.
   - The val slice is built once (a few-minute scan, with a progress bar)
     and **cached to `artifacts/value_net/val_slice.pt`** — later runs load
     it instantly. The cache key includes the data/batch config, so changing
     those triggers one rebuild.
   - Sample preparation is CPU-bound (~24k rows/s per worker); raise
     `[value_net] num_workers` if the GPU is starved. Useful workers are
     capped by the dataset's file count (20) — the loader auto-splits files
     across workers.
   - A ~3.5M-param net can't saturate a big GPU at batch 1024 (per-step
     Python overhead dominates); larger batches raise throughput. When
     scaling batch N×, scale `max_lr` by ~√N and divide `train_steps` by N
     to keep the sample budget fixed. Reference point: batch 6144,
     `max_lr 7e-4`, ~27k samples/s on a 24 GB card — one ~200M-sample epoch
     in about 2 hours.
3. **Blend at search time** — the eval script loads the net optionally and
   every search evaluation becomes
   `value = (1 − α) · model_value_head + α · value_net`:

   ```bash
   POLICIES="value_search_halving" ELO=1800 TAG=vnet ./eval_best_checkpoint.sh \
     --search-budget 1024 --search-max-depth 6 \
     --value-net-checkpoint artifacts/value_net/value_net_best.pt   # alpha defaults to 1.0
   ```

   `sweep_value_net_alpha.sh` loops an α grid with collision-free tags and
   prints a summary table. `--value-net-alpha` sweeps the blend (0 = pure
   model head, 1 = pure net) — α = 0.25 is the measured best (see the
   results section above);
   both knobs are recorded in the output JSON's `run_config.value_net`. With
   no checkpoint configured, eval behavior is unchanged. Terminal positions
   (mate/stalemate/claimable draws) keep their exact values regardless.

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
- Prefix K/V caching is per-turn only: the cache is rebuilt each model turn (no cross-turn reuse) and games are played sequentially (no cross-game batching).
- The big model's value labels are raw game outcomes (noisy); the standalone value net mitigates this at inference, but the trunk itself still trains on outcome labels.
- Checkpoints trained before the placement-aware board encoding / 1,970-token vocab are incompatible with current code (check out an older commit to evaluate them; keep a copy of the old `[model]` block and pass `--config` when evaluating old checkpoints after architecture changes).

## References

- `PLAN.md` for roadmap.
- `EVAL_SPEC.md` for evaluation design.
- `TRAINING_EVENT_SCHEMA.md` for input schema details.
- `FEN_TO_BOARD_STATE.md` for board-state encoding details.
- `VALUE_HEAD_OPTIONS.md` for value-head design notes.
