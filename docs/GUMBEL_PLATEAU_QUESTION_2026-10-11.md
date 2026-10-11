# Question: why has our Gumbel AlphaZero self-play loop stopped improving?

I'm running a Gumbel AlphaZero style self-play loop for chess (a small project, one RTX 5070 Ti). It started from a strong supervised network. It improved quickly for about 40 iterations and has been flat for about 230 iterations since. Changing the training side (more optimizer steps, a higher lr, fresher data, EMA) didn't move it.

I want to understand **why** it plateaued, and whether this is expected for Gumbel search at our scale or a sign that something in our setup differs from the paper in a way that matters. Please reason from the numbers below, separate what they show from what you're inferring, and name the measurements that would tell the hypotheses apart.

---

## 1. The search we use (Gumbel MuZero / mctx semantics, Danihelka et al., ICLR 2022)

We use our own Rust/Python reimplementation of DeepMind `mctx` `gumbel_muzero_policy`, audited against upstream mctx at commit `88f9205`. Chosen actions and root visits match exactly across 22 multi-depth synthetic trees, all sequential-halving schedules for m = 1–64 and budgets 1–512, and depth-one fixtures. Max target difference is 2e-16.

- **Root:** Gumbel-top-k sampling of m = 16 candidates, then sequential halving over the visit budget. Candidates are ranked by `g + logit + σ(completedQ)`.
- **Interior nodes:** deterministic selection, `argmax(π_improved − N/(1+ΣN))`, as in mctx.
- **Completed Q:** missing Q values use the mixed value approximation (raw value blended with the prior-weighted mean of visited Q).
- **σ transform:** `σ(q) = (c_visit + max_b N(b)) · c_scale · q`, with c_visit = 50.
- **Our deviation, adopted after measurement:** `rescale_values = false`, i.e. raw Q in [−1, 1] instead of min-max normalised Q, with `c_scale = 0.5`.
  - Min-max normalisation stretched near-ties. When all children's Q values were within 0.02, the target's KL from the prior was about 2× larger than when they spanned more than 0.5.
  - Raw Q at 0.5 matched the old target entropy and made that KL increase steadily with the real Q spread. Strength against Stockfish 2600 was within one SE of the old setting (37% vs 41%).
- **Policy target:** `softmax(logits + σ(completedQ))` over all legal moves. No Gumbel term enters it.
- **Move played in self-play:** the sequential-halving recommendation, with Gumbel noise on at the root. There is no extra temperature or visit-count exploration early in the game; the paper's training runs used some.
- **Evaluation:** zero Gumbel noise for both players.
- **Value of a node:** the network's WDL head, read as an expected score in [−1, 1] for the side to move. Terminal positions use the exact outcome. Backups flip the sign once per edge.
- **Seed sensitivity:** changing only the Gumbel seed flips about 33% of chosen moves at our budgets. bf16 vs fp32 flips 6.8%.

### Paper settings vs ours

These are the paper settings as I understand them; please correct anything wrong.

| | Paper (Gumbel MuZero / AlphaZero) | Ours |
|---|---|---|
| Root candidates m | 16 (min(n, 16)) | 16 |
| c_visit / c_scale | 50 / 1.0 on min-max normalised Q | 50 / 0.5 on raw Q |
| Simulations in training | Go/chess: 200 for Gumbel and 800 for MuZero baselines; ablations down to 2 | 256 (earlier lineage: 512, then 1024) |
| Interior selection | deterministic improved-policy matching | same |
| Policy target | softmax(logits + σ(completedQ)) | same |
| Value target | game outcome z (MuZero: n-step bootstrapped) | 0.9·z + 0.1·root search WDL, plus TD(λ) auxiliary heads |
| Scale | DeepMind scale: millions of games, ~10⁶ optimizer steps with batches of thousands | ~85k games, ~3.8k optimizer steps of ~2.7k positions (~10M position exposures) |
| Start | from scratch | from a strong supervised network |
| Exploration while acting | Gumbel sampling (+ visit-count sampling early in AlphaZero-style runs) | Gumbel sampling only |

## 2. Network and starting point

- **Architecture:** a 52.4M-parameter HSTU-style transformer over the game's move history (512-token context). The board is encoded by square attention, then a per-square LayerNorm, then a flatten into a Linear(4096 → d_model). Policy head over a move vocabulary, masked to legal moves; WDL value head; three auxiliary TD(λ) value heads (λ = 0.8, 0.9, 0.95).
- **Initial net ("53250"):** supervised on Lichess games of 2000+ average Elo, with Stockfish-derived value targets. On its own, plain Gumbel at 512 simulations scores 25.5% against Stockfish at UCI Elo 2600.
- **Openings:** every self-play game starts from one of 4,030 human Lichess game prefixes. Self-play takes over at ply 20–120, median 42, so the loop only ever sees middlegames reached by humans. Each prefix has been reused about 21 times.

## 3. Training recipe (current)

- **Optimizer:** StableAdamW at a constant lr: 1e-4 until iteration 189, then 2e-4. Weight decay 0.01, gradient clip 2.0 (mean gradient norm 1.4–2.3, so clipping rarely binds).
- **Batch:** about 2.7k supervised positions per optimizer step. Whole games are packed into 4 × 1024-token microbatches.
- **Iteration:** collection keeps launching games until 8,192 positions have completed. The ~160 games still in flight then finish, so an iteration yields ~28.6k positions (~300 games, ~95 plies each).
- **Training per iteration, until iteration 189:** reuse 1 over a 1M-position replay buffer, sampled uniformly. About 10 optimizer steps per iteration; 37% of positions were never trained on.
- **Training per iteration, from iteration 189:** reuse 2, a 500k buffer and "fresh-first" sampling. Every position from the current iteration is trained once, plus an equal number of old positions. About 21 steps per iteration.
- **Losses:**
  - policy cross-entropy against the search target;
  - value WDL cross-entropy (weight 1.0) against 0.9 × game result + 0.1 × root search WDL;
  - auxiliary TD heads (weight 0.15);
  - KataGo-style policy-surprise weighting (fraction 0.5, cap 3) on the policy loss only.
- **Weights:** collection always uses the latest raw weights. An EMA of the weights (0.995 per step) is evaluated alongside them but never used to play self-play games.

## 4. Results

**Screens:** each screen is 100 games of the current net against the fixed initial net 53250. Both sides use 256 simulations and zero Gumbel noise. The games cover 50 colour-swapped opening pairs, and each score's 95% interval is about ±7 points.

| Iterations | Screens | Mean score vs 53250 | Linear trend (points per 100 iterations) |
|---|---|---|---|
| 0–60 | 12 | 64.1% (59% at iteration 3, about 65–70% by iteration 36) | **+14.0 ± 4.8** |
| 40–273 | 46 | 67.0% | **+0.7 ± 0.7** |
| 100–273 | 34 | 67.5% | −0.6 ± 1.1 |
| 189–273 (new recipe) | 16 | 67.4% | −3.7 ± 3.5 |

EMA screens from iteration 152 on average about 69%. The last three EMA screens were 61%, 65% and 65%.

**Against Stockfish 2600** (plain Gumbel at 512 simulations, 100 games, raw Q at 0.5):

| Network | Score |
|---|---|
| 53250 (start) | 25.5% |
| actor at iteration 208 | 31.0% |
| EMA at iteration 255 | 34.5% (21 wins / 27 draws / 52 losses) |
| EMA at iteration 255, "tactical" search at 512 | **40.5%** (19 / 43 / 38) |
| Earlier, independent lineage from the same start (512 → 1024 simulations, ~3k steps) | plateaued at 35–37% |

"Tactical" search adds root forcing, a forcing-move prior floor, a minimax backup weight of 0.5 and own_width 5. So the **same weights** score 6 points more with better search: the network has headroom that plain search doesn't reach.

Two lineages from the same start, at 256 and 512–1024 simulations, plateau at the same level, about 35% against SF2600.

## 5. Training metrics

Definitions:
- `H(target)`: entropy of the search policy target.
- `KL_learner = KL(target ‖ network being trained now)`, measured on the training batches. These partly overlap data the network has already trained on.
- `KL_actor = KL(target ‖ the prior of the network that played the game)`, stored at collection time.
- Policy loss = `H(target) + KL_learner`.

| Iterations | Policy loss | H(target) | KL_learner | KL_actor | KL_actor − KL_learner | Net policy entropy | Value loss | Win/loss accuracy (decisive positions) | Draw rate |
|---|---|---|---|---|---|---|---|---|---|
| 0–10 | 1.682 | 0.890 | 0.792 | 0.824 | 0.032 | 1.756 | 0.814 | 0.815 | 9.9% |
| 10–40 | 1.674 | 0.909 | 0.765 | 0.816 | 0.051 | 1.786 | 0.561 | 0.826 | 10.8% |
| 40–100 | 1.647 | 0.944 | 0.704 | 0.731 | 0.027 | 1.760 | 0.535 | 0.844 | 11.9% |
| 100–160 | 1.613 | 0.936 | 0.677 | 0.703 | 0.026 | 1.725 | 0.520 | 0.850 | 12.6% |
| 160–189 | 1.587 | 0.930 | 0.657 | 0.681 | 0.024 | 1.696 | 0.524 | 0.848 | 13.6% |
| 189–230 (new recipe) | 1.574 | 0.932 | 0.642 | 0.681 | 0.039 | 1.674 | 0.515 | 0.855 | 12.7% |
| 230–273 | 1.558 | 0.931 | 0.627 | 0.674 | 0.047 | 1.655 | 0.496 | 0.857 | 12.7% |

Observations:
- `H(target)` is flat at about 0.93. All of the fall in policy loss comes from `KL_learner`.
- `KL_actor` fell from 0.82 to 0.68 by about iteration 160 and has been flat since. Strength stopped improving earlier, at about iteration 40.
- The network's own entropy (1.66) stays far above the target's (0.93). It sharpens slowly.
- Value loss and win/loss accuracy keep improving slowly (0.815 → 0.857) after strength flattened. Earlier, on a Stockfish-labelled test set, the value head also kept improving after strength plateaued.
- Game length (~95 plies) and draw rate (~12–13%) are flat.

**Earlier history** (a different start, the older non-flattened ckpt34, under a buggier setup): the loop *lost* strength. A frozen-evaluator decomposition showed the value head degrading and the policy not improving at all. The current lineage does improve at first, so it is a different regime.

## 6. My questions

1. **The chasing gap.** When the updated network plays, its 256-simulation search moves away from it by the same ~0.68 nats again. The network is chasing a target that moves with it. Is that a problem, or exactly what a healthy loop looks like? What would a healthy loop's `KL_actor` and `H(target)` curves look like over time?
2. **The entropy gap.** The network's entropy (1.66) stays far above the target's (0.93) and only slowly sharpens. Does that mean it's learning how concentrated search is rather than which moves search finds? Could much of the 0.68 nats be irreducible target noise? A Gumbel-seed change flips about 33% of chosen moves, so different seeds on the same position may give quite different targets. Or does it reflect knowledge that one forward pass can't amortise (tactics)? How would you measure the split?
3. **Value-to-policy flow.** The policy only sees the value through Q in the search targets: there's no gradient path except the shared trunk. Does an improving value head (lower loss, better win/loss accuracy) actually improve the policy targets? How would we tell whether that distillation is happening or the value gains are calibration that doesn't change move rankings? Raw Q at c_scale 0.5 means σ ≈ (50 + maxN) · 0.5 · q, so a 0.02 Q gap is about 1–1.5 logits at our visit counts.
4. **What the metrics show.** Do these metrics clearly show that "the network gets better at copying 256-simulation plain search of itself, but that search no longer finds moves that win more games"? Or is that only an inference from flat screens? Which direct measurements of target quality would settle it? For example: Stockfish agreement or centipawn loss of the raw-policy argmax and of the 256-simulation search move, on a fixed position set, across actors from different iterations.
5. **The central question: why aren't we following the paper's improvement curve?** "Training harder against the same kind of target is a lever we've already maxed out." The Gumbel paper says policy improvement is guaranteed in expectation, so the loop should keep improving. Which of our differences is most likely responsible?
   - Scale: about 2–3 orders of magnitude fewer games and steps.
   - Starting from a strong supervised net, so self-play only refines it.
   - Only 4,030 human-middlegame openings and no extra early exploration.
   - 256 simulations with m = 16.
   - Raw-Q σ at 0.5 instead of normalised Q at 1.0.
   - A value target that is mostly game outcome, with ~10 independent results per optimizer step.
   - Policy-surprise weighting.
   - Network capacity.
   - Or is the plateau simply what Gumbel AlphaZero does at this budget?

   For each candidate, say what evidence in the numbers above supports or contradicts it, and propose the cheapest experiment that would tell it apart from the others.

---

## 7. Added measurement: target noise and target quality against Stockfish

`scripts/measure_target_noise.py`; output in `artifacts/eval/target-noise-2026-10-11/`.

**Setup.** 600 positions, one random ply from each of the newest self-play games (iteration ≤ 283). Each position was searched 4× with training Gumbel noise and 1× with zero noise, at 256 simulations in bf16, by both the start net (53250) and the current actor (iteration 283). Stockfish 18 at depth 14, multipv over all legal moves (~25 per position), gives every move an expected score; "loss" is the best move's expected score minus that move's.

| | Start (53250) | Now (iteration 283) |
|---|---|---|
| KL(target ‖ prior), mean over seeds | 0.860 | 0.697 |
| … seed noise: mean_k KL(t_k ‖ mean target) | 0.064 (7.5%) | 0.060 (8.6%) |
| … agreed signal: KL(mean target ‖ prior) | 0.796 | 0.637 |
| TV distance between two seeds' targets | 0.105 | 0.095 |
| Chosen move differs between two seeds | 38% | 42% |
| **Prior: expected loss under the policy** | **0.081** | **0.083** |
| **Prior: argmax is Stockfish's best move** | **77.2%** | **76.7%** |
| **Prior: probability mass on moves within 0.02 of best** | **0.809** | **0.805** |
| Search target (mean over seeds): expected loss | 0.027 | 0.020 |
| Search target: argmax is Stockfish's best move | 82.8% | 84.7% |
| Search target: probability mass on moves within 0.02 of best | 0.896 | 0.913 |
| Zero-noise target: expected loss | 0.028 | 0.018 |
| Played move, noisy search: loss | 0.025 | 0.019 |
| Played move, zero-noise search: loss | 0.024 | 0.017 |
| Stockfish's best move is outside the prior's top 16 | 16.5% | 15.3% |
| …and that costs ≥ 0.02 expected score (true blind spot) | 0.5% | 0.0% |

**Findings:**
1. **Seed noise is small,** about 8% of the target–prior KL. (Pairwise KL is 0.41, but heavy tails inflate it: a move that one seed refutes gets near-zero mass, while in another seed, which never visited it, it keeps its blended value. TV is the fairer measure.)
2. **Zero noise loses nothing measurable here** (0/600 positions with a costly blind spot), and its targets and moves are slightly *better* than the noisy ones.
3. **The search targets improved over training** (expected loss 0.027 → 0.020), and they're about 4× better than the prior.
4. **The raw policy prior did not improve against Stockfish at all** (expected loss 0.081 → 0.083; argmax agreement 77% → 77%), even though its KL to the search target fell from 0.86 to 0.70 on these same positions. The policy has moved toward its search's preferences among moves Stockfish rates as roughly equal, not toward better moves.

**Revised question:** the loop's measured gains go through the value head, which improves the search. The policy head absorbs the search target in KL, but not in move quality. Why would distilling a target that is 4× better than the prior not improve the prior's move quality, and what would make it do so?
