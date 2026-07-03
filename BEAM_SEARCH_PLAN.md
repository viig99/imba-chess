# Beam Search Inference Plan (not yet implemented)

Status: **planned** — implement after the v3 retrain (piece-square board encoder input fix + moves-left aux head) has a well-trained value head (epoch 3+). Deeper search multiplies value-head quality, so testing it on an undertrained head measures noise, not the algorithm.

## Motivation

`value_search_d2` is fixed-shape: root top-K=16, one opponent reply level, minimax backup.
Scaling it naively (k per level, depth d) costs k^d evaluations and spends most of them
on implausible lines. A beam caps the evaluation budget per level at a constant B, so
depth 3–4 costs one batched forward pass per level regardless of branching — and lines
*compete* for expansion instead of every node getting equal width.

## Algorithm: beam-by-prior + value-backup + refutation guarantees

Breadth-first over levels; at each level, one batched model forward over ≤B positions
(reuses the existing 4096-token chunked forward in `scripts/eval_vs_stockfish.py`).

1. **Beam by plausibility, not value.** Score a partial line by the cumulative policy
   log-prob of the moves along it (both sides). Keep the top-B lines for expansion.
   Do NOT rank the beam by value estimates — at opponent levels that retains lines
   where the opponent cooperatively blunders and prunes the refutations (max-over-noise
   selection bias; see the λ=0 Goodhart collapse, score 0.12 vs 0.405).
2. **Refutation floor at opponent nodes.** Every surviving line must keep at least the
   opponent's top-1..2 policy replies plus ALL forcing replies (captures, checks,
   promotions), even if that spends beam slots. A line whose best refutation was
   beam-pruned is believed for the wrong reason. This is the hand-crafted analogue of
   MCTS's exploration bonus.
3. **Value enters only at backup.** After the tree is built, run exact minimax over the
   realized tree with the existing value-dominant scoring:
   `score = backed_up_value + λ · path_policy_log_prob`.
   Terminal positions scored exactly (mate/stalemate/repetition guards, as in d2 today).

## Knobs (sweepable)

- `B` — beam width / eval budget per level. Start 256.
- `depth` — levels. Start 3, try 4.
- `λ` — policy-prior weight in backup (current best: 0.05).
- refutation floor size (opponent top-r, r ∈ {1, 2}).

## Eval protocol

- A/B against `value_search_d2` (K=16, λ=0.05) at **matched wall-clock per move**,
  same checkpoint, 100 games vs SF 1400 (±0.05 SE), same openings config.
- If beam-d3 ≥ d2 + 0.05: depth is paying → consider batched MCTS next.
- If beam-d3 ≈ or < d2: value head is still the binding constraint → prioritize
  SF-annotated value labels (distillation) over more search.

## Known risks

- Human-imitation prior is off-distribution for Stockfish's replies; the refutation
  floor is the mitigation. If beam misses show up in debug traces (`--debug-trace-games`),
  raise r before raising B.
- Odd depth ends on our move (optimism bias); prefer even final level or extend
  forcing lines one level (mini-quiescence).

## MCTS-lite: sequential halving at the root (the preferred variant)

Root move choice is a fixed-budget best-arm-identification bandit: arms = candidate
moves, a "pull" = one value-head evaluation spent deepening that move's subtree, and
only the final chosen move is scored. Sequential halving is the canonical algorithm for
that objective, and it batches exactly like beam (each round = even allocation over a
known survivor set = one batched forward pass). This is the root allocation used by
Gumbel MCTS (Danihelka et al. 2022, "Policy improvement by planning with Gumbel"),
minus the Gumbel-noise candidate sampling — that part exists for training-time policy
improvement; for deterministic inference, top-m by prior is the right candidate set.

```
candidates = top-m root moves by policy prior (m = 16) + forcing moves
budget     = N value-head evaluations total (N = 256)
rounds     = ceil(log2(m))                  (m=16 -> 4 rounds)

for each round:
    per_arm = (N / rounds) / len(candidates)
    for each surviving candidate (all arms in one batched forward per level):
        grow that move's subtree by per_arm evaluations:
          - expand its most plausible continuations (prior-ordered),
          - refutation floor at opponent nodes (top-r replies + all forcing),
          - evaluate new leaves with the value head
        arm_score = minimax backup over the arm's tree so far
                    + lambda * log_prior(root move)
    candidates = top half of candidates by arm_score

play the last surviving move (or argmax after the final round)
```

Behavior: obvious losers die after ~4 evaluations; the final two candidates get ~64
evaluations of deepening each — depth adaptivity emerges from halving without any
per-node bandit machinery. Round count is the open-loop/closed-loop dial:

- `rounds = 1` -> exactly the beam plan above (all allocation from prior, no feedback)
- `rounds = log2(m)` -> sequential halving (feedback after every round)

**Implementation note:** build ONE mode with a `halving_rounds` knob; beam is the
rounds=1 setting, so the A/B between beam and MCTS-lite is a config sweep, not two
implementations.

## Relation to full MCTS/PUCT (further out)

PUCT closes the loop per-evaluation instead of per-round: each simulation walks the tree
picking the child maximizing `Q + c · P(prior) · sqrt(N_parent)/(1+N_child)`, backing up
running means. Strictly the most informed allocation, but the per-pull argmax serializes
on shared (Q, N) statistics — batching it needs virtual-loss machinery, and its UCB-style
exploration term optimizes cumulative reward, which is the wrong bandit objective for
move selection anyway (Gumbel MCTS's motivating observation). Sequential halving above
captures most of the value at a fraction of the complexity. No training is required to
use any of these (they consume the existing policy prior + value head); AlphaZero-style
*training* on search outputs (visit counts as policy targets) is a much bigger lift and
out of scope.
