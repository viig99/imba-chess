# MCTS-lite (Sequential Halving) Search + Strategy Isolation — Design

## Purpose

`value_search_d2` (fixed root top-K=16, one opponent reply level, minimax backup)
beats greedy by +0.13 score rate vs SF1400 on the current v3 epoch-6 checkpoint
(0.34 vs 0.21), clearing the gate in `BEAM_SEARCH_PLAN.md` for investing in
deeper search. This implements that plan's preferred variant — sequential
halving at the root with beam-by-prior tree growth — and, as groundwork,
isolates all move-selection strategies out of `scripts/eval_vs_stockfish.py`
into a dedicated module so the eval script can plug in greedy / value_rerank /
value_search_d2 / value_search_halving uniformly.

Baselines this must beat (same checkpoint, 100 games vs SF1400, seed 42):
greedy 0.21, value_search_d2 (K=16, λ=0.05) 0.34.

## Part 1: Strategy module — `src/imba_chess/eval/search.py`

All strategies move behind one evaluator interface:

```python
class PositionEval(NamedTuple):
    value_stm: float                 # value-head scalar, side-to-move POV
    legal_moves: list[chess.Move]    # legal moves that map to the move vocab
    legal_log_priors: list[float]    # log-softmax policy over those moves

class PositionEvaluator(Protocol):
    def extend(self, handle: Any, board_before: chess.Board, move: chess.Move) -> Any: ...
    def evaluate(self, batch: list[tuple[Any, chess.Board]]) -> list[PositionEval]: ...
```

- `handle` is opaque to the search module. In the eval script it is a
  `_SequenceHistory` clone; in unit tests it is whatever the dummy needs.
  `extend(handle, board_before, move)` returns the handle for the position
  after `move` (history append + record played move).
- `evaluate` is batched: one call may cover positions from many search arms;
  the script's adapter routes it through the existing chunked
  `_forward_last_token_outputs` (4096-token chunks, unchanged).

Module contents:

- `select_greedy(...)` — trivial argmax (moved for uniformity).
- `select_value_rerank(...)` — moved from the script, behavior unchanged.
- `select_value_search_d2(...)` — moved from the script, behavior unchanged;
  remains the A/B baseline.
- `select_value_search_halving(...)` — new (Part 2).
- Shared helpers: terminal-value scoring (mate/stalemate/claimable draws,
  root POV), WDL-logits→scalar conversion.

`scripts/eval_vs_stockfish.py` keeps everything model-shaped: checkpoint
loading, `_SequenceHistory`, jagged batch construction/merging, the chunked
forward, and a `PositionEvaluator` adapter over them. `_select_model_move`
shrinks to: root forward → project legal moves → dispatch by policy name →
stats/debug bookkeeping.

Existing d2/rerank tests in `tests/test_eval_vs_stockfish.py` keep running
end-to-end through the script (dummy model → dispatch → adapter → module) and
serve as the regression guard for the extraction. Naive evaluation only:
prefix-computation reuse (caching the shared game-history encoding) is
explicitly deferred until the algorithm shows a score-rate win.

## Part 2: `value_search_halving` algorithm

**Setup.** Root candidates = top-`m` legal moves by policy prior (default
m=16) plus any forcing moves (captures, checks, promotions) outside that
top-m. Immediate-mate short-circuit exactly as d2 today (return the mating
move with zero evaluator calls). Rounds = `halving_rounds`: explicit `1` is
the pure beam variant; `0` (default) means auto = `ceil(log2(m))`.

**Per round.**

- `per_arm = (search_budget / rounds) / len(surviving_arms)`, recomputed each
  round so budget from eliminated arms flows to survivors.
- Each arm owns a max-heap of unevaluated positions keyed by cumulative
  policy log-prob along the path (both sides' moves).
- The round pops up to `per_arm` positions per arm and evaluates all arms'
  pops in one batched `evaluate` call.
- For each evaluated position, push its children onto that arm's heap:
  - **Opponent to move** (refutation floor): top-`search_refutation_top_r`
    replies by prior **plus all forcing replies**, even when that exceeds r.
  - **Our move**: top-`search_expand_top` by prior; the heap's plausibility
    ordering decides which actually get evaluated.
  - Terminal children get exact scores and are never sent to the evaluator.
  - Depth is capped at `search_max_depth` plies below the arm root (default
    4 — even, per the plan's optimism-bias note).

**Scoring and halving.** After each round, arm score = negamax backup over
the realized tree — a node with evaluated children takes `max(-child)`, a
frontier leaf stands on its own value-head estimate, terminals are exact —
converted to root POV, plus `λ · log_prior(root_move)` using the existing
`value_rerank_lambda`. Keep the top `ceil(len/2)` arms per round. After the
final round, play the argmax arm.

**Value never selects within the tree** — beam ordering is by prior only;
value enters only at backup and root-arm comparison. This is the plan's
guard against the max-over-noise selection bias (the λ=0 Goodhart collapse,
0.12 vs 0.405).

**Determinism.** No sampling anywhere; ties break by move order; eval runs
are reproducible seed-for-seed like d2.

**Debug.** Per-arm rows (move, evals spent, max tree depth reached,
backed-up value, final score, round eliminated) flow into `debug_info`,
printed by the existing `--debug-trace-games` console tracing. Replay-HTML
overlay of considered moves stays a future extension (same extension point
as the game-animation spec).

## Part 3: Config, CLI, testing, A/B protocol

### Config (`[eval_vs_stockfish]`), each with a matching CLI override

```toml
model_move_policy = "value_search_d2"   # unchanged default; new option "value_search_halving"
search_budget = 256          # N: total position evaluations per move
search_top_m = 16            # root candidates by prior (+ forcing)
halving_rounds = 0           # 0 = auto ceil(log2(m)); 1 = pure beam
search_refutation_top_r = 2  # opponent reply floor (forcing always included)
search_expand_top = 3        # our-side children pushed per expansion
search_max_depth = 4         # plies below arm root
```

`value_rerank_lambda` / `value_rerank_top_k` keep their existing meanings for
the existing policies; halving reuses `value_rerank_lambda` as its λ.

### Testing

1. Existing greedy/rerank/d2 script tests unchanged — regression guard for
   the extraction refactor.
2. Unit tests for `select_value_search_halving` against a scripted dummy
   `PositionEvaluator` (hand-built priors/values, no torch model):
   - halving eliminates a clearly bad arm after round 1;
   - refutation floor keeps a low-prior forcing refutation that flips an
     arm's score;
   - mate-in-one short-circuits with zero evaluator calls;
   - budget accounting is exact (evaluator sees ≤ search_budget positions);
   - `halving_rounds=1` yields pure-beam allocation (no mid-search
     elimination).
3. One integration test through `_select_model_move` with a dummy model.

### A/B protocol (from BEAM_SEARCH_PLAN.md)

Same checkpoint (`best_hr10_checkpoint_6_hr10=0.9131`), 100 games vs SF1400,
same seed/openings, via `eval_best_checkpoint.sh` with
`POLICIES="value_search_halving"`. Budget 256 ≈ d2's current per-move eval
count, so wall-clock stays comparable (~30–40 s/game).

Decision rule:
- **≥ 0.39** (d2 + 0.05): depth pays → prefix caching and larger budgets
  become worth building.
- **≈ 0.34 or below**: value head is the binding constraint → pivot to
  SF-annotated value distillation instead of more search.

Cheap extra sweep point: `halving_rounds=1` (beam) vs auto, to attribute the
gain between the feedback loop and the deeper tree.

## Out of scope

- Prefix-computation reuse / KV-style caching (deferred until the algorithm
  wins on score rate).
- Full PUCT / per-simulation MCTS (explicitly deferred in the plan).
- Training on search outputs (AlphaZero-style) — out of scope entirely.
- Replay-HTML overlay of considered moves (existing extension point, later).
