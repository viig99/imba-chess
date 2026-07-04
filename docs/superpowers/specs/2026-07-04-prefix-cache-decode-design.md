# Prefix-Cache Decode for Search Inference — Design

## Purpose

Every search evaluation today re-encodes the entire game history (T tokens,
~80–150 mid-game) to read outputs at one new token, so per-move search cost
is O(budget × T). All evaluations share the root history as a prefix and
differ only in their last 1–`max_depth` tokens. Caching the prefix
computation makes each evaluation O(1 new token): ~80–150× fewer FLOPs per
evaluation, expected 10–30× wall-clock on the search portion (the model is
small enough to be launch/memory-bound). Goal: ~44 s/game at halving
budget 256 vs SF1400 drops to single-digit seconds — or equivalently,
budgets of 1024–4096 become affordable at today's wall-clock. This lands
before the SF1800 evals.

**No training impact.** Same weights, same checkpoints, no retraining. The
training forward path is untouched except an additive `return_kv` option.
bf16 float-ordering drift (~1e-3 logits) is the only numeric difference,
proven exact in fp32 by tests.

## Why the architecture allows exact caching

Verified properties of `SequentialTransductionUnitJagged`
(`src/imba_chess/model/hstu_attention.py`):

- Everything except attention is per-token pointwise: layernorms normalize
  over the feature dim only; `u,v,q,k = silu(uvqk(LN(x)))` of that token
  alone. A cached token's post-silu K/V never changes when tokens are
  appended.
- Attention is causal (doc-masked flex_attention): token t's output depends
  only on tokens ≤ t.
- Positions: absolute learned embedding at input (known for any new token:
  prefix length + suffix depth, same clamp) + T5-style relative bias
  indexed by `k_idx − q_idx` (`_ps_w[h, (k_pos − q_pos) + max_seq_len − 1]`,
  clamped) — a gather for decode, no flex machinery needed.
- flex_attention's default scale (`1/sqrt(head_dim)` = `1/sqrt(attention_dim)`)
  must be replicated in the decode matmul; the trunk equivalence test pins it.

## Key structural fact

The root forward that `_select_model_move` already runs once per turn ends
with the transient "current position" token — which is exactly the token
every candidate sequence starts from. Verified against the uncached flow:

- root batch = `[BOS, played…, state(current) prev=last_move]`
- candidate  = root batch + `[state(after cand) prev=cand]` (one token)
- reply      = candidate + one more token, etc.

So the root forward **is** the prefill (with `return_kv=True`), and every
search node adds exactly one token relative to its parent. In all three
strategies, parents are always evaluated before children, so a node's
ancestor K/V chain is always complete when it is evaluated.

## Cache structure: shared prefix + per-node suffixes (chosen approach)

- One immutable prefix K/V per model turn: per layer, the post-silu
  K `[T, H, d_qk]` and V `[T, H, d_v]` from the root forward
  (~12 KB/token across 6 layers; ≤ ~6 MB at the 512-token cap).
- Each search-node handle stores only its own single-token `(k, v)` per
  layer after it is evaluated, plus a parent pointer; a node's suffix cache
  is materialized by walking ≤ `max_depth` ancestors.
- Wave decode attention, per layer, hand-rolled plain torch (no
  flex_attention, eager, no compile requirement, no 4096-token chunk cap):
  queries `[B, H, 1, d]` vs broadcast prefix `[1, H, T, d]` and padded
  suffixes `[B, H, s_max, d]` (s_max ≤ max_depth, padding masked with −inf),
  relative bias added to both parts, softmax over the concatenation,
  weighted sum of both V parts.

Rejected alternatives: contiguous per-node cache copies (~0.5 GB of copy
traffic per move for zero compute savings); paged/vLLM-style KV (overkill
for ≤ 6-token suffixes).

## Part 1: Model-side API

`SequentialTransductionUnitJagged`:
- `forward(x, block_mask, return_kv=False)` — additive flag; when set, also
  returns the post-silu per-head K and V for all tokens. Training callers
  unaffected.
- `forward_decode(x_new, prefix_kv, suffix_kv, suffix_mask, q_positions,
  prefix_positions, suffix_positions)` — computes u/v/q/k for the new
  tokens exactly as `forward` does, runs the two-part attention above with
  the same relative bias and scale, then the existing
  norm/u-gate/output-proj/residual. Returns layer output plus the new
  tokens' (k, v).

`HSTUChessModel`:
- `forward(..., return_kv=True)` — threads per-layer K/V out of the trunk.
- `forward_decode(prefix_caches, new_token_batch, suffix_caches, positions)`
  — `_build_content` on the new tokens (unchanged, per-token), absolute
  position embedding at `prefix_len + depth` (same clamp), per-layer
  `forward_decode`, then policy logits + value logits at the new tokens
  (same heads).

Dropout is inactive in eval mode; decode asserts `not self.training`.

## Part 2: Eval-side — `CachedPositionEvaluator`

Replaces `_HistoryPositionEvaluator` outright (no config flag, no fallback;
rollback is git). Implements the same `PositionEvaluator` protocol —
`src/imba_chess/eval/search.py` and all strategies are untouched.

- Constructed fresh per model turn inside `_select_model_move`, seeded with
  the prefix K/V from the root forward (`return_kv=True`) and the prefix
  length T.
- `extend(handle, board_before, move)` → O(1): returns a node handle
  `(parent, move)`. No `_SequenceHistory` cloning.
- `evaluate(batch)` → one decode wave:
  - per node: features from `BoardStateEncoder.encode(node_board)` +
    `prev_move_id = vocab[move]` + `seq_token_id = EVENT` + position
    `T + depth`;
  - gather ancestor suffix K/V via parent pointers, pad to wave max;
  - `model.forward_decode`; store each node's new (k, v) on its handle;
  - project legal moves + log-softmax per node board exactly as today
    (empty `PositionEval` on the no-vocab-move `RuntimeError`, as today).

Deleted by this change: `_HistoryPositionEvaluator`,
`_forward_last_token_outputs`, `_merge_single_sequence_batches`,
`_SEARCH_EVAL_MAX_TOKENS_PER_CHUNK`, and `_SequenceHistory.clone()`
(existed only for candidate histories). The game loop's `_SequenceHistory`
bookkeeping for the root batch stays.

Precision: decode runs under the same CUDA bf16 autocast as the root
forward today.

Deferred (YAGNI for v1): cross-turn prefix reuse (re-prefilling each turn
costs ~10% of the post-cache budget); compiling the decode path.

## Part 3: Testing

Equivalence gate on a tiny real `HSTUChessModel` (random weights, fp32,
CPU, tolerance ≤ 1e-5):

1. **Trunk**: `forward(return_kv=True)` on a prefix + `forward_decode` of
   suffix tokens ≡ full `forward` over prefix+suffix at the suffix
   positions. Pins scale, relative-bias gather, and position indexing.
   Cover suffix depths 1–4 and a wave mixing depths.
2. **Evaluator**: `CachedPositionEvaluator` `PositionEval`s ≡ explicitly
   built candidate sequences through full forwards (values and legal
   log-priors).
3. **Strategy**: identical chosen moves through `_select_model_move` on
   scripted positions, cached vs full-forward reference computed in-test.

Dummy-model rewrite in `tests/test_eval_vs_stockfish.py`: scripted dummies
gain `forward(..., return_kv=...)` and `forward_decode`, keyed on the new
token's features directly (prev_move_id / turn_id) — simpler than today's
merged-batch introspection. `forward_calls` assertions change meaning
(1 prefill + 1 per decode wave) and are updated with comments. Selection
logic coverage is unaffected — it lives in `tests/test_search.py`'s
torch-free dummies, unchanged.

Performance validation (manual, post-merge, after the local GPU frees up):
same checkpoint, halving @ 256 vs SF1400, ~5 games; acceptance ≥ 5× faster
per move at mid-game lengths (expectation 10–30×). README performance
claims wait for this measurement.

## Out of scope

- Cross-turn prefix cache reuse.
- Compiling the decode path.
- Any training-path change beyond the additive `return_kv` flag.
- Batching across concurrent games.
