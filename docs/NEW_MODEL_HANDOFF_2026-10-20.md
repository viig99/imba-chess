# Handoff: a board-centric model for faster, better-generalizing self-play (2026-10-20)

**Your job is to explore and choose the best architecture for this project:** latency first, then how well
the network reasons about the position. The design below is a starting point from one day of
diagnostics, not a decision. Measure, compare alternatives, and replace any of it where the evidence says
so. A faster model makes everything else easier: pretraining, self-play throughput, and every experiment
after it.

## Why: the current model memorizes, it does not generalize

Diagnostics from 2026-10-19/20 (`artifacts/eval/learning-diagnostics-2026-10-19/`, details in
`docs/SEARCH_METHODS.md`).

**The search target is good, and learning it does not transfer.**
- Gumbel's search target beats the raw prior by a wide margin on the 4,110-root golden test set:
  target regret 0.046 vs 0.117 against Stockfish.
- Its corrections are consistent: noisy targets keep a ≥0.02-better top move 87% of the time.
- The network can learn its targets, but the learning does not transfer:
  - **Parity is fine.** The training forward reproduces the priors stored during play (KL 0.00024).
  - **Capacity is fine.** Overfitting 100 games flips 97% of the top moves the target asks for.
  - **No transfer to unseen games.** Distilling a frozen 454k-position snapshot with 10% of games held out,
    for 12 passes: training-game KL 0.79 → 0.14, but held-out KL 0.77 → 1.23 and held-out top-1 60% → 54%.
    There is no gain even on the first pass.
  - **No transfer within a game.** Positions held out *inside* training games behave exactly like unseen
    games.
  - **The golden-set gain is only sharpening.** Prior regret falls because mass leaves bad moves; the top
    move does not improve.

**Training levers do not change transfer.**
- **Learning rate (tested separately, at label mix 0.25):** 2e-4, 5e-5 and 2e-5 all give no held-out
  transfer; 2e-4 actively harms.
- The rest were single changes at lr 5e-5 and label mix 0.5, paired against two training seeds; seed noise is
  ≈0.002 in search move regret.
- **Neutral:** surprise weighting on or off, weight decay 0 to 0.1, aux TD heads, logit-space Q regression.
- **Harmful:** dropout added to a net trained without it, game-result-heavy value labels, stronger surprise
  or flip weighting, value-only training.
- **Weak positive:** 4× gradient accumulation, the only arm whose held-out KL improved.
- **The search is already tuned:** no PUCT variant beats Gumbel as a teacher (variance-RPO only ties it,
  with near one-hot targets), and `value_scale` 0.5 (raw Q) is at its optimum.

**What did help:** online, after the recipe moved to lr 5e-5 and value mix 0.5, the live run gained for
the first time (head-to-head ema-500 vs ema-554: 0.554 [0.516, 0.591]). The loop can improve. But the
network's ability to turn search corrections into general knowledge is the ceiling.

**The leading suspect: the architecture reasons about the board in only two thin layers.**

## The current model and its spatial limitation

`src/imba_chess/model/hstu_model.py`, config `config/imba_chess_v4_aux3.toml`, ~54.4M parameters.

1. A joint (piece, square) embedding, 13 × 64 entries, 64-dim. This is the v3 fix for a placement-blind
   bag-of-material bug.
2. **Spatial reasoning: 2 square-attention blocks, 64-dim, 4 heads**, each with a 64→128→64 MLP. About 0.1M
   parameters, and the only place squares interact. Square attention has no geometric bias.
3. Readout: the 64 square vectors are flattened (4,096) and projected by one linear layer to a 1,024-dim
   token per position (v4; it replaced mean pooling).
4. Side to move, castling, en passant, the halfmove/fullmove counters and the previous move are added to
   that token.
5. **8 HSTU layers, 1,024-dim, attend across the game's moves, not squares**, with a 512-token context. The
   heads are a policy over a 1,970-entry UCI vocabulary (`artifacts/move_vocab_static_uci.json`), a WDL
   value head (width 512, 2 blocks), 3 auxiliary TD value heads and moves-left.

Nearly all capacity models the move history. Tactics such as forks, pins, skewers and discovered attacks
are relations between pieces on squares, often several hops deep, and two narrow attention layers plus a
flatten cannot build general detectors for them. The big sequence model can still memorize "this history →
this move", which matches the measurements. The prior also depends on history: the same position reached
by a different move order changes its top move 7% of the time (identical history: 0.1%).

**Lc0 and Chessformer, as we understand them (verify against their papers and code):**
- 64 square tokens through 10-15+ attention layers, hundreds of dimensions wide.
- Attention biased by board geometry (Chessformer's geometric attention bias; Lc0's newer nets use
  smolgen-style dynamic biases).
- A from-square/to-square policy head.
- Memory notes on Chessformer/Maia: GAB and the from-to head are the takeaways.

## Goals for the new model

1. **Latency first.** Measure evaluations per second in the real self-play setting (5 collection workers
   sharing a 16 GB 5070 Ti under CUDA MPS, Gumbel 256). The live baseline is about 82k evals/s, about 14.8k
   games/h. Match or beat it; faster is a direct win for pretraining, self-play and every experiment.
2. **Position reasoning.** Square-token attention with geometry built in. The test is whether the model
   *generalizes* search corrections, not just its golden-set score.
3. **History: k = 2 previous positions as input features, not tokens.** Each square token carries the piece
   now and on the last 2 plies as extra input channels. Train with random history dropout so the model
   cannot lean on history. Also add side to move, castling, en passant, the fifty-move counter and the
   repetition count.
   - Each evaluation is then stateless: a forward over 64 tokens. There is no KV cache, so the per-worker
     KV arenas (~2.75 GB each, the reason collection is capped at 5 × 32 games) go away.
   - Batches are fixed-shape, which suits CUDA graphs.
   - An evaluation cache keyed by position (plus the k planes) becomes possible.
   - Compare against k = 0 (Markov plus repetition flags) in the pilot.
4. **The same outputs the search needs.**
   - **Per legal move:** a log prior over the UCI vocabulary.
   - **Per position:** a side-to-move WDL.
   - See `PositionEval` in `src/imba_chess/eval/search.py`: value_stm, legal moves/UCIs/ids, log priors,
     forcing flags, WDL.
   - The native Gumbel search is model-agnostic. Only the evaluator/runtime is HSTU-specific: decode waves,
     prefix KV, root carry.

## Open questions for you to explore (do not take these as settled)

- **Width and depth for the latency budget.** Rough FLOPs: the current one-token decode is ≈0.1 GFLOP per
  evaluation; a 64-token model is ≈0.35 GFLOP at 192 × 6 and ≈0.8 at 256 × 8. The current collector is
  partly CPU- and launch-bound, so FLOPs alone will not predict wall time. Benchmark.
- **Attention bias:** a static geometric bias (Chessformer GAB), relative-position tables, or smolgen-style
  dynamic biases. Also the head count and attention dimension.
- **Policy head:** from-to bilinear over square tokens plus promotions, mapped to the existing vocabulary, or
  something else.
- **Value head:** pooling, plus WDL, moves-left, and whether to keep the aux TD heads (they were neutral
  in today's tests).
- **History:** k = 2 with dropout vs k = 0, and the input encoding of the history planes.
- **Precision and kernels:** bf16, CUDA graphs, fused attention, eval-cache size and hit rate in the tree.
- **Pretraining speed:** position-level batches; distillation from the current model as a warm start; how
  much of the Lichess corpus is needed. The current model's base was ckpt34 (34 epochs × 10k steps at
  40,960-token batches, `max_lr` 5e-4) plus 53,250 flatten-continuation steps. The aim is 1-2 days on the
  5070 Ti.

## Suggested path, with gates

1. **Build (no GPU):** the model, a position-level dataset adapter (Lichess games → per-position samples
   with k planes and Stockfish value targets from the existing pipeline), training through the existing
   supervised loop where possible, and a stateless evaluator usable by `golden_raw`-style scoring,
   `scripts/compare_search_methods.py` and collection.
2. **Latency benchmark (minutes of GPU):** candidate sizes × realistic batch sizes vs the current decode
   path. **Gate:** at least as fast in collection.
3. **Pilot pretrain (a few hours).**
   - **Gate A:** golden-set raw prior and value at or near actor-290's (prior regret 0.117, argmax regret
     0.080, value MAE 0.164).
   - **Gate B, the decisive one:** fine-tune the pilot on a frozen self-play snapshot with held-out games,
     as `artifacts/eval/learning-diagnostics-2026-10-19/absorb.py` does for HSTU, and check that held-out
     KL and the flip rate improve. The current model fails this. If the pilot fails it too, the problem is
     data scale, not architecture; stop and report.
4. **Full pretrain (1-2 days),** then self-play from it with `run_self_play.py --initialize ... --stream-from
   <earlier runs>`. Judge it by head-to-head against ema-554, golden-set search regret and the raw golden
   curve every 25 iterations.

## Constraints and conventions

- **The live Gumbel self-play run owns the GPU.** Ask the session running it for a window (it pauses with
  STOP plus a drain) before any GPU job; a side job beside collection has OOM-crashed it before. Report back
  when done.
- **Develop in a git worktree.** The live run imports the main tree on restart.
- **Fail fast.** No catch-and-continue.
- **Commit and push only when the user asks.**
- **Run new value or policy labels through the golden set first.**
- **Use Sonnet or larger for any subagents.**
- **Key artifacts:**
  - golden set: `artifacts/eval/golden-v1-2026-10-18/`;
  - frozen snapshot and diagnostics: `artifacts/eval/learning-diagnostics-2026-10-19/`;
  - live run: `artifacts/self_play/gumbel-iwdl-mix025-s256-lr2e4-from-it554-2026-10-18/`, on the current
    recipe `config/self_play.toml` (lr 5e-5, value_search_mix 0.5).
