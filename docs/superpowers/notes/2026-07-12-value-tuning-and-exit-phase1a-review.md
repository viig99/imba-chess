# Value-head tuning, ExIt Phase 1a review, and adaptive search lambda (2026-07-12)

Session summary: one hyperparameter change adopted, one feature removed as dead code, one search-time idea tried and reverted, and a literature-grounded reassessment of why search-backed value distillation (ExIt Phase 1a) hasn't paid off yet. All code changes are in git history (`0bb9cc3`..`be1f9e1` on `main`); this note is the narrative behind them.

## 1. `value_weight_alpha`: 0.9 → 0.1 (adopted)

The value head's per-token loss weight is `progress ^ value_weight_alpha`, where `progress` is a token's fractional position within its game. The production default (`0.9`) was close to a full linear recency discount, suppressing gradient on early/mid-game positions much more aggressively than the data actually supports.

**Method:** trained two 10,000-step fine-tunes from the same checkpoint (`best_hr10_checkpoint_23`), same data, only `alpha` differing (0.9 control vs 0.1 treatment). Evaluated both with a new diagnostic (`scripts/eval_value_by_progress.py`) that buckets held-out value loss into ten game-progress deciles — deliberately *unweighted* per bucket, so the comparison doesn't inherit either run's own training-time weighting.

**Result:** `alpha=0.1` beat `alpha=0.9` in 7 of 10 buckets; overall unweighted mean loss 0.8596 vs 0.8784 (~2.1% better), concentrated in early/mid-game, with a small late-game cost. Adopted as the new default in `config/imba_chess.toml`.

**Open question:** only validated over 10,000 fine-tune steps from an already-strong checkpoint, not from-scratch training. Tonight's overnight job (`scripts/nightly_alpha01_train_and_eval.sh`) continues training under `alpha=0.1` toward ~step 310,000 and then plays 200 games vs. Stockfish (2200 ladder) — first real evidence on whether this shows up as playing strength, not just held-out loss.

## 2. ExIt Phase 1a (search-backed value distillation): negative, but now better understood

**What was tried:** `scripts/generate_search_rollouts.py` runs the production search (`value_search_halving`, budget 2048/depth 8) against ~5,800 human games from one month, recording a search-backed value estimate at ~1-in-8 sampled plies (55,939 rows total). `compute_blended_value_target` blends that estimate with the real game outcome via `beta`: `target = (1-beta)*outcome + beta*searched_wdl`. Fine-tuned `beta in {0.5, 0.75}` at both `alpha=0.9` and `alpha=0.1`.

**Result:** held-out value_loss got *worse* with higher beta, consistently, at both alpha settings. Not a fluke of one config.

**Why, per a literature review done this session (not just intuition):**
- AlphaGo/AlphaZero (podcast deep-dive on the actual algorithm) and Gumbel MuZero (Danihelka et al., ICLR 2022 — read in full) both use search to refine the **policy** target (visit-count / completed-Q distillation); value training stays on real self-play outcomes, untouched by search. Gumbel MuZero in particular is a close structural analog to our own root-level sequential-halving design, and it still doesn't touch value.
- Two more papers were found showing search→**value** distillation genuinely *can* work: Veness et al. (NeurIPS 2009, "Bootstrapping from Game Tree Search" — the Meep chess program reached master level training a linear heuristic purely from alpha-beta-backed self-play) and Willemsen et al. ("Value targets in off-policy AlphaZero: a new greedy backup" — A0GB beats the standard outcome target on Connect-Four/Breakthrough, and finds the optimal policy in a tabular domain where the standard target fails). EfficientZero V2 (Wang et al., ICML 2024, read in full) also does this via Search-Based Value Estimation (SVE).
- The common thread in all three working examples: they run inside a **continual, iterating self-play loop**, and where stated explicitly, the motivation is correcting **staleness** — a replay-buffer entry's outcome becomes a worse label as the policy that generated it falls further behind the current one. Meep also labels *every node visited during search*, not just the root.
- **None of that holds in our setup.** Rollouts were generated once, from one fixed checkpoint, against a static human-game dataset that doesn't go stale the way a self-play buffer does. Only the root of each sampled ply gets labeled. The mechanism that makes search→value pay off elsewhere is largely absent from how Phase 1a was run.

**Revised plan, not abandoned:** the technique isn't disproven, the one-shot offline framing is the likely culprit. Next attempt (not started) should look more like periodic re-generation with the current checkpoint (closing the loop, Expert-Iteration-style) and/or whole-subtree labeling, rather than more data through the same flat, one-shot, root-only pipeline.

## 3. Visit-adaptive search lambda (`c_visit`): tried, reverted

Idea: instead of a fixed `value_rerank_lambda` throughout a search, shrink its weight as an arm accumulates `evals_spent` within that search (`effective_lam = lam * c_visit / (c_visit + evals_spent)`), borrowed from Gumbel MuZero's visit-adaptive `sigma(q_hat)`. Implemented in `HalvingConfig`/`_score_arm` with a unit test proving the mechanism (a low-prior, higher-value arm can survive elimination under the adaptive version that it wouldn't under fixed lambda).

**Screening (budget=256, cheap):** a 12-cell grid (`lam0 in {0.05,0.1,0.2}` x `c_visit in {none,25,50,100}`) at 15 games/cell suggested several configs beating the fixed-lambda baseline; a 45-game re-run of the top 3 showed most of that gap was noise, but `lam0=0.1, c_visit=100` held up as consistently above a coin flip.

**Critical finding:** `c_visit`'s effect depends on the *ratio* `evals_spent / c_visit`, and `evals_spent` scales linearly with `search_budget`. Carrying `c_visit=100` (tuned at budget=256, where max `evals_spent` per arm is ~60) unscaled to production budget=2048 (where max `evals_spent` is ~480) collapsed search quality: 0.10 score rate vs. SF2200, against an established 0.595 baseline — the same Goodhart-style failure the original `lambda=0` sweep found years ago, arrived at indirectly. Re-derived a budget-scaled value (`c_visit=800 = 100 * 2048/256`) which tested at 0.65 on a 10-game smoke test, but a 100-game production confirmation (killed at 86/100, score trending to ~0.57) showed it did **not** hold up against the 0.595 baseline either.

**Outcome:** reverted completely (`be1f9e1`) — removed `HalvingConfig.c_visit`, the CLI flag, and both tests. `value_rerank_lambda=0.05` (fixed, no adaptivity) remains the only supported behavior. Worth remembering as a general lesson: a small-budget hyperparameter screen is not a substitute for testing at the actual deployment budget, even when the scaling relationship is mathematically well-understood.

## 4. Standalone value-net: removed (dead code)

Audited and confirmed the standalone `ValueNet`/Stockfish-eval-distilled model (`src/imba_chess/model/value_net.py`, `scripts/train_value_net.py`) was trained (checkpoints existed) but never actually wired into any shipped config — `value_net_checkpoint` was always unset, so `value_net_alpha` sitting in configs was a silent no-op. Removed entirely (`25cd103`): the model, its trainer, its dataset loader, the blend logic in `position_evaluator.py`, all CLI/config plumbing, and their tests. The README's "Standalone value net" section and the `[value_net]` config section are gone accordingly.

## Staged plan going forward

1. **Tonight** (in progress): `alpha=0.1` continues training + plays 200 games vs. SF2200 — first playing-strength evidence for the alpha change.
2. **If that holds up:** a few more nights of rollout generation (still from a fixed checkpoint) + beta-blend fine-tuning on top of the improved `alpha=0.1` base, to see whether Phase 1a's conclusion changes once the base value head is itself better-calibrated.
3. **Only if that shows real gains:** invest in closing the loop — generate rollouts nightly with whatever the current-best checkpoint is (not a fixed starting point), fine-tune during the day, regenerate the next night with the updated model. This is real infrastructure work, worth doing only once there's evidence the loop pays off rather than just adding complexity, per lesson 3 above.
