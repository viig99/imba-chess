# Search-Backed Policy Distillation (Expert Iteration, Phase 1b) — Design

## Purpose

Phase 1a (`2026-07-07-expert-iteration-distillation-design.md`) distilled
`value_search_halving`'s per-position value backup into the value head via a
`β`-blend against real game outcomes. It was tried at both `α=0.9` and the
newly-adopted `α=0.1` base and, at production scale (100 games, SF2200,
budget=2048, depth=8), it did not beat the 0.595 pure-head baseline at any
`β` tried — held-out value loss got *worse* with higher `β`, consistently.
The abandoned `c_visit` (visit-adaptive search-lambda) experiment failed the
same way for a different reason: a ratio tuned at one search budget silently
meant something different at another.

A multi-session literature review (Gumbel MuZero, EfficientZero V2,
Veness/Meep 2009, Willemsen/A0GB, plus an external discussion thread the
user brought back citing Spigler 2024's Proximal Policy Distillation)
converged on a specific, better-grounded explanation: every literature
example of search→**value** distillation that actually works either runs
inside a continual self-play loop or exists to correct off-policy replay
staleness — neither property holds in Phase 1a's one-shot offline setup.
Search→**policy** distillation, by contrast, has a *proven* single-round
improvement guarantee that holds even from a frozen checkpoint (Danihelka
et al., "Policy improvement by planning with Gumbel", ICLR 2022,
arXiv:2202.01344 — read in full this session). This design is Phase 1b:
distill `value_search_halving`'s root-arm search outcome into the **policy**
head instead of the value head, using the same rollout infrastructure
already built for Phase 1a.

## Relationship to prior work: Gumbel MuZero

Two mechanisms from the paper are adopted directly, and one is deliberately
simplified. Quoting/paraphrasing the paper's own formulas (its §3.3–3.4,
§4):

**Root candidate selection must be stochastic, not deterministic top-k.**
The paper's own counterexample (§3.2, "Example 1"): `q=(0,0,1)`,
`π=(0.5,0.3,0.2)`, deterministic top-2 by prior picks `{0,1}` (both `q=0`),
missing the only good move (`q=1`) and scoring *worse* than the raw prior
(`E[q]=0 < 0.2`). Their fix, the Gumbel-Top-k trick (§2, eq. 3-5): sample
`n` actions without replacement by adding i.i.d. `Gumbel(0)` noise to each
action's log-prior and taking the top-`n` by the perturbed score — an
unbiased sample from `π`. This is already implemented in this codebase
(`HalvingConfig.gumbel_root_sampling`, default `True` for rollout
generation only, `select_value_search_halving`'s `_gumbel_top_k_order`) —
no new work needed here, called out for context.

**The distillation target — `completedQ`, adopted with one simplification.**
Eq. 10: `completedQ(a) = q(a)` if `a` was visited (searched), else `v_π`
(the value network's own unsearched estimate) for every other action in the
full action space. Eq. 11: `π' = softmax(logits + σ(completedQ))`, with `σ`
a monotonically increasing transformation; Eq. 8 gives their concrete
instantiation, `σ(q̂(a)) = (c_visit + max_b N(b))·c_scale·q̂(a)` — a scale
that grows with the search's own realized max visit count, held constant
(`c_visit=50, c_scale=1.0`) across every simulation-count setting they test
(2 to 200), because it's relative to *that search's own* statistics, not an
absolute constant carried over from a different budget's tuning run (the
mistake `c_visit`, the unrelated abandoned search-lambda feature, made).

Our `α` (the paper's `σ`, renamed to avoid clashing with this repo's
existing `value_weight_alpha`) is simplified to a **single fixed constant**,
swept over a small grid rather than made visit-adaptive — the project's own
explicit, repeated lesson from the `c_visit` revert, and the user's
standing instruction to prefer the simplest implementation the paper's
actual guarantee supports. See §Sigma sweep protocol below.

**Scope reduction: arm-only, not full action space.** The paper's
`completedQ` covers every legal move. Our rollout rows store only the arms
search actually examined (`arm_move_uci`/`arm_backed_value`/`arm_evals_spent`/
`arm_log_prior` — Gumbel-sampled `top_m` plus a forcing-move floor, ~16-20
per position, variable length). Building the full-legal-move version would
require a new rollout field (the complete legal move list per sampled ply)
and a `v_π`-fill/gather step over 20-40+ moves. Per the paper's own math,
`v_π` applied uniformly across every *non-arm* action only ever acts as a
shared baseline for that subgroup relative to the individually-scored arms
— it does not change which non-arm moves are relatively favored against
each other. Phase 1b restricts the target to the arm subset only:
every move outside the searched arms keeps whatever probability mass the
model's own current logits already assign it, unchanged. This is exactly
the existing Phase 1a value-loss pattern (`has_rollout_value_target`/
`value_target_soft`), applied to a policy gather instead of a fixed 3-class
softmax, and needs no new rollout data.

**Target base: live, not frozen.** The paper's guarantee is stated relative
to the *same* network doing the planning — policy improves relative to
itself. Our rollout's `arm_log_prior` is frozen from whichever checkpoint
generated it (currently `checkpoint_23`). At the first training step of a
Phase 1b run resumed from that same checkpoint, frozen and live are
identical; they diverge only as training moves the checkpoint away from
where the rollout was generated — the same staleness axis Phase 1a's
post-mortem was about. Phase 1b uses **live** current-model logits
(gathered at the arm vocab ids, detached for the target side) as the
target's base, tracking that drift instead of freezing it. `arm_log_prior`
stays in the parquet as provenance/debugging data, unused by this loss.

## Roadmap update

Supersedes Phase 1b as originally sketched in the 2026-07-07 doc's Part 5
(`evals_spent`-normalized visit-count target with a confidence-margin gate
against the human move). That version predates this session's Gumbel MuZero
reading; the `completedQ`-softmax-KL formulation below is the current
design and the one to build.

| Phase | What changes | Status |
|---|---|---|
| 1a | Value target: `blend(real_outcome, search_backed_value; β)` | Built, evaluated, did not beat baseline at any β. Not reverted (β=0 is a no-op default); deprioritized pending new evidence. |
| **1b** (this doc) | Policy target: `completedQ`-softmax over searched arms, KL-distilled, mixed with existing human CE | Designed now, build next |
| 2 | Value improvement via real outcomes of games played by the Phase-1b-improved policy+search (self-play or eval games), or n-step bootstrapped search values at a *different* ply than computed | Deferred pending 1b's result — explicitly out of scope here |

## Part 1: Rollout data — already built, no changes

`src/imba_chess/data/rollout_store.py` / `scripts/generate_search_rollouts.py`
already store everything Phase 1b needs: per-arm `move_uci`/`backed_value`
(=`q̂`)/`evals_spent`/`log_prior` (variable-length, no padding/truncation —
fixed this session), full search-config provenance
(`search_budget`/`search_top_m`/`search_max_depth`/`search_refutation_top_r`/
`search_expand_top`/`search_lam`), and `--gumbel-root-sampling` (default
`True` for generation). A 5-night rollout-generation cron job against
`checkpoint_23` is producing the corrected-schema data this design consumes;
no new generation work is in scope here.

### Checkpoint-consistency guard (added post-review, 2026-07-13)

Wu, Han & Cai ("Lightning OPD", NVIDIA, arXiv:2604.13010) prove that
on-policy distillation requires **teacher consistency**: the model that
produced the SFT/reference training data and the model that scores
distillation targets must be the same, or a gradient bias `G·σ_Δ`
(Theorems 3.8-3.9) degrades training — worse for offline/fixed-rollout
variants, since their drift term scales with `χ²(π_θ‖π_ref)` over
model-generated trajectories that grows as the student moves away from the
frozen rollout distribution. Their proof is for trajectory-level,
advantage-weighted policy-gradient distillation over the model's *own*
autoregressive rollouts — a materially different structure from Phase 1b's
per-position supervised soft-label KL over *human* game positions (an
external dataset, never generated by the model, so no `χ²(π_θ‖π_ref)`-style
drift term exists in our loss at all). The precise bound does not transfer
directly. The underlying practical concern does, by loose analogy: a
rollout's `arm_backed_value`/`q̂` reflects whichever checkpoint's search
produced it, and training against that target while resumed from a
*different* checkpoint would be a real reference mismatch, just not the
exact one this theorem bounds.

Phase 1b's current single-round design already satisfies this trivially by
construction — `checkpoint_23` both generates the rollouts and initializes
training, so there is only one checkpoint involved. This only becomes a
live risk if/when a future iterative loop (Part 2 territory: nightly
rollout regeneration from the current-best checkpoint) is built. Guarding
against it now is cheap: `RolloutRow.checkpoint` already stores the
generating checkpoint's path (present since Phase 1a's original schema).
Any training run that loads a `rollout_path` must assert every row's
`checkpoint` field matches the checkpoint training resumes from, raising a
clear error on mismatch rather than silently training against a
stale-teacher target. See §Testing and §Out of scope.

## Part 2: Target construction — new module

`src/imba_chess/data/policy_target_kl.py` (sibling to
`value_target_blend.py`), one pure-data function, no torch:

```python
POLICY_KL_MAX_ARMS = 24  # safely above search_top_m=16 + typical forcing-floor extras

def arm_vocab_ids_and_qhat(
    row: RolloutRow, move_vocab: MoveVocab, max_arms: int = POLICY_KL_MAX_ARMS,
) -> tuple[list[int], list[float], list[bool]]:
    """Maps arm_move_uci -> move_vocab id, pads/truncates to max_arms.

    Returns (arm_ids, arm_qhat, arm_mask), each length max_arms. Any arm
    whose move resolves to move_vocab's <unk> fallback is excluded (not
    included with a bogus id) -- an UNK slot in the target's softmax would
    corrupt it, and this is a real code path, not a hypothetical: MoveVocab
    always has an <unk> fallback for a move string that fails to map. Rows
    with more than max_arms real arms keep the first max_arms in the row's
    existing order (search's own return order -- Gumbel-sampled top_m first,
    forcing-floor extras appended after). Padding slots get a dummy id (0)
    and arm_qhat=0.0, masked out via arm_mask=False.
    """
```

## Part 3: Loss integration

**`event_builder.py`** — extends the existing `rollout_lookup`-gated block
(already present for `has_rollout_value_target`) to also call
`arm_vocab_ids_and_qhat` and attach four new per-token batch arrays:
`policy_kl_arm_ids [seq_len, MAX_ARMS]`, `policy_kl_arm_qhat [seq_len, MAX_ARMS]`,
`policy_kl_arm_mask [seq_len, MAX_ARMS]`, `has_rollout_policy_target [seq_len]`.
Same `if self.rollout_lookup is not None:` gate as today; zero cost when
rollouts aren't configured.

**`hstu_model.py`** — new block inserted after the existing policy-loss
computation (`policy_logits` and `valid_mask` already in scope at that
point):

```python
if has_rollout_policy_target is not None:  # batch key presence, not tensor content
    student_arm_logits = torch.gather(policy_logits, dim=-1, index=policy_kl_arm_ids)
    target_arm_logits = student_arm_logits.detach() + policy_kl_sigma * policy_kl_arm_qhat
    neg_inf_fill = torch.finfo(student_arm_logits.dtype).min
    masked_target_logits = target_arm_logits.masked_fill(~policy_kl_arm_mask, neg_inf_fill)
    masked_student_logits = student_arm_logits.masked_fill(~policy_kl_arm_mask, neg_inf_fill)
    target = F.softmax(masked_target_logits, dim=-1)          # 0 exactly on padding slots
    student_log_probs = F.log_softmax(masked_student_logits, dim=-1)
    per_token_policy_kl_loss = -(target * student_log_probs).sum(dim=-1)

    policy_kl_token_weights = (
        has_rollout_policy_target.to(per_token_policy_kl_loss.dtype)
        * valid_mask.to(per_token_policy_kl_loss.dtype)
    )
    policy_kl_loss_sum = (per_token_policy_kl_loss * policy_kl_token_weights).sum()
    policy_kl_weight_sum = policy_kl_token_weights.sum().clamp_min(1.0)
    policy_kl_loss = policy_kl_loss_sum / policy_kl_weight_sum
    output["policy_kl_loss"] = policy_kl_loss
    total_loss = total_loss + expert_iteration.policy_kl_weight * policy_kl_loss
```

Masking mechanics: padding slots get `-inf`-filled logits (via `dtype`'s
min, not literal `-inf`, to stay finite-arithmetic-safe under autocast) on
*both* target and student before their softmax/log_softmax, so they receive
exactly `0` probability on both sides — no separate masked-softmax helper
needed, and no risk of a padding slot contributing a spurious KL term. The
`weighted_mean`/`clamp_min(1.0)` reduction is copied verbatim from the
existing `value_loss_sum`/`value_weight_sum` pattern immediately above this
block in the same file, for consistency.

Branches only on the batch's static key structure (`is not None`, a Dynamo
guard), not on tensor content — the same discipline this repo's own
`d39b678` fix ("remove data-dependent branch breaking
`torch.compile(fullgraph=True)`") already established and enforces
elsewhere in this file. `torch.gather`/`torch.where` over fixed
`MAX_ARMS`-width tensors keeps shapes static regardless of how many real
arms a given token has.

The human policy CE over the full ~1970-move vocab (existing, unchanged) is
always active on every token; this KL term is purely additive, only on
rollout-covered tokens — mixing is required, not optional, since that prior
entropy is what feeds the search's own `search_refutation_top_r` forcing
floor at inference time.

## Part 4: Config

Extends the existing `ExpertIterationConfig` (`beta`'s sibling section, not
a new one — the rollout schema was deliberately built so value-only,
policy-only, both, or neither can run from the same parquet):

```python
@dataclass(frozen=True)
class ExpertIterationConfig:
    rollout_path: Optional[str] = None
    beta: float = 0.0                    # existing value-blend weight
    policy_kl_weight: float = 0.0        # new; 0.0 = today's exact behavior
    policy_kl_sigma: float = 1.0         # new; fixed constant, not visit-adaptive
```

Both new fields default off; every existing config stays byte-identical.

## Part 5: Sigma sweep & checkpoint-selection protocol

**Base checkpoint:** `checkpoint_23` (`best_hr10_checkpoint_23_hr10=0.9564.pt`,
α=0.9, the strongest known checkpoint). The `value_weight_alpha` question
stays closed; this doesn't reopen it.

**Probe sweep:** 3 short fine-tunes from `checkpoint_23`, 10,000 steps each
(matching the α=0.9→0.1 probe's own precedent for direct comparability),
`policy_kl_weight` fixed at `0.1` across all three (isolates `σ` as the one
swept variable — the KL term's natural scale differs from the 3-class value
loss's `value_loss_weight=0.15`, since it's a ~24-way softmax rather than a
3-way one, so `0.15` is not a safe transplant), `σ ∈ {0.5, 1.0, 2.0}`. New
configs `config/imba_chess_exit_policy_sigma050.toml` /
`sigma100.toml` / `sigma200.toml`, mirroring the existing
`imba_chess_exit_probe_beta0XX.toml` naming convention, each with its own
`checkpoints_exit_policy_sigmaXXX/` dir.

**Probe evaluation — two signals:**
1. Held-out policy-KL loss on val split (`scripts/eval_policy_kl_loss.py`,
   new, structurally identical to `eval_value_by_progress.py` but reporting
   this loss instead of value loss) — cheap, fast first-pass filter.
2. Cheap-budget SF-ladder spot check: `search_budget=256`, `depth=4`, ~20-30
   games at SF1800 — same "cheap screen before expensive confirm" shape
   already used for the `c_visit` budget=256 screening pass. Directional
   signal only, not a final verdict.

**Winner → full run:** the best-looking `σ` on both signals gets a full
training run (same `--max-steps` shape as the α probe's follow-up).
Existing periodic `last_checkpoint_*.pt` saving is left untouched — no
Ignite training-loop changes.

**Final checkpoint selection is manual and offline, never `hr@10`:** a
successful search-distilled policy is *expected* to diverge from human
moves exactly where search finds their mistakes, so `hr@10` may flatline or
drop while the SF2200 ladder score rises — using it as a selection signal
would actively discard the checkpoints where Phase 1b is working. After the
full run: take a handful of late-training snapshots, run each through the
cheap-budget spot check to shortlist 1-2 candidates, then confirm the
actual pick with the full 100-game/SF2200/budget=2048/depth=8 protocol —
identical to how α=0.1 was judged this session.

## Testing

- Checkpoint-consistency guard: loading a `rollout_path` whose rows'
  `checkpoint` field doesn't match the training run's resume checkpoint
  raises a clear error; a matching checkpoint loads without incident. This
  is a training-time (`train.py`) check, not a per-row data-transform test.
- `arm_vocab_ids_and_qhat`: correct id mapping; padding (mask=False on
  padding slots, dummy id doesn't collide with a real move); truncation
  (row with >24 arms keeps exactly the first 24, in order); UNK exclusion
  (a fabricated unmappable UCI string is dropped, not included with a bogus
  id).
- `event_builder.py`: a rollout row with policy arms produces the four new
  batch arrays at the right token position; a token with no rollout
  coverage gets `has_rollout_policy_target=False` and an all-padding row —
  same shape as the existing `has_rollout_value_target` tests.
- `hstu_model.py`:
  - **Backward-compat invariant (critical):** `policy_kl_weight=0.0` (the
    default) produces byte-identical `total_loss`/gradients to today's
    model — same pattern as the existing `beta=0.0` invariant test. This is
    what protects every config that doesn't opt in.
  - Detach correctness: target-side gather contributes no gradient (a
    moving-target/degenerate-loss regression would show up as gradients
    flowing twice through the same logits).
  - Masking: rows with `has_rollout_policy_target=False` or fully-masked
    arm slots contribute exactly zero to the loss sum.
  - Compiles and trains correctly under `torch.compile(fullgraph=True)` —
    verified by actually compiling a tiny model with this loss active, not
    just inspecting the code for branches.
- `test_expert_iteration_end_to_end.py`: one new case with both `beta>0`
  and `policy_kl_weight>0` set simultaneously on the same rollout row — both
  loss terms present, finite, nonzero, and a gradient step actually moves
  the relevant logits, proving "value-only, policy-only, or both from the
  same rollout" works end to end.

## Acceptance protocol

Same three-stage funnel discipline as Phase 1a: label/loss-level probe
signal (§Part 5) → cheap-budget spot check → full 100-game SF2200/2048/d8
confirmation against the 0.595 baseline. `hr@10` is never used as a
selection or acceptance signal anywhere in this pipeline — only the live
Stockfish-ladder score decides.

## Out of scope

- Phase 2 (value improvement via real outcomes of search-augmented-policy
  games, or n-step bootstrapped search values at a different ply than
  computed) — deferred pending Phase 1b's result.
- The paper's full visit-adaptive `σ(q̂(a)) = (c_visit + max_b N(b))·c_scale·q̂(a)`
  — deliberately simplified to a fixed constant per §Relationship to prior
  work; revisit only if a fixed `σ` demonstrably underperforms.
- Full-legal-move-space `completedQ` with `v_π` fill for non-arm moves — the
  arm-only simplification per §Relationship to prior work; would need a new
  rollout field (full legal move list per sampled ply) if ever pursued.
- Automated in-training-loop checkpoint selection (a periodic Stockfish
  ladder eval inside the Ignite loop) — manual offline spot-checking chosen
  instead, avoiding GPU/CPU contention with training on the single shared
  3070 Ti and avoiding new Ignite-loop engineering.
- Any change to `value_weight_alpha`, the standalone value net, or Phase
  1a's `β`-blend itself — all orthogonal, already-shipped or already-decided
  infrastructure.
- Regenerating rollouts mid-Phase-1b from an improving checkpoint (would
  turn this into an iterative/online loop) — this doc covers one round;
  the coarse-grained online-loop idea discussed this session stays deferred
  pending evidence any single round pays off. The checkpoint-consistency
  guard (§Part 1) is built now specifically because it's cheap, not because
  Phase 1b needs it yet — it protects whichever future round first resumes
  training from a checkpoint that may differ from the one that generated
  its rollouts.
