> Superseded by [the September 22 handoff](SELF_PLAY_INVESTIGATION_HANDOFF_2026-09-22.md), which includes the auxiliary-value run and completed SF2600 evaluations.

# Fresh-agent prompt: why is self-play not improving imba-chess?

You are taking over an ongoing chess-learning investigation. Work in
`/home/vigi99/CodeDir/imba-chess`. Diagnose why self-play fails to improve playing
strength and often regresses despite search improving a frozen policy. Read the
actual code, configurations, logs and artifacts; distinguish measured findings
from hypotheses, and actively look for bottlenecks we have not considered.

Do not just recommend familiar hyperparameter changes. Trace the full chain:
**position/history → neural predictions → search → stored targets/outcomes →
replay sampling → optimizer updates → changed predictions → playing strength.**
Identify where the benefit is lost using controlled, reproducible measurements.

## Starting checkpoint and user intent

Use the **trained flattened-board checkpoint** below as the default starting point
for NEW self-play analysis and experiments. The user explicitly wants this new
architecture/checkpoint going forward; do not default back to old ckpt34 or actor54.
Old checkpoints are historical controls, not the current research starting point.
Initialize stage-2 from these model weights; do not treat this supervised checkpoint
as a resumable stage-2 optimizer/replay state.

- Frozen checkpoint: `artifacts/self-play-handoff-2026-09-21/last_checkpoint_51750.pt`
- Update: 51,750 of the new supervised continuation.
- SHA-256: `b688ccdc16989873127a9bad5e124c1829cbde2ed396ea7266ec2cbda0cb469d`
- Identity: `artifacts/self-play-handoff-2026-09-21/checkpoint.json`
- Model/training config: `config/imba_chess_v4_laptop.toml`
- Exact validation config/log/results: `artifacts/self-play-handoff-2026-09-21/`
- This snapshot is immutable and separate from rotating live checkpoints. If a
  newer checkpoint is deliberately chosen, freeze it, record its hash, remeasure
  its metrics and clearly identify that change. Never evaluate a moving file.

The user considers high HR@16 sufficient to investigate self-play now. Respect
that research direction, while distinguishing human-move top-16 recall from
value correctness, legal-search candidate coverage and proven playing strength.
No matched SF2400 non-regression result exists yet for this new checkpoint.
Using it as the research default is not evidence that it passed that gate.

Metrics measured directly on the frozen update-51,750 checkpoint on September 21:
512 held-out games, 38,677 policy target positions, BF16 evaluation, identical
validation game set to the original baseline (batch packing differs).

| Metric | New frozen checkpoint | Original ckpt34 |
|---|---:|---:|
| MRR | 0.645487 | 0.684549 |
| HR@1 | 47.3615% | 51.5371% |
| HR@3 | 76.2598% | 80.0889% |
| HR@5 | 86.2813% | 89.5028% |
| HR@10 | 95.3771% | 96.9749% |
| HR@16 | 98.4875% | 99.0408% |
| Policy CE | 1.670444 | 1.493551 |
| Value CE | 0.619123 | 0.609560 |

These are checkpoint-matched measurements, not metrics from a nearby training step.
A separate full validation at update 50,000 covered 100,000 games / 7,607,425 policy
target positions: MRR 0.646899, HR@1 47.7528%, HR@16 98.3681%, policy CE 1.661322,
value CE 0.617768. We have not run the original baseline on that full set here.


The original baseline metrics were measured using the verified function-equivalent
flattened copy of ckpt34 at `artifacts/flatten-board-ckpt34/initial.pt`. That file
contains the old predictions, not the subsequently trained candidate. HR@N here
means recall of the observed human next move among the full vocabulary's top N
logits; it is not an engine move-quality metric. MRR is mean reciprocal rank.

## Current architecture and training history

Repository commit: `5b05214` (pushed to origin/main); inspect current HEAD/worktree.

- 52,388,996 unique parameters, static 1,970-entry UCI move vocabulary.
- Each board: 64 joint piece/square embeddings, width 64; two 4-head square-attention
  blocks; per-square LayerNorm; flatten 64×64 → 4096; learned Linear(4096,1024).
- Add embeddings for previous move, turn, castling, en passant, move-clock buckets
  and sequence token type; scale content and add learned positional embeddings.
- Eight causal HSTU history blocks, width 1024, 16 attention heads, dimensions 64
  per head; final shared LayerNorm. One history token per position, NOT per square.
- Policy: one bias-free Linear(1024,1970), tied to previous-move embedding weights.
- Value: Linear(1024,512), two residual pre-norm MLP blocks (512→1024→SiLU→512),
  LayerNorm, Linear(512,3), output order loss/draw/win. Search scalar is P(win)−P(loss).
- Auxiliary moves-left head: 1024→512→SiLU→1, supervised in stage1, unused by search.

The ONLY architectural change was replacing mean pooling into a 64-dimensional
board summary followed by 64→1024 projection with flattening followed by
4096→1024 projection. This added 4,128,768 parameters. Initially, the new projection
was initialized with 64 copies of W/64 and unchanged bias, preserving the old
function. CPU tests and a real FP32 GPU batch verified preservation (maximum
policy probability difference about5.13e-6; value probability difference1.43e-6).

Flattening is now the sole architecture. Pooling switches, standalone migration
CLI and automatic runtime mean-pool conversion have been removed. Generic strict
checkpoint loading and weights-only warm starts remain. Original mean-pooled
checkpoints cannot be loaded directly into current code. Historical audits have
frozen source snapshots. For old comparisons use the verified migrated ckpt34
copy or recreate historical execution in isolation; do not reintroduce a dual
architecture into production just to rerun an old experiment.

The continuation history matters:
- A short 1e-6 run made small validation improvements; it was discarded as the
  active branch when the user explicitly requested restarting from pristine
  ckpt34 at its saved LR 0.0004151649149149149.
- The active flattened branch started with original-equivalent weights and a
  **fresh optimizer**, at that constant LR. It does not restore ckpt34's old Adam
  moments or its decaying OneCycle schedule.
- Original ckpt34 used 40,960 tokens/update. Laptop continuation used 1024 tokens,
  then 1536 from update 10,000, then 1280 from 11,000 onward. No gradient accumulation.
- Current settings: 1280 total tokens/batch, maximum sequence length 512, positional
  capacity 513, four training workers, BF16 autocast with FP32 weights, full-model
  compilation disabled, weight decay .01 and clipping 1.0. Both heads and shared
  features train jointly. Preserve these facts when interpreting recovery.
- Policy accuracy fell sharply after the high-rate restart, then recovered only
  partially. For example: 42.83% at 1,500; 46.25% at 8,500; 46.92% at 14,000. Do not claim
  that more training necessarily restores baseline or that this is already an
  optimized starting checkpoint.
- This is still supervised Lichess/Stockfish training, not a new self-play run.
  Live run directory: `artifacts/checkpoints_v4_flatten_ckpt34_lr/`;
  current process/config in `active-run.json`; latest log `batch1280-training.log`.
  Training was paused at user request after update 53,250 on September 21.
  The saved optimizer/scheduler checkpoint is
  `artifacts/checkpoints_v4_flatten_ckpt34_lr/last_checkpoint_53250.pt`.
  GPU is available for investigation. The frozen, directly evaluated research
  starting point above remains update 51,750 unless deliberately changed.

## Existing evidence: search can help, learning still fails

### 1. Frozen-model search versus greedy policy

`artifacts/eval/search-improvement-2026-09-20/REPORT.md`:
Frozen old ckpt34, scale .1, top-m 16, depth 32, zero evaluation Gumbel noise,
FP32/TF32 off, 25 distinct openings with colors swapped per budget:

| Search simulations | W/D/L vs same model greedy | Score | Paired 95% interval |
|---|---|---|---|
|32|38/9/3|85.0%|77–92%|
|200|43/4/3|90.0%|83–96%|

Search improved this frozen model's playing strength. That does not prove that
its training targets transfer into a stronger learner.

### 2. Search targets scored independently

Same campaign: 100 positions across 10 complete **actor54-generated held-out monitor
continuations**, generated as fallback because historical remote replay could
not be accessed. These were NOT proven consumed training examples.

All legal moves were scored by unrestricted Stockfish, 1 thread, 64 MiB, fresh hash,
100,000 nodes per legal move, preserving full histories, move alignment and mates.
Mean search-target minus generating-policy expected-score gain:
- Unweighted: +3.926 percentage points; game-bootstrap 95% interval[+1.631,+6.391].
- Surprise-weighted: +5.452 points; interval[+1.891,+9.159].
- Positions improved 44%, worsened 4%, unchanged 52%.
- CP comparisons only where ALL legal moves have finite CP:45 positions.

These are small-sample engine proxies, not match win-rate gains. A few harmful
corrections remain. Do not infer successful learning transfer from these targets.

### 3. Controlled policy/value source swap: strongest localization so far

Read `docs/POLICY_REGRESSION_RESULTS_2026-09-20.md` and
`artifacts/eval/policy-regression-2026-09-20/REPORT.md`.
Old detached-value actor54 versus old ckpt34; 50 distinct monitor openings,
color swaps, 100 games per arm, 512 simulations, scale .1,top-m 16,depth 32,zero noise,
FP32,TF32 off. Every arm faced a fixed ckpt34-policy/ckpt34-value opponent.

| Arm | Policy source | Value source | W/D/L | Score |
|---|---|---|---|---|
|00|ckpt34|ckpt34|43/14/43|50.0%|
|10|actor54|ckpt34|44/11/45|49.5%|
|01|ckpt34|actor54|28/13/59|34.5%|
|11|actor54|actor54|33/9/58|37.5%|

Joint opening-pair bootstrap contrasts:
- Policy effect:−0.5pp,95% interval[−7.5,+7.0].
- Value effect:−15.5pp,[−23.0,−7.0].
- Both:−12.5pp,[−20.0,−4.5].
- Interaction:+3.5pp,[−8.0,+14.5].

Each source ran its **complete network** at root and every evaluated node, with
independent continuation/KV caches and identical path histories. No head transplant
or cross-model hidden representation sharing. Terminal values came from rules.
Same-source composition matched ordinary search; exact budgets and perspectives
were tested. All 400 games completed. This identifies harmful value predictions
under that protocol; it does NOT isolate the private value head from its encoder,
prove a particular value-loss defect, or establish universal additive contributions.

A separate 100-game greedy actor54-vs-ckpt34 match scored 51.5% (30/43/27;
paired95% interval 45.5–57.5): inconclusive for raw-policy strength regression.
On the cached 100 positions, actor54 policy quality declined by−0.902pp for the
full distribution and−3.046pp for the greedy move. Three extreme losses account
for almost all net distribution decline. This selected population is not all chess.

The 11 arm repeated the earlier 37.5% result on the same inputs; do not count it
as an independent replication or double the sample size.

### 4. Value targets, gradients, clipping and detachment

Read `docs/SELF_PLAY_VALUE_AUDIT_2026-09-20.md` and
`artifacts/loss_audit_2026-09-19/` (the scale .03 run, not detached actor54).

- Stage1 `winpercent_wdl` maps CP through a fixed logistic into loss/win with
  **draw probability identically zero**. It is not Stockfish's full WDL distribution.
- Stage2 trains accepted continuation positions on actual terminal one-hot WDL,
  from the side-to-move perspective; opening-prefix tokens are context only.
- At the start of the audited.03 run: observed draw fraction .328, predicted draw
  probability~.00001, value CE 5.713. A substantial target-semantics transition.
- Policy CE fell 1.29229→1.21871 while target entropy fell 1.12160→1.03039;
  learner-target KL rose .17069→.18832. Falling CE did not demonstrate better fit.
- Value CE .77129→.75845; Brier .44520→.44692. Whole-game batches have highly
  correlated labels and variable outcome mix. Some large raw gradient norms.
- Nine completed replay games (three per outcome), 1,873 positions: independent
  board/side-to-move label checks passed. Sampled evidence, not blanket immunity.
- Three fixed-batch probes: shared-trunk value/policy raw gradient norm ratios
  about 2–5×; gradient cosines were positive, not evidence of opposing gradients.
- Both joint training and a detached-value-feature run were tried; detached
  actor54 still regressed. Do not propose detachment as an untried fix or declare
  shared-gradient interference the established primary cause.
- Detaching value gradients does not freeze the value function: policy updates
  still change its input features. Freezing just the private head also does not
  freeze values. A stable value control requires a complete frozen evaluator.
- Current source reverted detachment. Joint losses update the shared trunk;
  private value-head gradients and combined backbone/policy gradients are clipped
  separately. These are parameter groups, not isolation of loss contributions.
- A `value_weight=0`/frozen-private-head path and frozen policy-surprise comparison
  were prepared. Inventory actual outputs before claiming completed results:
  `artifacts/eval/remote5090-q01-value0-frozen-2026-09-20/` and
  `artifacts/policy_surprise/ckpt34-frozen-pilot/`. Preparation is not evidence.

Earlier human-outcome training used progress-based value loss weighting; engine
position-target training removed it. A historical alpha .1 vs .9 experiment favored
milder weighting on held-out loss. We discussed reintroducing mild weighting for
self-play, but did NOT establish it as the cause or apply it as the current fix.
Discounting outcome targets and weighting loss are different interventions.

### 5. Gumbel/MCTS, mctx, exploration and scale audits

Read `docs/GUMBEL_MCTX_AUDIT_2026-09-19.md` and
`artifacts/gumbel_audit_2026-09-19/`.
Pinned mctx reference: `88f92056a420c2673bed282f5a0c00211f126e78`.

- 22 synthetic multi-depth alternating-player trees: exact selected moves/root
 visit vectors; target/Q errors at floating-point rounding scale.
- 200 randomized Q/root/interior cases and 32,768 scheduling cases passed.
- Checked sequential-halving budget scheduling, completed/missing Q, root scoring,
 interior selection, policy targets, alternating value perspective, terminal
 absorption and deterministic depth-cutoff reuse.
- Compact legal-action representation matched. Padded invalid actions can change
 upstream mctx's Q normalization extrema:9/2,136 cached targets changed in one
 counterfactual, max TV .5653, mean TV .000911. Not a measured strength explanation.
- Local Q transform uses node-local range normalization and `(50+max_visits)*scale`.
 Tiny-Q amplification is worth targeted testing but resembles the reference;
 simple scalar rescaling can be canceled by normalization.
- Evaluation formerly used Gumbel noise; default was corrected to zero. Regression
 survived: actor73 noisy 32.0% vs zero 31.5%, actor53 noisy 38.0% vs zero 37.0%.
- Training collector uses noisy Gumbel search and plays its recommendation. The
 reference paper also used early-game visit-count exploration. Compare actual
 exploration behavior, not just algorithm names.
- Policy-surprise roundoff bug was fixed: normalize saved priors/targets in double
 precision; KL<=1e-12 is numerical zero. The affected weighting was absent from
 the original scale 1 run, so that bug cannot explain every regression.
- Our surprise method weights policy loss only, using stored actor priors and full
 game normalization; it is not an exact copy of KataGo's sampling pipeline.

`docs/SEARCH_SCALE_SCREEN_2026-09-19.md`: exploratory engine-scored scale sweep,
200 simulations, training noise, 20 pilot + 30 held-out human-prefix positions.
Independent 30-position target gains: scale .01 +.83pp; .03 +2.07pp; .1 +2.34pp;
.3 +1.46pp;1 −.05pp. The .1−.03 difference was inconclusive. Worst-position
losses grew with aggressive scale. Four-times-higher Stockfish scoring preserved
broad findings on a ten-position subset. This does not identify a universal
optimal scale, and smaller-scale online runs also failed to establish improvement.

LightZero was previously discussed as a comparison, including:
- https://github.com/opendilab/LightZero/tree/main/zoo/board_games/chess
- https://github.com/opendilab/LightZero/blob/main/zoo/board_games/chess/config/chess_alphazero_bot_mode_config.py
- Its separate Gumbel MuZero implementation under `lzero/`.

Locate any existing local comparison evidence, then read and pin relevant upstream
code yourself. No completed quantitative LightZero parity report was located for
this handoff; do not invent one. Distinguish chess AlphaZero/bot-mode examples from
Gumbel MuZero and do not transplant hyperparameters across different algorithms.
The mctx audit is strong component evidence, not proof the whole learning system
is correct. Revisit audited components when new end-to-end evidence warrants it.

### 6. Historical learning-transfer test remains blocked

Actor54 learner-state tensors matched its frozen checkpoint and 460 games had
positive per-game training exposure counts. However, genuine earlier detached-run
replay trajectories and qualifying generating checkpoints were unavailable locally;
read-only remote SSH attempts were refused. No historical transfer conclusion was
made. New actor54-generated targets or the available scale .03 replay are invalid
substitutes for those consumed examples. `transfer` in the diagnostic CLI is an
exposure/provenance gate, not a completed historical scoring pipeline.

For NEW experiments from the flattened checkpoint, fix this evidence gap by
saving complete trajectories, generating priors, exact search targets, outcomes,
checkpoint ancestry, sampler choices and positive exposure counts from the start.

## What to investigate next

Build an evidence-ranked hypothesis list, including alternatives to our existing
value/architecture explanation. Prefer small experiments that distinguish causes:

1. Establish search benefit and value behavior on the NEW checkpoint; do not assume
   historical ckpt34/actor54 results transfer to it.
2. Freeze replay and measure before/after on identical targets: policy KL, engine
   distribution quality, greedy quality, WDL loss/calibration and search strength.
   Separate improvement on consumed examples from generalization on held-out data.
3. Isolate policy learning with a complete frozen value network; compare value
   updates under a fixed policy. Reuse full-network composition, not head swapping.
4. Inspect search leaves/siblings as well as replay roots: value ordering, draw
   calibration, saturation, target scale, completed-Q behavior and out-of-distribution
   search states. Engine expectations are proxies, not calibrated self-play outcomes.
5. Audit effective optimization: token/game weighting, batch correlation, actual
   parameter updates after Adam/clipping, optimizer state, fresh-vs-stale replay,
   actor lag, reuse/exposure accounting, schedule timebase, shared feature drift,
   target sharpening, catastrophic forgetting and small-batch/high-LR effects.
6. Check histories and state information all the way through replay/prefill/decode:
   side-to-move signs, pre/post-move alignment, legal IDs, cache ownership, repetition,
   draw rules, context limits, terminal versus administrative stops, target serialization
   precision, train/eval distribution differences, and leakage between monitor starts
   and training inputs. Consider missing features and bottlenecks we have not named.
7. Distinguish genuine algorithm bugs from unfavorable objectives, insufficient or
   biased data, weak exploration and inappropriate evaluation. Do not assume the
   narrow board summary was the root cause merely because it was changed.

Keep comparisons matched on starting weights, data, exposure and opponent/search
protocol. Record runtime and neural work as well as tree budgets. Use distinct
opening pairs with color swaps, complete games and clustered confidence intervals;
leave administrative limits/errors unlabeled. Do not optimize repeatedly against
only the old 50 monitor openings without a fresh confirmation set.

For strong-engine position scoring, existing audits used unrestricted Stockfish,
1 thread,64MiB,fresh hash,100k nodes per legal move; preserve mate scores and only
compare CP when all relevant moves have finite CP. Keep SF2400 match testing
separate from unrestricted engine-scored target diagnostics and Gumbel match tests.
The historical halving recipe is pinned in `config/eval_flatten_sf2400.toml`.

## Deliverables and working constraints

Start by reconciling this handoff with actual files and run provenance. Produce:
- An inventory of completed experiments versus planned/blocked ones.
- A concise causal map of supported findings, unknowns and discriminating tests.
- Results of the highest-information feasible diagnostics on the frozen new model.
- A concrete explanation supported by evidence, or a narrowed uncertainty with
  the next decisive experiment; do not manufacture a single cause.
- If a fix is justified, a small isolated implementation with meaningful tests
  and matched before/after evaluation before adopting it in ongoing training.

Use existing standalone diagnostics where suitable; no unnecessary parallel
production implementations or sweeping rewrites. Preserve old artifacts. Some
September20 audit scripts/docs/tests remain **untracked locally** and were excluded
from the flattened-model commit; they may be absent in a fresh remote clone.
Re-read `git status` and archive needed local evidence before relocating work.
Original mean-pooled checkpoints require archived historical code, not current
strict loaders. Do not silently discard mismatched weights.

The laptop is an 8 GiB RTX 3070 Ti; supervised training is now paused. Check current
processes/configuration before GPU work; GPU operations require execution outside
the sandbox in this environment. Do not resume or reconfigure pretraining merely
to test a conjecture. Keep diagnostic training isolated from the saved learner. Save results
incrementally with identities that reject incompatible resumes.

Primary local reading list:
- `src/imba_chess/self_play/{collector,dataset,losses,trainer,streaming,runtime}.py`
- `src/imba_chess/eval/{gumbel_search,inference_runtime,position_evaluator}.py`
- Native search selectors/backup and `tests/fixtures/gumbel/`
- `src/imba_chess/model/{hstu_model,hstu_attention,checkpoint}.py`
- `docs/GUMBEL_MCTX_AUDIT_2026-09-19.md`
- `docs/POLICY_REGRESSION_RESULTS_2026-09-20.md`
- `docs/SELF_PLAY_VALUE_AUDIT_2026-09-20.md`
- `docs/SEARCH_SCALE_SCREEN_2026-09-19.md`
- `docs/POLICY_SURPRISE_WEIGHTING.md`
- `docs/SELF_PLAY_STREAMING_RUN_2026-09-17.md`
- `docs/SELF_PLAY_STREAMING_ASSESSMENT_2026-09-18.md`
- `docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md`
- `docs/VALUE_TARGET_WINPERCENT_HANDOFF.md`
- `docs/FLATTEN_BOARD_CONTINUATION.md`
- `artifacts/eval/search-improvement-2026-09-20/REPORT.md`
- `artifacts/eval/policy-regression-2026-09-20/REPORT.md`

Treat historical recommendations in those documents as dated proposals. The
current task is to find why learning is not producing stronger play, not to defend
our previous hypotheses or assume that a high HR@16 settles readiness.
