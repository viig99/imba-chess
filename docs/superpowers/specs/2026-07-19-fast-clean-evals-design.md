# Fast clean evals: batching + node-limited Stockfish (design, 2026-07-19)

## Problem

The SF-ladder eval protocol has two structural flaws exposed on 2026-07-19:

1. **Load sensitivity**: Stockfish moves on a 0.05s wall-clock budget, so
   opponent strength depends on background machine load. Measured directly:
   identical code (proven 98/98 move-identical) scored 0.575@100 on a loaded
   machine vs ~0.42-0.52 on clean samples. Every historical number carries an
   unknown load bias, including the 0.595 anchor.
2. **Speed**: single-game evals (~16-25 s/game at 2048/d8 even after the
   search speedups) make large samples expensive — yet 30-game samples have
   ±0.09 SE and produced a false regression scare (0.333 vs 0.519 on
   behaviorally identical code). Phase-1b needs to resolve ~0.03 deltas,
   i.e. 200-300-game runs, routinely.

Additionally, `opening_random_plies=1` randomizes the game's first ply — so
in half the games OUR side opens with a random move as White (observed
`b1a3`), plausibly explaining persistent as-White underperformance in
2026-07-19 samples.

## Decisions (user-approved 2026-07-19)

- **Node-limited Stockfish** replaces time-limited: `--stockfish-nodes` set
  by calibration (below). Opponent strength becomes load- and
  time-invariant; parallel SF instances become legitimate.
- **Calibration to current strength**: probe SF2200's actual nodes-per-move
  distribution at 0.05s on the idle machine (~10 instrumented games), take
  the median → config `stockfish_nodes`. The new protocol lands at
  approximately the clean-anchor strength (~0.52@2048/d8), so baselines
  carry over approximately. (Alternative "clean round number" rejected: an
  unpredictable first anchor muddies all prior comparisons.)
- **Eval batching**: G concurrent eval games in one process, mirroring the
  rollout architecture (same BatchScheduler, same merged executors — their
  2026-07-19 relocation to src/ was the enabling move).
- **fp32 evals**: with batching amortizing GPU overhead, fp32 is nearly free
  (rollout-proven); ends bf16 near-tie noise in measurements.
- **Opening-plies A/B**: re-anchor runs at `opening_random_plies` 1 AND 0,
  with a diversity check at 0 (unique game move-sequences — SF's UCI_Elo
  skill-randomization should diversify; verify, don't assume). If 0 is
  diverse and stronger, it becomes the default.
- Standing user authorization (2026-07-19): execute this plan end-to-end
  autonomously incl. re-anchor eval runs; report major out-of-scope
  deviations; minor optimizations at implementer discretion.

## Architecture

The eval game loop (`scripts/eval_vs_stockfish.py`, `_run_segment`) becomes a
game coroutine yielding THREE `WorkRequest` kinds:

- `root_eval` / `decode_wave` — the existing merged executors from
  `src/imba_chess/eval/merged_executors.py`, unchanged.
- `sf_move` (new) — payload: (slot's engine handle, board/limit info). The
  executor fans out all pending SF requests concurrently via a small thread
  pool (UCI pipes are I/O-bound): one tick costs ~one SF think, not G.
  Node-limited SF makes this fair by construction.

Engine lifecycle: G `chess.engine.SimpleEngine` instances, one per scheduler
slot, spawned at segment start, reused across that slot's games, closed at
segment end. Ladder segments stay sequential; batching is within a segment.
Failure policy per repo rule (fail fast, no silent errors): an engine or
executor error kills the run loudly; no catch-and-continue.

Color alternation, per-game bookkeeping (result attribution, coverage
stats, debug traces, JSON aggregation) stay per-coroutine; completed-game
accounting flows through the scheduler's stream-order emission exactly as
rollouts do.

## Gates

- Suite green throughout; new scheduler/coroutine unit tests CPU-only.
- **Move-probe gate (the strong one)**: fixed positions through eval-config
  search — batched (G>1) vs single-game evaluation must produce
  byte-identical move choices at fp32 (the same probe methodology that
  settled the 2026-07-19 regression scare, promoted to a permanent test
  asset where feasible, else a scripted controller gate).
- G=1 structural gate: the batched driver at G=1 plays complete games
  end-to-end with sane aggregates (score-level byte-identity is impossible:
  SF is stochastic by design via skill-randomization).
- Calibration gate: the node-limited opponent's strength is validated by the
  re-anchor itself landing near the clean time-based anchor (~±0.07 of
  0.52); a large deviation is investigated before adopting the protocol.

## Re-anchor protocol (runs under standing authorization)

Sequence after implementation gates pass, machine otherwise idle:
1. Calibration probe (~10 games, instrumented nps/nodes recording).
2. 200-game run: node-limited, fp32, G=8, opening_random_plies=1.
3. 200-game run: same, opening_random_plies=0, plus duplicate-game-rate
   diversity report.
4. Adopt: `stockfish_nodes` + fp32 + chosen opening setting into config as
   the new standard; record all numbers in this spec's Results.

## Out of scope

- Budget 4096 A/B — explicitly a follow-up AFTER the new protocol lands
  (first clean experiment on the new baseline).
- Rollout-side changes of any kind (rollout pipeline is done and gated).
- Multi-elo ladder parallelism across segments (within-segment batching only).
- Any change to training or Phase-1b target construction.

## Expected outcome

200-game evals in ~15-25 min (vs ~85), opponent strength invariant to
machine load, fp32-deterministic model decisions, and a defensible new
baseline for checkpoint_23 from which Phase-1b deltas can be measured at
±0.03 resolution without protocol caveats.
