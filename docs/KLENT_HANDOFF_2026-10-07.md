# KLENT self-play handoff (2026-10-07)

This note is for whoever continues the KLENT experiments on **fedora-box**. That desktop (RTX 5070 Ti 16 GB, Ryzen 9900X, 64 GB RAM) is now the only experiment machine.

**Rule from the user: run nothing on the laptop. If fedora-box is unreachable, stop and report; do not fall back to the laptop.**

## 1. What KLENT is here

KLENT (arXiv 2602.10894, reference code github.com/KazukiOhta/klent) is self-play **without search**. A per-move action-value head Q, trained on λ-returns, defines a closed-form improved policy, so each move costs one forward pass.

- Improved policy: π′ ∝ exp((β·logits + Q) / (α + β)) over legal moves. Self-play samples from π′, and the policy is trained toward π′ with cross-entropy.
- λ-returns: G_t = r + γ((1−λ)V_{t+1} + λG_{t+1}), with γ = −1 (side to move flips) and λ = e^(−1/τ).
- Bootstrap value: V = Σπ′·Q.
- Fixed point: the policy settles at π ∝ exp(Q/α). **α must match the size of the learned Q gaps between moves.**

| Where | What |
|---|---|
| `src/imba_chess/klent/targets.py` | `improved_policy`, `lambda_returns` |
| `engine.py` | `SlotEngine`: N game slots, each with its own bf16 KV cache; one decode token per slot per step |
| `selfplay.py` | `SelfPlay.collect`: native cozy-chess rules, sampling from π′, λ-returns; supports forced human-prefix plies |
| `train.py` | `KlentTrainer`: compiled forward+loss, bf16 autocast, StableAdamW, gradient accumulation, trunk-gradient probes |
| `config.py` | `KlentConfig`, the TOML loader |
| `run.py` | `KlentRun`: the collect → train → checkpoint loop; resumable |
| `scripts/run_klent.py` | entry point (`--save-every`, `--keep-snapshots`) |
| `scripts/klent_eval_sidecar.py` | evals running next to training; see section 4 |
| `scripts/klent_tensorboard.py` | follows `metrics.jsonl` and writes `<run>/tb` |
| `scripts/bench_klent.py` | throughput benchmark (`--slots 256 384 ...`) |
| Model hooks | `enable_action_value_head`, `action_value_head_blocks/width`, `tie_policy_embeddings`, `value_source = "pi_q"` (search uses V = Σπ·Q) |
| Eval hooks | `--model-move-policy policy` (argmax π) and `policy_q` (argmax β·logit + Q) in `eval_vs_stockfish.py` and `match_two_checkpoints.py`; `--value-source pi_q` |

**Deliberate deviation from the paper:** we keep a WDL value head next to Q, so the existing Gumbel and Stockfish evals still work.

## 2. Run currently going on fedora-box

`artifacts/klent/scratch-adv-5070ti-2026-10-07`, config `config/klent_scratch_adv.toml`, code at commit 2c0d415. It started 2026-10-07 07:34 EDT.

- **Init:** random weights (no imitation), HSTU v4 (8 layers, d 1024, about 52M parameters), policy head not tied to the move embeddings.
- **Heads:** the WDL value head is trained (value_weight 1.0). Q is a linear tanh readout in **advantage mode**: it predicts A = Q − V, with target G − V(s) and V detached.
- **KLENT settings:** α 0.03, β 0.3, τ 8, bootstrap Σπ′·Q (V + Σπ′A), no Q warm-up.
- **Data:** 256 slots, every game from the initial position, 512-ply cap scored as a draw, 2²¹ positions per iteration, 400 iterations (838,860,800 positions).
- **Optimisation:** lr 1e-3 constant, wd 0, grad clip 1.0, batch 4096 tokens, no accumulation. Training is bf16 autocast and compiled; self-play is mixed bf16 (fp32 weights under autocast, bf16 KV cache, π′ computed in fp32).
- **Speed:** about 110 s per iteration (self-play about 80 s, about 26k positions/s; training about 30 s at 35k tokens/s). That puts the end around **2026-10-07 20:00 EDT**. GPU 74%, 5.3 GB during self-play and 9.4 GB peak during training. 384 or 512 slots would fit in memory, but the expected gain is ≤5% (decode cost grows with slots × the longest cached game), so it was not changed.

Processes (all started with `nohup setsid`):

```
run_klent.py --config config/klent_scratch_adv.toml --output $R --save-every 10 --keep-snapshots 5
klent_tensorboard.py $R --follow                # -> $R/tb
klent_eval_sidecar.py $R --every 40 --final-iteration 400 --elo 1320 --baseline "" --follow
tensorboard --logdir $R --port 6008 --bind_all  # http://fedora-box:6008
```

The logs are `$R/run.log`, `$R/eval_sidecar.log` and `$R/tb_sidecar.log`. The eval sidecar uses SF **1320**, not 2600, because a from-scratch run scores zero against 2600 for a long time.

First iterations:

| it | kl_1 | ent_1 | q_loss | value_loss | mate / draw / 512-cap |
|---|---|---|---|---|---|
| 1 | 0.000 | 2.91 | 0.077 | 0.33 | 977 / 5081 / 214 |
| 3 | 0.317 | 2.27 | 0.093 | 0.49 | 1397 / 2684 / 1478 |
| 5 | 0.259 | 2.08 | 0.073 | 0.43 | 1154 / 3566 / 1151 |
| 7 | 0.203 | 2.01 | 0.058 | 0.38 | 1034 / 4360 / 905 |

kl_1 is 0 at iteration 1 because Q starts at zero. Policy loss ≈ ent_1 + kl_1, as expected when the policy is trained toward π′.

**What to watch:**
- kl_1 should stay small and positive, and ent_1 should fall slowly.
- The share of 512-ply-cap games should keep falling. If it is still around 30% after about 20 iterations, the draw-scored cap is teaching the policy to shuffle pieces.
- Mates should eventually rise.
- The h2h_prev score at each 40-iteration point should stay above 0.5.

**Resume after a crash or reboot:** rerun the same `run_klent.py` command. It resumes from `$R/checkpoint.pt` at the last iteration boundary, and only the settings in `_RESUME_OVERRIDES` (`run.py`) may change on resume: slots, lr, batch_tokens, dtypes, accumulation and so on. Restart the sidecars as well; they skip evals already done.

## 3. Results so far (all on the laptop 3070 Ti, from `flatten53250`)

Ruler notes: greedy (no-search) SF2600 scores sit near the floor and are noisy (±2-3% at 200 games). **Head-to-head against the run's own earlier snapshot**, 400 games with paired openings and colour reversal, SE ≈ 0.021, is the sensitive ruler. The user wants progress judged that way, plus an external Stockfish anchor.

1. **Paper α 0.03 fails from a pretrained start.** The learned Q gaps are tiny (top-1 vs top-2 about 0.012), so the policy flattens: entropy 1.5 → 2.7, decisive games 60% → 25-40%, greedy SF2600 about 8% → 1%. Seen twice.
2. **α 0.003 overshoots** (entropy 1.5 → 1.1 by iteration 7 and still falling). **α 0.01 works:** entropy settles near 0.8, and iteration 80 beat iteration 40 by 59.5% ± 2.1%.
3. **Training the value head on self-play outcomes erased its move discrimination** (one-ply −V(s′) spread 0.157 → 0.04). Freezing it (value_weight 0) kept 0.126, and the Q gaps grew about 4×. *This is why the current scratch run, which has no good V to protect, is the test of a learned V plus advantage-mode Q.*
4. **β 0.3 beats β 0.1** with a linear Q: 64.6% ± 2.0% head-to-head at iteration 40 (+104 Elo), with greedy SF2600 6.25% vs 2.25%.
5. **τ 16 is worse than τ 8:** 41.4% ± 1.9% at iteration 40, and q_loss about 1.8× higher.
6. **A private Q MLP collapses to position value:** within-position spread 0.02 vs 0.10, and correlation with the policy 0.2-0.3 vs 0.86. Keep the linear readout.
7. **The linear Q mostly echoes the policy:** about 75% of its move variance is explained by the logits, and its independent part does not agree with a one-ply 53250 teacher (partial correlation −0.10 to +0.06). Improvement is largely self-sharpening.
8. **Best 53250 recipe:** `config/klent_53k_best.toml` (α 0.01, β 0.3, τ 8, frozen V, linear Q). The streaming-starts + ×4 accumulation variant ran for 200 iterations (`artifacts/klent/53k-laptop-a001-streaming-b03-2026-10-05`):

   | snapshot | greedy policy | greedy policy_q | vs previous kept |
   |---|---|---|---|
   | 40 | 3.5% | 4.5% | — |
   | 80 | 5.0% | 6.0% | 54.1% ± 2.1 |
   | 120 | 4.75% | 3.75% | 52.3% ± 2.1 |
   | 160 | 5.5% | 4.75% | 53.0% ± 2.2 |
   | 200 | 5.75% | 5.75% | 53.6% ± 2.2 |

   Greedy columns are against SF2600 over 200 games. At iteration 40 it lost to the iteration-40 control without streaming or accumulation (46.1% ± 2.1%).

   Iteration 200 with Gumbel 512 scored **14.0%** using the value head and **11.75%** using V = Σπ·Q, against **28.5%** for 53250 itself. Head-to-head with greedy policy_q, **iteration 200 vs 53250 was 42.9% ± 2.1% (−50 Elo)**.

   **Conclusion:** KLENT from 53250 keeps beating its own earlier versions (+16 to +29 Elo per 40 iterations) but has not passed the supervised start in absolute strength. That motivated the from-scratch run in section 2: learned V, advantage-mode Q, no inherited imitation prior.

## 4. Eval sidecar behaviour

`klent_eval_sidecar.py` copies every `--every`-th snapshot to `<run>/keep/`, where it survives the rolling pruning. For each kept snapshot it then runs:
- greedy `policy` and `policy_q` evals against SF `--elo` (200 games each);
- `h2h_prev`: 400 games against the previous kept snapshot.

When the `--final-iteration` snapshot exists it runs Gumbel 512 (`--gumbel-mode`, default `gumbel512_piq`) on that snapshot only, then exits. Results go to `<run>/sf2600/summary.jsonl` (the directory name is historical; each row has an `elo` field) and `<run>/tb_sf2600`.

Gotchas already fixed:
- the ladder config overrides `--stockfish-elo` and `--games`, so the sidecar uses `--ladder-elos` and `--ladder-games-per-segment`;
- `--baseline ""` skips the baseline;
- eval raw-Q-trained actors with raw Q.

## 5. fedora-box setup notes

- uv is at `~/.local/bin/uv` and is not on the PATH of non-login shells. `uv sync --extra dev` installs pytest. torch 2.14+cu130 supports sm_120.
- gcc-c++ was installed 2026-10-07; Inductor needs it for compiled training.
- Stockfish 18 is at `/usr/bin/stockfish`, the same build as the laptop.
- Already copied over: `artifacts/checkpoints_keep/`, `artifacts/klent/` (the laptop runs above), the corpus, syzygy and the move vocab.
- Test suite: 1902 tests passed. The one known failure is the extended CUDA test `test_decode_workspace::test_stable_placement_and_selective_gather[cuda-compiled]`, at 3.1e-6 vs a 1e-6 tolerance. That is compiled-vs-eager noise on Blackwell.
- The box rebooted unexpectedly once during setup. Check `journalctl -b -1` if it happens again. **Never change suspend or power settings yourself.**
- Launch long jobs with `nohup setsid ... < /dev/null &`.
- Standing preferences:
  - fail fast, no catch-and-continue;
  - no unbounded-memory risk for small speedups;
  - subagents on Sonnet, never Haiku;
  - commit and push after code changes;
  - snapshots every 10 iterations, keep at most 5.

## 6. Open next steps

- Watch the scratch run (section 2). At iterations 40 and 80, check h2h_prev and greedy scores against SF1320. If it learns, raise the Stockfish Elo for later evals. Compare the scratch run with the 53250 runs head-to-head (`match_two_checkpoints.py`, `--model-move-policy policy_q`), not only against Stockfish.
- Ideas not yet tried:
  - advantage-mode Q from 53250 with a frozen V;
  - an α schedule;
  - a one-ply teacher distilled into Q.
- Optional: benchmark 384 or 512 slots only at a natural restart point (`bench_klent.py --slots 256 384 512`). The gain is expected to be small.
