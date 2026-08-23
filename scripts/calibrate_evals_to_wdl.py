#!/usr/bin/env python3
"""Fit centipawns -> WDL from this corpus, and emit evals in the rollout schema.

Two jobs, deliberately in one place so the calibration that produced a target
travels with it:

1. **Calibrate.** For every Stockfish eval that also has a known game result,
   bin by side-to-move centipawns and read off the empirical P(win/draw/loss).
   No functional form is assumed and none is borrowed from Leela -- this
   corpus has ~477k such positions, which is plenty, and its own draw rate and
   conversion rate are what matter. Two facts fall out that no borrowed curve
   would give you: at |cp| > 800 these players convert only 86.7%, and
   MATE-scored positions convert 85.1%. The game outcome is a very noisy
   function of true position quality, which is precisely the label noise the
   value head is stuck with.

2. **Emit.** Write the calibrated targets in `rollout_store`'s schema, so
   `load_rollout_lookup` / `EventBuilder` / `compute_blended_value_target`
   consume them with no code changes and `beta` means exactly what it already
   means. Only the value fields are populated; the policy/arm fields are left
   empty, which `_build_rollout_policy_targets` already treats as "no target"
   (every arm masked -> token skipped).

`compute_blended_value_target` reads only `p_draw0` out of
`root_wdl_unsearched` and reconstructs win/loss from `backed_value`, so
setting root_wdl_unsearched = (P(loss), P(draw), P(win)) and
backed_value = P(win) - P(loss) makes beta=1 reproduce the calibrated vector
exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from imba_chess.data.rollout_store import RolloutRow, write_rollout_parquet

# Mate scores are not centipawns and must not be folded into that scale; they
# get their own two buckets, signed by who is mating.
MATE_BUCKET = 1e9


def fit_calibration(train_evals: Path, *, bins: int) -> dict:
    d = pd.read_parquet(train_evals)
    cp = d[d["cp_stm"].notna()].copy()
    # Quantile edges: dense where the data is, which is near cp=0.
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(cp["cp_stm"], qs))
    idx = np.clip(np.searchsorted(edges, cp["cp_stm"], side="right") - 1, 0, len(edges) - 2)
    cp["bin"] = idx
    out = []
    for b, grp in cp.groupby("bin"):
        o = grp["real_outcome_stm"].to_numpy()
        out.append({
            "cp_center": float(grp["cp_stm"].median()),
            "n": int(len(o)),
            "p_loss": float((o == -1).mean()),
            "p_draw": float((o == 0).mean()),
            "p_win": float((o == 1).mean()),
        })
    out.sort(key=lambda r: r["cp_center"])

    mate = d[d["mate_stm"].notna()]
    mate_cal = {}
    for sign, grp in mate.groupby(np.sign(mate["mate_stm"])):
        o = grp["real_outcome_stm"].to_numpy()
        mate_cal[str(int(sign))] = {
            "n": int(len(o)),
            "p_loss": float((o == -1).mean()),
            "p_draw": float((o == 0).mean()),
            "p_win": float((o == 1).mean()),
        }
    return {"cp_bins": out, "mate": mate_cal, "source": str(train_evals)}


def apply_calibration(cal: dict, cp_stm, mate_stm):
    """Vectorised lookup -> (p_loss, p_draw, p_win) arrays."""
    centers = np.array([b["cp_center"] for b in cal["cp_bins"]])
    P = np.array([[b["p_loss"], b["p_draw"], b["p_win"]] for b in cal["cp_bins"]])
    cp = np.asarray(cp_stm, dtype=float)
    # Linear interpolation between bin centers, flat outside the range.
    out = np.stack([np.interp(cp, centers, P[:, k]) for k in range(3)], axis=1)
    mate = np.asarray(mate_stm, dtype=float)
    has_mate = ~np.isnan(mate)
    for sign_key, vals in cal["mate"].items():
        sel = has_mate & (np.sign(mate) == float(sign_key))
        out[sel] = [vals["p_loss"], vals["p_draw"], vals["p_win"]]
    return out / out.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-evals", type=Path, required=True,
                    help="eval parquet used to FIT the calibration")
    ap.add_argument("--apply-to", type=Path, required=True,
                    help="eval parquet to convert (may be the same file)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--calibration-json", type=Path, default=None)
    ap.add_argument("--bins", type=int, default=48)
    args = ap.parse_args()

    cal = fit_calibration(args.train_evals, bins=args.bins)
    if args.calibration_json:
        args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_json.write_text(json.dumps(cal, indent=2))

    d = pd.read_parquet(args.apply_to)
    wdl = apply_calibration(cal, d["cp_stm"].to_numpy(dtype=float),
                            d["mate_stm"].to_numpy(dtype=float))
    rows = [
        RolloutRow(
            game_id=str(gid), ply=int(ply),
            human_move_uci="", human_move_backed_value=None,
            real_outcome_stm=int(out),
            best_arm_move_uci="",
            best_arm_backed_value=float(p_win - p_loss),
            root_wdl_unsearched=(float(p_loss), float(p_draw), float(p_win)),
            arm_move_uci=(), arm_backed_value=(), arm_evals_spent=(), arm_log_prior=(),
            search_budget=0, search_top_m=0, search_max_depth=0,
            checkpoint="lichess-stockfish-eval",
        )
        for gid, ply, out, (p_loss, p_draw, p_win) in zip(
            d["game_id"], d["ply"], d["real_outcome_stm"], wdl
        )
    ]
    write_rollout_parquet(rows, args.output)
    print(f"calibration bins : {len(cal['cp_bins'])}  (fit on {args.train_evals.name})")
    print(f"mate buckets     : "
          f"{ {k: round(v['p_win'], 3) for k, v in cal['mate'].items()} } P(win)")
    print(f"rows written     : {len(rows)} -> {args.output}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
