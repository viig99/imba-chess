"""Classify where zero-noise Gumbel search loses its biggest engine-scored mistakes.

Reads a finished `audit_search_settings` run (positions + full-strength Stockfish
labels for every legal move), re-runs one setting with full search output on the
worst positions and on a well-played control sample, and records for each:
policy rank of the engine's good moves, whether search considered them, their
search Q versus the chosen move, and the raw network one-ply value of both.
Q and values are converted to expected score (q + 1) / 2 to compare with engine
WDL expectation. Engine labels are proxies, not ground truth. Eval only.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics

import chess
import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval import cozy_bridge
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import file_hash

GOOD_MARGIN = 0.02  # moves within 2 points of the engine's best count as good


def classify(record, top_m):
    """Mutually exclusive cause of a search choice, ordered from policy to value."""
    if record["chosen_is_good"]:
        return "chose_good_move"
    if record["best_good_prior_rank"] > top_m:
        return "policy_miss"  # no good move among the considered root candidates
    if record["best_good_q"] is None:
        return "candidate_unvisited"
    if record["best_good_q"] < record["chosen_q"]:
        return "value_misorder"  # search Q prefers the worse move
    return "selection_override"  # good move had higher Q but prior + sigma(Q) chose otherwise


def analyse(result, moves, scores, top_m):
    exp = {m: scores[m]["expectation"] for m in moves}
    best = max(exp.values())
    good = {m for m in moves if exp[m] >= best - GOOD_MARGIN}
    order = sorted(range(len(moves)), key=lambda i: -result["root_log_priors"][i])
    rank = {moves[i]: r + 1 for r, i in enumerate(order)}
    q = {m: (result["qvalues"][i] if result["visits"][i] else None) for i, m in enumerate(moves)}
    visits = dict(zip(moves, result["visits"]))
    chosen = result["move_uci"]
    good_by_q = [m for m in good if q[m] is not None]
    best_good = max(good_by_q, key=lambda m: q[m]) if good_by_q else min(good, key=rank.get)
    record = dict(
        chosen=chosen, chosen_expectation=exp[chosen], best_expectation=best,
        regret=best - exp[chosen], chosen_is_good=chosen in good, good_moves=sorted(good),
        chosen_prior_rank=rank[chosen], best_good_move=best_good,
        best_good_prior_rank=min(rank[m] for m in good), best_good_visits=visits[best_good],
        chosen_visits=visits[chosen], chosen_q=q[chosen], best_good_q=q[best_good],
        root_value_expectation=(result["root_value"] + 1) / 2,
    )
    record["cause"] = classify(record, top_m)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", type=Path, required=True, help="audit_search_settings output dir")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--setting", default="c_visit=50,top_m=16")
    p.add_argument("--blunder-regret", type=float, default=0.10)
    p.add_argument("--control-regret", type=float, default=0.01)
    p.add_argument("--controls", type=int, default=150)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    screen = json.loads((args.screen / "results.json").read_text())
    if screen["identity"]["checkpoint"] != file_hash(args.checkpoint):
        raise ValueError("checkpoint differs from the screened one")
    c_visit, top_m = (float(x.split("=")[1]) for x in args.setting.split(","))
    top_m = int(top_m)
    rows = {r["position_id"]: r["metrics"] for r in screen["search"] if r["setting"] == args.setting}
    blunders = sorted(i for i, m in rows.items() if m["selected_regret"] > args.blunder_regret)
    calm = sorted(i for i, m in rows.items() if m["selected_regret"] <= args.control_regret)
    controls = sorted(random.Random(42).sample(calm, min(args.controls, len(calm))))
    cfg = load_config(args.config)
    config = replace(cfg.search, simulations=screen["identity"]["simulations"],
                     value_scale=screen["identity"]["value_scale"], maxvisit_init=c_visit, top_m=top_m)
    one_ply = replace(config, simulations=1)
    torch.set_num_threads(4)
    runtime, _ = load_runtime(cfg, args.checkpoint, args.device)
    tasks, out = {}, {}

    def history_for(prefix):
        history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
        board = chess.Board()
        for uci in prefix:
            history.append_observed_position(board)
            history.record_played_move(uci)
            board.push_uci(uci)
        return board, history

    def factory():
        for group, ids in (("blunder", blunders), ("control", controls)):
            for pid in ids:
                pos = screen["stockfish"][pid]
                board, history = history_for(pos["prefix"])
                ids_, _, ucis, _, _ = cozy_bridge.project_legal_moves(cozy_bridge.board_to_cozy(board), runtime.move_vocab)
                key = json.dumps([pid, "root"])
                tasks[key] = (group, pid, dict(zip(ids_, ucis)))
                yield key, runtime.search(board=board, history=history, actor_id="blunders", game_id=key,
                                          config=config, noise=0.0)

    def complete(key, result):
        group, pid, mapping = tasks.pop(key)
        raw = asdict(result)
        moves = [mapping[i] for i in raw["legal_ids"]]
        rec = analyse(raw, moves, screen["stockfish"][pid]["scores"], top_m)
        if rec["chosen"] != rows[pid]["selected"]:
            raise RuntimeError(f"rerun chose {rec['chosen']} but screen chose {rows[pid]['selected']} at {pid}")
        out[pid] = dict(group=group, position_id=pid, **rec)

    def error(key, exc):
        raise RuntimeError(key) from exc

    BatchScheduler(game_factory=iter(factory()), executors=runtime.executors, concurrent_games=32,
                   on_game_done=complete, on_game_error=error, completion_order=True).run()

    # Raw network value of the chosen and best-good child positions (1-simulation root value).
    def child_factory():
        for pid, rec in out.items():
            for which, move_key in (("chosen", "chosen"), ("best_good", "best_good_move")):
                prefix = screen["stockfish"][pid]["prefix"] + [rec[move_key]]
                board, history = history_for(prefix)
                if board.is_game_over(claim_draw=True):
                    rec[which + "_raw_expectation"] = None
                    continue
                key = json.dumps([pid, which])
                tasks[key] = (pid, which)
                yield key, runtime.search(board=board, history=history, actor_id="blunders", game_id=key,
                                          config=one_ply, noise=0.0)

    def child_complete(key, result):
        pid, which = tasks.pop(key)
        # Child value is from the opponent's perspective; negate for the root player.
        out[pid][which + "_raw_expectation"] = (1 - result.root_value) / 2

    BatchScheduler(game_factory=iter(child_factory()), executors=runtime.executors, concurrent_games=32,
                   on_game_done=child_complete, on_game_error=error, completion_order=True).run()

    summary = {}
    for group in ("blunder", "control"):
        recs = [r for r in out.values() if r["group"] == group]
        causes = {}
        for r in recs:
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        misjudged = [r for r in recs if r["chosen_raw_expectation"] is not None
                     and r["best_good_raw_expectation"] is not None and not r["chosen_is_good"]]
        summary[group] = dict(
            positions=len(recs), causes=causes,
            mean_regret=statistics.mean(r["regret"] for r in recs),
            best_good_prior_rank_median=statistics.median(r["best_good_prior_rank"] for r in recs),
            root_value_minus_engine_best=statistics.mean(r["root_value_expectation"] - r["best_expectation"] for r in recs),
            raw_value_prefers_chosen=(statistics.mean(
                float(r["chosen_raw_expectation"] > r["best_good_raw_expectation"]) for r in misjudged)
                if misjudged else None),
            raw_value_misjudged_positions=len(misjudged),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "records.json", dict(
        screen=str(args.screen), setting=args.setting, checkpoint=file_hash(args.checkpoint),
        good_margin=GOOD_MARGIN, records=sorted(out.values(), key=lambda r: (r["group"], -r["regret"]))))
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
