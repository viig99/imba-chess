"""Paired held-out intervals for two fixed training seeds, updated as arms finish.

Resample source games jointly across both seeds. This measures held-out sampling
uncertainty conditional on these two trained runs, not uncertainty over new
training seeds. Report the individual seed effects separately.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.golden import read_labels
from imba_chess.eval.search_study import policy_metrics


ORDER = ("gumbel", "puct_variance_rpo", "puct_variance", "puct_rpo", "puct")
METRICS = ("target_regret", "move_regret", "root_wdl_mae", "root_wdl_mse")


def paired_interval(differences, games, samples=2000):
    """Two seed-paired effects; a bootstrap unit includes all plies and both seeds."""
    if len(differences) != 2 or set(differences[0]) != set(differences[1]) or not differences[0]:
        raise ValueError("need identical nonempty position cohorts for both seeds")
    groups = defaultdict(list)
    for pid in sorted(differences[0]):
        groups[games[pid]].append((differences[0][pid] + differences[1][pid]) / 2)
    units = [(sum(values), len(values)) for values in groups.values()]
    rng = random.Random(1042)
    draws = []
    for _ in range(samples):
        chosen = rng.choices(units, k=len(units))
        draws.append(sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    draws.sort()
    seed_effects = [statistics.fmean(d.values()) for d in differences]
    return dict(mean_difference=statistics.fmean(seed_effects), per_seed_differences=seed_effects,
                lower=draws[int(samples * .025)], upper=draws[int(samples * .975)],
                negative_favors_candidate=True, source_games=len(units),
                interval_scope="95% paired source-game bootstrap, conditional on these two training seeds")


def predictions(root, arm):
    return root / "raw-learning-test" / arm / "phase-003/predictions.json"


def report(first, second, positions, labels, available):
    initial_manifests = [json.loads((p / "raw-learning-test/initial/manifest.json").read_text())
                         for p in (first, second)]
    if initial_manifests[0] != initial_manifests[1]:
        raise ValueError("held-out evaluation inputs differ between seeds")
    repeat = json.loads((second / "manifest.json").read_text())
    ids = {p["position_id"] for p in positions}
    games = {p["position_id"]: p["game_id"] for p in positions}
    metrics = {}
    for name in available:
        metrics[name] = []
        for root in (first, second):
            rows = json.loads(predictions(root, name).read_text())
            if len(rows) != len(ids) or {r["position_id"] for r in rows} != ids:
                raise ValueError("prediction coverage differs from held-out cohort")
            metrics[name].append({r["position_id"]: policy_metrics(r, labels[r["position_id"]]) for r in rows})
    result = dict(seeds=[repeat["first_seed"], repeat["seed"]], positions=len(ids),
        complete=len(available) == len(ORDER), arms={},
        interpretation="Intervals condition on the two observed trained runs. Per-seed effects show training variability; two seeds cannot reliably estimate a training-seed confidence interval.")
    for name in available:
        means = [{m: statistics.fmean(v[m] for v in seed.values()) for m in METRICS}
                 for seed in metrics[name]]
        entry = dict(per_seed_metrics=means,
                     mean_metrics={m: statistics.fmean(x[m] for x in means) for m in METRICS})
        if name != "gumbel":
            entry["versus_gumbel"] = {m: paired_interval([
                {pid: candidate[pid][m] - baseline[pid][m] for pid in ids}
                for candidate, baseline in zip(metrics[name], metrics["gumbel"])], games)
                for m in METRICS}
        result["arms"][name] = entry
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=Path, default=Path("artifacts/eval/search-study-2026-10-18"))
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    path = args.second / "paired-two-seed-intervals.json"
    status = args.second / "interval-report-status.json"
    last_signature, labels, positions = None, None, None
    deadline = time.monotonic() + 8 * 3600
    while True:
        queue = json.loads((args.second / "gpu-queue.json").read_text())
        if any(v.get("status") in ("failed", "partial") for v in queue.values()):
            atomic_json(status, dict(status="blocked_by_learning_queue"))
            raise SystemExit("learning queue needs attention")
        available = [n for n in ORDER if all(predictions(root, n).exists() for root in (args.first, args.second))]
        signature = tuple((str(predictions(root, n)), predictions(root, n).stat().st_mtime_ns)
                          for n in available for root in (args.first, args.second))
        if "gumbel" in available and signature != last_signature:
            if labels is None:
                golden = Path("artifacts/eval/golden-v1-2026-10-18")
                positions = [x for x in json.loads((golden / "positions.json").read_text())["positions"] if x["split"] == "test"]
                labels = read_labels(golden / "reference-200k/labels.jsonl")
            result = report(args.first, args.second, positions, labels, available)
            atomic_json(path, result)
            atomic_json(status, dict(status="complete" if result["complete"] else "partial", arms=available))
            print(json.dumps(dict(event="paired_report_updated", arms=available)), flush=True)
            last_signature = signature
            if result["complete"]:
                return
        elif not available:
            atomic_json(status, dict(status="waiting_for_second_seed_baseline"))
        if not args.watch:
            return
        if time.monotonic() > deadline:
            raise SystemExit("report wait timed out")
        time.sleep(10)


if __name__ == "__main__":
    main()
