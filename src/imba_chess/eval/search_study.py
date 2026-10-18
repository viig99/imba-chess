"""Common Stockfish-reference metrics for every search backend (no torch)."""

import math
import random
import statistics
from collections import defaultdict


def policy_metrics(row, reference):
    moves, policy = row["legal_ucis"], row["policy"]
    if set(moves) != set(reference["moves"]) or len(set(moves)) != len(moves):
        raise ValueError("search and reference legal actions differ")
    if len(policy) != len(moves) or any(not math.isfinite(p) or p < 0 for p in policy) or abs(sum(policy) - 1) > 1e-5:
        raise ValueError("invalid search policy")
    scores = [reference["moves"][m]["expectation"] for m in moves]
    best, worst = max(scores), min(scores)
    regret = [best - s for s in scores]
    chosen = moves.index(row["move_uci"])
    logs = row["root_log_priors"]
    priors = [math.exp(x - max(logs)) for x in logs]
    total = sum(priors)
    priors = [p / total for p in priors]
    result = dict(move_regret=regret[chosen], target_regret=sum(p * r for p, r in zip(policy, regret)),
                  prior_regret=sum(p * r for p, r in zip(priors, regret)),
                  prior_argmax_regret=regret[max(range(len(moves)), key=priors.__getitem__)],
                  best_move=float(regret[chosen] <= 1e-12),
                  good_mass=sum(p for p, r in zip(policy, regret) if r <= 0.02),
                  blunder_005=float(regret[chosen] >= 0.05),
                  target_entropy=-sum(p * math.log(p) for p in policy if p),
                  target_kl=sum(p * (math.log(p) - math.log(max(q, 1e-38))) for p, q in zip(policy, priors) if p),
                  visited_actions=sum(n > 0 for n in row["visits"]),
                  informative=float(best - worst >= 0.02))
    root_score = reference["root"]["expectation"]
    for name in ("root_wdl", "search_wdl"):
        wdl = row.get(name)
        if wdl is not None:
            expectation = wdl[2] + 0.5 * wdl[1]
            result[name + "_mae"] = abs(expectation - root_score)
            result[name + "_mse"] = (expectation - root_score) ** 2
    # search_wdl averages every simulation, so it is pulled toward the values of
    # moves Gumbel's halving later discards. The played edge's Q is comparable
    # across methods; q = W - L, so its expected score is (1 + q) / 2.
    qvalues = row.get("qvalues")
    if qvalues is not None and row["visits"][chosen]:
        expectation = (1 + qvalues[chosen]) / 2
        result["played_q_mae"] = abs(expectation - root_score)
        result["played_q_mse"] = (expectation - root_score) ** 2
    return result


def summarize(rows, positions, references):
    positions = {p["position_id"]: p for p in positions}
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    output = {}
    for variant, group in groups.items():
        per_position = []
        for row in group:
            p = positions[row["position_id"]]
            per_position.append(dict(position_id=row["position_id"], game_id=p["game_id"],
                                     source=p["source"], phase=p["phase"],
                                     **policy_metrics(row, references[row["position_id"]])))
        metrics = [k for k in per_position[0] if k not in ("position_id", "game_id", "source", "phase")]
        def aggregate(selected):
            return dict(positions=len(selected), **{k: statistics.fmean(r[k] for r in selected) for k in metrics}) if selected else {}
        output[variant] = dict(all=aggregate(per_position),
                               informative=aggregate([r for r in per_position if r["informative"]]),
                               by_source={s: aggregate([r for r in per_position if r["source"] == s]) for s in sorted({r["source"] for r in per_position})},
                               by_phase={s: aggregate([r for r in per_position if r["phase"] == s]) for s in sorted({r["phase"] for r in per_position})})
    return output


def paired_regret_interval(baseline, candidate, references, positions, *, metric="target_regret", variant="eval", samples=2000):
    """Candidate minus baseline; bootstrap source games, not correlated plies."""
    a = {r["position_id"]: r for r in baseline if r["variant"] == variant}
    b = {r["position_id"]: r for r in candidate if r["variant"] == variant}
    if set(a) != set(b):
        raise ValueError("paired comparison needs identical completed positions")
    games = defaultdict(list)
    metadata = {p["position_id"]: p for p in positions}
    for pid in sorted(a):
        delta = policy_metrics(b[pid], references[pid])[metric] - policy_metrics(a[pid], references[pid])[metric]
        games[metadata[pid]["game_id"]].append(delta)
    if not games:
        return {}
    groups = list(games.values())
    rng = random.Random(1042)
    draws = []
    for _ in range(samples):
        selected = rng.choices(groups, k=len(groups))
        draws.append(sum(sum(g) for g in selected) / sum(len(g) for g in selected))
    draws.sort()
    return dict(candidate_minus_baseline=statistics.fmean(v for g in groups for v in g),
                lower=draws[int(samples * 0.025)], upper=draws[int(samples * 0.975)],
                positions=len(a), source_games=len(games), negative_favors_candidate=True)
