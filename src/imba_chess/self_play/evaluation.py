"""Restartable matched-prefix screens. Promotion requires completed opening pairs."""

from collections import defaultdict
import random
from pathlib import Path


from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler
from .collector import play_game
from .seeds import stable_hash


class EvaluationProtocolError(RuntimeError):
    """An unlabeled game cannot finish under this fixed evaluation protocol."""


PERMANENT_LIMITS = {"game_limit", "context_limit"}


def paired_interval(results, *, pairs, seed=42, samples=10000):
    grouped = defaultdict(list)
    for row in results:
        if row["status"] != "completed":
            continue
        score = (row["outcome_white"] * (1 if row["candidate_white"] else -1) + 1) / 2
        grouped[row["pair"]].append(score)
    if len(grouped) != pairs or any(len(v) != 2 for v in grouped.values()):
        raise ValueError("evaluation requires every planned color-swapped pair")
    scores = [sum(grouped[i]) / 2 for i in range(pairs)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(scores, k=pairs)) / pairs for _ in range(samples))
    return dict(
        score=sum(scores) / pairs,
        lower=draws[int(0.025 * samples)],
        upper=draws[min(samples - 1, int(0.975 * samples))],
        pairs=pairs,
    )


def decision(screen, confirmation=None):
    if screen["upper"] < 0.45:
        return "rollback_stop"
    if confirmation is not None and confirmation["lower"] > 0.5:
        return "promote"
    return "retain_best"


def evaluate_pair_checkpoints(
    *,
    candidate,
    best,
    candidate_id,
    best_id,
    seeds,
    config,
    max_positions,
    output,
    pairs,
    should_stop=lambda: False,
):
    if len(seeds) < pairs or any(s.split != "monitor" for s in seeds[:pairs]):
        raise ValueError(f"evaluation needs {pairs} held-out source-game prefixes")
    if len({s.source_id for s in seeds[:pairs]}) != pairs:
        raise ValueError("evaluation prefixes must have distinct source games")
    if any(
        getattr(runtime, "algorithm", "gumbel") != "gumbel"
        for runtime in (candidate, best)
    ):
        raise ValueError("self-play screening requires Gumbel for the entire match")
    output = Path(output)
    import json

    identity = dict(
        candidate=candidate_id,
        best=best_id,
        config=config.identifier,
        seeds=[s.seed_id for s in seeds[:pairs]],
        pairs=pairs,
        inference=dict(
            candidate=getattr(candidate, "options", {}),
            best=getattr(best, "options", {}),
            algorithm="gumbel",
            simulations=config.search.simulations,
            exploration="gumbel_noise",
            precision="float32",
            tf32=False,
        ),
    )
    state = (
        json.loads(output.read_text())
        if output.exists()
        else dict(identity=identity, results={})
    )
    if state["identity"] != identity:
        raise ValueError(
            "evaluation progress belongs to different checkpoints/protocol"
        )

    def check_protocol_failure():
        failures = {
            key: row["termination"]
            for key, row in state["results"].items()
            if row["status"] != "completed"
            and row.get("termination") in PERMANENT_LIMITS
        }
        if failures:
            state["protocol_failure"] = failures
            state.pop("interval", None)
            atomic_json(output, state)
            raise EvaluationProtocolError(
                f"Evaluation protocol failed: {failures}. Outcomes remain unlabeled; "
                "revise the limits/protocol and use a new evaluation output. "
                "Retrying unchanged inputs cannot complete these games."
            )

    # Also recognizes capped games written by versions predating this check.
    check_protocol_failure()
    runtimes = {candidate_id: candidate, best_id: best}

    def executor(kind):
        def execute(payloads):
            groups = defaultdict(list)
            for i, payload in enumerate(payloads):
                groups[payload[0][0]].append((i, payload))
            out = [None] * len(payloads)
            for actor, entries in groups.items():
                values = runtimes[actor].executors[kind]([p for _, p in entries])
                for (i, _), value in zip(entries, values):
                    out[i] = value
            return out

        return execute

    def factory():
        for pair, seed in enumerate(seeds[:pairs]):
            for candidate_white in (True, False):
                key = f"{pair}:{int(candidate_white)}"
                if (
                    key in state["results"]
                    and state["results"][key]["status"] == "completed"
                ):
                    continue
                if should_stop():
                    return

                def actors(turn, white=candidate_white):
                    return (
                        (candidate_id, candidate) if turn == white else (best_id, best)
                    )

                yield (
                    key,
                    play_game(
                        seed=seed,
                        game_id=stable_hash(str(identity) + key),
                        actor_id=candidate_id,
                        runtime=candidate,
                        search_config=config.search,
                        max_positions=max_positions,
                        max_game_plies=config.collection.max_game_plies,
                        run_seed=config.run.seed,
                        config_id=config.identifier,
                        should_stop=should_stop,
                        actor_for_turn=actors,
                    ),
                )

    def done(key, game):
        pair, white = key.split(":")
        state["results"][key] = dict(
            pair=int(pair),
            candidate_white=bool(int(white)),
            status=game["status"] if game else "unfinished",
            outcome_white=game.get("outcome_white") if game else None,
            termination=game.get("termination") if game else "error",
        )
        atomic_json(output, state)
        check_protocol_failure()

    BatchScheduler(
        game_factory=iter(factory()),
        executors={
            kind: executor(kind)
            for kind in set(candidate.executors) | set(best.executors)
        },
        concurrent_games=config.collection.concurrent_games,
        completion_order=True,
        on_game_done=done,
        on_game_error=lambda *args: None,
    ).run()
    rows = list(state["results"].values())
    if len(rows) != 2 * pairs or any(r["status"] != "completed" for r in rows):
        return None
    interval = paired_interval(rows, pairs=pairs, seed=config.run.seed)
    state["interval"] = interval
    atomic_json(output, state)
    return interval
