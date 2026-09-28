"""Observed-trajectory regret and bounded restart storage; no neural inference.

The caller owns persistence: buffer mutations and launch acknowledgement must be
published in the same atomic consumer-state transaction.
"""

from dataclasses import asdict
import math

import chess

from .seeds import Seed, source_split, stable_hash


class RegretProtocolError(ValueError):
    """A completed trajectory cannot supply the required raw selected-move Q."""


def suffix_regrets(game):
    """One backward pass, using raw selected Q and the current player's outcome."""
    if game.get("status") != "completed":
        return None
    targets = game["targets"]
    if (
        not targets
        or len(targets) != len(game["moves"])
        or game["takeover_ply"] != len(game["prefix_moves"])
        or game["outcome_white"] not in (-1, 0, 1)
    ):
        raise RegretProtocolError("invalid completed regret trajectory")
    result = [0.0] * len(targets)
    total = 0.0
    for t in range(len(targets) - 1, -1, -1):
        target = targets[t]
        try:
            ids, qs = target["legal_ids"], target["qvalues"]
            if (
                len(ids) != len(qs)
                or len(set(ids)) != len(ids)
                or target["move_uci"] != game["moves"][t]
            ):
                raise ValueError("selected-move alignment")
            q = qs[ids.index(target["move_id"])]
            if not math.isfinite(q) or abs(q) > 1 + 1e-6:
                raise ValueError("selected Q must be finite and in [-1, 1]")
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise RegretProtocolError(
                f"invalid selected raw Q at continuation ply {t}"
            ) from exc
        z = game["outcome_white"] * (1 if (game["takeover_ply"] + t) % 2 == 0 else -1)
        total += (q - z) ** 2
        result[t] = total / (len(targets) - t)
    return result


def history_id(prefix):
    return stable_hash(" ".join(prefix))


def empty_buffer():
    return dict(
        entries=[],
        next_admission=0,
        admissions=0,
        replacements=0,
        refreshes=0,
        stale_refreshes=0,
        unusable_retirements=0,
    )


class RegretBuffer:
    def __init__(self, state, config, *, max_positions, max_game_plies, max_depth):
        self.state, self.config = state, config
        self.max_positions = max_positions
        self.max_game_plies, self.max_depth = max_game_plies, max_depth

    def fits(self, prefix):
        # _SequenceHistory has BOS plus one token per inherited ply; collection
        # needs one further observed position plus its search-depth allowance.
        return (
            0 < len(prefix) < self.max_game_plies
            and len(prefix) + 2 + self.max_depth <= self.max_positions
        )

    def usable(self, entry):
        seed = Seed(**entry["seed"])
        if not self.fits(seed.prefix_moves) or seed.split != "train":
            return False
        try:
            board = seed.board()
            if any(not move for move in board.move_stack):
                return False  # python-chess accepts null moves in push_uci.
        except ValueError:
            return False
        return True

    def sample(self, rng):
        entries = self.state["entries"]
        usable = [e for e in entries if self.usable(e)]
        self.state["unusable_retirements"] += len(entries) - len(usable)
        entries[:] = usable
        positive = [e for e in entries if e["priority"] > 0]
        if not positive:
            return None
        logs = [math.log(e["priority"]) for e in positive]
        peak = max(logs)
        weights = [math.exp((v - peak) / self.config.temperature) for v in logs]
        return rng.choices(positive, weights=weights, k=1)[0]

    def admit(self, seed, priority, *, parent_game_id, parent_ply):
        """At most one admission per ordinary completion; oldest loses ties."""
        entries = self.state["entries"]
        key = history_id(seed.prefix_moves)
        if any(e["history_id"] == key for e in entries):
            return False
        if len(entries) >= self.config.capacity:
            victim = min(entries, key=lambda e: (e["priority"], e["admission_id"]))
            if priority <= victim["priority"]:
                return False
            entries.remove(victim)
            self.state["replacements"] += 1
        entries.append(
            dict(
                seed=asdict(seed),
                history_id=key,
                priority=priority,
                parent_game_id=parent_game_id,
                parent_ply=parent_ply,
                admission_id=self.state["next_admission"],
                refresh_count=0,
            )
        )
        self.state["next_admission"] += 1
        self.state["admissions"] += 1
        return True

    def observe(self, game, launch):
        if game.get("status") != "completed" or game.get("split") != "train":
            return
        if source_split(game["source_id"]) != "train":
            raise RegretProtocolError("regret trajectory source/split mismatch")
        regrets = suffix_regrets(game)
        parent = launch.get("restart_parent")
        prefix = list(game["prefix_moves"])
        board = chess.Board()
        try:
            for uci in prefix:
                if not chess.Move.from_uci(uci):
                    raise ValueError("null inherited move")
                board.push_uci(uci)
            buffered = {e["history_id"] for e in self.state["entries"]}
            best = None
            for t, uci in enumerate(game["moves"]):
                if board.is_game_over(claim_draw=True):
                    raise ValueError("terminal position inside trajectory")
                if (
                    parent is None
                    and self.fits(prefix)
                    and history_id(prefix) not in buffered
                ):
                    # Strict comparison preserves earliest-ply ties.
                    if best is None or regrets[t] > best[0]:
                        best = (regrets[t], list(prefix))
                if not chess.Move.from_uci(uci):
                    raise ValueError("null continuation move")
                board.push_uci(uci)
                prefix.append(uci)
            outcome = board.outcome(claim_draw=True)
            actual = (
                None
                if outcome is None
                else (0 if outcome.winner is None else (1 if outcome.winner else -1))
            )
            if actual != game["outcome_white"]:
                raise ValueError("trajectory does not end at the recorded outcome")
        except ValueError as exc:
            raise RegretProtocolError("invalid full regret trajectory") from exc
        if parent is not None:
            for entry in self.state["entries"]:
                if (entry["history_id"], entry["admission_id"]) == (
                    parent["history_id"],
                    parent["admission_id"],
                ):
                    alpha = self.config.ema_alpha
                    entry["priority"] = (1 - alpha) * entry[
                        "priority"
                    ] + alpha * regrets[0]
                    entry["refresh_count"] += 1
                    self.state["refreshes"] += 1
                    return
            self.state["stale_refreshes"] += 1
            return  # Evicted admissions must never be resurrected.
        if best is not None:
            priority, prefix = best
            seed = Seed(
                history_id(prefix),
                game["source_id"],
                prefix,
                len(prefix),
                game["split"],
                game["corpus_id"],
            )
            self.admit(
                seed, priority, parent_game_id=game["game_id"], parent_ply=len(prefix)
            )

    def report(self):
        priorities = sorted(e["priority"] for e in self.state["entries"])
        return dict(
            **{
                k: v
                for k, v in self.state.items()
                if k not in ("entries", "next_admission")
            },
            size=len(priorities),
            positive_entries=sum(p > 0 for p in priorities),
            priority_min=min(priorities, default=0),
            priority_max=max(priorities, default=0),
            priority_mean=sum(priorities) / max(1, len(priorities)),
            priority_p50=priorities[len(priorities) // 2] if priorities else 0,
            priority_p95=priorities[
                min(len(priorities) - 1, int(0.95 * len(priorities)))
            ]
            if priorities
            else 0,
        )
