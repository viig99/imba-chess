"""One sequential search per game, inference batched across independent games."""

from collections import Counter, deque
from dataclasses import asdict
import random
import time

import chess

from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval.gumbel_search import REFERENCE_REVISION
from imba_chess.eval.position_evaluator import (
    _SequenceHistory,
)
from imba_chess.eval.search import terminal_value_for_color
from .seeds import stable_hash


def terminal_outcome(board):
    value = terminal_value_for_color(board, color=board.turn)
    if value is None:
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        raise RuntimeError("native/python terminal convention mismatch")
    return int(
        value if board.turn == chess.WHITE else -value
    ), outcome.termination.name.lower()


def play_game(
    *,
    seed,
    game_id,
    actor_id,
    runtime,
    search_config,
    max_positions,
    max_game_plies=512,
    run_seed=42,
    config_id="",
    should_stop=lambda: False,
    on_position=lambda result, elapsed: None,
    actor_for_turn=None,
    gumbel_noise=True,
):
    board = seed.board()
    history = _SequenceHistory(
        move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder
    )
    replay_board = chess.Board()
    for uci in seed.prefix_moves:
        history.append_observed_position(replay_board)
        history.record_played_move(uci)
        replay_board.push_uci(uci)
    game = dict(
        schema_version=1,
        game_id=game_id,
        seed_id=seed.seed_id,
        source_id=seed.source_id,
        actor_id=actor_id,
        prefix_moves=seed.prefix_moves,
        takeover_ply=seed.takeover_ply,
        split=seed.split,
        corpus_id=seed.corpus_id,
        moves=[],
        targets=[],
        outcome_white=None,
        status="unfinished",
        termination="",
        run_seed=run_seed,
        config_id=config_id,
        search_config=asdict(search_config),
        reference_revision=REFERENCE_REVISION,
        inference_config=getattr(runtime, "options", {}),
    )
    rng = random.Random(f"{run_seed}:{game_id}")
    try:
        while True:
            terminal = terminal_outcome(board)
            if terminal is not None:
                game.update(
                    status="completed",
                    outcome_white=terminal[0],
                    termination=terminal[1],
                )
                return game
            if should_stop():
                game["termination"] = "interrupted"
                break
            if len(history.seq_token_id) + 1 + search_config.max_depth > max_positions:
                game["termination"] = "context_limit"
                break
            if len(board.move_stack) >= max_game_plies:
                game["termination"] = "game_limit"
                break
            current_actor, current_runtime = (
                (actor_id, runtime)
                if actor_for_turn is None
                else actor_for_turn(board.turn)
            )
            start = time.perf_counter()
            result = yield from current_runtime.search(
                board=board,
                history=history,
                actor_id=current_actor,
                game_id=game_id,
                config=search_config,
                rng=rng,
                should_stop=should_stop,
                **({} if gumbel_noise else {"noise": 0.0}),
            )
            move = chess.Move.from_uci(result.move_uci)
            if move not in board.legal_moves:
                raise ValueError("search produced illegal move")
            on_position(result, time.perf_counter() - start)
            game["targets"].append(asdict(result))
            game["moves"].append(result.move_uci)
            history.append_observed_position(board)
            history.record_played_move(result.move_uci)
            board.push(move)
    except InterruptedError:
        game["termination"] = "interrupted"
    except Exception as exc:
        game["termination"] = "error"
        game["error"] = f"{type(exc).__name__}: {exc}"
    # Administrative stops must not leak training labels.
    game["searched_positions"] = len(game["moves"])
    game["targets"] = []
    return game


class CollectionMetrics:
    def __init__(self):
        self.counts = Counter()
        self.terminations = Counter()
        self.latencies = deque(maxlen=4096)
        self.start = time.perf_counter()

    def position(self, result, elapsed):
        self.counts["searched_positions"] += 1
        self.counts["neural_evaluations"] += result.neural_evaluations
        self.counts["simulations"] += result.simulations
        self.counts["terminal_hits"] += result.terminal_hits
        self.counts["depth_cutoffs"] += result.depth_cutoffs
        self.latencies.append(elapsed)

    def done(self, game):
        self.counts[game["status"] + "_games"] += 1
        self.terminations[game["termination"]] += 1
        if game["status"] == "completed":
            self.counts["completed_positions"] += len(game["moves"])
            if game["split"] == "train":
                self.counts["usable_positions"] += len(game["moves"])
                self.counts["training_positions"] += len(game["moves"])

    def report(self):
        elapsed = max(time.perf_counter() - self.start, 1e-9)
        latencies = sorted(self.latencies)
        return dict(
            self.counts,
            seconds=elapsed,
            terminations=dict(self.terminations),
            searched_positions_per_hour=3600
            * self.counts["searched_positions"]
            / elapsed,
            usable_positions_per_hour=3600 * self.counts["usable_positions"] / elapsed,
            neural_evaluations_per_second=self.counts["neural_evaluations"] / elapsed,
            move_latency_p50=latencies[len(latencies) // 2] if latencies else 0,
            move_latency_p95=latencies[
                min(len(latencies) - 1, int(len(latencies) * 0.95))
            ]
            if latencies
            else 0,
        )


def collect(
    *,
    seeds,
    runtime,
    config,
    actor_id,
    store,
    max_positions,
    game_count=None,
    should_launch=lambda: True,
    should_stop=lambda: False,
    iteration=0,
    on_game=lambda game: None,
    skip_ids=(),
    metrics=None,
    concurrent_games=None,
    start_sampler=None,
):
    if concurrent_games is not None and concurrent_games < 1:
        raise ValueError("concurrent_games must be positive")
    metrics = metrics or CollectionMetrics()
    skipped = set(skip_ids)
    active = {}
    seed_order = list(seeds)
    if not seed_order and start_sampler is None:
        raise ValueError("collection requires at least one seed")
    if start_sampler is not None:
        start_sampler.begin_phase(iteration, actor_id, skipped)
    if game_count is None:
        # Sample sources uniformly without replacement in each iteration;
        # fixed-size audits retain manifest order for comparable benchmarks.
        random.Random(f"{config.run.seed}:{iteration}:sources").shuffle(seed_order)

    def factory():
        index = 0
        while should_launch() and (game_count is None or index < game_count):
            if (
                game_count is None
                and metrics.counts["training_positions"]
                >= config.collection.fresh_positions
            ):
                return
            if start_sampler is None:
                seed = seed_order[index % len(seed_order)]
                gid = stable_hash(
                    f"{config.run.seed}:{iteration}:{index}:{seed.seed_id}:{actor_id}"
                )
            else:
                try:
                    seed, gid = start_sampler.next_launch(iteration, actor_id)
                except InterruptedError:
                    return
            index += 1
            if gid in skipped:
                continue
            active[gid] = dict(
                game_id=gid,
                seed_id=seed.seed_id,
                source_id=seed.source_id,
                actor_id=actor_id,
                status="unfinished",
                termination="error",
                moves=[],
                targets=[],
                outcome_white=None,
            )
            yield (
                gid,
                play_game(
                    seed=seed,
                    game_id=gid,
                    actor_id=actor_id,
                    runtime=runtime,
                    search_config=config.search,
                    max_positions=max_positions,
                    max_game_plies=config.collection.max_game_plies,
                    run_seed=config.run.seed,
                    config_id=config.identifier,
                    should_stop=should_stop,
                    on_position=metrics.position,
                ),
            )

    def done(gid, game):
        metadata = active.pop(gid)
        if game is None:
            game = metadata
        game["iteration"] = iteration
        metrics.done(game)
        if game["status"] == "completed":
            store.add(game)
        elif start_sampler is not None and game.get("termination") not in (
            "interrupted",
            "error",
        ):
            start_sampler.retire(gid, game.get("termination", "unknown"))
        on_game(game)

    try:
        BatchScheduler(
            game_factory=iter(factory()),
            executors=runtime.executors,
            concurrent_games=(
                config.collection.concurrent_games
                if concurrent_games is None
                else concurrent_games
            ),
            on_game_done=done,
            on_game_error=lambda gid, exc: None,
            completion_order=True,
        ).run()
    except Exception as exc:
        # A failed merged executor aborts every still-live game. Keep their
        # identities/reasons visible while leaving them completely unlabeled.
        for gid in list(active):
            active[gid]["error"] = f"{type(exc).__name__}: {exc}"
            done(gid, active[gid])
        raise
    finally:
        getattr(runtime, "clear_caches", lambda: None)()
        store.flush()
        if start_sampler is not None:
            start_sampler.reconcile(store.seen)
    return metrics
