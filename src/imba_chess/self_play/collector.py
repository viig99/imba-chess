"""One sequential search per game, inference batched across independent games."""

from collections import Counter, defaultdict, deque
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
        self.buckets = defaultdict(Counter)
        self.restarted_positions = set()

    def launched(self, metadata):
        if "starting_bucket" in metadata:
            counts = self.buckets[metadata["starting_bucket"]]
            counts["launch_attempts"] += 1
            counts["fallbacks"] += (
                metadata["requested_bucket"] != metadata["starting_bucket"]
            )
            if "restart_parent" in metadata:
                self.restarted_positions.add(metadata["restart_parent"]["history_id"])

    def position(self, result, elapsed, bucket=None):
        self.counts["searched_positions"] += 1
        self.counts["neural_evaluations"] += result.neural_evaluations
        self.counts["simulations"] += result.simulations
        self.counts["terminal_hits"] += result.terminal_hits
        self.counts["depth_cutoffs"] += result.depth_cutoffs
        self.latencies.append(elapsed)
        if bucket is not None:
            counts = self.buckets[bucket]
            for key in (
                "neural_evaluations",
                "simulations",
                "terminal_hits",
                "depth_cutoffs",
            ):
                counts[key] += getattr(result, key)
            counts["searched_positions"] += 1
            counts["search_seconds"] += elapsed

    def done(self, game):
        self.counts[game["status"] + "_games"] += 1
        self.terminations[game["termination"]] += 1
        if "starting_bucket" in game:
            counts = self.buckets[game["starting_bucket"]]
            counts["finished_games"] += 1
            counts[game["status"] + "_games"] += 1
            counts["limited_games"] += game["termination"] in (
                "context_limit",
                "game_limit",
            )
            length = len(game["moves"])
            counts["continuation_plies"] += length
            counts["max_continuation_plies"] = max(
                counts["max_continuation_plies"], length
            )
            if game["status"] == "completed":
                counts["completed_continuation_plies"] += length
        if game["status"] == "completed":
            self.counts["completed_positions"] += len(game["moves"])
            if game["split"] == "train":
                self.counts["usable_positions"] += len(game["moves"])
                self.counts["training_positions"] += len(game["moves"])

    def report(self):
        elapsed = max(time.perf_counter() - self.start, 1e-9)
        latencies = sorted(self.latencies)
        report = dict(
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
        if self.buckets:
            report["starting_buckets"] = {
                bucket: dict(
                    counts,
                    completion_rate=counts["completed_games"]
                    / max(1, counts["finished_games"]),
                    limit_rate=counts["limited_games"]
                    / max(1, counts["finished_games"]),
                    mean_continuation_plies=counts["continuation_plies"]
                    / max(1, counts["finished_games"]),
                    mean_completed_continuation_plies=counts[
                        "completed_continuation_plies"
                    ]
                    / max(1, counts["completed_games"]),
                    search_position_share=counts["searched_positions"]
                    / max(1, self.counts["searched_positions"]),
                    neural_evaluation_share=counts["neural_evaluations"]
                    / max(1, self.counts["neural_evaluations"]),
                )
                for bucket, counts in sorted(self.buckets.items())
            }
            report["distinct_restarted_positions"] = len(self.restarted_positions)
        return report


class CollectionPhase:
    """Launch, completion and metrics bookkeeping for one collection phase.

    Owned by the coordinating process in both the in-process collector and
    the multi-worker one: streamed starts, regret state and replay writes stay
    here; only game play runs elsewhere.
    """

    def __init__(
        self,
        *,
        config,
        actor_id,
        store,
        max_positions,
        start_sampler,
        game_count=None,
        should_launch=lambda: True,
        iteration=0,
        on_game=lambda game: None,
        skip_ids=(),
        metrics=None,
    ):
        self.config, self.actor_id, self.store = config, actor_id, store
        self.start_sampler, self.game_count = start_sampler, game_count
        self.should_launch, self.iteration, self.on_game = should_launch, iteration, on_game
        self.metrics = metrics or CollectionMetrics()
        self.skipped = set(skip_ids)
        self.active = {}
        self.launches = 0
        self.exhausted = False
        if config.regret is None:
            start_sampler.begin_phase(iteration, actor_id, self.skipped)
        else:
            start_sampler.begin_phase(
                iteration, actor_id, store, max_positions=max_positions
            )

    def next_launch(self):
        """(game_id, seed, metadata) for the next game; None once launching has ended."""
        while not self.exhausted:
            if not self.should_launch() or (
                self.game_count is not None and self.launches >= self.game_count
            ):
                break
            if (
                self.game_count is None
                and self.metrics.counts["training_positions"]
                >= self.config.collection.fresh_positions
            ):
                break
            try:
                seed, gid = self.start_sampler.next_launch(self.iteration, self.actor_id)
            except InterruptedError:
                break
            self.launches += 1
            if gid in self.skipped:
                continue
            self.active[gid] = dict(
                game_id=gid,
                seed_id=seed.seed_id,
                source_id=seed.source_id,
                actor_id=self.actor_id,
                status="unfinished",
                termination="error",
                moves=[],
                targets=[],
                outcome_white=None,
            )
            metadata = self.start_sampler.launch_metadata(gid)
            self.active[gid].update(metadata)
            self.metrics.launched(metadata)
            return gid, seed, metadata
        self.exhausted = True
        return None

    def done(self, gid, game):
        config, store = self.config, self.store
        metadata = self.active.pop(gid)
        if game is None:
            game = metadata
        else:
            game.update(
                {
                    k: metadata[k]
                    for k in (
                        "requested_bucket",
                        "starting_bucket",
                        "restart_parent",
                        "exploration_seed",
                    )
                    if k in metadata
                }
            )
        game["iteration"] = self.iteration
        self.metrics.done(game)
        if game["status"] == "completed":
            if config.regret is not None:
                from .regret import suffix_regrets

                suffix_regrets(game)  # Fail explicitly before replay serialization.
            store.add(game)
            if config.regret is not None and gid in store.seen:
                self.start_sampler.reconcile(store)
        elif game.get("termination") not in (
            "interrupted",
            "error",
        ):
            self.start_sampler.retire(gid, game.get("termination", "unknown"))
        self.on_game(game)

    def abort(self, exc):
        # A failed executor or worker aborts every still-live game. Keep their
        # identities/reasons visible while leaving them completely unlabeled.
        for gid in list(self.active):
            self.active[gid]["error"] = f"{type(exc).__name__}: {exc}"
            self.done(gid, self.active[gid])

    def finish(self):
        self.store.flush()
        self.start_sampler.reconcile(
            self.store.seen if self.config.regret is None else self.store
        )


def collect(
    *,
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
    start_sampler,
):
    if concurrent_games is not None and concurrent_games < 1:
        raise ValueError("concurrent_games must be positive")
    phase = CollectionPhase(
        config=config,
        actor_id=actor_id,
        store=store,
        max_positions=max_positions,
        start_sampler=start_sampler,
        game_count=game_count,
        should_launch=should_launch,
        iteration=iteration,
        on_game=on_game,
        skip_ids=skip_ids,
        metrics=metrics,
    )

    def factory():
        while (launch := phase.next_launch()) is not None:
            gid, seed, metadata = launch
            yield gid, launch_game(
                runtime=runtime,
                config=config,
                config_id=config.identifier,
                actor_id=actor_id,
                seed=seed,
                game_id=gid,
                max_positions=max_positions,
                should_stop=should_stop,
                on_position=lambda result, elapsed, bucket=metadata.get(
                    "starting_bucket"
                ): phase.metrics.position(result, elapsed, bucket),
            )

    try:
        BatchScheduler(
            game_factory=iter(factory()),
            executors=runtime.executors,
            concurrent_games=(
                config.collection.concurrent_games
                if concurrent_games is None
                else concurrent_games
            ),
            on_game_done=phase.done,
            on_game_error=lambda gid, exc: None,
            completion_order=True,
        ).run()
    except Exception as exc:
        phase.abort(exc)
        raise
    finally:
        getattr(runtime, "clear_caches", lambda: None)()
        phase.finish()
    return phase.metrics


def launch_game(
    *, runtime, config, config_id, actor_id, seed, game_id, max_positions,
    should_stop, on_position,
):
    """One self-play game coroutine with the run's search and collection limits."""
    return play_game(
        seed=seed,
        game_id=game_id,
        actor_id=actor_id,
        runtime=runtime,
        search_config=config.search,
        max_positions=max_positions,
        max_game_plies=config.collection.max_game_plies,
        run_seed=config.run.seed,
        config_id=config_id,
        should_stop=should_stop,
        on_position=on_position,
    )
