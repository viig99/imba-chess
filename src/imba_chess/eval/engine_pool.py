"""Engine pool + concurrent sf_move executor for the fast-clean-evals batched
driver (`docs/superpowers/plans/2026-07-19-fast-clean-evals.md` Task 2).

`EnginePool` owns one engine handle per `BatchScheduler` slot, spawned via a
caller-supplied `spawn` callable so tests can substitute fakes and production
code supplies e.g. `lambda: chess.engine.SimpleEngine.popen_uci(sf_path)`.
This matches Task 3's engine-ownership design: a slot's next game reuses that
slot's engine, exactly generalizing today's one-engine-for-all-games
behavior at `--concurrent-games 1` to `--concurrent-games G`.

`make_sf_move_executor` builds the `sf_move` kind's merged executor for
`BatchScheduler` (see `batch_scheduler.py`): each tick it receives one
`(engine, board, limit)` payload per live slot with a pending engine turn and
fans them out concurrently across a `ThreadPoolExecutor`, since each engine's
`play()` blocks on its own subprocess I/O and G independent slot-owned
engines can "think" in parallel rather than serially.

Deliberately torch-free AND `chess`/`chess.engine`-free: engine handles,
boards, limits, and results are all typed `Any` rather than
`chess.engine.SimpleEngine` / `chess.Board` / `chess.engine.Limit` /
`chess.engine.PlayResult`. This module only ever calls `engine.play(board,
limit)` and `engine.quit()` -- it never constructs or inspects a
`chess.engine` type itself -- so there is nothing gained by importing
`chess.engine` purely for annotations, and real value in NOT doing so: it
keeps this module importable (and its tests runnable) with fake engine
objects and zero `python-chess`/subprocess dependency, matching the brief's
"no torch, no chess.engine subprocess in tests" test requirement.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class EnginePool:
    """Owns `size` engine handles, one per scheduler slot, spawned via `spawn`.

    Spawn timing: EAGER, in `__init__` -- all `size` engines are spawned
    immediately at construction, not lazily on first `engine_for_slot` call.
    The pool exists to give every slot a stable, long-lived engine for the
    whole run; eager spawn means a broken `spawn` (e.g. the Stockfish binary
    missing, or engine startup failing) fails loudly at pool construction,
    before any game has been scheduled, rather than silently deep into a run
    on whichever slot's first game happens to reach its first engine turn --
    fail-fast per repo policy, and simpler to reason about than lazy spawn's
    "first call wins" per-slot race if `engine_for_slot` were ever called
    concurrently.
    """

    def __init__(self, *, spawn: Callable[[], Any], size: int) -> None:
        self._engines: list[Any] = [spawn() for _ in range(size)]
        # FIFO of free physical slot indices, for acquire()/release() (Task
        # 3's eval driver). Starts with every slot free, ascending order --
        # deterministic, easy to reason about in tests -- and shrinks/grows
        # as games check engines out and back in.
        self._free_slots: list[int] = list(range(size))

    def engine_for_slot(self, slot_index: int) -> Any:
        """Return the engine owned by `slot_index`. Stable across calls: the
        same slot always gets back the identical engine object spawned for
        it at construction."""
        return self._engines[slot_index]

    def acquire(self) -> tuple[int, Any]:
        """Check out a free engine, returning `(slot_index, engine)`.

        Task 3's eval driver calls this exactly when the `BatchScheduler`
        admits a new game into a live slot (i.e. inside the game factory,
        synchronously, before the game's coroutine object is even created)
        and `release`s the slot exactly when that game's coroutine finishes
        or raises. Because the scheduler never holds more than
        `concurrent_games` games live at once (see batch_scheduler.py's
        `_fill_slots` invariant), and acquire/release bracket a game's exact
        live-slot lifetime 1:1, this guarantees no two *simultaneously live*
        games ever share a physical engine -- unlike assigning engines by
        static `game_index % size` round-robin, which breaks the instant two
        games have different durations (a fast game N and a still-running
        earlier game can both compute the same `% size` slot while both are
        live). Raises if every slot is already checked out: that is a caller
        bug (more concurrently-live games than pool size), not a condition
        to silently block or reuse a live engine for.
        """
        if not self._free_slots:
            raise RuntimeError(
                "EnginePool.acquire: no free engine slot available "
                f"(pool size={len(self._engines)}, all checked out)"
            )
        slot_index = self._free_slots.pop(0)
        return slot_index, self._engines[slot_index]

    def release(self, slot_index: int) -> None:
        """Return `slot_index`'s engine to the free pool.

        The engine object itself is untouched (still alive, ready for its
        next game) -- only free-list bookkeeping changes. Safe to call from
        a generator's `finally` block on both the normal-completion and
        exception paths (see `scripts/eval_vs_stockfish.py`'s
        `_release_engine_on_finish`).
        """
        self._free_slots.append(slot_index)

    def close(self) -> None:
        """Quit every engine, even if some `quit()` calls raise; then
        re-raise the FIRST error encountered.

        Fail-fast, but never at the cost of leaking a live engine process:
        every engine's `quit()` is attempted regardless of an earlier
        engine's failure, so a single misbehaving engine can never prevent
        its siblings from being shut down. Only after every `quit()` has
        been attempted does this re-raise the first exception seen (later
        exceptions from other engines are not swallowed silently -- they are
        deliberately not re-raised so exactly one exception surfaces, per
        the brief's "raise the FIRST error after closing all" contract --
        but they did fire during the loop, so nothing was skipped).
        """
        first_error: Exception | None = None
        for engine in self._engines:
            try:
                engine.quit()
            except Exception as exc:  # noqa: BLE001 - must not skip remaining engines
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _play_one(payload: tuple[Any, Any, Any]) -> Any:
    engine, board, limit = payload
    return engine.play(board, limit)


def make_sf_move_executor(*, pool_threads: int) -> Callable[[list[Any]], list[Any]]:
    """Build the `sf_move` kind's merged executor for `BatchScheduler`.

    Each payload is `(engine, board, limit)`; `limit` is passed straight
    through to `engine.play(board, limit)` uninterpreted (a
    `chess.engine.Limit` or an equivalent dict in production, whatever a
    test fake wants -- this module never inspects it).

    Payloads fan out across a fresh `ThreadPoolExecutor(max_workers=
    pool_threads)` for the duration of this one call (the returned executor
    is stateless across calls -- no thread pool or other state is kept alive
    between ticks/invocations); results come back in the SAME order as
    `payloads`, regardless of completion order.

    Fail-fast: if any engine's `play()` raises, `ThreadPoolExecutor`'s
    context manager blocks on `shutdown(wait=True)` on the way out
    regardless of whether an exception is propagating, so every OTHER
    payload's `play()` still runs to completion before the exception
    reaches the caller -- no orphaned background work, and no engine call
    silently skipped or swallowed.
    """

    def executor(payloads: list[Any]) -> list[Any]:
        with ThreadPoolExecutor(max_workers=pool_threads) as pool:
            futures = [pool.submit(_play_one, payload) for payload in payloads]
            results = [future.result() for future in futures]
        return results

    return executor
