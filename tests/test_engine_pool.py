"""Tests for `imba_chess.eval.engine_pool` (EnginePool + make_sf_move_executor).

Fake engines only -- no subprocess, no `chess.engine`, no Stockfish binary,
no torch. Concurrency proof uses ~0.05s sleeps with a generous margin so it
stays robust under CI scheduling jitter.
"""

from __future__ import annotations

import time

import pytest

from imba_chess.eval.engine_pool import EnginePool, make_sf_move_executor


class _FakeEngine:
    """Records every `play`/`quit` call; sleeps in `play` to simulate
    engine "thinking" time; can be configured to raise from either."""

    def __init__(self, *, play_sleep=0.0, play_result=None, play_exc=None, quit_exc=None):
        self.play_calls: list[tuple] = []
        self.quit_calls = 0
        self._play_sleep = play_sleep
        self._play_result = play_result
        self._play_exc = play_exc
        self._quit_exc = quit_exc

    def play(self, board, limit):
        self.play_calls.append((board, limit))
        if self._play_sleep:
            time.sleep(self._play_sleep)
        if self._play_exc is not None:
            raise self._play_exc
        return self._play_result

    def quit(self):
        self.quit_calls += 1
        if self._quit_exc is not None:
            raise self._quit_exc


# ---------------------------------------------------------------------------
# EnginePool
# ---------------------------------------------------------------------------


def test_engine_pool_spawns_size_engines_eagerly_at_construction():
    spawned: list[_FakeEngine] = []

    def spawn():
        engine = _FakeEngine()
        spawned.append(engine)
        return engine

    pool = EnginePool(spawn=spawn, size=3)

    # Eager spawn: all `size` engines exist immediately after __init__, not
    # only after engine_for_slot is first called.
    assert len(spawned) == 3
    assert [pool.engine_for_slot(i) for i in range(3)] == spawned


def test_engine_for_slot_is_stable_per_slot():
    engines = [_FakeEngine() for _ in range(3)]
    it = iter(engines)
    pool = EnginePool(spawn=lambda: next(it), size=3)

    for slot in range(3):
        first = pool.engine_for_slot(slot)
        second = pool.engine_for_slot(slot)
        assert first is second
        assert first is engines[slot]
    # Distinct slots got distinct engines.
    assert len({id(pool.engine_for_slot(i)) for i in range(3)}) == 3


def test_engine_pool_close_quits_all_engines():
    engines = [_FakeEngine() for _ in range(4)]
    it = iter(engines)
    pool = EnginePool(spawn=lambda: next(it), size=4)

    pool.close()

    assert all(engine.quit_calls == 1 for engine in engines)


def test_engine_pool_close_quits_all_even_when_one_quit_raises_then_reraises_first():
    e0 = _FakeEngine(quit_exc=ValueError("boom0"))
    e1 = _FakeEngine()
    e2 = _FakeEngine(quit_exc=RuntimeError("boom2"))
    engines = [e0, e1, e2]
    it = iter(engines)
    pool = EnginePool(spawn=lambda: next(it), size=3)

    with pytest.raises(ValueError, match="boom0"):
        pool.close()

    # Every engine's quit() was attempted despite e0's failure -- no engine
    # was left un-quit because an earlier sibling raised.
    assert e0.quit_calls == 1
    assert e1.quit_calls == 1
    assert e2.quit_calls == 1


def test_acquire_returns_distinct_slots_and_release_frees_for_reuse():
    engines = [_FakeEngine() for _ in range(3)]
    it = iter(engines)
    pool = EnginePool(spawn=lambda: next(it), size=3)

    slot0, engine0 = pool.acquire()
    slot1, engine1 = pool.acquire()
    slot2, engine2 = pool.acquire()

    # Every concurrently-checked-out acquire() gets a distinct slot/engine.
    assert {slot0, slot1, slot2} == {0, 1, 2}
    assert {id(engine0), id(engine1), id(engine2)} == {id(e) for e in engines}

    # Once released, a slot's engine is available again -- same physical
    # engine object, not a new one (EnginePool never respawns).
    pool.release(slot1)
    slot_reused, engine_reused = pool.acquire()
    assert slot_reused == slot1
    assert engine_reused is engine1


def test_acquire_raises_when_pool_exhausted():
    pool = EnginePool(spawn=lambda: _FakeEngine(), size=2)
    pool.acquire()
    pool.acquire()

    with pytest.raises(RuntimeError, match="no free engine slot"):
        pool.acquire()


def test_acquire_release_never_double_checks_out_under_variable_durations():
    # Regression for the exact bug a static `game_idx % size` round robin
    # hits: acquire/release must stay correct even when "games" (here,
    # acquire/release calls interleaved in an order that does NOT match
    # acquire order) finish in a different order than they started --
    # e.g. slot 0 stays checked out for a long time while slots 1 and 2
    # cycle through several short "games" -- no two simultaneously-held
    # slots may ever collide.
    pool = EnginePool(spawn=lambda: _FakeEngine(), size=2)

    slot_a, engine_a = pool.acquire()  # long-running "game"
    slot_b, engine_b = pool.acquire()  # short "game" 1
    assert {slot_a, slot_b} == {0, 1}
    pool.release(slot_b)

    slot_c, engine_c = pool.acquire()  # short "game" 2 -- must reuse slot_b's index
    assert slot_c == slot_b
    assert engine_c is engine_b
    # The long-running game's slot/engine must be untouched throughout.
    assert pool.engine_for_slot(slot_a) is engine_a
    pool.release(slot_c)
    pool.release(slot_a)


# ---------------------------------------------------------------------------
# make_sf_move_executor
# ---------------------------------------------------------------------------

_THINK = 0.05
_N = 4


def _payloads(engines):
    return [(engine, f"board_{i}", f"limit_{i}") for i, engine in enumerate(engines)]


def test_sf_move_executor_fans_out_concurrently_not_serially():
    engines = [_FakeEngine(play_sleep=_THINK, play_result=f"move_{i}") for i in range(_N)]
    executor = make_sf_move_executor(pool_threads=_N)

    t0 = time.perf_counter()
    results = executor(_payloads(engines))
    elapsed = time.perf_counter() - t0

    assert results == [f"move_{i}" for i in range(_N)]
    # Serial execution would take ~= _N * _THINK; concurrent fan-out should
    # take ~= one think. Generous margin (2x one think) for CI jitter, while
    # still being well under _N * _THINK (0.2s) if it were serialized.
    assert elapsed < _THINK * 2, f"elapsed={elapsed} suggests serial, not concurrent, execution"


def test_sf_move_executor_preserves_payload_order_regardless_of_completion_order():
    # Deliberately reverse the sleep durations so the engine submitted LAST
    # finishes FIRST -- result order must still track payload (submission)
    # order, not completion order.
    sleeps = [0.04, 0.03, 0.02, 0.01]
    engines = [
        _FakeEngine(play_sleep=sleep, play_result=f"move_{i}") for i, sleep in enumerate(sleeps)
    ]
    executor = make_sf_move_executor(pool_threads=len(engines))

    results = executor(_payloads(engines))

    assert results == [f"move_{i}" for i in range(len(engines))]


def test_sf_move_executor_propagates_exception_and_other_futures_still_complete():
    good0 = _FakeEngine(play_sleep=_THINK, play_result="move_0")
    bad = _FakeEngine(play_sleep=_THINK, play_exc=RuntimeError("engine1 failed"))
    good2 = _FakeEngine(play_sleep=_THINK, play_result="move_2")
    engines = [good0, bad, good2]
    executor = make_sf_move_executor(pool_threads=len(engines))

    with pytest.raises(RuntimeError, match="engine1 failed"):
        executor(_payloads(engines))

    # The exception was not swallowed (it propagated), and it did not
    # prevent the other engines' play() calls from running to completion.
    assert good0.play_calls == [("board_0", "limit_0")]
    assert bad.play_calls == [("board_1", "limit_1")]
    assert good2.play_calls == [("board_2", "limit_2")]


def test_sf_move_executor_is_stateless_across_calls():
    executor = make_sf_move_executor(pool_threads=2)

    engines_a = [_FakeEngine(play_result="a0"), _FakeEngine(play_result="a1")]
    assert executor(_payloads(engines_a)) == ["a0", "a1"]

    # A prior call raising must not leave the executor (or some shared pool)
    # broken for subsequent calls.
    bad_engines = [_FakeEngine(play_exc=RuntimeError("boom")), _FakeEngine(play_result="ok")]
    with pytest.raises(RuntimeError, match="boom"):
        executor(_payloads(bad_engines))

    engines_b = [_FakeEngine(play_result="b0"), _FakeEngine(play_result="b1")]
    assert executor(_payloads(engines_b)) == ["b0", "b1"]


def test_sf_move_executor_handles_empty_payload_list():
    executor = make_sf_move_executor(pool_threads=2)
    assert executor([]) == []
