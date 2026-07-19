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
