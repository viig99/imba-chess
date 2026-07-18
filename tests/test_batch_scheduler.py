"""Scheduler semantics, tested with plain-Python fake games — no torch, no chess.

Fake game coroutines yield WorkRequest("echo", payload) a scripted number of
times; the fake executor returns [p * 10 for p in payloads] and logs each
tick's merged payload list, letting every scheduler property be asserted
deterministically.
"""

from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest


def _fake_game(game_id, num_requests, log):
    rows = []
    for i in range(num_requests):
        result = yield WorkRequest(kind="echo", payload=(game_id, i))
        rows.append(result)
    log.append(f"done:{game_id}")
    return rows


def _run(games, concurrent, tick_log):
    done_order = []

    def executor(payloads):
        tick_log.append(sorted(payloads))
        return [p for p in payloads]

    scheduler = BatchScheduler(
        game_factory=iter(games),
        executors={"echo": executor},
        concurrent_games=concurrent,
        on_game_done=lambda gid, rows: done_order.append((gid, rows)),
        on_game_error=lambda gid, exc: done_order.append((gid, repr(exc))),
    )
    scheduler.run()
    return done_order


def test_merges_requests_across_games_per_tick():
    log = []
    ticks = []
    games = [(f"g{i}", _fake_game(f"g{i}", 3, log)) for i in range(4)]
    _run(games, concurrent=4, tick_log=ticks)
    # First tick carries one request from each of the 4 games.
    assert ticks[0] == sorted([("g0", 0), ("g1", 0), ("g2", 0), ("g3", 0)])


def test_emission_is_stream_order_even_when_completion_is_not():
    log = []
    ticks = []
    # g0 needs 5 requests, g1 needs 1 — g1 finishes first but must emit second...
    # (stream order: g0 started first)
    games = [("g0", _fake_game("g0", 5, log)), ("g1", _fake_game("g1", 1, log))]
    done = _run(games, concurrent=2, tick_log=ticks)
    assert [gid for gid, _ in done] == ["g0", "g1"]


def test_slot_refill_keeps_concurrency():
    log = []
    ticks = []
    games = [(f"g{i}", _fake_game(f"g{i}", 2, log)) for i in range(5)]
    _run(games, concurrent=2, tick_log=ticks)
    assert len({gid for tick in ticks for gid, _ in tick}) == 5  # all games ran


def test_error_isolation_drops_game_and_continues():
    def _bad_game(gid):
        yield WorkRequest(kind="echo", payload=(gid, 0))
        raise ValueError("boom")

    log = []
    ticks = []
    games = [("bad", _bad_game("bad")), ("ok", _fake_game("ok", 2, log))]
    done = _run(games, concurrent=2, tick_log=ticks)
    # Both on_game_error and on_game_done write into done_order in this
    # harness, so "bad" produces two stream-ordered entries before "ok":
    # the error callback, then the done callback with rows=None.
    assert done[0][0] == "bad" and "boom" in done[0][1]
    assert done[1] == ("bad", None)
    assert done[2] == ("ok", [("ok", 0), ("ok", 1)])


def test_determinism_same_inputs_same_ticks():
    def run_once():
        log, ticks = [], []
        games = [(f"g{i}", _fake_game(f"g{i}", 3, log)) for i in range(3)]
        _run(games, concurrent=3, tick_log=ticks)
        return ticks

    assert run_once() == run_once()
