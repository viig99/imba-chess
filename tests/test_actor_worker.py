"""Tests for the multiprocess-eval-actors Task 1 protocol + worker:
`imba_chess.eval.actor_protocol` and `imba_chess.eval.actor_worker`.

Two kinds of coverage:
  - a permanent, subprocess-based regression test that these two modules
    (and everything they import) stay torch-free -- a same-process
    meta-path hook can't be trusted here, since by the time any test in this
    suite runs, some earlier test has already imported torch, and `sys.
    modules` is checked before any import finder ever runs.
  - an in-process "fake server" driving `run_eval_worker` end-to-end over a
    real `multiprocessing.Pipe()` (worker on a background thread, this
    test's main thread playing the GPU-server role), asserting message
    ordering, node-id/parent-id chain shape, summary fragments, and that SF
    turns go through the injected fake engine.
"""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import textwrap
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine
import pytest

from imba_chess.eval.actor_protocol import (
    GameDone,
    RootEvalRequest,
    RootEvalResponse,
    WaveRequest,
    WaveResponse,
    WorkerFinished,
)
from imba_chess.eval.actor_worker import (
    _WorkerSearchNode,
    _build_engine,
    _legal_vocab_projection,
    _log_softmax_f32,
    run_eval_worker,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_VOCAB_PATH = REPO_ROOT / "artifacts" / "move_vocab_static_uci.json"


def test_actor_protocol_and_worker_stay_torch_free() -> None:
    """Permanent regression test: `import imba_chess.eval.actor_protocol` /
    `actor_worker` must never pull torch into `sys.modules`, even
    transitively through package `__init__`s. Run in a clean subprocess
    (not this test process) since pytest's own collection has already
    imported torch-dependent modules elsewhere in the suite by the time any
    single test runs -- `sys.modules` is consulted before any `sys.
    meta_path` finder, so an in-process hook here would prove nothing.
    """
    script = textwrap.dedent(
        """
        import sys

        class _TorchBlocker:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError(f"torch import blocked: {name}")
                return None

        sys.meta_path.insert(0, _TorchBlocker())

        import imba_chess.eval.actor_protocol
        import imba_chess.eval.actor_worker

        assert "torch" not in sys.modules, sorted(
            m for m in sys.modules if m == "torch" or m.startswith("torch.")
        )
        print("TORCH_FREE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "TORCH_FREE_OK" in result.stdout


# --------------------------------------------------------------------------
# Fake in-process "server": answers RootEvalRequest/WaveRequest with all-zero
# raw "legal_logits", one per requested legal_vocab_id -- the worker itself
# now computes the legal-move projection (movegen + UCI-sort + vocab lookup)
# BEFORE sending the request (profile-driven thin-down,
# docs/superpowers/sdd/thin-report.md), so the fake server no longer needs to
# reconstruct a board or know anything about legality at all; it only needs
# to know how many ids were requested per row. All-zero logits make the
# worker's own log-softmax (`actor_worker._log_softmax_f32`) come out
# uniform (log(1/n) per entry) -- the same scripted distribution the
# pre-thin-down fake server built explicitly -- without this test needing to
# duplicate that math.
# --------------------------------------------------------------------------


class _FakeEngine:
    """Minimal `chess.engine.SimpleEngine`-shaped test double: always plays
    the (deterministic) first legal move in python-chess's own iteration
    order, records call count for assertions."""

    def __init__(self) -> None:
        self.play_calls = 0
        self.quit_calls = 0

    def play(self, board: chess.Board, limit) -> SimpleNamespace:
        self.play_calls += 1
        move = next(iter(board.legal_moves))
        return SimpleNamespace(move=move)

    def quit(self) -> None:
        self.quit_calls += 1


def _run_fake_server(conn, received: list) -> None:
    """Plays the GPU-server role for `run_eval_worker`: answers every
    RootEvalRequest/WaveRequest with a scripted (but position-legal)
    projection, and records every message it receives (in receipt order) for
    the test's assertions. Stops after the worker's `WorkerFinished`."""
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        received.append(msg)
        if isinstance(msg, RootEvalRequest):
            conn.send(
                RootEvalResponse(
                    turn_id=msg.turn_id,
                    value_stm=0.1,
                    legal_logits=[0.0] * len(msg.legal_vocab_ids),
                )
            )
        elif isinstance(msg, WaveRequest):
            rows = [(0.2, [0.0] * len(row.legal_vocab_ids)) for row in msg.rows]
            conn.send(WaveResponse(rows=rows))
        elif isinstance(msg, WorkerFinished):
            break
        elif isinstance(msg, GameDone):
            continue
        else:  # pragma: no cover - fail fast on an unexpected message shape
            raise AssertionError(f"unexpected message from worker: {msg!r}")


def _run_two_short_games(*, model_move_policy: str) -> tuple[list, dict]:
    """Drives `run_eval_worker` over a real `multiprocessing.Pipe()` for 2
    short (max_plies=2) games, worker on a background thread, this thread
    playing the fake server. Returns (received_messages, engine)."""
    import multiprocessing

    parent_conn, child_conn = multiprocessing.Pipe()
    received: list = []
    fake_engine = _FakeEngine()

    worker_config = {
        "worker_id": 7,
        "game_indices": [0, 1],
        "seed": 123,
        "max_plies": 2,
        "opening_random_plies": 0,
        "model_move_policy": model_move_policy,
        "value_rerank_top_k": 1,
        "value_rerank_lambda": 0.0,
        "vocab_path": str(STATIC_VOCAB_PATH),
        "vocab_include_unk": False,
        "engine": {
            "fake_engine_factory": lambda: fake_engine,
            "stockfish_limit": {"time": 0.01},
        },
    }

    server_thread = threading.Thread(target=_run_fake_server, args=(parent_conn, received))
    server_thread.start()
    try:
        run_eval_worker(child_conn, worker_config)
    finally:
        server_thread.join(timeout=30)
        assert not server_thread.is_alive(), "fake server thread did not finish"

    return received, {"engine": fake_engine}


def test_two_short_games_end_to_end_message_ordering_and_summaries() -> None:
    received, ctx = _run_two_short_games(model_move_policy="value_search_d2")

    # -- Overall message shape: game0 [root, wave, wave, GameDone], game1
    # [root, wave, wave, GameDone], WorkerFinished. game0's model plays
    # WHITE (game_idx even) so its model turn is ply 0 (before any SF
    # call); game1's model plays BLACK so SF moves first (not over the
    # pipe at all -- the fake engine is called directly), then the model's
    # one turn. Wave count per turn is policy-shaped (value_search_d2:
    # exactly 2 EvalRequests per turn -- board1 candidates, then board2
    # replies) with top_k=1 keeping each wave's row count small; asserted
    # as "at least 1 row" rather than an exact count so this isn't coupled
    # to incidental legal-move ordering.
    kinds = [type(m).__name__ for m in received]
    assert kinds.count("RootEvalRequest") == 2, kinds
    assert kinds.count("WaveRequest") == 4, kinds
    assert kinds.count("GameDone") == 2, kinds
    assert kinds[-1] == "WorkerFinished", kinds

    # Root before its own turn's waves, per turn: every WaveRequest in the
    # stream must be preceded (not necessarily immediately) by a
    # RootEvalRequest with the SAME turn_id, and no WaveRequest ever
    # precedes ITS turn's root.
    seen_root_turn_ids: set[int] = set()
    for msg in received:
        if isinstance(msg, RootEvalRequest):
            seen_root_turn_ids.add(msg.turn_id)
        elif isinstance(msg, WaveRequest):
            assert msg.turn_id in seen_root_turn_ids, (
                f"WaveRequest(turn_id={msg.turn_id}) arrived before its "
                f"turn's RootEvalRequest; roots seen so far: {seen_root_turn_ids}"
            )

    # turn_id is a strictly increasing, worker-lifetime-scoped counter (one
    # model turn per game here): game0's turn, then game1's turn.
    root_turn_ids = [m.turn_id for m in received if isinstance(m, RootEvalRequest)]
    assert root_turn_ids == sorted(root_turn_ids)
    assert len(set(root_turn_ids)) == 2

    # -- Node-id/parent-id chain shape: within each turn, wave 1's rows are
    # all children of the turn's root prefix (parent_id is None); wave 2's
    # rows all have parent_id set, and every such parent_id refers to a
    # node_id actually seen in that SAME turn's wave 1 (the exact
    # `_CachedNode`-style parent-link discipline `_WorkerSearchNode.extend`
    # is meant to mirror).
    wave_requests_by_turn: dict[int, list[WaveRequest]] = {}
    for msg in received:
        if isinstance(msg, WaveRequest):
            wave_requests_by_turn.setdefault(msg.turn_id, []).append(msg)
    assert len(wave_requests_by_turn) == 2
    for turn_id, waves in wave_requests_by_turn.items():
        assert len(waves) == 2, (turn_id, waves)
        wave1, wave2 = waves
        assert len(wave1.rows) >= 1
        wave1_node_ids = {row.node_id for row in wave1.rows}
        for row in wave1.rows:
            assert row.parent_id is None
        assert len(wave2.rows) >= 1
        for row in wave2.rows:
            assert row.parent_id is not None
            assert row.parent_id in wave1_node_ids
        # node ids are unique within a turn (fresh evaluator per turn).
        all_ids = [row.node_id for row in wave1.rows] + [row.node_id for row in wave2.rows]
        assert len(all_ids) == len(set(all_ids))

    # -- summary fragments: both games truncate at max_plies=2 before
    # completion (incomplete), exactly one model turn each, opposite
    # colors (game_idx 0 -> model white, game_idx 1 -> model black), and
    # the fake server's legality always agrees with the worker's own live
    # board (legal_moves_total == legal_moves_mapped_total).
    game_done = [m for m in received if isinstance(m, GameDone)]
    assert {m.game_idx for m in game_done} == {0, 1}
    assert {m.worker_id for m in game_done} == {7}
    for msg in game_done:
        frag = msg.summary_fragment
        assert frag["games"] == 1
        assert frag["completed_games"] == 0
        assert frag["incomplete_games"] == 1
        assert frag["total_plies"] == 2
        assert frag["model_turns"] == 1
        assert frag["turns_with_no_vocab_legal_move"] == 0
        assert frag["legal_moves_total"] == frag["legal_moves_mapped_total"]
        assert frag["legal_moves_total"] > 0

    by_game_idx = {m.game_idx: m.summary_fragment for m in game_done}
    assert by_game_idx[0]["games_as_white"] == 1
    assert by_game_idx[0]["games_as_black"] == 0
    assert by_game_idx[1]["games_as_white"] == 0
    assert by_game_idx[1]["games_as_black"] == 1

    # -- SF turns went through the injected fake engine, synchronously,
    # not through the conn pipe at all: game0 (model=white) has exactly 1
    # SF turn (ply 1), game1 (model=black) has exactly 1 SF turn (ply 0).
    assert ctx["engine"].play_calls == 2
    assert ctx["engine"].quit_calls == 1


def test_greedy_policy_sends_no_wave_requests() -> None:
    """Sanity check on the policy dispatch in `_select_model_move`: greedy
    never builds a `_WaveEvaluator`, so no WaveRequest should ever be sent,
    only the per-turn root round trip."""
    received, _ctx = _run_two_short_games(model_move_policy="greedy")
    kinds = [type(m).__name__ for m in received]
    assert kinds.count("RootEvalRequest") == 2
    assert kinds.count("WaveRequest") == 0
    assert kinds.count("GameDone") == 2
    assert kinds[-1] == "WorkerFinished"


def test_worker_search_node_extend_mirrors_cached_node_parent_semantics() -> None:
    """Unit check on `_WorkerSearchNode`/`_WaveEvaluator.extend` in
    isolation (no pipe): `None` (root prefix) and a real parent handle both
    produce the expected `(node_id, parent_id)` chain, matching
    `position_evaluator._CachedNode`'s `parent is None` convention."""
    from imba_chess.eval.actor_worker import _WaveEvaluator

    class _StubMoveVocab:
        def encode(self, uci: str) -> int:
            return {"e2e4": 101, "e7e5": 202}[uci]

    class _StubBoardStateEncoder:
        pass

    evaluator = _WaveEvaluator(
        conn=None,
        worker_id=0,
        turn_id=0,
        move_vocab=_StubMoveVocab(),
        board_state_encoder=_StubBoardStateEncoder(),
    )

    root_child = evaluator.extend(None, "e2e4")
    assert isinstance(root_child, _WorkerSearchNode)
    assert root_child.node_id == 0
    assert root_child.parent_id is None
    assert root_child.move_vocab_id == 101

    grandchild = evaluator.extend(root_child, "e7e5")
    assert grandchild.node_id == 1
    assert grandchild.parent_id == root_child.node_id
    assert grandchild.move_vocab_id == 202

    # A handle from an unrelated/opaque object (not a _WorkerSearchNode) is
    # treated the same as None, mirroring _CachedNode.extend's isinstance guard.
    other_root_child = evaluator.extend(object(), "e2e4")
    assert other_root_child.parent_id is None


def test_build_engine_quits_on_configure_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_build_engine`'s production path (`stockfish_path`, not
    `fake_engine_factory`): if `engine.configure(...)` raises right after a
    successful `popen_uci`, the already-spawned engine must be quit()
    before the exception propagates -- otherwise that Stockfish subprocess
    leaks (run_eval_worker's own try/finally can't help here: it starts
    only after `_build_engine` returns, so this guard has to live inside
    `_build_engine` itself). `popen_uci` itself is monkeypatched out (no
    real Stockfish binary needed for this unit test) to return a fake
    engine whose `.configure()` always raises."""

    class _ConfigureFailsEngine:
        def __init__(self) -> None:
            self.quit_calls = 0

        def configure(self, options: dict) -> None:
            raise RuntimeError("bad UCI option")

        def quit(self) -> None:
            self.quit_calls += 1

    fake_engine = _ConfigureFailsEngine()
    monkeypatch.setattr(
        chess.engine.SimpleEngine,
        "popen_uci",
        staticmethod(lambda path: fake_engine),
    )

    with pytest.raises(RuntimeError, match="bad UCI option"):
        _build_engine(
            {"stockfish_path": "/fake/stockfish", "stockfish_options": {"Threads": 2}}
        )

    assert fake_engine.quit_calls == 1


# ---------------------------------------------------------------------------
# Task 3 addendum: `_install_sigterm_handler` (this module).
#
# Task 3's orchestrator (`scripts/eval_vs_stockfish.py`'s
# `_terminate_worker_processes`) SIGTERMs every worker on any fail-fast
# supervision trigger. A bare SIGTERM skips this module's own `finally:
# engine.quit()` -- the test below proves, with a REAL spawned process (not
# an in-process call: signal delivery/handling is exactly the mechanism
# under test), that the installed handler turns that into a clean unwind
# instead: `engine.quit()` still runs even though the worker is blocked
# inside `engine.play()` when SIGTERM arrives.
# ---------------------------------------------------------------------------


class _SigtermTestFakeEngine:
    """Sleeps in `play()` long enough for the test to deliver SIGTERM while
    the worker is blocked inside it; `quit()` writes a marker file so the
    test (a different process) can observe it ran."""

    def __init__(self, quit_marker_path: str) -> None:
        self._quit_marker_path = quit_marker_path

    def play(self, board: chess.Board, limit) -> SimpleNamespace:
        time.sleep(5.0)
        return SimpleNamespace(move=next(iter(board.legal_moves)))  # pragma: no cover

    def quit(self) -> None:
        Path(self._quit_marker_path).write_text("quit-called", encoding="utf-8")


def _sigterm_test_fake_engine_factory(*, quit_marker_path: str) -> _SigtermTestFakeEngine:
    """Top-level (picklable-under-spawn) factory; bound to `quit_marker_path`
    via `functools.partial` (also picklable, unlike a closure) since the
    protocol's `fake_engine_factory` is a zero-arg callable."""
    return _SigtermTestFakeEngine(quit_marker_path)


def test_sigterm_while_blocked_in_engine_play_still_quits_engine(tmp_path: Path) -> None:
    quit_marker = tmp_path / "quit.marker"
    worker_config = {
        "worker_id": 0,
        # game_idx=1 -> model plays BLACK (game_idx % 2 == 1) -> ply 0 is an
        # SF turn, so the worker calls engine.play() immediately -- no pipe
        # round trip (and so no fake server) is needed before the kill.
        "game_indices": [1],
        "seed": 0,
        "max_plies": 4,
        "model_move_policy": "greedy",
        "vocab_path": str(STATIC_VOCAB_PATH),
        "engine": {
            "fake_engine_factory": partial(
                _sigterm_test_fake_engine_factory, quit_marker_path=str(quit_marker)
            ),
            "stockfish_limit": {"time": 0.01},
        },
    }

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=run_eval_worker, args=(child_conn, worker_config))
    proc.start()
    child_conn.close()
    try:
        time.sleep(1.0)  # let the worker reach engine.play()'s 5s sleep
        assert proc.is_alive(), "worker exited before SIGTERM could be sent"
        proc.terminate()  # SIGTERM
        proc.join(timeout=10.0)
        assert not proc.is_alive(), "worker did not exit after SIGTERM"
        assert quit_marker.exists(), (
            "engine.quit() did not run after SIGTERM -- the installed "
            "handler should have unwound run_eval_worker's finally block"
        )
    finally:
        parent_conn.close()
        if proc.is_alive():  # pragma: no cover - safety net only
            proc.kill()
            proc.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Profile-driven thin-down (docs/superpowers/sdd/thin-report.md): the
# legal-move projection (movegen + UCI-sort + vocab lookup) and the
# log-softmax over the server's raw logits both moved from the server into
# this module. These two tests pin their correctness independently of the
# end-to-end worker/fake-server tests above: `_legal_vocab_projection` must
# match `position_evaluator._project_legal_logits_cozy`'s mapped+sorted
# output exactly (over many real random-playout positions, not just the
# opening), and `_log_softmax_f32` must match `torch.log_softmax`'s float32
# result to the fp32-exactness suite's 1e-6 bar. Both tests import torch/
# position_evaluator locally (this module itself must stay torch-free at
# import time -- test_actor_protocol_and_worker_stay_torch_free above is the
# permanent regression test for that -- but nothing prevents a TEST
# function, as opposed to the module under test, from importing torch for
# its own reference computation).
# ---------------------------------------------------------------------------


def test_legal_vocab_projection_matches_project_legal_logits_cozy_over_random_playouts() -> None:
    import random

    import cozy_chess as cc

    from imba_chess.data.move_vocab import MoveVocab
    from imba_chess.eval import cozy_bridge
    from imba_chess.eval.position_evaluator import _project_legal_logits_cozy

    move_vocab = MoveVocab.build_static()
    rng = random.Random(0)
    positions_checked = 0
    for _game in range(8):
        board = chess.Board()
        for _ply in range(30):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            cozy_board = cozy_bridge.board_to_cozy(board)

            vocab_ids, moves, ucis = _legal_vocab_projection(cozy_board, move_vocab)

            # Reference: the pre-thin-down server-side projection, driven off
            # a dummy logits tensor (only the move-mapping/order matters
            # here, not the values) -- fp32-exactness of the actual gathered
            # VALUES is covered separately by tests/test_actor_server.py.
            import torch

            dummy_logits = torch.arange(len(move_vocab), dtype=torch.float32)
            _legal_logits, ref_moves, ref_ucis, _total, _mapped = (
                _project_legal_logits_cozy(
                    logits=dummy_logits, cozy_board=cozy_board, move_vocab=move_vocab
                )
            )
            assert ucis == ref_ucis
            assert [str(m) for m in moves] == [str(m) for m in ref_moves]
            assert vocab_ids == [move_vocab.token_to_id[u] for u in ucis]
            positions_checked += 1
    assert positions_checked > 100


def test_log_softmax_f32_matches_torch_log_softmax_fp32() -> None:
    import random

    import torch

    rng = random.Random(0)
    for n in (0, 1, 2, 5, 30, 128):
        raw = [rng.uniform(-20.0, 20.0) for _ in range(n)]
        # Round-trip through float32 first: raw_logits on the wire are
        # produced by torch.Tensor.tolist() on a float32 tensor, so a
        # genuine test input must be exactly float32-representable, same as
        # production.
        raw_f32 = torch.tensor(raw, dtype=torch.float32).tolist()

        got = _log_softmax_f32(raw_f32)
        expected = torch.log_softmax(torch.tensor(raw_f32, dtype=torch.float32), dim=0)

        assert len(got) == n
        torch.testing.assert_close(
            torch.tensor(got, dtype=torch.float32), expected, atol=1e-6, rtol=1e-6
        )
