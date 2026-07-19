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

import math
import subprocess
import sys
import textwrap
import threading
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
from imba_chess.eval.actor_worker import _WorkerSearchNode, _build_engine, run_eval_worker

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
# Fake in-process "server": answers RootEvalRequest/WaveRequest with values
# derived from a board reconstructed from the request's own board-state
# fields (piece_ids/turn_id/castle_id), so the worker's real game state and
# the fake server's answers always agree on which moves are actually legal.
# En-passant is deliberately not reconstructed (harmless: it can only ever
# make the returned legal-move set a strict subset of the true one, never
# include an illegal move, and these test games are far too short for ep to
# be reachable anyway).
# --------------------------------------------------------------------------

_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def _board_from_state(piece_ids: list[int], turn_id: int, castle_id: int) -> chess.Board:
    board = chess.Board.empty()
    for square, code in enumerate(piece_ids):
        if code == 0:
            continue
        color = chess.WHITE if code <= 6 else chess.BLACK
        piece_type = _PIECE_TYPES[(code - 1) % 6]
        board.set_piece_at(square, chess.Piece(piece_type, color))
    board.turn = chess.WHITE if turn_id == 0 else chess.BLACK
    rights = 0
    if castle_id & 1:
        rights |= chess.BB_H1
    if castle_id & 2:
        rights |= chess.BB_A1
    if castle_id & 4:
        rights |= chess.BB_H8
    if castle_id & 8:
        rights |= chess.BB_A8
    board.castling_rights = rights
    return board


def _scripted_projection(board: chess.Board, *, value_stm: float) -> tuple[float, list[str], list[float]]:
    legal = sorted(board.legal_moves, key=lambda m: m.uci())
    ucis = [m.uci() for m in legal]
    n = len(ucis)
    log_priors = [math.log(1.0 / n)] * n if n else []
    return value_stm, ucis, log_priors


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
            piece_ids = msg.batch_arrays["piece_ids"][-1]
            turn_id_field = msg.batch_arrays["turn_id"][-1]
            castle_id = msg.batch_arrays["castle_id"][-1]
            board = _board_from_state(piece_ids, turn_id_field, castle_id)
            value_stm, ucis, log_priors = _scripted_projection(board, value_stm=0.1)
            conn.send(
                RootEvalResponse(
                    turn_id=msg.turn_id,
                    value_stm=value_stm,
                    legal_ucis=ucis,
                    legal_log_priors=log_priors,
                )
            )
        elif isinstance(msg, WaveRequest):
            rows = []
            for row in msg.rows:
                board = _board_from_state(
                    row.board_state["piece_ids"],
                    row.board_state["turn_id"],
                    row.board_state["castle_id"],
                )
                rows.append(_scripted_projection(board, value_stm=0.2))
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
