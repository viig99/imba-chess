"""Torch-free eval actor worker: the per-game loop, ported from
`scripts/eval_vs_stockfish.py`'s `_play_game` coroutine minus the
`BatchScheduler` `WorkRequest` yields it used for intra-process cross-game
batching. Each worker here handles exactly ONE game at a time and talks to
the GPU inference server (Task 2) synchronously over a `multiprocessing.
Pipe()` connection: send a `RootEvalRequest`/`WaveRequest`, block on
`conn.recv()` for the matching response. Cross-worker batching happens
server-side (worker-id polling order), not here -- so there is no
generator/coroutine machinery to port at all for the outer game loop, only
for the search-tree evaluator shim (`_WaveEvaluator`, below).

MUST STAY TORCH-FREE: `run_eval_worker` is the entry point spawned into a
separate process (`multiprocessing.get_context("spawn")`, Task 3) that never
loads torch. `tests/test_actor_worker.py::test_actor_worker_and_protocol_are_torch_free`
is a permanent regression test for this via a `sys.meta_path` import-blocking
hook run in a subprocess (a same-process hook can't be trusted: pytest
itself has already imported torch by the time any test runs, and `sys.modules`
is checked before any finder). Do not import
`imba_chess.eval.position_evaluator`, `imba_chess.eval.metrics`, `torch`, or
anything that imports them, from this module. (`imba_chess/eval/__init__.py`
and `imba_chess/data/__init__.py` were made lazy -- PEP 562 module
`__getattr__` -- specifically so that importing this module's dependencies,
`imba_chess.data.board_state`/`move_vocab`/`models` and
`imba_chess.eval.search`/`cozy_bridge`, does not transitively import torch
merely by running those packages' `__init__`.)
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass

import chess
import chess.engine
import cozy_chess as cc

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.event_builder import BOS_TOKEN_ID, EVENT_TOKEN_ID, TARGET_IGNORE_INDEX
from imba_chess.data.models import BoardTokenConfig
from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab
from imba_chess.eval import cozy_bridge, search
from imba_chess.eval.actor_protocol import (
    GameDone,
    RootEvalRequest,
    RootEvalResponse,
    WaveRequest,
    WaveResponse,
    WaveRow,
    WorkerFinished,
)


class _PlainSequenceHistory:
    """Torch-free twin of `position_evaluator._SequenceHistory`: identical
    incremental BOS+event token bookkeeping, but every tensor field of
    `_build_single_batch()` stays a plain Python list/int here -- this dict
    IS `RootEvalRequest.batch_arrays`, sent across the pipe as-is (the
    server tensorizes it on receipt; Task 2)."""

    def __init__(
        self, *, worker_id: int, move_vocab: MoveVocab, board_state_encoder: BoardStateEncoder
    ) -> None:
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder
        self._game_id = f"actor_worker_{worker_id}"

        self.seq_token_id: list[int] = [BOS_TOKEN_ID]
        self.piece_ids: list[list[int]] = [[0] * 64]
        self.turn_id: list[int] = [0]
        self.castle_id: list[int] = [0]
        self.ep_file_id: list[int] = [0]
        self.halfmove_bucket_id: list[int] = [0]
        self.fullmove_bucket_id: list[int] = [0]
        self.prev_move_id: list[int] = [self._move_vocab.start_id]
        self.target_move_id: list[int] = [TARGET_IGNORE_INDEX]
        self.played_by_elo: list[int] = [0]

        self._prev_move_id_for_next_token = self._move_vocab.start_id

    def append_observed_position(self, board: chess.Board) -> None:
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)

    def record_played_move(self, move_uci: str) -> None:
        self._prev_move_id_for_next_token = int(self._move_vocab.encode(move_uci))

    def _append_from_state(self, state) -> None:
        self.seq_token_id.append(EVENT_TOKEN_ID)
        self.piece_ids.append(list(state.piece_ids))
        self.turn_id.append(int(state.turn_id))
        self.castle_id.append(int(state.castle_id))
        self.ep_file_id.append(int(state.ep_file_id))
        self.halfmove_bucket_id.append(int(state.halfmove_bucket_id))
        self.fullmove_bucket_id.append(int(state.fullmove_bucket_id))
        self.prev_move_id.append(int(self._prev_move_id_for_next_token))
        self.target_move_id.append(TARGET_IGNORE_INDEX)
        self.played_by_elo.append(0)

    def _pop_last(self) -> None:
        self.seq_token_id.pop()
        self.piece_ids.pop()
        self.turn_id.pop()
        self.castle_id.pop()
        self.ep_file_id.pop()
        self.halfmove_bucket_id.pop()
        self.fullmove_bucket_id.pop()
        self.prev_move_id.pop()
        self.target_move_id.pop()
        self.played_by_elo.pop()

    def _build_single_batch(self) -> dict:
        total_tokens = len(self.seq_token_id)
        return {
            "game_id": [self._game_id],
            "game_result_white": [0],
            "num_games": 1,
            "total_tokens": total_tokens,
            "seq_lens": [total_tokens],
            "seq_offsets": [0, total_tokens],
            "piece_ids": [list(row) for row in self.piece_ids],
            "seq_token_id": list(self.seq_token_id),
            "turn_id": list(self.turn_id),
            "castle_id": list(self.castle_id),
            "ep_file_id": list(self.ep_file_id),
            "halfmove_bucket_id": list(self.halfmove_bucket_id),
            "fullmove_bucket_id": list(self.fullmove_bucket_id),
            "prev_move_id": list(self.prev_move_id),
            "target_move_id": list(self.target_move_id),
            "played_by_elo": list(self.played_by_elo),
        }

    def build_batch_for_current_position(self, board: chess.Board) -> dict:
        # Add transient current-position token for next-move prediction only.
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)
        try:
            return self._build_single_batch()
        finally:
            self._pop_last()


class _WorkerSearchNode:
    """Search-tree node handle minted worker-side: flattened twin of
    `position_evaluator._CachedNode` (parent link + the move that led here),
    carrying plain ints instead of a Python object reference + KV tensor, so
    it can be reasoned about purely from protocol responses. `move_vocab_id`
    is stashed here (rather than re-derived later) so `_WaveEvaluator.
    evaluate` can fill `WaveRow.prev_move_vocab_id` without re-encoding.
    """

    __slots__ = ("node_id", "parent_id", "move_vocab_id")

    def __init__(self, node_id: int, parent_id: int | None, move_vocab_id: int) -> None:
        self.node_id = node_id
        self.parent_id = parent_id
        self.move_vocab_id = move_vocab_id


def _match_cozy_moves(cozy_board: "cc.Board", legal_ucis: list[str]) -> list["cc.Move"]:
    """Reverse of `position_evaluator._project_legal_logits_cozy`'s UCI
    derivation: the server sends back standard (castling-normalized) UCI
    strings, not `cc.Move` objects -- the wire protocol is chess-object-free
    -- so reconstructing `search.PositionEval.legal_moves` means matching
    each returned UCI against THIS position's own cozy movegen. One dict per
    node (cheap; every search-wave position is distinct anyway), exactly as
    the task brief specifies.
    """
    uci_to_move = {
        cozy_bridge.cozy_move_to_uci(cozy_board, move): move
        for move in cozy_board.generate_moves()
    }
    try:
        return [uci_to_move[uci] for uci in legal_ucis]
    except KeyError as exc:
        raise RuntimeError(
            f"server returned a UCI not legal on this position: {exc}"
        ) from exc


class _WaveEvaluator:
    """`search.PositionEvaluator` implementation that speaks the wire
    protocol instead of calling a model: `evaluate()` packages its batch
    into one `WaveRequest`, blocks for the matching `WaveResponse`, and
    rebuilds `search.PositionEval` rows from it; `extend()` mints child node
    handles the way `CachedPositionEvaluator.extend`/`_CachedNode` do, just
    carrying `(node_id, parent_id, move_vocab_id)` instead of a KV tensor.

    Node ids are scoped to one turn: a fresh instance is built per model
    turn (mirrors `CachedPositionEvaluator` being "constructed fresh each
    model turn"), so ids restart at 0 every turn -- the server's
    per-(worker_id, turn_id) KV tree can be released wholesale once the turn
    is done (Task 2's `release_turn`).
    """

    def __init__(
        self,
        *,
        conn,
        worker_id: int,
        turn_id: int,
        move_vocab: MoveVocab,
        board_state_encoder: BoardStateEncoder,
    ) -> None:
        self._conn = conn
        self._worker_id = worker_id
        self._turn_id = turn_id
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder
        self._next_node_id = 0

    def extend(self, handle, move_uci: str) -> "_WorkerSearchNode":
        """`handle` is opaque to the caller (search.py); `None` (or
        anything that isn't a `_WorkerSearchNode`, matching
        `_CachedNode.extend`'s own `isinstance` guard) means "parent is the
        turn's root prefix"."""
        parent = handle if isinstance(handle, _WorkerSearchNode) else None
        node_id = self._next_node_id
        self._next_node_id += 1
        return _WorkerSearchNode(
            node_id=node_id,
            parent_id=None if parent is None else parent.node_id,
            move_vocab_id=int(self._move_vocab.encode(move_uci)),
        )

    def evaluate(
        self, batch: list[tuple["_WorkerSearchNode", "cc.Board"]]
    ) -> list["search.PositionEval"]:
        if not batch:
            return []
        rows: list[WaveRow] = []
        boards: list["cc.Board"] = []
        for handle, cozy_board in batch:
            board_state = self._board_state_encoder.encode_cozy(cozy_board)
            rows.append(
                WaveRow(
                    node_id=handle.node_id,
                    parent_id=handle.parent_id,
                    prev_move_vocab_id=handle.move_vocab_id,
                    board_state=vars(board_state),
                )
            )
            boards.append(cozy_board)
        self._conn.send(
            WaveRequest(worker_id=self._worker_id, turn_id=self._turn_id, rows=rows)
        )
        response = self._conn.recv()
        if not isinstance(response, WaveResponse):
            raise RuntimeError(
                f"actor worker {self._worker_id}: expected WaveResponse, got {response!r}"
            )
        if len(response.rows) != len(batch):
            raise RuntimeError(
                f"actor worker {self._worker_id}: WaveResponse row count "
                f"{len(response.rows)} != request row count {len(batch)}"
            )
        results: list[search.PositionEval] = []
        for (value_stm, legal_ucis, legal_log_priors), cozy_board in zip(
            response.rows, boards
        ):
            legal_moves = _match_cozy_moves(cozy_board, legal_ucis)
            results.append(
                search.PositionEval(
                    value_stm=float(value_stm),
                    legal_moves=legal_moves,
                    legal_ucis=list(legal_ucis),
                    legal_log_priors=list(legal_log_priors),
                )
            )
        return results


@dataclass
class _EvalSummaryFragment:
    """Torch-free, self-contained twin of `scripts/eval_vs_stockfish.py`'s
    `EvalSummary` -- same field set, duplicated rather than imported (that
    script imports torch at module scope; importing it here would break
    this module's torch-free contract, same rationale as
    `_main_with_hard_exit_on_crash`'s own precedent for duplicating a small
    amount of code across those two scripts). One instance per finished
    game (`games == 1`); `GameDone.summary_fragment` is `dataclasses.
    asdict()` of it, foldable into a running total the same way
    `scripts/eval_vs_stockfish.py`'s `_accumulate_summary` folds one game's
    fragment today.
    """

    games: int = 0
    completed_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_as_white: int = 0
    games_as_black: int = 0
    wins_as_white: int = 0
    losses_as_white: int = 0
    draws_as_white: int = 0
    wins_as_black: int = 0
    losses_as_black: int = 0
    draws_as_black: int = 0
    incomplete_games: int = 0
    total_plies: int = 0
    model_turns: int = 0
    legal_moves_total: int = 0
    legal_moves_mapped_total: int = 0
    turns_with_no_vocab_legal_move: int = 0


def _update_summary_fragment(
    summary: _EvalSummaryFragment,
    *,
    result: str,
    model_color: bool,
    completed: bool,
    plies: int,
) -> None:
    """Verbatim port of `scripts/eval_vs_stockfish.py`'s `_update_summary`
    onto `_EvalSummaryFragment` (`model_color` is `chess.WHITE`/`chess.
    BLACK`, i.e. `bool`, exactly as there)."""
    summary.games += 1
    summary.total_plies += int(plies)
    if model_color == chess.WHITE:
        summary.games_as_white += 1
    else:
        summary.games_as_black += 1

    if not completed:
        summary.incomplete_games += 1
        return

    summary.completed_games += 1
    if result == "1/2-1/2":
        summary.draws += 1
        if model_color == chess.WHITE:
            summary.draws_as_white += 1
        else:
            summary.draws_as_black += 1
        return

    model_won = (model_color == chess.WHITE and result == "1-0") or (
        model_color == chess.BLACK and result == "0-1"
    )
    if model_won:
        summary.wins += 1
        if model_color == chess.WHITE:
            summary.wins_as_white += 1
        else:
            summary.wins_as_black += 1
    else:
        summary.losses += 1
        if model_color == chess.WHITE:
            summary.losses_as_white += 1
        else:
            summary.losses_as_black += 1


def _build_engine_limit(limit_config: dict | None) -> "chess.engine.Limit":
    """Torch-free, dict-driven twin of `scripts/eval_vs_stockfish.py`'s
    `_build_engine_limit` (which reads argparse `Namespace` fields --
    `worker_config` is plain data instead, so the analogous knobs travel as
    a plain `{"time": ...}` / `{"nodes": ...}` / `{"depth": ...}` dict)."""
    limit_config = limit_config or {}
    kwargs: dict[str, float | int] = {}
    if "time" in limit_config:
        kwargs["time"] = float(limit_config["time"])
    if "nodes" in limit_config:
        kwargs["nodes"] = int(limit_config["nodes"])
    if "depth" in limit_config:
        kwargs["depth"] = int(limit_config["depth"])
    if not kwargs:
        kwargs["time"] = 0.05
    return chess.engine.Limit(**kwargs)


def _build_engine(engine_config: dict):
    """Builds this worker's own Stockfish `SimpleEngine` -- OR, test-only,
    calls a caller-supplied zero-arg factory instead.

    Design choice (spawn-compat; task brief asked this be decided and
    documented here): production workers are spawned via `multiprocessing.
    get_context("spawn")` (Task 3), which pickles `worker_config` whole into
    the child process, so anything it carries must be picklable.
    `engine_config["stockfish_path"]` (a plain string) is; a live engine
    object or a closure is not. So the fake-engine path is a separate,
    explicitly test-only key, `engine_config["fake_engine_factory"]` -- a
    zero-arg callable returning an object with `.play(board, limit)` (an
    object with a `.move` attribute) and `.quit()`. This works unmodified
    for Task 1's tests, which call `run_eval_worker` directly in-process (no
    real spawn, so a local closure/lambda factory is fine); a caller that
    needs a fake engine under REAL spawn (e.g. a future spawn-based
    integration test) must supply a picklable module-level function
    reference instead of a closure. Production `worker_config` never sets
    this key at all -- it always supplies `stockfish_path`.
    """
    factory = engine_config.get("fake_engine_factory")
    if factory is not None:
        return factory()
    stockfish_path = engine_config["stockfish_path"]
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
    options = engine_config.get("stockfish_options")
    if options:
        engine.configure(options)
    return engine


def _select_model_move(
    *,
    conn,
    worker_id: int,
    turn_id: int,
    board: chess.Board,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    history: _PlainSequenceHistory,
    policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    halving_config: "search.HalvingConfig | None",
) -> tuple[chess.Move, dict]:
    """Torch-free, protocol-driven twin of `scripts/eval_vs_stockfish.py`'s
    `_select_model_move`: the model forward becomes one `RootEvalRequest`/
    `RootEvalResponse` round trip (the server does the forward AND the
    vocab projection -- it owns the logits -- so `legal_ucis`/
    `legal_log_priors` come back already UCI-sorted), and
    `CachedPositionEvaluator` becomes `_WaveEvaluator`. Everything
    downstream -- `search.select_greedy`/`select_value_rerank`/
    `select_value_search_d2`/`select_value_search_halving` -- is called
    exactly as `_select_model_move` calls it, since `_WaveEvaluator`
    satisfies the same `search.PositionEvaluator` protocol
    `CachedPositionEvaluator` does.
    """
    batch_arrays = history.build_batch_for_current_position(board)
    conn.send(
        RootEvalRequest(worker_id=worker_id, turn_id=turn_id, batch_arrays=batch_arrays)
    )
    response = conn.recv()
    if not isinstance(response, RootEvalResponse) or response.turn_id != turn_id:
        raise RuntimeError(
            f"actor worker {worker_id}: expected RootEvalResponse(turn_id={turn_id}), "
            f"got {response!r}"
        )

    total_legal_moves = len(list(board.legal_moves))
    legal_moves = [chess.Move.from_uci(uci) for uci in response.legal_ucis]
    legal_log_priors = list(response.legal_log_priors)
    mapped_legal_moves = len(legal_moves)
    if mapped_legal_moves == 0:
        raise RuntimeError(
            f"actor worker {worker_id}: no legal moves mapped to vocab ids for "
            f"current board (turn={turn_id}, total_legal={total_legal_moves})."
        )

    if policy == "greedy":
        chosen_index = search.select_greedy(legal_log_priors)
    elif policy in ("value_rerank", "value_search_d2", "value_search_halving"):
        evaluator = _WaveEvaluator(
            conn=conn,
            worker_id=worker_id,
            turn_id=turn_id,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
        )
        if policy == "value_rerank":
            chosen_index, _rows = search.select_value_rerank(
                evaluator=evaluator,
                root_handle=None,
                board=board,
                legal_moves=legal_moves,
                legal_log_priors=legal_log_priors,
                top_k=value_rerank_top_k,
                lam=value_rerank_lambda,
            )
        elif policy == "value_search_d2":
            chosen_index, _rows = search.select_value_search_d2(
                evaluator=evaluator,
                root_handle=None,
                board=board,
                legal_moves=legal_moves,
                legal_log_priors=legal_log_priors,
                top_k=value_rerank_top_k,
                lam=value_rerank_lambda,
            )
        else:
            if halving_config is None:
                raise ValueError("policy=value_search_halving requires halving_config")
            chosen_index, _rows = search.select_value_search_halving(
                evaluator=evaluator,
                root_handle=None,
                board=board,
                legal_moves=legal_moves,
                legal_log_priors=legal_log_priors,
                config=halving_config,
            )
    else:
        raise ValueError(f"Unknown model move policy: {policy}")

    debug_info = {
        "total_legal_moves": total_legal_moves,
        "mapped_legal_moves": mapped_legal_moves,
    }
    return legal_moves[chosen_index], debug_info


def _play_one_game(
    *,
    conn,
    worker_id: int,
    game_idx: int,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    engine,
    engine_limit: "chess.engine.Limit",
    policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    halving_config: "search.HalvingConfig | None",
    opening_random_plies: int,
    max_plies: int,
    rng: random.Random,
    turn_counter: "itertools.count",
) -> _EvalSummaryFragment:
    """One game's synchronous core: a direct (non-generator) port of
    `scripts/eval_vs_stockfish.py`'s `_play_game`, minus the `WorkRequest`
    yields -- each worker plays exactly one game at a time (cross-worker
    batching happens server-side, not via an in-process scheduler), so
    there is nothing to yield through: model turns block on
    `_select_model_move`'s pipe round trip(s) directly, and SF turns call
    this worker's own `engine.play(...)` directly (no thread-pool executor
    needed -- SF overlap across DIFFERENT workers is automatic, each in its
    own OS process). Debug tracing / game-saving (`_play_game`'s
    `debug_trace_games`/`save_games_dir` hooks) are out of Task 1's scope:
    not part of the produced interfaces or required tests.
    """
    summary = _EvalSummaryFragment()
    board = chess.Board()
    history = _PlainSequenceHistory(
        worker_id=worker_id, move_vocab=move_vocab, board_state_encoder=board_state_encoder
    )
    model_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
    completed = True
    plies = 0

    while not board.is_game_over(claim_draw=True):
        if plies >= max_plies:
            completed = False
            break
        if plies < opening_random_plies:
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
        elif board.turn == model_color:
            turn_id = next(turn_counter)
            move, debug_info = _select_model_move(
                conn=conn,
                worker_id=worker_id,
                turn_id=turn_id,
                board=board,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                history=history,
                policy=policy,
                value_rerank_top_k=value_rerank_top_k,
                value_rerank_lambda=value_rerank_lambda,
                halving_config=halving_config,
            )
            summary.model_turns += 1
            summary.legal_moves_total += int(debug_info["total_legal_moves"])
            summary.legal_moves_mapped_total += int(debug_info["mapped_legal_moves"])
            if int(debug_info["mapped_legal_moves"]) == 0:
                summary.turns_with_no_vocab_legal_move += 1
        else:
            result = engine.play(board, engine_limit)
            if result.move is None:
                raise RuntimeError("Stockfish returned no move.")
            move = result.move

        history.append_observed_position(board)
        history.record_played_move(move.uci())
        board.push(move)
        plies += 1

    result = board.result(claim_draw=True) if completed else "*"
    _update_summary_fragment(
        summary, result=result, model_color=model_color, completed=completed, plies=plies
    )
    return summary


def run_eval_worker(conn, worker_config: dict) -> None:
    """Actor worker entry point (Task 1): plays `worker_config["game_indices"]`
    sequentially, one at a time, talking to the GPU inference server over
    `conn` (a `multiprocessing.Pipe()` connection, or any duck-typed object
    exposing `.send`/`.recv`). Sends one `GameDone` per finished game (in
    `game_indices` order) and one `WorkerFinished` at the very end.

    `worker_config` (plain dict, picklable -- built by the orchestrator,
    Task 3):
      - "worker_id": int
      - "game_indices": list[int] -- this worker's assigned games, in play order
      - "seed": int -- combined with worker_id to seed this worker's opening-ply rng
      - "max_plies": int
      - "opening_random_plies": int (default 0)
      - "model_move_policy": "greedy" | "value_rerank" | "value_search_d2" | "value_search_halving"
      - "value_rerank_top_k": int (default 1)
      - "value_rerank_lambda": float (default 0.0)
      - "halving_config": dict of `search.HalvingConfig` fields, or None
        (required iff model_move_policy == "value_search_halving")
      - "vocab_path": str
      - "vocab_include_unk": bool (default False)
      - "board_state_config": dict of `data.models.BoardTokenConfig` fields, or None
      - "engine": dict, see `_build_engine`/`_build_engine_limit`:
          - "stockfish_path": str (production)
          - "stockfish_options": dict (UCI options, optional)
          - "fake_engine_factory": Callable[[], Any] (test-only, see `_build_engine`)
          - "stockfish_limit": dict, see `_build_engine_limit`

    Fail-fast (repo policy): no exception here is caught and swallowed. A
    dead pipe, a malformed response, an engine crash -- all propagate to the
    caller (Task 3's orchestrator kills the whole run on any worker
    exception/EOF). The only `finally` is `engine.quit()`, so a real
    Stockfish subprocess is never leaked even when a game raises.
    """
    worker_id = int(worker_config["worker_id"])
    game_indices = list(worker_config["game_indices"])
    seed = int(worker_config.get("seed", 0))
    max_plies = int(worker_config["max_plies"])
    opening_random_plies = int(worker_config.get("opening_random_plies", 0))
    policy = str(worker_config["model_move_policy"])
    value_rerank_top_k = int(worker_config.get("value_rerank_top_k", 1))
    value_rerank_lambda = float(worker_config.get("value_rerank_lambda", 0.0))
    halving_config_dict = worker_config.get("halving_config")
    halving_config = (
        search.HalvingConfig(**halving_config_dict) if halving_config_dict is not None else None
    )

    move_vocab = load_or_create_static_move_vocab(
        path=worker_config["vocab_path"],
        include_unk=bool(worker_config.get("vocab_include_unk", False)),
    )
    board_state_config_dict = worker_config.get("board_state_config")
    board_state_encoder = BoardStateEncoder(
        BoardTokenConfig(**board_state_config_dict)
        if board_state_config_dict
        else BoardTokenConfig()
    )

    engine_config = worker_config["engine"]
    engine = _build_engine(engine_config)
    engine_limit = _build_engine_limit(engine_config.get("stockfish_limit"))
    rng = random.Random(seed + worker_id)
    turn_counter = itertools.count()

    try:
        for game_idx in game_indices:
            summary = _play_one_game(
                conn=conn,
                worker_id=worker_id,
                game_idx=game_idx,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                engine=engine,
                engine_limit=engine_limit,
                policy=policy,
                value_rerank_top_k=value_rerank_top_k,
                value_rerank_lambda=value_rerank_lambda,
                halving_config=halving_config,
                opening_random_plies=opening_random_plies,
                max_plies=max_plies,
                rng=rng,
                turn_counter=turn_counter,
            )
            conn.send(
                GameDone(
                    worker_id=worker_id,
                    game_idx=game_idx,
                    summary_fragment=asdict(summary),
                )
            )
    finally:
        engine.quit()

    conn.send(WorkerFinished(worker_id=worker_id))
