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
import signal
from dataclasses import asdict, dataclass

import chess
import chess.engine
import cozy_chess as cc
import numpy as np

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
    server tensorizes it on receipt; Task 2).

    `server_prefix_len` (incremental-root-KV optimization; see
    `docs/superpowers/sdd/increm-report.md`) tracks how many tokens the
    SERVER's persisted per-worker prefix KV covers, as of this game's last
    root request -- `None` means "no root request sent yet for this game",
    the signal `_select_model_move` uses to send a FULL
    `RootEvalRequest.batch_arrays` instead of an incremental one. A fresh
    instance is constructed per game (`actor_worker._play_one_game`), so
    this starts `None` for every game with no explicit reset needed --
    "new game = fresh full root" falls out of that structurally.
    """

    def __init__(
        self, *, worker_id: int, move_vocab: MoveVocab, board_state_encoder: BoardStateEncoder
    ) -> None:
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder
        self._game_id = f"actor_worker_{worker_id}"
        self.server_prefix_len: int | None = None

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

    def build_incremental_tokens_for_current_position(
        self, board: chess.Board
    ) -> tuple[dict, int]:
        """Incremental-root-KV twin of `build_batch_for_current_position`:
        appends the same transient current-position token, but returns only
        the TAIL rows added since `self.server_prefix_len` (the server's
        persisted prefix length as of this game's PREVIOUS root request)
        instead of the whole sequence -- exactly the fields one
        `RootEvalRequest.incremental_tokens` step needs
        (`actor_server._service_incremental_root` feeds them straight into
        `hstu_model.HSTUChessModel.forward_decode`'s `new_token_batch`, one
        row at a time). The row count (k) is WHATEVER the actual length
        delta turns out to be -- normally 2 (this worker's own prior move
        settling into history + the opponent's reply) but never assumed;
        see `RootEvalRequest`'s own docstring.

        Returns `(incremental_arrays, new_total_len)` -- `new_total_len` is
        what the caller should record via `self.server_prefix_len =
        new_total_len` once the server has confirmed the extension (i.e.
        after a matching `RootEvalResponse` comes back), mirroring
        `build_batch_for_current_position`'s own append-then-pop discipline
        so `history`'s committed bookkeeping is never mutated by evaluating
        a position that hasn't actually been played yet.
        """
        if self.server_prefix_len is None:
            raise RuntimeError(
                "build_incremental_tokens_for_current_position called "
                "before any full root request was registered for this "
                "game -- use build_batch_for_current_position for the "
                "game's first root request instead."
            )
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)
        try:
            since = self.server_prefix_len
            new_total_len = len(self.seq_token_id)
            if new_total_len <= since:
                raise RuntimeError(
                    f"incremental root request has no new tokens: history "
                    f"length {new_total_len} <= server_prefix_len {since} "
                    "(protocol/bookkeeping bug)."
                )
            tokens = {
                "piece_ids": [list(row) for row in self.piece_ids[since:]],
                "seq_token_id": list(self.seq_token_id[since:]),
                "turn_id": list(self.turn_id[since:]),
                "castle_id": list(self.castle_id[since:]),
                "ep_file_id": list(self.ep_file_id[since:]),
                "halfmove_bucket_id": list(self.halfmove_bucket_id[since:]),
                "fullmove_bucket_id": list(self.fullmove_bucket_id[since:]),
                "prev_move_id": list(self.prev_move_id[since:]),
            }
            return tokens, new_total_len
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


def _legal_vocab_projection(
    cozy_board: "cc.Board", move_vocab: MoveVocab
) -> tuple[list[int], list["cc.Move"], list[str]]:
    """Torch-free worker-side mirror of
    `position_evaluator._project_legal_logits_cozy`'s move-mapping + UCI-sort
    semantics (profile-driven thin-down, see
    `docs/superpowers/sdd/thin-report.md`): the worker holds the real cozy
    board for every search-tree node (and can cheaply derive one for the
    root from its own live `chess.Board`), so it -- not the server -- runs
    movegen, derives each move's castling-normalized UCI
    (`cozy_bridge.cozy_move_to_uci`), keeps only vocab-mapped moves, and
    sorts the (uci, vocab_id, move) triples jointly by UCI. This is the
    exact same "drop unmapped, then UCI-sort" discipline
    `_project_legal_logits_cozy` applies, just computed before the request
    is sent instead of after a round trip, and never touching a tensor.

    Unlike `_project_legal_logits_cozy`, this never raises on an empty
    mapped set -- it mirrors `CachedPositionEvaluator.consume_decode_result`'s
    own `except RuntimeError: legal_moves, legal_ucis, log_priors = [], [], []`
    for decode-wave rows (empty result, no crash); `_select_model_move` below
    keeps its own explicit fail-fast raise for the root case, exactly as
    before.
    """
    legal_moves_all = list(cozy_board.generate_moves())
    ucis_all = [cozy_bridge.cozy_move_to_uci(cozy_board, move) for move in legal_moves_all]
    triples: list[tuple[str, int, "cc.Move"]] = []
    for move, uci in zip(legal_moves_all, ucis_all):
        vocab_id = move_vocab.token_to_id.get(uci)
        if vocab_id is not None:
            triples.append((uci, int(vocab_id), move))
    triples.sort(key=lambda t: t[0])
    legal_ucis = [t[0] for t in triples]
    legal_vocab_ids = [t[1] for t in triples]
    legal_moves = [t[2] for t in triples]
    return legal_vocab_ids, legal_moves, legal_ucis


def _log_softmax_f32(raw_logits: list[float]) -> list[float]:
    """Worker-side pure-Python (numpy) twin of
    `torch.log_softmax(tensor.float(), dim=0)` over the ~30 raw legal-move
    logits the server now returns unprojected (profile-driven thin-down --
    see `docs/superpowers/sdd/thin-report.md` for the fp32-parity evidence).

    Computed in float64 (numerically stable max-subtract + log-sum-exp, the
    same algebraic form ATen's float32 log_softmax kernel uses internally,
    just at higher precision throughout) and only rounded to float32 at the
    very end. `raw_logits` entries are Python floats produced by
    `torch.Tensor.tolist()` on a float32 tensor -- an exact, lossless
    float32->float64 widening, so the float64 computation here operates on
    the identical bit values torch's own kernel would. log-softmax is
    well-conditioned for a vector this small (~30 legal moves), so the
    float64-then-round-to-float32 result matches torch's native float32
    computation to within a few ULPs -- comfortably inside the 1e-6
    tolerance the fp32-exactness test suite requires (verified directly
    against `torch.log_softmax` in `tests/test_actor_worker.py`).
    """
    if not raw_logits:
        return []
    shifted = np.asarray(raw_logits, dtype=np.float64)
    shifted = shifted - shifted.max()
    logsumexp = np.log(np.exp(shifted).sum())
    return (shifted - logsumexp).astype(np.float32).tolist()


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

    Profile-driven thin-down (`docs/superpowers/sdd/thin-report.md`):
    `evaluate()` now computes each row's legal-move projection ITSELF
    (`_legal_vocab_projection`, off the real cozy board it already holds)
    before sending the `WaveRequest`, and stashes the resulting
    `(legal_moves, legal_ucis)` in `self._pending_legal`, keyed by node_id --
    the server never sees a board or a move string, only the vocab ids and,
    on the way back, raw logits. `WaveResponse` rows are matched back to
    their node via `self._pending_legal.pop(node_id)` rather than positional
    zip-with-boards, so a future response-reordering bug (there isn't one
    today -- `WaveResponse.rows` is documented request order) would raise a
    KeyError instead of silently mismatching a position's legal moves.
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
        self._pending_legal: dict[int, tuple[list["cc.Move"], list[str]]] = {}

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
        node_ids: list[int] = []
        for handle, cozy_board in batch:
            legal_vocab_ids, legal_moves, legal_ucis = _legal_vocab_projection(
                cozy_board, self._move_vocab
            )
            self._pending_legal[handle.node_id] = (legal_moves, legal_ucis)
            board_state = self._board_state_encoder.encode_cozy(cozy_board)
            rows.append(
                WaveRow(
                    node_id=handle.node_id,
                    parent_id=handle.parent_id,
                    prev_move_vocab_id=handle.move_vocab_id,
                    board_state=vars(board_state),
                    legal_vocab_ids=legal_vocab_ids,
                )
            )
            node_ids.append(handle.node_id)
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
        for node_id, (value_stm, legal_logits) in zip(node_ids, response.rows):
            legal_moves, legal_ucis = self._pending_legal.pop(node_id)
            results.append(
                search.PositionEval(
                    value_stm=float(value_stm),
                    legal_moves=legal_moves,
                    legal_ucis=legal_ucis,
                    legal_log_priors=_log_softmax_f32(list(legal_logits)),
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
        # popen_uci already succeeded (the subprocess is live) by the time
        # configure() runs; if configure() itself raises (a bad UCI option
        # value, engine protocol error, ...) that subprocess must still be
        # reaped here rather than left running -- quit-on-failure, then
        # re-raise unchanged (fail-fast preserved, nothing swallowed).
        try:
            engine.configure(options)
        except Exception:
            engine.quit()
            raise
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
    `RootEvalResponse` round trip, and `CachedPositionEvaluator` becomes
    `_WaveEvaluator`. Everything downstream -- `search.select_greedy`/
    `select_value_rerank`/`select_value_search_d2`/
    `select_value_search_halving` -- is called exactly as
    `_select_model_move` calls it, since `_WaveEvaluator` satisfies the same
    `search.PositionEvaluator` protocol `CachedPositionEvaluator` does.

    Incremental-root-KV optimization (`docs/superpowers/sdd/increm-report.md`):
    the `RootEvalRequest` this sends is FULL (`batch_arrays`) only for this
    game's first model turn (`history.server_prefix_len is None`); every
    later turn sends an INCREMENTAL request (`incremental_tokens`) carrying
    only the tokens committed since the previous one, and the server
    extends its own persisted prefix KV for this worker's game instead of
    re-forwarding the whole game so far.

    Profile-driven thin-down (`docs/superpowers/sdd/thin-report.md`): the
    vocab projection (movegen + UCI-sort + vocab-id lookup) is now computed
    HERE, worker-side, off this worker's own live `board` (converted to
    cozy once), exactly as `_WaveEvaluator.evaluate` does per node -- the
    server only ever sees the resulting `legal_vocab_ids` and gathers raw
    logits at them; `legal_log_priors` is then this worker's own
    `_log_softmax_f32` over those raw logits, not a server-computed field.
    """
    # NOTE: legal_moves here must stay python-chess chess.Move objects (the
    # root's `board` is a chess.Board, pushed via board.push(move) by
    # _play_one_game, and search.select_* at the root level is called with
    # board=<chess.Board> throughout) -- _legal_vocab_projection's own
    # `legal_moves` return is cozy cc.Move (used at the WAVE/decode-tree
    # level below, Stage 3's cozy-only search tree), so it is deliberately
    # discarded here in favor of rebuilding chess.Move from the (still
    # UCI-sorted) uci strings, exactly as the pre-thin-down code built them
    # from the server's response.legal_ucis.
    cozy_board = cozy_bridge.board_to_cozy(board)
    legal_vocab_ids, _cozy_legal_moves, legal_ucis = _legal_vocab_projection(
        cozy_board, move_vocab
    )
    legal_moves = [chess.Move.from_uci(uci) for uci in legal_ucis]

    # Incremental-root-KV optimization (docs/superpowers/sdd/increm-report.md):
    # this game's FIRST root request sends the full sequence
    # (history.server_prefix_len is None); every subsequent one sends only
    # the new tokens committed since the last root request, and the server
    # extends its own persisted per-worker prefix KV instead of re-forwarding
    # the whole game so far.
    common_request_fields = dict(
        worker_id=worker_id, turn_id=turn_id, legal_vocab_ids=legal_vocab_ids
    )
    if history.server_prefix_len is None:
        batch_arrays = history.build_batch_for_current_position(board)
        new_server_prefix_len = int(batch_arrays["total_tokens"])
        request = RootEvalRequest(**common_request_fields, batch_arrays=batch_arrays)
    else:
        incremental_tokens, new_server_prefix_len = (
            history.build_incremental_tokens_for_current_position(board)
        )
        request = RootEvalRequest(
            **common_request_fields,
            incremental_tokens=incremental_tokens,
            prefix_len_before=history.server_prefix_len,
        )
    conn.send(request)
    response = conn.recv()
    if not isinstance(response, RootEvalResponse) or response.turn_id != turn_id:
        raise RuntimeError(
            f"actor worker {worker_id}: expected RootEvalResponse(turn_id={turn_id}), "
            f"got {response!r}"
        )
    history.server_prefix_len = new_server_prefix_len

    total_legal_moves = len(list(board.legal_moves))
    legal_log_priors = _log_softmax_f32(list(response.legal_logits))
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


def _handle_sigterm(signum, frame) -> None:  # noqa: ARG001 - signal handler signature
    raise SystemExit(143)  # 128 + SIGTERM(15), conventional shell exit code


def _install_sigterm_handler() -> None:
    """Task 3's orchestrator supervises workers fail-fast: on any dead-
    worker/pipe-EOF/server exception it terminates every worker process --
    SIGTERM first, then SIGKILL for stragglers after a grace period (see
    `scripts/eval_vs_stockfish.py`'s `_terminate_worker_processes`). A bare
    SIGTERM's default disposition kills this process immediately, skipping
    `run_eval_worker`'s own `finally: engine.quit()` below and leaking that
    worker's live Stockfish subprocess (reparented to init/a subreaper once
    its parent -- this worker -- is gone).

    Installing a handler that raises `SystemExit` instead means the signal
    -- delivered between bytecode instructions, or interrupting a blocking
    syscall like `conn.recv()`/`engine.play()`'s I/O (PEP 475: a Python
    signal handler that raises skips the automatic EINTR retry) -- unwinds
    through whatever try/finally is currently on the stack exactly like any
    other exception, including `run_eval_worker`'s own `finally:
    engine.quit()`. So a SIGTERM-based termination still quits this
    worker's engine cleanly; only the orchestrator's final SIGKILL
    escalation (unblockable by design -- no handler can intercept it) leaks
    the Stockfish child, and only for a worker that didn't exit within the
    SIGTERM grace period. Documented, not solved, in
    `_terminate_worker_processes`'s own docstring.

    Guarded for the (non-spawn) case where `run_eval_worker` is called
    directly, in-process, off the main thread -- Task 1's own tests do this
    (`tests/test_actor_worker.py`'s `_run_two_short_games`, worker on the
    test's main thread with a background-thread fake server, so this
    actually lands on the main thread there too, but `signal.signal` is a
    hard `ValueError` off it) -- `signal.signal` only works on the main
    thread of the main interpreter, and there is no real OS process signal
    to intercept in that scenario anyway, so skipping silently is correct.
    """
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except ValueError:
        pass


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
    exception/EOF). `engine.quit()` runs in a `finally` covering the whole
    game loop, so a real Stockfish subprocess is never leaked by anything
    raised while playing; `_build_engine` additionally quits-then-re-raises
    if `engine.configure(...)` itself fails right after a successful
    `popen_uci` (a window this `finally` can't reach, since it starts only
    once `engine` already exists). `_build_engine_limit` runs BEFORE
    `_build_engine` for the same reason: it can't leak anything since it
    never spawns a subprocess, so validating/building the limit first means
    a bad `stockfish_limit` dict never reaches "subprocess already spawned,
    then raise" territory at all.
    """
    _install_sigterm_handler()
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

    rng = random.Random(seed + worker_id)
    turn_counter = itertools.count()

    engine_config = worker_config["engine"]
    # Limit first, engine second: _build_engine_limit is pure (no
    # subprocess), so if it raises (a malformed "stockfish_limit" dict)
    # there is nothing live yet to leak. Building the engine only after the
    # limit is known-good, immediately before the try/finally below, means
    # the ONLY thing the finally's engine.quit() ever needs to cover is
    # "the engine object already exists" -- no window between engine
    # construction and try-entry where an unrelated raise could skip quit().
    engine_limit = _build_engine_limit(engine_config.get("stockfish_limit"))
    engine = _build_engine(engine_config)
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
