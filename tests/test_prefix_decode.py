from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imba_chess.model import create_batch_block_mask
from imba_chess.model.hstu_attention import SequentialTransductionUnitJagged
from imba_chess.model.position_embedding import PositionEmbedding

ATOL = 1e-5
RTOL = 1e-5


def _layer() -> SequentialTransductionUnitJagged:
    torch.manual_seed(0)
    return SequentialTransductionUnitJagged(
        embedding_dim=32,
        linear_hidden_dim=8,
        attention_dim=8,
        dropout_ratio=0.0,
        num_heads=2,
        max_seq_len=64,
    ).eval()


def _full_forward(layer, x):
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, x.size(0)]),
        total_tokens=x.size(0),
        device=x.device,
    )
    return layer(x=x, block_mask=block_mask)


def test_forward_return_kv_output_unchanged():
    layer = _layer()
    x = torch.randn(10, 32)
    full = _full_forward(layer, x)
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, 10]), total_tokens=10, device=x.device
    )
    out, (k, v) = layer(x=x, block_mask=block_mask, return_kv=True)
    torch.testing.assert_close(out, full, atol=ATOL, rtol=RTOL)
    assert k.shape == (2, 10, 8)  # [H, S, attention_dim]
    assert v.shape == (2, 10, 8)  # [H, S, linear_hidden_dim]


def test_layer_decode_matches_full_forward_token_by_token():
    layer = _layer()
    S, T = 13, 9  # prefill 9 tokens, decode tokens 9..12 sequentially
    x = torch.randn(S, 32)
    full = _full_forward(layer, x)

    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, T]), total_tokens=T, device=x.device
    )
    out_prefix, (prefix_k, prefix_v) = layer(
        x=x[:T], block_mask=block_mask, return_kv=True
    )
    torch.testing.assert_close(out_prefix, full[:T], atol=ATOL, rtol=RTOL)

    suffix_k_parts: list[torch.Tensor] = []
    suffix_v_parts: list[torch.Tensor] = []
    for i in range(T, S):
        if suffix_k_parts:
            suffix_k = torch.cat(suffix_k_parts, dim=2)  # [1, H, s, d]
            suffix_v = torch.cat(suffix_v_parts, dim=2)
            s = suffix_k.size(2)
            suffix_positions = torch.arange(T, T + s).view(1, s)
            suffix_mask = torch.ones(1, s, dtype=torch.bool)
        else:
            suffix_k = suffix_v = suffix_positions = suffix_mask = None
        x_out, k_new, v_new = layer.forward_decode(
            x[i : i + 1],
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            q_positions=torch.tensor([i]),
            suffix_k=suffix_k,
            suffix_v=suffix_v,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
        )
        torch.testing.assert_close(x_out.squeeze(0), full[i], atol=ATOL, rtol=RTOL)
        suffix_k_parts.append(k_new)
        suffix_v_parts.append(v_new)


def test_layer_decode_batched_wave_with_mixed_suffix_lengths():
    layer = _layer()
    T = 7
    prefix = torch.randn(T, 32)
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, T]), total_tokens=T, device=prefix.device
    )
    _, (prefix_k, prefix_v) = layer(x=prefix, block_mask=block_mask, return_kv=True)

    # Node A: depth 0 (no suffix). Node B: depth 1 (one ancestor token).
    tok_a = torch.randn(1, 32)
    tok_b_parent = torch.randn(1, 32)
    tok_b = torch.randn(1, 32)

    # References via full forwards over explicit sequences.
    full_a = _full_forward(layer, torch.cat([prefix, tok_a]))[T]
    full_b = _full_forward(layer, torch.cat([prefix, tok_b_parent, tok_b]))[T + 1]

    # Evaluate B's parent first to obtain its (k, v).
    _, kp, vp = layer.forward_decode(
        tok_b_parent,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        q_positions=torch.tensor([T]),
    )

    # One wave containing A (depth 0, padded suffix) and B (depth 1).
    x_new = torch.cat([tok_a, tok_b])  # [2, 32]
    suffix_k = torch.cat([torch.zeros_like(kp), kp])  # [2, H, 1, d]
    suffix_v = torch.cat([torch.zeros_like(vp), vp])
    suffix_positions = torch.tensor([[0], [T]])
    suffix_mask = torch.tensor([[False], [True]])
    x_out, _, _ = layer.forward_decode(
        x_new,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        q_positions=torch.tensor([T, T + 1]),
        suffix_k=suffix_k,
        suffix_v=suffix_v,
        suffix_positions=suffix_positions,
        suffix_mask=suffix_mask,
    )
    torch.testing.assert_close(x_out[0], full_a, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(x_out[1], full_b, atol=ATOL, rtol=RTOL)


def test_position_embedding_at_positions_matches_forward():
    torch.manual_seed(1)
    pe = PositionEmbedding(max_seq_len=16, embedding_dim=8, dropout_rate=0.0).eval()
    content = torch.randn(5, 8)
    offsets = torch.tensor([0, 5])
    full = pe(content, offsets)
    picked = pe.at_positions(content, torch.arange(5))
    torch.testing.assert_close(picked, full, atol=ATOL, rtol=RTOL)
    # Clamp behavior matches forward's clamp.
    over = pe.at_positions(content[:1], torch.tensor([99]))
    ref = pe.at_positions(content[:1], torch.tensor([15]))
    torch.testing.assert_close(over, ref, atol=ATOL, rtol=RTOL)


from imba_chess.config import ModelConfig
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


def _tiny_model(vocab_size: int = 32) -> HSTUChessModel:
    torch.manual_seed(2)
    config = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=8,
            attention_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            max_position_embeddings=64,
            enable_value_head=True,
        ),
        move_vocab_size=vocab_size,
    )
    return HSTUChessModel(config).eval()


def _random_token_ids(n: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "piece_ids": torch.randint(0, 13, (n, 64), generator=g),
        "seq_token_id": torch.randint(0, 2, (n,), generator=g),
        "turn_id": torch.randint(0, 2, (n,), generator=g),
        "castle_id": torch.randint(0, 16, (n,), generator=g),
        "ep_file_id": torch.randint(0, 9, (n,), generator=g),
        "halfmove_bucket_id": torch.randint(0, 50, (n,), generator=g),
        "fullmove_bucket_id": torch.randint(0, 100, (n,), generator=g),
        "prev_move_id": torch.randint(0, 32, (n,), generator=g),
    }


def _full_batch(token_ids: dict[str, torch.Tensor]) -> dict:
    n = token_ids["piece_ids"].size(0)
    batch = dict(token_ids)
    batch.update(
        {
            "total_tokens": n,
            "seq_offsets": torch.tensor([0, n]),
            "target_move_id": torch.full((n,), -100, dtype=torch.long),
        }
    )
    return batch


def test_model_decode_matches_full_forward_over_depths():
    model = _tiny_model()
    T, max_depth = 9, 4
    ids = _random_token_ids(T + max_depth, seed=7)
    prefix_ids = {key: value[:T] for key, value in ids.items()}

    with torch.no_grad():
        full = model(_full_batch(ids), return_loss=False)
        prefill = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)

    prefix_kv = prefill["kv_caches"]
    suffix_kv = None
    suffix_positions = suffix_mask = None
    for depth in range(max_depth):
        i = T + depth
        step_ids = {key: value[i : i + 1] for key, value in ids.items()}
        with torch.no_grad():
            out = model.forward_decode(
                new_token_batch=step_ids,
                positions=torch.tensor([i]),
                prefix_kv=prefix_kv,
                suffix_kv=suffix_kv,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
        torch.testing.assert_close(
            out["logits"].squeeze(0), full["logits"][i], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out["value_logits"].squeeze(0),
            full["value_logits"][i],
            atol=ATOL,
            rtol=RTOL,
        )
        # Grow the suffix cache with this token's per-layer (k, v).
        if suffix_kv is None:
            suffix_kv = [(k, v) for k, v in out["kv"]]
        else:
            suffix_kv = [
                (torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2))
                for (sk, sv), (k, v) in zip(suffix_kv, out["kv"])
            ]
        s = suffix_kv[0][0].size(2)
        suffix_positions = torch.arange(T, T + s).view(1, s)
        suffix_mask = torch.ones(1, s, dtype=torch.bool)


def test_model_decode_mixed_depth_wave():
    model = _tiny_model()
    T = 8
    ids = _random_token_ids(T + 3, seed=11)  # prefix + [a, b_parent, b]
    prefix_ids = {key: value[:T] for key, value in ids.items()}
    tok_a = {key: value[T : T + 1] for key, value in ids.items()}
    tok_bp = {key: value[T + 1 : T + 2] for key, value in ids.items()}
    tok_b = {key: value[T + 2 : T + 3] for key, value in ids.items()}

    seq_a = {key: torch.cat([prefix_ids[key], tok_a[key]]) for key in ids}
    seq_b = {
        key: torch.cat([prefix_ids[key], tok_bp[key], tok_b[key]]) for key in ids
    }
    with torch.no_grad():
        full_a = model(_full_batch(seq_a), return_loss=False)
        full_b = model(_full_batch(seq_b), return_loss=False)
        prefill = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)
        parent_out = model.forward_decode(
            new_token_batch=tok_bp,
            positions=torch.tensor([T]),
            prefix_kv=prefill["kv_caches"],
        )
        wave_ids = {key: torch.cat([tok_a[key], tok_b[key]]) for key in ids}
        suffix_kv = [
            (
                torch.cat([torch.zeros_like(k), k], dim=0),
                torch.cat([torch.zeros_like(v), v], dim=0),
            )
            for k, v in parent_out["kv"]
        ]
        wave = model.forward_decode(
            new_token_batch=wave_ids,
            positions=torch.tensor([T, T + 1]),
            prefix_kv=prefill["kv_caches"],
            suffix_kv=suffix_kv,
            suffix_positions=torch.tensor([[0], [T]]),
            suffix_mask=torch.tensor([[False], [True]]),
        )
    torch.testing.assert_close(
        wave["logits"][0], full_a["logits"][T], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        wave["logits"][1], full_b["logits"][T + 1], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        wave["value_logits"][1], full_b["value_logits"][T + 1], atol=ATOL, rtol=RTOL
    )


import importlib.util
import sys
from pathlib import Path

import chess

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.eval import cozy_bridge
from imba_chess.eval.position_evaluator import _project_legal_logits_cozy
from imba_chess.eval.search import PositionEval, select_value_search_d2


def _load_eval_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_vs_stockfish.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_vs_stockfish_script_pd", script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FullForwardReferenceEvaluator:
    """Reference PositionEvaluator: rebuilds every sequence and runs a full
    forward — the uncached ground truth the cached path must reproduce.

    Unlike CachedPositionEvaluator, this reference needs the board BEFORE
    every move on the handle's path (not just the final position) to build
    _SequenceHistory incrementally -- but the PositionEvaluator protocol's
    extend(handle, move_uci) carries no board. So `handle` here is just the
    list of ucis played since root_board, and evaluate() replays them on a
    copy of root_board to reconstruct each intermediate board_before.
    """

    def __init__(
        self, *, module, model, move_vocab, board_state_encoder, played, root_board
    ):
        self._module = module
        self._model = model
        self._move_vocab = move_vocab
        self._encoder = board_state_encoder
        self._played = played  # list[(board_before, move_uci)] real game so far
        self._root_board = root_board  # py board the extend-chain's handle=None roots at

    def _fresh_history(self):
        history = self._module._SequenceHistory(
            move_vocab=self._move_vocab, board_state_encoder=self._encoder
        )
        for board_before, move_uci in self._played:
            history.append_observed_position(board_before)
            history.record_played_move(move_uci)
        return history

    def extend(self, handle, move_uci):
        path = list(handle) if handle is not None else []
        return path + [move_uci]

    def evaluate(self, batch):
        results = []
        for handle, cozy_board in batch:
            board = chess.Board(cozy_board.fen())
            history = self._fresh_history()
            replay_board = self._root_board.copy()
            for move_uci in handle:
                history.append_observed_position(replay_board)
                history.record_played_move(move_uci)
                replay_board.push_uci(move_uci)
            assert replay_board.board_fen() == board.board_fen()
            full_batch = history.build_batch_for_current_position(board)
            with torch.no_grad():
                out = self._model(full_batch, return_loss=False)
            logits = out["logits"][-1]
            value_stm = self._module._value_scalar_from_logits(
                out["value_logits"][-1]
            )
            try:
                # cozy-native projection (Stage 3 Task 4/5 contract:
                # PositionEval.legal_moves are always cc.Move, tree levels
                # included -- search.py's tree carries no python-chess board
                # to fall back on for a legacy-py-Move PositionEval anymore).
                legal_logits, legal_moves, legal_ucis, _, _ = _project_legal_logits_cozy(
                    logits=logits, cozy_board=cozy_board, move_vocab=self._move_vocab
                )
                log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
            except RuntimeError:
                legal_moves, legal_ucis, log_priors = [], [], []
            results.append(PositionEval(value_stm, legal_moves, legal_ucis, log_priors))
        return results


def _static_vocab():
    from imba_chess.data.move_vocab import all_possible_uci_moves

    return MoveVocab.build(
        all_possible_uci_moves(), config=MoveVocabConfig(include_unk=False)
    )


def test_cached_evaluator_matches_full_forward_reference():
    module = _load_eval_script_module()
    move_vocab = _static_vocab()
    model = _tiny_model(vocab_size=len(move_vocab))
    encoder = BoardStateEncoder()

    board = chess.Board()
    played = []
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=encoder
    )
    for move_uci in ["e2e4", "e7e5", "g1f3"]:
        played.append((board.copy(stack=False), move_uci))
        history.append_observed_position(board)
        history.record_played_move(move_uci)
        board.push_uci(move_uci)

    root_batch = history.build_batch_for_current_position(board)
    with torch.no_grad():
        prefill = model(root_batch, return_loss=False, return_kv=True)

    cached = module.CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=prefill["kv_caches"],
        prefix_len=int(root_batch["total_tokens"]),
    )
    reference = _FullForwardReferenceEvaluator(
        module=module,
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        played=played,
        root_board=board,
    )

    # Depth 1: two candidate moves. Depth 2: one reply under each.
    candidates = [chess.Move.from_uci("b8c6"), chess.Move.from_uci("d7d6")]
    cached_handles = [cached.extend(None, move.uci()) for move in candidates]
    ref_handles = [reference.extend(None, move.uci()) for move in candidates]
    boards1 = []
    for move in candidates:
        board1 = board.copy()
        board1.push(move)
        boards1.append(board1)
    cozy_boards1 = [cozy_bridge.board_to_cozy(b1) for b1 in boards1]

    cached_evals = cached.evaluate(list(zip(cached_handles, cozy_boards1)))
    ref_evals = reference.evaluate(list(zip(ref_handles, cozy_boards1)))
    for got, want in zip(cached_evals, ref_evals):
        assert abs(got.value_stm - want.value_stm) < 1e-5
        assert got.legal_ucis == want.legal_ucis
        for a, b in zip(got.legal_log_priors, want.legal_log_priors):
            assert abs(a - b) < 1e-5

    # One depth-2 node under each candidate, evaluated in a single wave.
    replies = [list(b1.legal_moves)[0] for b1 in boards1]
    cached2 = [
        cached.extend(handle, reply.uci())
        for handle, reply in zip(cached_handles, replies)
    ]
    ref2 = [
        reference.extend(handle, reply.uci())
        for handle, reply in zip(ref_handles, replies)
    ]
    boards2 = []
    for b1, reply in zip(boards1, replies):
        b2 = b1.copy()
        b2.push(reply)
        boards2.append(b2)
    cozy_boards2 = [cozy_bridge.board_to_cozy(b2) for b2 in boards2]
    cached_evals2 = cached.evaluate(list(zip(cached2, cozy_boards2)))
    ref_evals2 = reference.evaluate(list(zip(ref2, cozy_boards2)))
    for got, want in zip(cached_evals2, ref_evals2):
        assert abs(got.value_stm - want.value_stm) < 1e-5
        for a, b in zip(got.legal_log_priors, want.legal_log_priors):
            assert abs(a - b) < 1e-5


def test_strategy_picks_identical_move_cached_vs_reference():
    module = _load_eval_script_module()
    move_vocab = _static_vocab()
    model = _tiny_model(vocab_size=len(move_vocab))
    encoder = BoardStateEncoder()

    board = chess.Board()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=encoder
    )
    root_batch = history.build_batch_for_current_position(board)
    with torch.no_grad():
        prefill = model(root_batch, return_loss=False, return_kv=True)
        root_logits = prefill["logits"][-1]
    legal_logits, legal_moves, _, _ = module._project_legal_logits(
        logits=root_logits, board=board, move_vocab=move_vocab
    )
    log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()

    cached = module.CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=prefill["kv_caches"],
        prefix_len=int(root_batch["total_tokens"]),
    )
    reference = _FullForwardReferenceEvaluator(
        module=module,
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        played=[],
        root_board=board,
    )

    chosen_cached, _ = select_value_search_d2(
        evaluator=cached,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=log_priors,
        top_k=4,
        lam=0.05,
    )
    chosen_ref, _ = select_value_search_d2(
        evaluator=reference,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=log_priors,
        top_k=4,
        lam=0.05,
    )
    assert legal_moves[chosen_cached].uci() == legal_moves[chosen_ref].uci()


def test_project_legal_logits_returns_moves_sorted_by_uci():
    from imba_chess.eval.position_evaluator import _project_legal_logits

    move_vocab = _static_vocab()
    logits = torch.arange(len(move_vocab), dtype=torch.float32)
    board = chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    legal_logits, legal_moves, _, _ = _project_legal_logits(
        logits=logits, board=board, move_vocab=move_vocab
    )
    ucis = [m.uci() for m in legal_moves]
    assert ucis == sorted(ucis)
    # Alignment: each row of legal_logits must be the vocab logit of the
    # SAME-index move. Since logits == arange(vocab_size), the vocab id of a
    # move's uci equals its logit value directly.
    for row, move in zip(legal_logits.tolist(), legal_moves):
        assert row == move_vocab.token_to_id[move.uci()]


def test_project_legal_logits_cozy_matches_py_variant_incl_castling():
    """Direct unit test of _project_legal_logits_cozy (Stage 3 Task 4),
    mirroring test_project_legal_logits_returns_moves_sorted_by_uci but from
    a cozy board -- on a castling-rich position where cozy's own internal
    move encoding (e1h1/e1a1, king-takes-rook) differs from the standard
    UCI (e1g1/e1c1) cozy_move_to_uci must emit and the projection must sort
    by.
    """
    from imba_chess.eval.position_evaluator import (
        _project_legal_logits,
        _project_legal_logits_cozy,
    )

    move_vocab = _static_vocab()
    logits = torch.arange(len(move_vocab), dtype=torch.float32)
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)
    cozy_board = cozy_bridge.board_to_cozy(board)

    legal_logits_py, legal_moves_py, _, _ = _project_legal_logits(
        logits=logits, board=board, move_vocab=move_vocab
    )
    (
        legal_logits_cozy,
        legal_moves_cozy,
        legal_ucis_cozy,
        total_cozy,
        mapped_cozy,
    ) = _project_legal_logits_cozy(
        logits=logits, cozy_board=cozy_board, move_vocab=move_vocab
    )

    py_ucis = [m.uci() for m in legal_moves_py]
    assert "e1g1" in py_ucis and "e1c1" in py_ucis  # sanity: this fen has both castles
    # Same canonical UCI order as the python-chess variant (Task 1's
    # cross-movegen invariant), standard-UCI castling included.
    assert legal_ucis_cozy == py_ucis
    assert legal_ucis_cozy == sorted(legal_ucis_cozy)
    assert len(legal_moves_cozy) == len(legal_ucis_cozy)

    # Alignment: legal_logits row i is the vocab logit of legal_moves[i] /
    # legal_ucis[i] (index-aligned) for the sorted cozy result, and
    # legal_ucis really is cozy_move_to_uci(legal_moves[i]) -- not some
    # independently-derived string that happens to match.
    for row, uci in zip(legal_logits_cozy.tolist(), legal_ucis_cozy):
        assert row == move_vocab.token_to_id[uci]
    for move, uci in zip(legal_moves_cozy, legal_ucis_cozy):
        assert cozy_bridge.cozy_move_to_uci(cozy_board, move) == uci

    assert total_cozy == len(list(cozy_board.generate_moves()))
    assert mapped_cozy == len(legal_moves_py)
    torch.testing.assert_close(legal_logits_cozy, legal_logits_py)


def test_project_legal_logits_cozy_raises_when_nothing_maps_to_vocab():
    from imba_chess.eval.position_evaluator import _project_legal_logits_cozy

    empty_vocab = MoveVocab.build([], config=MoveVocabConfig(include_unk=False))
    board = chess.Board()
    cozy_board = cozy_bridge.board_to_cozy(board)
    logits = torch.zeros(len(empty_vocab))
    with pytest.raises(RuntimeError):
        _project_legal_logits_cozy(logits=logits, cozy_board=cozy_board, move_vocab=empty_vocab)
