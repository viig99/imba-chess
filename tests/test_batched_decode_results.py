import chess
import pytest
import torch

from imba_chess.eval import cozy_bridge
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    consume_batched_decode_results,
)
from imba_chess.eval.merged_executors import (
    _split_decode_output,
    _merge_decode_requests,
)
from tests.test_self_play import VOCAB, ENCODER


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)])
def test_batched_projection_preserves_results_and_arena_ownership(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(129)
    counts = [1, 2, 3]
    groups = []
    for _ in range(2):
        evaluators = []
        requests = []
        for n in counts:
            evaluator = CachedPositionEvaluator(
                model=None,
                move_vocab=VOCAB,
                board_state_encoder=ENCODER,
                device=torch.device(device),
                dtype=torch.float32,
                prefix_kv=[
                    (
                        torch.zeros(2, 3, 8, device=device),
                        torch.zeros(2, 3, 8, device=device),
                    )
                    for _ in range(2)
                ],
                prefix_len=3,
            )
            batch = []
            for uci in ["e2e4", "d2d4", "g1f3"][:n]:
                board = chess.Board()
                board.push_uci(uci)
                batch.append(
                    (evaluator.extend(None, uci), cozy_bridge.board_to_cozy(board))
                )
            evaluators.append(evaluator)
            requests.append(evaluator.build_decode_request(batch))
        groups.append((evaluators, requests))
    # Deferred construction must preserve all merged inputs exactly.
    from dataclasses import replace

    regular = groups[0][1]
    deferred = [
        replace(
            r,
            new_token_batch={k: v.tolist() for k, v in r.new_token_batch.items()},
            positions=r.positions.tolist(),
        )
        for r in regular
    ]
    expected, actual_inputs = (
        _merge_decode_requests(regular),
        _merge_decode_requests(deferred),
    )
    for key, value in expected.new_token_batch.items():
        torch.testing.assert_close(
            value, actual_inputs.new_token_batch[key], atol=0, rtol=0
        )
    torch.testing.assert_close(
        expected.positions, actual_inputs.positions, atol=0, rtol=0
    )
    total = sum(counts)
    out = dict(
        logits=torch.randn(total, len(VOCAB), device=device),
        value_logits=torch.randn(total, 3, device=device),
        kv=[
            (
                torch.randn(total, 2, 1, 8, device=device),
                torch.randn(total, 2, 1, 8, device=device),
            )
            for _ in range(2)
        ],
    )
    a, ar = groups[0]
    b, br = groups[1]
    reference = [
        e.consume_decode_result(r, o)
        for e, r, o in zip(a, ar, _split_decode_output(out, counts))
    ]
    actual = consume_batched_decode_results([(e, None) for e in b], br, out)
    for ea, eb, ra, rb, refs, got in zip(a, b, ar, br, reference, actual):
        torch.testing.assert_close(ea._arena.k, eb._arena.k, atol=0, rtol=0)
        torch.testing.assert_close(ea._arena.v, eb._arena.v, atol=0, rtol=0)
        assert [n.arena_chain for n in ra.nodes] == [n.arena_chain for n in rb.nodes]
        for x, y in zip(refs, got):
            assert x.legal_ids == y.legal_ids and x.legal_ucis == y.legal_ucis
            assert x.legal_forcing == y.legal_forcing
            assert x.value_stm == pytest.approx(y.value_stm, abs=1e-6)
            torch.testing.assert_close(
                torch.tensor(x.legal_log_priors),
                torch.tensor(y.legal_log_priors),
                atol=1e-6,
                rtol=1e-6,
            )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)])
@pytest.mark.parametrize("retain_layers", [False, True])
def test_suffix_layer_packing_preserves_mixed_group_padding(device, retain_layers):
    from types import SimpleNamespace

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(89)
    requests = []
    for group, (count, length) in enumerate([(1, 0), (2, 1), (1, 3)]):
        layers = (
            (
                torch.randn(3, count, 2, length, 4, device=device),
                torch.randn(3, count, 2, length, 5, device=device),
            )
            if length
            else None
        )
        requests.append(
            SimpleNamespace(
                nodes=list(range(count)),
                new_token_batch={"turn_id": torch.arange(count)},
                positions=torch.arange(count),
                prefix_len=group + 2,
                prefix_kv=[
                    (
                        torch.randn(2, group + 2, 4, device=device),
                        torch.randn(2, group + 2, 5, device=device),
                    )
                    for _ in range(3)
                ],
                suffix_kv=list(zip(layers[0].unbind(0), layers[1].unbind(0)))
                if layers
                else None,
                suffix_layers=layers if retain_layers else None,
                suffix_positions=torch.arange(length, device=device).expand(count, -1)
                if length
                else None,
                suffix_mask=torch.ones(count, length, dtype=torch.bool, device=device)
                if length
                else None,
            )
        )
    reference = _merge_decode_requests(requests)
    candidate = _merge_decode_requests(requests, batch_suffix=True)
    for actual, expected in zip(candidate.suffix_kv, reference.suffix_kv):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(
        candidate.suffix_positions, reference.suffix_positions, atol=0, rtol=0
    )
    assert torch.equal(candidate.suffix_mask, reference.suffix_mask)
