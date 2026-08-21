"""Batched per-wave projection must equal the per-node math it replaces.

consume_decode_result ran, per node, a torch.softmax for the value scalar plus
torch.tensor + index_select + log_softmax for the priors. At budget 2048 a wave
carries ~1,321 nodes over ~31-element tensors, so dispatch dominated: measured
15.0 us/node of kv cat, 11.9 us/node of value-scalar + PositionEval, and 7.0
us/node of gather + log_softmax (scripts/probe_project_phases.py).

These tests pin the batched replacements against the exact per-node
expressions, including the ragged case (nodes have different legal-move counts,
so the batch is padded and masked).
"""

import pytest

torch = pytest.importorskip("torch")

from imba_chess.eval.position_evaluator import (
    _batched_legal_log_priors,
    _batched_value_scalars,
    _value_scalar_from_logits,
)


def test_batched_value_scalars_matches_per_node():
    torch.manual_seed(0)
    value_logits = torch.randn(37, 3)
    expected = [_value_scalar_from_logits(value_logits[r]) for r in range(37)]
    got = _batched_value_scalars(value_logits)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-9)


def test_batched_value_scalars_extreme_logits():
    """Saturated WDL logits must not produce nan/inf through the batched path."""
    value_logits = torch.tensor(
        [[100.0, 0.0, -100.0], [-100.0, 0.0, 100.0], [0.0, 0.0, 0.0]]
    )
    got = _batched_value_scalars(value_logits)
    expected = [_value_scalar_from_logits(value_logits[r]) for r in range(3)]
    assert got == pytest.approx(expected, abs=1e-9)
    assert all(-1.0 <= g <= 1.0 for g in got)


def _per_node_log_priors(logits, id_lists):
    """Verbatim old per-node expression."""
    out = []
    for row, ids in enumerate(id_lists):
        t = torch.tensor(ids, device=logits.device, dtype=torch.long)
        legal = logits[row].index_select(0, t)
        out.append(torch.log_softmax(legal.float(), dim=0).tolist())
    return out


def test_batched_log_priors_matches_per_node_ragged():
    """The realistic shape: every node has a different legal-move count."""
    torch.manual_seed(1)
    V = 1970
    id_lists = [
        list(range(0, 31)),
        list(range(100, 108)),  # short row -> padded
        list(range(500, 545)),  # long row -> sets M
        [7],                    # single legal move
    ]
    logits = torch.randn(len(id_lists), V)
    expected = _per_node_log_priors(logits, id_lists)
    got = _batched_legal_log_priors(logits, id_lists)

    assert [len(g) for g in got] == [len(x) for x in id_lists], (
        "padded rows must be trimmed back to each node's own move count"
    )
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-6)
        # A log-softmax must still normalize over exactly that node's moves.
        assert sum(torch.tensor(g).exp().tolist()) == pytest.approx(1.0, abs=1e-5)


def test_batched_log_priors_uniform_lengths():
    """Equal-length rows: no padding involved, so this must be tight."""
    torch.manual_seed(2)
    id_lists = [list(range(i, i + 20)) for i in (0, 40, 80)]
    logits = torch.randn(3, 500)
    got = _batched_legal_log_priors(logits, id_lists)
    expected = _per_node_log_priors(logits, id_lists)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-6)


def test_batched_log_priors_padding_cannot_leak():
    """Padded slots must contribute exactly nothing.

    Regression guard: if padding used index 0 without masking to -inf, a large
    logit at vocab id 0 would bleed into every short row's normalizer.
    """
    V = 64
    logits = torch.full((2, V), -5.0)
    logits[:, 0] = 50.0  # huge value at the id used as padding filler
    id_lists = [[10, 11, 12, 13], [20]]
    got = _batched_legal_log_priors(logits, id_lists)
    expected = _per_node_log_priors(logits, id_lists)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-6)
    # The single-move row is normalized over one move: log(1) == 0.
    assert got[1] == pytest.approx([0.0], abs=1e-6)
