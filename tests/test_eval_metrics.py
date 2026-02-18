import math

import pytest

torch = pytest.importorskip("torch")

from imba_chess.eval.metrics import (
    BatchCount,
    GameCount,
    NextMoveCrossEntropy,
    NextMoveMRR,
    NextMoveTokenCount,
    NextMoveTopKAccuracy,
)


def _output():
    logits = torch.tensor(
        [
            [0.1, 4.0, 0.3, 0.2, 0.1],  # target=1 rank=1
            [0.1, 1.5, 0.3, 0.2, 1.0],  # target=4 rank=2
            [2.0, 0.1, 0.1, 0.1, 0.1],  # ignored
            [3.0, 4.0, 5.0, 0.1, 0.1],  # target=0 rank=3
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 4, -100, 0], dtype=torch.long)
    return {"logits": logits, "targets": targets, "num_games": 2.0}


def test_next_move_metrics_with_ignore_index():
    output = _output()

    loss = NextMoveCrossEntropy(ignore_index=-100)
    top1 = NextMoveTopKAccuracy(k=1, ignore_index=-100)
    top3 = NextMoveTopKAccuracy(k=3, ignore_index=-100)
    mrr = NextMoveMRR(ignore_index=-100)
    token_count = NextMoveTokenCount(ignore_index=-100)
    batch_count = BatchCount()
    game_count = GameCount()

    metrics = [loss, top1, top3, mrr, token_count, batch_count, game_count]
    for metric in metrics:
        metric.reset()
        metric.update(output)

    assert token_count.compute() == 3.0
    assert batch_count.compute() == 1.0
    assert game_count.compute() == 2.0
    assert top1.compute() == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert top3.compute() == pytest.approx(1.0, rel=1e-6)
    assert mrr.compute() == pytest.approx((1.0 + 0.5 + (1.0 / 3.0)) / 3.0, rel=1e-6)

    valid_mask = output["targets"] != -100
    expected_loss = torch.nn.functional.cross_entropy(
        output["logits"][valid_mask],
        output["targets"][valid_mask],
        reduction="mean",
    ).item()
    assert loss.compute() == pytest.approx(expected_loss, rel=1e-6)
    assert math.exp(loss.compute()) > 0.0


def test_topk_metric_validates_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        NextMoveTopKAccuracy(k=0, ignore_index=-100)


def test_metrics_raise_when_all_targets_ignored():
    output = {
        "logits": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        "targets": torch.tensor([-100], dtype=torch.long),
        "num_games": 1.0,
    }
    metrics = [
        NextMoveCrossEntropy(ignore_index=-100),
        NextMoveTopKAccuracy(k=1, ignore_index=-100),
        NextMoveMRR(ignore_index=-100),
    ]
    for metric in metrics:
        metric.reset()
        with pytest.raises(ValueError, match="no valid targets"):
            metric.update(output)
