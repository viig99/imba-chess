from __future__ import annotations

import chess
import pytest

torch = pytest.importorskip("torch")

from imba_chess.model.value_net import (
    ValueNet,
    ValueNetConfig,
    board_material_count,
    cp_to_wdl,
)


def test_cp_to_wdl_sums_to_one_and_is_symmetric():
    for cp in [-4000, -300, -50, 0, 50, 300, 4000]:
        for material in [17, 40, 58, 78]:
            p_loss, p_draw, p_win = cp_to_wdl(cp, material)
            assert abs(p_loss + p_draw + p_win - 1.0) < 1e-9
            assert 0.0 <= p_draw <= 1.0
            # Symmetry: negating cp swaps win and loss exactly.
            q_loss, q_draw, q_win = cp_to_wdl(-cp, material)
            assert abs(p_win - q_loss) < 1e-9
            assert abs(p_draw - q_draw) < 1e-9


def test_cp_to_wdl_monotone_in_cp():
    for material in [20, 58, 78]:
        wins = [cp_to_wdl(cp, material)[2] for cp in range(-500, 501, 50)]
        assert all(b >= a for a, b in zip(wins, wins[1:]))


def test_cp_to_wdl_anchor_and_extremes():
    # SF normalized-cp convention: +100 cp == 50% win probability.
    _, _, p_win = cp_to_wdl(100, 58)
    assert abs(p_win - 0.5) < 1e-6
    # Equal position: win and loss mass are equal and small vs draw.
    p_loss, p_draw, p_win = cp_to_wdl(0, 58)
    assert abs(p_win - p_loss) < 1e-9
    assert p_draw > p_win
    # Huge advantage saturates.
    assert cp_to_wdl(4000, 58)[2] > 0.99
    # Extreme cp values beyond the clamp do not blow up.
    assert cp_to_wdl(100000, 58)[2] == pytest.approx(cp_to_wdl(20000, 58)[2], abs=1e-6)


def test_board_material_count():
    assert board_material_count(chess.Board()) == 78  # 16P + 4N*3 + 4B*3 + 4R*5 + 2Q*9
    assert board_material_count(chess.Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")) == 0


def test_value_net_forward_shapes_and_determinism():
    torch.manual_seed(0)
    net = ValueNet(ValueNetConfig(dim=32, num_heads=2, num_layers=2)).eval()
    batch = {
        "piece_ids": torch.randint(0, 13, (5, 64)),
        "turn_id": torch.randint(0, 2, (5,)),
        "castle_id": torch.randint(0, 16, (5,)),
        "ep_file_id": torch.randint(0, 9, (5,)),
        # Extra keys must be ignored (the eval wave batch carries them).
        "prev_move_id": torch.zeros(5, dtype=torch.long),
        "seq_token_id": torch.zeros(5, dtype=torch.long),
    }
    with torch.no_grad():
        out1 = net(batch)
        out2 = net(batch)
    assert out1.shape == (5, 3)
    torch.testing.assert_close(out1, out2)


def test_value_net_turn_changes_output():
    torch.manual_seed(1)
    net = ValueNet(ValueNetConfig(dim=32, num_heads=2, num_layers=2)).eval()
    base = {
        "piece_ids": torch.randint(0, 13, (1, 64)),
        "castle_id": torch.zeros(1, dtype=torch.long),
        "ep_file_id": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        white = net({**base, "turn_id": torch.tensor([0])})
        black = net({**base, "turn_id": torch.tensor([1])})
    assert not torch.allclose(white, black)


def test_trainer_smoke_one_step():
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_value_net.py"
    spec = importlib.util.spec_from_file_location("train_value_net_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    net = ValueNet(ValueNetConfig(dim=32, num_heads=2, num_layers=1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    batch = {
        "piece_ids": torch.randint(0, 13, (8, 64)),
        "turn_id": torch.randint(0, 2, (8,)),
        "castle_id": torch.randint(0, 16, (8,)),
        "ep_file_id": torch.randint(0, 9, (8,)),
        "wdl_target": torch.softmax(torch.randn(8, 3), dim=-1),
    }
    before = [p.detach().clone() for p in net.parameters()]
    loss = module.train_step(net, batch, optimizer, grad_clip_norm=1.0)
    assert torch.isfinite(torch.tensor(loss))
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(before, net.parameters())
    )
