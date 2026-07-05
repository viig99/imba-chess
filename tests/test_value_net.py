import torch

from imba_chess.model.value_net import ValueNet, ValueNetConfig


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
