#!/usr/bin/env python3
"""Train the standalone position-only WDL value net on Stockfish evals.

Lean flat-batch supervised loop: no jagged packing, no game parsing.
Usage: python scripts/train_value_net.py [--config config/imba_chess.toml]
       [--steps N] [--device cuda|cpu|auto]
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.position_eval_dataset import PositionEvalDataset
from imba_chess.model.value_net import ValueNet, ValueNetConfig


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def train_step(model, batch, optimizer, *, grad_clip_norm: float, autocast_ctx=None) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    with ctx:
        logits = model(batch)
        loss = soft_cross_entropy(logits, batch["wdl_target"].to(logits.device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def validate(model, batches) -> tuple[float, float]:
    # Scores the fixed val slice materialized at startup — a consistent
    # tracking metric, not a full-holdout average.
    model.eval()
    losses, correct, total = [], 0, 0
    for batch in batches:
        logits = model(batch)
        targets = batch["wdl_target"].to(logits.device)
        losses.append(float(soft_cross_entropy(logits, targets)))
        correct += int((logits.argmax(-1) == targets.argmax(-1)).sum())
        total += int(targets.size(0))
    mean_loss = sum(losses) / max(1, len(losses))
    accuracy = correct / max(1, total)
    return mean_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()

    cfg = load_repo_config(args.config).value_net
    steps = int(args.steps if args.steps is not None else cfg.train_steps)
    device_arg = args.device or cfg.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    torch.manual_seed(cfg.seed)

    model = ValueNet(
        ValueNetConfig(dim=cfg.dim, num_heads=cfg.num_heads, num_layers=cfg.num_layers)
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ValueNet params: {num_params / 1e6:.2f}M | device: {device}")

    def make_loader(split: str) -> DataLoader:
        dataset = PositionEvalDataset(
            split=split,
            depth_min=cfg.depth_min,
            dataset_name=cfg.dataset_name,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            seed=cfg.seed,
            val_permille=cfg.val_permille,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers if split == "train" else 0,
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader("train")

    # Materialize the fixed val slice once. The val stream keeps only
    # val_permille/1000 of rows, so filling it scans ~200x its size — minutes
    # of one-time work; re-scanning on every validate() call would cost that
    # every 5k steps. ~50 MB of tensors buys ~1 s validations instead.
    print("materializing val slice (one-time stream scan) ...")
    val_batches = []
    for batch in make_loader("val"):
        val_batches.append(batch)
        if len(val_batches) >= cfg.val_batches:
            break
    print(f"val slice ready: {len(val_batches)} batches of {cfg.batch_size}")

    try:
        from optimi import StableAdamW

        optimizer = StableAdamW(
            model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay,
            kahan_sum=True,
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, total_steps=steps, pct_start=0.05
    )

    use_amp = device.type == "cuda" and cfg.dtype in {"bfloat16", "float16"}
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=getattr(torch, cfg.dtype))
        if use_amp
        else None
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tb"))
    model_config_payload = {
        "dim": cfg.dim, "num_heads": cfg.num_heads, "num_layers": cfg.num_layers
    }

    def save(name: str, step: int, val_loss: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "config": model_config_payload,
                "step": step,
                "val_loss": val_loss,
            },
            checkpoint_dir / name,
        )

    best_val = float("inf")
    step = 0
    t0 = time.time()
    while step < steps:
        for batch in train_loader:
            loss = train_step(
                model, batch, optimizer,
                grad_clip_norm=cfg.grad_clip_norm, autocast_ctx=autocast_ctx,
            )
            scheduler.step()
            step += 1
            if step % cfg.log_every_steps == 0:
                rate = step / (time.time() - t0)
                print(f"step {step}/{steps} loss {loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} ({rate:.1f} it/s)")
                writer.add_scalar("train/loss", loss, step)
            if step % cfg.val_every_steps == 0 or step == steps:
                val_loss, val_acc = validate(model, val_batches)
                print(f"  val loss {val_loss:.4f} acc {val_acc:.3f}")
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/acc", val_acc, step)
                save("value_net_last.pt", step, val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    save("value_net_best.pt", step, val_loss)
            if step >= steps:
                break
    writer.close()
    print(f"done: best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
