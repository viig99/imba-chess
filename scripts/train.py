#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import (
    LichessDataset,
    build_event_dataloader,
    load_or_create_static_move_vocab,
)
from imba_chess.eval import create_next_move_evaluator
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ignite trainer for chess next-move prediction."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from (or to load in eval-only mode).",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "both"],
        default="val",
        help="Used only with --eval-only.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override training.device config.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Override training.dtype config.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override training.compile_model config.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional cap for eval iterations per run.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_arg]


def _make_dataset(config, *, split: str) -> LichessDataset:
    dataset_cfg = replace(config.dataset, split=split)
    return LichessDataset(
        min_avg_elo=dataset_cfg.min_avg_elo,
        split=dataset_cfg.split,
        dataset_name=dataset_cfg.dataset_name,
        train_start_month=dataset_cfg.train_start_month,
        train_end_month=dataset_cfg.train_end_month,
        val_start_month=dataset_cfg.val_start_month,
        val_end_month=dataset_cfg.val_end_month,
        test_start_month=dataset_cfg.test_start_month,
        test_end_month=dataset_cfg.test_end_month,
        val_max_games=dataset_cfg.val_max_games,
        test_max_games=dataset_cfg.test_max_games,
        cache_dir=dataset_cfg.cache_dir,
        parquet_batch_size=dataset_cfg.parquet_batch_size,
        max_seq_len=dataset_cfg.max_seq_len,
        return_dataclasses=dataset_cfg.return_dataclasses,
        board_state_config=config.board_state,
    )


def _build_optimizer(model: torch.nn.Module, config, *, device: torch.device):
    kwargs: dict[str, Any] = {
        "lr": float(config.training.max_lr),
        "weight_decay": float(config.training.weight_decay),
    }
    if device.type == "cuda":
        kwargs["fused"] = bool(config.training.optimizer_fused)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_scheduler(optimizer: torch.optim.Optimizer, config):
    total_steps = int(config.training.epochs) * int(config.training.steps_per_epoch)
    max_lr = float(config.training.max_lr)
    lr_start_factor = float(config.training.lr_start_factor)
    lr_end_factor = float(config.training.lr_end_factor)
    pct_start = float(config.training.onecycle_pct_start)
    if lr_start_factor <= 0.0 or lr_end_factor <= 0.0:
        raise ValueError("lr_start_factor and lr_end_factor must be > 0")
    if pct_start <= 0.0 or pct_start >= 1.0:
        raise ValueError("onecycle_pct_start must be in (0, 1)")

    # Required shape:
    # start = lr_start_factor * max_lr, end = lr_end_factor * max_lr.
    peak_lr = lr_end_factor * max_lr
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=peak_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="linear",
        div_factor=lr_end_factor / lr_start_factor,
        final_div_factor=lr_start_factor / lr_end_factor,
        three_phase=False,
    )


def _score_hr10(engine: Engine) -> float:
    value = float(engine.state.metrics.get("top10_acc", float("nan")))
    if math.isnan(value):
        return float("-inf")
    return value


def _print_eval_metrics(split: str, metrics: dict[str, float]) -> None:
    print(f"{split} metrics:")
    print(f"  game_count: {int(metrics['game_count'])}")
    print(f"  token_count: {int(metrics['token_count'])}")
    print(f"  loss_ce: {metrics['loss_ce']:.6f}")
    print(f"  ppl: {metrics['ppl']:.4f}")
    print(f"  top1_acc: {metrics['top1_acc']:.6f}")
    print(f"  top3_acc: {metrics['top3_acc']:.6f}")
    print(f"  top5_acc: {metrics['top5_acc']:.6f}")
    print(f"  hr@10: {metrics['top10_acc']:.6f}")
    print(f"  mrr: {metrics['mrr']:.6f}")


def main() -> None:
    args = parse_args()
    repo_config = load_repo_config(args.config)
    if create_next_move_evaluator is None:
        raise ImportError(
            "pytorch-ignite is not available. Run `uv sync` and retry."
        )

    training_cfg = repo_config.training
    if args.device is not None:
        training_cfg = replace(training_cfg, device=args.device)
    if args.dtype is not None:
        training_cfg = replace(training_cfg, dtype=args.dtype)
    if args.compile is not None:
        training_cfg = replace(training_cfg, compile_model=bool(args.compile))
    repo_config = replace(repo_config, training=training_cfg)
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume <checkpoint_path>")

    device = _resolve_device(repo_config.training.device)
    dtype = _resolve_dtype(repo_config.training.dtype)
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    use_scaler = device.type == "cuda" and dtype == torch.float16

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path,
        include_unk=repo_config.vocab.include_unk,
    )
    val_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(repo_config, split="val"),
        config=repo_config,
        move_vocab=move_vocab,
    )
    test_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(repo_config, split="test"),
        config=repo_config,
        move_vocab=move_vocab,
    )

    model_cfg = build_hstu_chess_config(repo_config.model, move_vocab_size=len(move_vocab))
    model: torch.nn.Module = HSTUChessModel(model_cfg).to(device)
    if repo_config.training.compile_model:
        model = torch.compile(model, dynamic=True)

    val_evaluator = create_next_move_evaluator(
        model=model,
        device=device,
        dtype=dtype,
        ignore_index=repo_config.model.ignore_index,
        topk=(1, 3, 5, 10),
    )
    test_evaluator = create_next_move_evaluator(
        model=model,
        device=device,
        dtype=dtype,
        ignore_index=repo_config.model.ignore_index,
        topk=(1, 3, 5, 10),
    )

    def _run_eval_only() -> None:
        run_kwargs: dict[str, Any] = {}
        if args.max_eval_batches is not None:
            run_kwargs["epoch_length"] = args.max_eval_batches

        if args.eval_split in {"val", "both"}:
            val_evaluator.run(val_loader, max_epochs=1, **run_kwargs)
            _print_eval_metrics("val", val_evaluator.state.metrics)
        if args.eval_split in {"test", "both"}:
            test_evaluator.run(test_loader, max_epochs=1, **run_kwargs)
            _print_eval_metrics("test", test_evaluator.state.metrics)

    if args.eval_only:
        checkpoint = torch.load(args.resume, map_location="cpu")
        Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
        print(f"Loaded checkpoint for eval: {args.resume}")
        _run_eval_only()
        return

    train_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(repo_config, split="train"),
        config=repo_config,
        move_vocab=move_vocab,
    )
    optimizer = _build_optimizer(model, repo_config, device=device)
    scheduler = _build_scheduler(optimizer, repo_config)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def _train_step(engine: Engine, batch: dict[str, object]) -> dict[str, float]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            output = model(batch, return_loss=True)
            loss = output["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(repo_config.training.grad_clip_norm)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(repo_config.training.grad_clip_norm)
            )
            optimizer.step()

        scheduler.step()
        return {
            "loss": float(loss.detach().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

    trainer = Engine(_train_step)
    checkpoint_dir = Path(repo_config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_objects: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer": trainer,
        "scaler": scaler,
    }
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        Checkpoint.load_objects(to_load=checkpoint_objects, checkpoint=checkpoint)
        print(f"Resumed training from checkpoint: {args.resume}")

    tb_logger = TensorboardLogger(log_dir=str(checkpoint_dir / "tb"))
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=repo_config.training.log_every_steps),
        tag="train",
        output_transform=lambda output: output,
    )
    tb_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.COMPLETED,
        tag="val",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    best_ckpt_handler = Checkpoint(
        to_save=checkpoint_objects,
        save_handler=DiskSaver(
            str(checkpoint_dir), create_dir=True, require_empty=False
        ),
        filename_prefix="best_hr10",
        n_saved=int(repo_config.training.checkpoint_keep),
        global_step_transform=global_step_from_engine(trainer),
        score_function=_score_hr10,
        score_name="hr10",
    )
    val_evaluator.add_event_handler(Events.COMPLETED, best_ckpt_handler)

    @trainer.on(Events.ITERATION_COMPLETED(every=repo_config.training.eval_every_steps))
    def _run_periodic_val_eval(engine: Engine) -> None:
        run_kwargs: dict[str, Any] = {}
        if args.max_eval_batches is not None:
            run_kwargs["epoch_length"] = args.max_eval_batches
        val_evaluator.run(val_loader, max_epochs=1, **run_kwargs)
        _print_eval_metrics("val", val_evaluator.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _epoch_summary(engine: Engine) -> None:
        print(
            f"epoch={engine.state.epoch} iteration={engine.state.iteration} "
            f"loss={engine.state.output['loss']:.6f} lr={engine.state.output['lr']:.7f}"
        )

    try:
        print("Starting training with Ignite")
        print(
            f"  epochs={repo_config.training.epochs}, "
            f"steps_per_epoch={repo_config.training.steps_per_epoch}, "
            f"eval_every_steps={repo_config.training.eval_every_steps}"
        )
        trainer.run(
            train_loader,
            max_epochs=repo_config.training.epochs,
            epoch_length=repo_config.training.steps_per_epoch,
        )
    finally:
        tb_logger.close()


if __name__ == "__main__":
    main()
