#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, ProgressBar, global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import (
    LichessDataset,
    build_event_dataloader,
    load_or_create_static_move_vocab,
)
from imba_chess.eval import create_next_move_evaluator
from imba_chess.model import HSTUChessModel, build_hstu_chess_config

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)


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
    if bool(config.training.optimizer_fused):
        if device.type != "cuda":
            raise ValueError(
                "training.optimizer_fused=true requires CUDA device. "
                "Set optimizer_fused=false for CPU training."
            )
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_scheduler(optimizer: torch.optim.Optimizer, config):
    total_steps = int(config.training.epochs) * int(config.training.steps_per_epoch)
    max_lr = float(config.training.max_lr)
    lr_start_factor = float(config.training.lr_start_factor)
    lr_end_factor = float(config.training.lr_end_factor)
    warmup_first_epoch_fraction = float(
        config.training.onecycle_warmup_fraction_first_epoch
    )
    pct_start = warmup_first_epoch_fraction / max(1, int(config.training.epochs))
    if lr_start_factor <= 0.0 or lr_end_factor <= 0.0:
        raise ValueError("lr_start_factor and lr_end_factor must be > 0")
    if pct_start <= 0.0 or pct_start >= 1.0:
        raise ValueError(
            "Derived OneCycle pct_start must be in (0, 1). "
            "Check onecycle_warmup_fraction_first_epoch and epochs."
        )

    # Required shape:
    # start = lr_start_factor * max_lr, peak = max_lr, end = lr_end_factor * max_lr.
    peak_lr = max_lr
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=peak_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="linear",
        div_factor=1.0 / lr_start_factor,
        final_div_factor=lr_start_factor / lr_end_factor,
        three_phase=False,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_eval_runtime_config(
    config,
    *,
    val_max_games: int | None = None,
    test_max_games: int | None = None,
):
    dataset_cfg = config.dataset
    if val_max_games is not None:
        dataset_cfg = replace(dataset_cfg, val_max_games=val_max_games)
    if test_max_games is not None:
        dataset_cfg = replace(dataset_cfg, test_max_games=test_max_games)
    eval_num_workers = int(config.training.eval_num_workers)
    if eval_num_workers < 0:
        raise ValueError("training.eval_num_workers must be >= 0")
    dataloader_cfg = replace(
        config.dataloader,
        num_workers=eval_num_workers,
        pin_memory=False,
        prefetch_factor=(
            config.dataloader.prefetch_factor if eval_num_workers > 0 else None
        ),
        persistent_workers=(
            config.dataloader.persistent_workers if eval_num_workers > 0 else False
        ),
    )
    return replace(config, dataset=dataset_cfg, dataloader=dataloader_cfg)


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


def _validate_runtime_config(
    *,
    repo_config,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "training.device='cuda' requested but CUDA is not available."
        )
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError(
            "Non-float32 training on CPU is unsupported in this script. "
            "Use training.dtype='float32' or switch to CUDA."
        )
    if not bool(repo_config.training.deterministic_eval):
        raise ValueError(
            "training.deterministic_eval must be true. "
            "Stochastic eval mode is intentionally unsupported."
        )


def _run_deterministic_eval(
    *,
    evaluator: Engine,
    loader,
    seed: int,
    epoch_length: int | None,
) -> None:
    _set_seed(seed)
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        evaluator.run(loader, max_epochs=1, epoch_length=epoch_length)
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.deterministic = prev_deterministic


def main() -> None:
    args = parse_args()
    repo_config = load_repo_config(args.config)
    if create_next_move_evaluator is None:
        raise ImportError("pytorch-ignite is not available. Run `uv sync` and retry.")

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
    if repo_config.training.eval_every_steps < 1:
        raise ValueError("training.eval_every_steps must be >= 1")
    if repo_config.training.save_last_every_steps < 1:
        raise ValueError("training.save_last_every_steps must be >= 1")
    if repo_config.training.full_val_every_epochs < 1:
        raise ValueError("training.full_val_every_epochs must be >= 1")
    if repo_config.training.fast_val_max_games < 1:
        raise ValueError("training.fast_val_max_games must be >= 1")
    if repo_config.training.last_checkpoint_keep < 1:
        raise ValueError("training.last_checkpoint_keep must be >= 1")

    _set_seed(int(repo_config.training.seed))

    device = _resolve_device(repo_config.training.device)
    dtype = _resolve_dtype(repo_config.training.dtype)
    _validate_runtime_config(repo_config=repo_config, device=device, dtype=dtype)
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    eval_epoch_length = args.max_eval_batches

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path,
        include_unk=repo_config.vocab.include_unk,
    )
    eval_runtime_fast_val = _make_eval_runtime_config(
        repo_config,
        val_max_games=int(repo_config.training.fast_val_max_games),
    )
    eval_runtime_full_val = _make_eval_runtime_config(
        repo_config,
        val_max_games=repo_config.dataset.val_max_games,
    )
    eval_runtime_test = _make_eval_runtime_config(
        repo_config,
        test_max_games=repo_config.dataset.test_max_games,
    )
    fast_val_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(eval_runtime_fast_val, split="val"),
        config=eval_runtime_fast_val,
        move_vocab=move_vocab,
    )
    full_val_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(eval_runtime_full_val, split="val"),
        config=eval_runtime_full_val,
        move_vocab=move_vocab,
    )
    test_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(eval_runtime_test, split="test"),
        config=eval_runtime_test,
        move_vocab=move_vocab,
    )

    model_cfg = build_hstu_chess_config(
        repo_config.model, move_vocab_size=len(move_vocab)
    )
    model: torch.nn.Module = HSTUChessModel(model_cfg).to(device)
    if repo_config.training.compile_model:
        model = torch.compile(model, dynamic=True)

    fast_val_evaluator = create_next_move_evaluator(
        model=model,
        device=device,
        dtype=dtype,
        ignore_index=repo_config.model.ignore_index,
        topk=(1, 3, 5, 10),
    )
    full_val_evaluator = create_next_move_evaluator(
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

    fast_val_pbar = ProgressBar(persist=False, desc="val_fast")
    fast_val_pbar.attach(
        fast_val_evaluator,
        output_transform=lambda out: {"games": int(out["num_games"])},
    )
    full_val_pbar = ProgressBar(persist=False, desc="val_full")
    full_val_pbar.attach(
        full_val_evaluator,
        output_transform=lambda out: {"games": int(out["num_games"])},
    )
    test_pbar = ProgressBar(persist=False, desc="test")
    test_pbar.attach(
        test_evaluator,
        output_transform=lambda out: {"games": int(out["num_games"])},
    )

    def _run_eval_only() -> None:
        if args.eval_split in {"val", "both"}:
            _run_deterministic_eval(
                evaluator=full_val_evaluator,
                loader=full_val_loader,
                seed=int(repo_config.training.seed),
                epoch_length=eval_epoch_length,
            )
            _print_eval_metrics("val", full_val_evaluator.state.metrics)
        if args.eval_split in {"test", "both"}:
            _run_deterministic_eval(
                evaluator=test_evaluator,
                loader=test_loader,
                seed=int(repo_config.training.seed),
                epoch_length=eval_epoch_length,
            )
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
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    def _train_step(engine: Engine, batch: dict[str, object]) -> dict[str, float]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_games = int(batch["num_games"])
        prior_epoch_games = int(getattr(engine.state, "epoch_game_count", 0))
        engine.state.epoch_game_count = prior_epoch_games + batch_games
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            output = model(batch, return_loss=True)
            loss = output["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered at iteration {engine.state.iteration}"
            )

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
            "tokens": float(int(batch["total_tokens"])),
            "games": float(batch_games),
        }

    trainer = Engine(_train_step)

    @trainer.on(Events.EPOCH_STARTED)
    def _reset_epoch_game_count(engine: Engine) -> None:
        engine.state.epoch_game_count = 0

    train_pbar = ProgressBar(persist=True, desc="train")
    train_pbar.attach(
        trainer,
        output_transform=lambda out: {
            "loss": f"{out['loss']:.4f}",
            "lr": f"{out['lr']:.6f}",
            "tokens": int(out["tokens"]),
            "games": int(out["games"]),
        },
    )
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
        event_name=Events.ITERATION_COMPLETED(
            every=repo_config.training.log_every_steps
        ),
        tag="train",
        output_transform=lambda output: output,
    )
    tb_logger.attach_output_handler(
        fast_val_evaluator,
        event_name=Events.COMPLETED,
        tag="val_fast",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach_output_handler(
        full_val_evaluator,
        event_name=Events.COMPLETED,
        tag="val_full",
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
    full_val_evaluator.add_event_handler(Events.COMPLETED, best_ckpt_handler)

    last_ckpt_handler = Checkpoint(
        to_save=checkpoint_objects,
        save_handler=DiskSaver(
            str(checkpoint_dir), create_dir=True, require_empty=False
        ),
        filename_prefix="last",
        n_saved=int(repo_config.training.last_checkpoint_keep),
        global_step_transform=global_step_from_engine(trainer),
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(
            every=int(repo_config.training.save_last_every_steps)
        ),
        last_ckpt_handler,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=repo_config.training.eval_every_steps))
    def _run_periodic_fast_val_eval(engine: Engine) -> None:
        _run_deterministic_eval(
            evaluator=fast_val_evaluator,
            loader=fast_val_loader,
            seed=int(repo_config.training.seed),
            epoch_length=eval_epoch_length,
        )
        _print_eval_metrics("val_fast", fast_val_evaluator.state.metrics)

    @trainer.on(
        Events.EPOCH_COMPLETED(every=int(repo_config.training.full_val_every_epochs))
    )
    def _run_periodic_full_val_eval(engine: Engine) -> None:
        _run_deterministic_eval(
            evaluator=full_val_evaluator,
            loader=full_val_loader,
            seed=int(repo_config.training.seed),
            epoch_length=eval_epoch_length,
        )
        _print_eval_metrics("val_full", full_val_evaluator.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _epoch_summary(engine: Engine) -> None:
        print(
            f"epoch={engine.state.epoch} iteration={engine.state.iteration} "
            f"loss={engine.state.output['loss']:.6f} "
            f"lr={engine.state.output['lr']:.7f} "
            f"tokens={int(engine.state.output['tokens'])} "
            f"games_batch={int(engine.state.output['games'])} "
            f"games_epoch={int(getattr(engine.state, 'epoch_game_count', 0))}"
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
