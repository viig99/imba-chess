#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import torch

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import (
    LichessDataset,
    build_event_dataloader,
    load_or_create_static_move_vocab,
)
from imba_chess.model import (
    HSTUChessModel,
    build_hstu_chess_config,
    create_batch_block_mask,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HSTU chess forward/backward benchmark on one jagged batch."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--inspect-batches", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto picks cuda if available else cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile model with torch.compile before timing.",
    )
    parser.add_argument("--max-tokens-per-batch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
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


def _make_dataset(config) -> LichessDataset:
    return LichessDataset(
        min_avg_elo=config.dataset.min_avg_elo,
        split=config.dataset.split,
        dataset_name=config.dataset.dataset_name,
        cache_dir=config.dataset.cache_dir,
        parquet_batch_size=config.dataset.parquet_batch_size,
        max_seq_len=config.dataset.max_seq_len,
        return_dataclasses=config.dataset.return_dataclasses,
        board_state_config=config.board_state,
    )


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _format_count(value: int | float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return str(int(value))


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    idx = int(math.ceil(p * len(values))) - 1
    idx = max(0, min(len(values) - 1, idx))
    return sorted(values)[idx]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _estimate_flops(
    model_config, seq_lens: Iterable[int], move_vocab_size: int
) -> dict[str, float]:
    seq_lens = list(int(x) for x in seq_lens)
    total_tokens = float(sum(seq_lens))
    sum_tri = float(sum((l * (l + 1)) // 2 for l in seq_lens))

    d = float(model_config.model_dim)
    h = float(model_config.num_heads)
    linear = float(model_config.linear_hidden_dim)
    attn = float(model_config.attention_dim)
    layers = float(model_config.num_layers)
    vocab = float(move_vocab_size)

    uvqk_out = h * (2 * linear + 2 * attn)
    flops_uvqk = 2.0 * total_tokens * d * uvqk_out
    flops_qk = 2.0 * h * attn * sum_tri
    flops_av = 2.0 * h * linear * sum_tri
    flops_o = 2.0 * total_tokens * (h * linear) * d
    flops_gate = total_tokens * h * linear
    flops_per_layer = flops_uvqk + flops_qk + flops_av + flops_o + flops_gate

    flops_head = 2.0 * total_tokens * d * vocab
    flops_fwd_total = (layers * flops_per_layer) + flops_head

    return {
        "tokens": total_tokens,
        "sum_tri": sum_tri,
        "per_layer": flops_per_layer,
        "head": flops_head,
        "fwd_total": flops_fwd_total,
        "fwd_bwd_approx": 3.0 * flops_fwd_total,
    }


def _print_model_params(model: HSTUChessModel) -> None:
    rows: list[tuple[str, int]] = [
        ("piece_embedding", _count_params(model.piece_embedding)),
        ("square_embedding", _count_params(model.square_embedding)),
        ("seq_token_embedding", _count_params(model.seq_token_embedding)),
        ("turn_embedding", _count_params(model.turn_embedding)),
        ("castle_embedding", _count_params(model.castle_embedding)),
        ("ep_embedding", _count_params(model.ep_embedding)),
        ("halfmove_embedding", _count_params(model.halfmove_embedding)),
        ("fullmove_embedding", _count_params(model.fullmove_embedding)),
        ("prev_move_embedding", _count_params(model.prev_move_embedding)),
        ("position_embedding", _count_params(model.position_embedding)),
    ]
    for idx, layer in enumerate(model.layers):
        rows.append((f"hstu_layer_{idx}", _count_params(layer)))
    rows.extend(
        [
            ("final_norm", _count_params(model.final_norm)),
            ("prediction_head", _count_params(model.prediction_head)),
        ]
    )

    total = _count_params(model)
    print("\nModel parameters:")
    for name, count in rows:
        print(f"  {name:20s} {_format_count(count):>10s} ({count})")
    print(f"  {'TOTAL':20s} {_format_count(total):>10s} ({total})")


def _get_cpu_rss_mb() -> float:
    import resource

    # Linux ru_maxrss is kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    args = parse_args()
    repo_config = load_repo_config(args.config)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    dataloader_cfg = repo_config.dataloader
    model_cfg_section = repo_config.model

    if args.max_tokens_per_batch is not None:
        dataloader_cfg = replace(
            dataloader_cfg, max_tokens_per_batch=args.max_tokens_per_batch
        )
    if args.num_workers is not None:
        dataloader_cfg = replace(dataloader_cfg, num_workers=args.num_workers)
    if args.num_layers is not None:
        model_cfg_section = replace(model_cfg_section, num_layers=args.num_layers)

    repo_config = replace(
        repo_config, dataloader=dataloader_cfg, model=model_cfg_section
    )

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path,
        include_unk=repo_config.vocab.include_unk,
    )
    move_vocab_size = len(move_vocab)

    dataset = _make_dataset(repo_config)
    loader = build_event_dataloader(
        lichess_dataset=dataset,
        config=repo_config,
        move_vocab=move_vocab,
    )

    print(
        f"Collecting {args.inspect_batches} dataloader batches for sequence stats...",
        flush=True,
    )
    iterator = iter(loader)
    benchmark_batch = None
    sampled_seq_lens: list[int] = []
    for batch_idx in range(args.inspect_batches):
        batch = next(iterator)
        if benchmark_batch is None:
            benchmark_batch = batch
            print(
                f"  first batch ready: games={batch['num_games']}, tokens={batch['total_tokens']}",
                flush=True,
            )
        if args.inspect_batches > 1:
            print(
                f"  inspected batch {batch_idx + 1}/{args.inspect_batches}",
                flush=True,
            )
        sampled_seq_lens.extend(int(x) for x in batch["seq_lens"].tolist())
        if (batch_idx + 1) == args.inspect_batches:
            print("Dataloader inspection complete.", flush=True)

    if benchmark_batch is None:
        raise RuntimeError("No batch available from dataloader.")

    model_cfg = build_hstu_chess_config(
        repo_config.model, move_vocab_size=move_vocab_size
    )
    base_model = HSTUChessModel(model_cfg).to(device)
    base_model.train()
    total_params = _count_params(base_model)
    print(f"Model initialized: {_format_count(total_params)} params ({total_params})")

    compile_requested = bool(args.compile)
    if compile_requested:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        run_model: torch.nn.Module = torch.compile(base_model, dynamic=True)
        compile_enabled = True
    else:
        run_model = base_model
        compile_enabled = False

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    def amp_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _run_benchmark(
        model_to_run: torch.nn.Module,
        batch_to_run: dict[str, object],
    ) -> tuple[list[float], float, float, float]:
        seq_offsets = batch_to_run["seq_offsets"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
        block_mask = create_batch_block_mask(
            seq_offsets=seq_offsets,
            total_tokens=int(batch_to_run["total_tokens"]),  # type: ignore[arg-type]
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device)
        else:
            mem_before = _get_cpu_rss_mb()

        for _ in range(args.warmup_steps):
            model_to_run.zero_grad(set_to_none=True)
            with amp_context():
                out = model_to_run(
                    batch_to_run, block_mask=block_mask, return_loss=True
                )
                out["loss"].backward()

        timings: list[float] = []
        for _ in tqdm(range(args.benchmark_steps), desc="Benchmarking", leave=False):
            _sync()
            start = time.perf_counter()
            model_to_run.zero_grad(set_to_none=True)
            with amp_context():
                out = model_to_run(
                    batch_to_run, block_mask=block_mask, return_loss=True
                )
                out["loss"].backward()
            _sync()
            timings.append((time.perf_counter() - start) * 1000.0)

        if device.type == "cuda":
            mem_after = torch.cuda.memory_allocated(device)
            mem_peak = torch.cuda.max_memory_allocated(device)
            mem_peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            mem_after = _get_cpu_rss_mb()
            mem_peak = mem_after
            mem_peak_reserved = mem_after
        return (
            timings,
            mem_before,
            mem_after,
            mem_peak_reserved if device.type != "cuda" else mem_peak,
        )

    latencies_ms, mem_before_alloc, mem_after_alloc, mem_peak_alloc = _run_benchmark(
        run_model, benchmark_batch
    )

    if device.type == "cuda":
        mem_peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        mem_peak_reserved = mem_peak_alloc

    batch_tokens = int(benchmark_batch["total_tokens"])
    batch_games = int(benchmark_batch["num_games"])
    max_seq = max(sampled_seq_lens) if sampled_seq_lens else 0
    p95_seq = _percentile(sampled_seq_lens, 0.95) if sampled_seq_lens else 0
    suggested_max_seq = _round_up(max(128, max_seq), 64)

    benchmark_seq_lens = [int(x) for x in benchmark_batch["seq_lens"].tolist()]  # type: ignore[union-attr]
    flops = _estimate_flops(repo_config.model, benchmark_seq_lens, move_vocab_size)
    mean_ms = statistics.fmean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p90_ms = sorted(latencies_ms)[max(0, math.ceil(0.9 * len(latencies_ms)) - 1)]
    tokens_per_s = (batch_tokens / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0
    approx_tflops = (
        (flops["fwd_bwd_approx"] / (mean_ms / 1000.0)) / 1e12 if mean_ms > 0 else 0.0
    )

    print("HSTU forward/backward benchmark")
    print(f"  device: {device}")
    print(f"  dtype: {dtype}")
    print(f"  torch.compile requested: {compile_requested}")
    print(f"  torch.compile active: {compile_enabled}")
    print(f"  move_vocab_size: {move_vocab_size}")
    print(f"  benchmark batch: games={batch_games}, tokens={batch_tokens}")
    print(
        f"  sampled seq lens: count={len(sampled_seq_lens)}, max={max_seq}, p95={p95_seq}"
    )
    print(
        f"  model max_position_embeddings: {repo_config.model.max_position_embeddings}"
    )
    print(f"  suggested max_position_embeddings (from sample): ~{suggested_max_seq}")
    if repo_config.model.max_position_embeddings < max_seq:
        print(
            "  WARNING: model max_position_embeddings is below observed max sequence length."
        )

    _print_model_params(base_model)

    print("\nApprox FLOPs (from benchmark batch):")
    print(f"  per_hstu_layer: {_format_count(flops['per_layer'])}")
    print(f"  prediction_head: {_format_count(flops['head'])}")
    print(f"  forward_total: {_format_count(flops['fwd_total'])}")
    print(f"  forward+backward_approx: {_format_count(flops['fwd_bwd_approx'])}")

    print("\nLatency:")
    print(f"  mean fwd+bwd: {mean_ms:.2f} ms")
    print(f"  p50 fwd+bwd: {p50_ms:.2f} ms")
    print(f"  p90 fwd+bwd: {p90_ms:.2f} ms")
    print(f"  throughput: {tokens_per_s:.1f} tokens/s")
    print(f"  approx throughput: {approx_tflops:.3f} TFLOP/s (fwd+bwd est)")

    print("\nMemory:")
    if device.type == "cuda":
        print(f"  allocated before benchmark: {mem_before_alloc / (1024**2):.1f} MB")
        print(f"  allocated after benchmark:  {mem_after_alloc / (1024**2):.1f} MB")
        print(f"  peak allocated:             {mem_peak_alloc / (1024**2):.1f} MB")
        print(f"  peak reserved:              {mem_peak_reserved / (1024**2):.1f} MB")
    else:
        print(f"  process RSS before benchmark: {mem_before_alloc:.1f} MB")
        print(f"  process RSS after benchmark:  {mem_after_alloc:.1f} MB")


if __name__ == "__main__":
    main()
