"""Compare attention implementations on small chess inference workloads.

Run on the GPU host (no checkpoint needed):
  .venv/bin/python scripts/bench_attention.py --output artifacts/attention_bench.json

CPU validation only, without reporting substitute CPU performance:
  .venv/bin/python scripts/bench_attention.py --device cpu --check-only \
      --lengths 32 --root-games 1 2 --decode-games 1 3 --heads 4 --head-dim 16

Attention-only: projections, board encoding, search control, and Flex BlockMask
construction are excluded. Per-layer relative-bias lookup is included. Root
histories are packed exactly as in search; leaves have mixed prefix/depth lengths.
The custom leaf path calls the production helper; the custom root is a dense
reference implementation, not a production root path. "prepacked" variants omit
KV assembly; the other SDPA/Flex variants concatenate prefix/branch/self per call.
Compilation/first-call time is separate from synchronized, warmed timings.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.benchmark import Timer

from imba_chess.model.hstu_attention import GroupedDecodeCache, _one_query_attention


@dataclass
class Workload:
    description: dict
    functions: dict
    reference: torch.Tensor
    sdpa_probe: tuple


def _reference(q, k, v, ps_w, rel, keep, scale):
    # Independent float64 CPU reference, using the quantized input values for
    # BF16 cases. No candidate implementation serves as the correctness oracle.
    q, k, v = (x.cpu().double() for x in (q, k, v))
    bias = ps_w.cpu().double()[torch.arange(q.size(1))[None, :, None, None], rel.cpu()]
    scores = (q @ k.transpose(-2, -1)) * scale + bias
    return torch.softmax(scores.masked_fill(~keep.cpu(), -torch.inf), -1) @ v


def make_workload(phase, games, length, args, device, dtype):
    heads, dim, max_position = args.heads, args.head_dim, 513
    scale = dim ** -0.5
    # SiLU projections match the trunk's positive-skewed Q/K/V distribution.
    def sample(shape):
        return F.silu(torch.randn(shape, device=device, dtype=torch.float32)).to(dtype)

    ps_w = (torch.randn(heads, 2 * max_position - 1, device=device) * 0.02).to(dtype)
    if phase == "root":
        tokens = games * length
        q, k, v = (sample((1, heads, tokens, dim)) for _ in range(3))
        pos = torch.arange(tokens, device=device)
        allowed = (pos[:, None] >= pos[None, :]) & (
            pos[:, None] // length == pos[None, :] // length
        )
        keep = allowed[None, None]
        rel = (pos[None, :] - pos[:, None] + max_position - 1).clamp(
            0, 2 * max_position - 2
        )[None, None]

        def mask_mod(b, h, qi, ki):
            return (qi >= ki) & (qi // length == ki // length)

        def score_mod(score, b, h, qi, ki):
            index = (ki - qi + max_position - 1).clamp(0, 2 * max_position - 2)
            return score + ps_w[h, index].to(score.dtype)

        def additive():
            return ps_w[:, rel[0, 0]][None].masked_fill(~keep, -torch.inf)

        def custom():
            scores = (q @ k.transpose(-2, -1)) * scale + additive()
            return torch.softmax(scores.float(), -1).to(dtype) @ v

        def sdpa():
            return F.scaled_dot_product_attention(q, k, v, attn_mask=additive())

        block = create_block_mask(
            mask_mod, B=1, H=None, Q_LEN=tokens, KV_LEN=tokens, device=device
        )

        def flex():
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block)

        functions = dict(custom_dense_reference=custom, sdpa=sdpa, flex=flex)
        description = dict(phase=phase, games=games, history=length, q_tokens=tokens,
                           kv_tokens=tokens)
    else:
        suffix = args.suffix
        rows = torch.arange(games, device=device)
        lengths = length - rows % min(length, 4)
        depths = suffix - rows % min(suffix + 1, 4)
        q_positions = lengths + depths
        prefix_pos = torch.arange(length, device=device)[None, :]
        suffix_pos = lengths[:, None] + torch.arange(suffix, device=device)[None, :]
        prefix_rel = (prefix_pos - q_positions[:, None] + max_position - 1).clamp(
            0, 2 * max_position - 2
        )
        suffix_rel = (suffix_pos - q_positions[:, None] + max_position - 1).clamp(
            0, 2 * max_position - 2
        )
        prefix_fill = (prefix_pos >= lengths[:, None])[:, None, None, :]
        suffix_fill = (torch.arange(suffix, device=device)[None, :] >= depths[:, None])[
            :, None, None, :
        ]
        cache = GroupedDecodeCache(
            [], [], [], [], prefix_rel, prefix_fill, suffix_rel, suffix_fill
        )
        q, new_k, new_v = (sample((games, heads, 1, dim)) for _ in range(3))
        prefix_k, prefix_v = (sample((games, heads, length, dim)) for _ in range(2))
        suffix_k, suffix_v = (sample((games, heads, suffix, dim)) for _ in range(2))
        # Prepacked variants are a kernel-only lower bound. Split variants also
        # pay for assembling KV; production custom does not concatenate them.
        k = torch.cat((prefix_k, suffix_k, new_k), -2)
        v = torch.cat((prefix_v, suffix_v, new_v), -2)
        rel = torch.cat((prefix_rel, suffix_rel,
                         torch.full((games, 1), max_position - 1, device=device)), -1)
        rel = rel[:, None, None, :]
        keep = torch.cat((~prefix_fill, ~suffix_fill,
                          torch.ones(games, 1, 1, 1, device=device, dtype=torch.bool)), -1)
        key_count = length + suffix + 1

        def mask_mod(b, h, qi, ki):
            # Flex may evaluate callbacks on padded tile indices.
            return (ki < key_count) & keep[b, 0, 0, ki.clamp(max=key_count - 1)]

        def score_mod(score, b, h, qi, ki):
            index = rel[b, 0, 0, ki.clamp(max=key_count - 1)]
            return score + ps_w[h, index].to(score.dtype)

        def additive():
            return ps_w[:, rel[:, 0, 0]].permute(1, 0, 2)[:, :, None].masked_fill(
                ~keep, -torch.inf
            )

        def custom():
            return _one_query_attention(
                q, prefix_k, prefix_v, new_k, new_v, suffix_k, suffix_v, cache,
                ps_w, ps_w[:, max_position - 1].view(1, -1, 1, 1), scale,
            )

        def sdpa():
            return F.scaled_dot_product_attention(
                q, torch.cat((prefix_k, suffix_k, new_k), -2),
                torch.cat((prefix_v, suffix_v, new_v), -2), attn_mask=additive(),
            )

        def sdpa_prepacked():
            return F.scaled_dot_product_attention(q, k, v, attn_mask=additive())

        block = create_block_mask(
            mask_mod, B=games, H=None, Q_LEN=1, KV_LEN=length + suffix + 1, device=device
        )

        def flex():
            return flex_attention(
                q, torch.cat((prefix_k, suffix_k, new_k), -2),
                torch.cat((prefix_v, suffix_v, new_v), -2),
                score_mod=score_mod, block_mask=block,
            )

        def flex_prepacked():
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block)

        functions = dict(custom=custom, sdpa=sdpa, flex=flex,
                         sdpa_prepacked=sdpa_prepacked, flex_prepacked=flex_prepacked)
        description = dict(phase=phase, games=games, history=length, q_tokens=1,
                           kv_tokens=length + suffix + 1, suffix_capacity=suffix,
                           mixed_lengths=games > 1)

    return Workload(description, functions, _reference(q, k, v, ps_w, rel, keep, scale),
                    (q, k, v, additive()))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def summary(report):
    lines = ["| Phase | Games | History | Implementation | Mode | Median µs | Max error |",
             "|---|---:|---:|---|---|---:|---:|"]
    for row in report["cases"]:
        latency = f"{row['median_us']:.2f}" if "median_us" in row else row["status"]
        error = f"{row['max_abs_error']:.2g}" if "max_abs_error" in row else "—"
        lines.append(f"| {row['phase']} | {row['games']} | {row['history']} | "
                     f"{row['implementation']} | {row['mode']} | {latency} | {error} |")
    return "\n".join(lines) + "\n"


def benchmark_case(workload, args, device, dtype):
    rows, prepared = [], {}
    # CPU mode is explicitly correctness-only. In particular it does not make
    # claims about whether Inductor's CUDA Flex/SDPA kernels compile or run.
    modes = ["eager"] if args.check_only else args.modes
    for name, eager in workload.functions.items():
        for mode in modes:
            row = dict(**workload.description, implementation=name, mode=mode,
                       dtype=args.dtype, status="ok")
            rows.append(row)
            start = time.perf_counter()
            try:
                fn = torch.compile(eager, fullgraph=True, dynamic=True) if mode == "compiled" else eager
                synchronize(device)
                start = time.perf_counter()
                out = fn()
                synchronize(device)
                row["first_call_seconds"] = time.perf_counter() - start
                actual = out.cpu().double()
                error = actual - workload.reference
                row["max_abs_error"] = error.abs().max().item()
                row["rms_error"] = error.square().mean().sqrt().item()
                atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (0.02, 0.03)
                torch.testing.assert_close(actual, workload.reference, atol=atol, rtol=rtol)
                del out
                if args.check_only:
                    print(json.dumps(row), flush=True)
                    continue
                for _ in range(5):
                    fn()
                synchronize(device)
                before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                fn()
                synchronize(device)
                row["peak_extra_allocated_bytes"] = max(
                    0, torch.cuda.max_memory_allocated(device) - before
                )
                prepared[(name, mode)] = (fn, row, [])
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                           elapsed_seconds=time.perf_counter() - start)
                print(json.dumps(row), flush=True)

    # Randomly interleave warmed measurements to reduce ordering/clock bias.
    rng = random.Random(args.seed)
    for _ in range(args.repeats):
        order = list(prepared)
        rng.shuffle(order)
        for key in order:
            fn, row, trials = prepared[key]
            measurement = Timer(stmt="fn()", globals={"fn": fn}, num_threads=1).blocked_autorange(
                min_run_time=args.seconds
            )
            trials.append(dict(median_us=measurement.median * 1e6,
                               iqr_us=measurement.iqr * 1e6))
    for fn, row, trials in prepared.values():
        row["trials"] = trials
        row["median_us"] = statistics.median(t["median_us"] for t in trials)
        # Record actual SDPA dispatch outside the latency measurement. Compiled
        # graphs may hide ATen operator names; an empty list means unknown.
        if row["implementation"].startswith("sdpa"):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                fn()
                synchronize(device)
            row["attention_dispatch_ops"] = [
                e.key for e in prof.key_averages() if "scaled_dot_product" in e.key
            ]
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--root-games", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--decode-games", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--phases", nargs="+", choices=["root", "decode"], default=["root", "decode"])
    parser.add_argument("--modes", nargs="+", choices=["eager", "compiled"], default=["eager", "compiled"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--suffix", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=0.2, help="Minimum time per warmed trial.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/attention_bench.json"))
    args = parser.parse_args()
    if any(n < 1 for n in args.lengths + args.root_games + args.decode_games):
        parser.error("lengths and game counts must be positive")
    if not 0 <= args.suffix <= 32 or max(args.lengths) + args.suffix + 1 > 513:
        parser.error("suffix must be 0..32 and history + suffix + self must fit 513 positions")
    if min(args.heads, args.head_dim, args.repeats, args.seconds) <= 0:
        parser.error("heads, head dimension, repeats, and seconds must be positive")
    if args.device == "cpu" and not args.check_only:
        parser.error("CPU is supported only with --check-only; GPU performance requires CUDA")
    if "root" in args.phases and max(args.root_games) * max(args.lengths) > 1024:
        parser.error("packed roots are limited to 1024 total tokens, matching production defaults")
    if args.heads * 1024 ** 2 * 8 > 256 * 2 ** 20:
        parser.error("head count would exceed the bounded reference-memory budget")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not exposed to this session. Run this command in the GPU host terminal.")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    device=str(device), tf32=False, threads=1, arguments=vars(args).copy(),
                    scope="attention only; excludes projections, search and BlockMask construction",
                    reference="float64 CPU; same input tensor values", compiled_dynamic=True,
                    first_call="includes lazy compilation/cache loading and execution; not timed trials")
    metadata["arguments"]["output"] = str(args.output)
    if device.type == "cuda":
        metadata.update(gpu=torch.cuda.get_device_name(device),
                        capability=list(torch.cuda.get_device_capability(device)),
                        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    report = dict(metadata=metadata, cases=[])
    save_report(args.output, report)
    print(json.dumps(metadata), flush=True)
    with torch.inference_mode():
        for phase in args.phases:
            counts = args.root_games if phase == "root" else args.decode_games
            for games in counts:
                for length in args.lengths:
                    workload = make_workload(phase, games, length, args, device, dtype)
                    if device.type == "cuda":
                        q, k, v, bias = workload.sdpa_probe
                        params = torch.backends.cuda.SDPAParams(q, k, v, bias, 0.0, False, False)
                        workload.description["flash_sdpa_eligible"] = torch.backends.cuda.can_use_flash_attention(params)
                    rows = benchmark_case(workload, args, device, dtype)
                    report["cases"].extend(rows)
                    save_report(args.output, report)
                    del workload
                    # Cases are intentionally independent: reset compilation
                    # guards so a growing sweep cannot hit a recompile limit.
                    torch.compiler.reset()
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    if any(row["status"] != "ok" for row in report["cases"]):
        raise SystemExit(f"Some variants failed; see {args.output} (no failed variant is ranked).")
    if not args.check_only:
        table = summary(report)
        args.output.with_suffix(".md").write_text(
            f"GPU: {metadata['gpu']}; PyTorch: {metadata['torch_version']}; "
            f"dtype: {args.dtype}; TF32 disabled.\n\n"
            "Attention-only warmed wall latency; includes per-layer bias lookup. "
            "Split SDPA/Flex also include KV concatenation; prepacked variants omit it. "
            "Projections, BlockMask construction, search and compilation are excluded.\n\n"
            + table
        )
        print(table, flush=True)
    print(f"Saved {len(report['cases'])} checks/measurements to {args.output}", flush=True)


if __name__ == "__main__":
    main()
