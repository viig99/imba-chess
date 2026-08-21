"""Historical Timer comparison for padding per-node suffix paths.

This benchmark produced the evidence for the former FBGEMM implementation.
Production now stores one K/V row per node in `_KVArena` and gathers ancestor
rows directly, so it no longer constructs the `paths` representation measured
here. Keep this harness for regression/history and for evaluating future true
values+offsets jagged designs; FBGEMM remains optional.

Variants produce an identical `[B, L, H, max_suffix, d]` padded batch:
  A current   per-node F.pad, then one torch.stack
  B prealloc  one new_zeros, then per-node slice copy
  C nested    torch.nested.nested_tensor -> to_padded_tensor
  D fbgemm    values+offsets -> jagged_to_padded_dense, when installed

`torch.utils.benchmark.Timer` synchronizes CUDA correctly.

Run: `.venv/bin/python scripts/bench_wave_suffixes.py`
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

L, H, D = 8, 12, 64
MAX_SUFFIX = 4


def make_paths(b: int, device: str):
    """Per-node parent path K of shape [L, H, depth, D], depth in 0..MAX_SUFFIX."""
    rng = random.Random(0)
    depths = [rng.randint(0, MAX_SUFFIX) for _ in range(b)]
    paths = [torch.randn(L, H, d, D, device=device) if d else None for d in depths]
    return depths, paths


def variant_current(depths, paths, device):
    zero = torch.zeros(L, H, MAX_SUFFIX, D, device=device)
    rows = []
    for d, p in zip(depths, paths):
        if p is None:
            rows.append(zero)
            continue
        pad = MAX_SUFFIX - d
        rows.append(F.pad(p, (0, 0, 0, pad)) if pad else p)
    return torch.stack(rows, dim=0)


def variant_prealloc(depths, paths, device):
    out = torch.zeros(len(depths), L, H, MAX_SUFFIX, D, device=device)
    for i, (d, p) in enumerate(zip(depths, paths)):
        if p is not None:
            out[i, :, :, :d] = p
    return out


def variant_nested(depths, paths, device):
    # Ragged dim must lead, so carry tokens first: [depth, L, H, D].
    zero = torch.zeros(0, L, H, D, device=device)
    comps = [(p.permute(2, 0, 1, 3) if p is not None else zero) for p in paths]
    nt = torch.nested.nested_tensor(list(comps), device=device)
    padded = nt.to_padded_tensor(0.0, output_size=(len(depths), MAX_SUFFIX, L, H, D))
    return padded.permute(0, 2, 3, 1, 4)


def _fbgemm_ready():
    try:
        import fbgemm_gpu  # noqa: F401

        return hasattr(torch.ops.fbgemm, "jagged_to_padded_dense")
    except Exception:  # noqa: BLE001
        return False


def stage_jagged(depths, paths, device):
    """Pre-stage the flat token-first buffer FBGEMM's jagged op consumes.

    `jagged_to_padded_dense` expects 2-D values plus cumulative offsets.
    This isolates the historical padding kernel. The former production path
    first built full-path tensors per node, so staging `values` still required
    a concatenation outside the timed operation.
    """
    # fbgemm wants values 2-D: [total_tokens, inner_dense_size].
    parts = [
        p.permute(2, 0, 1, 3).reshape(-1, L * H * D) for p in paths if p is not None
    ]
    values = (
        torch.cat(parts, dim=0) if parts else torch.zeros(0, L * H * D, device=device)
    )
    offs = torch.zeros(len(depths) + 1, dtype=torch.long, device=device)
    offs[1:] = torch.tensor(depths, dtype=torch.long, device=device).cumsum(0)
    return values, offs


def variant_fbgemm(values, offs, b, device):
    padded = torch.ops.fbgemm.jagged_to_padded_dense(
        values, [offs], [MAX_SUFFIX], 0.0
    )  # [B, MAX_SUFFIX, L*H*D]
    return padded.view(b, MAX_SUFFIX, L, H, D).permute(0, 2, 3, 1, 4)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for b in (330, 990):
        depths, paths = make_paths(b, device)
        print(
            f"\n=== B={b} nodes, device={device}, "
            f"[L={L}, H={H}, max_suffix={MAX_SUFFIX}, d={D}] ==="
        )

        ref = variant_current(depths, paths, device)
        for name, fn in (("prealloc", variant_prealloc), ("nested", variant_nested)):
            try:
                got = fn(depths, paths, device)
                ok = got.shape == ref.shape and torch.equal(got.contiguous(), ref)
                print(f"  {name:<9} shape {tuple(got.shape)}  equals current: {ok}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<9} FAILED: {type(exc).__name__}: {exc}")

        variants = [
            ("A current", variant_current),
            ("B prealloc", variant_prealloc),
            ("C nested", variant_nested),
        ]
        if _fbgemm_ready():
            values, offs = stage_jagged(depths, paths, device)
            got = variant_fbgemm(values, offs, b, device)
            ok = got.shape == ref.shape and torch.equal(got.contiguous(), ref)
            print(f"  {'fbgemm':<9} shape {tuple(got.shape)}  equals current: {ok}")
            variants.append(
                (
                    "D fbgemm",
                    lambda _d, _p, _dev: variant_fbgemm(values, offs, b, device),
                )
            )
        else:
            print("  fbgemm    NOT AVAILABLE")

        for name, fn in variants:
            try:
                t = Timer(
                    stmt="fn(depths, paths, device)",
                    globals={
                        "fn": fn,
                        "depths": depths,
                        "paths": paths,
                        "device": device,
                    },
                    num_threads=1,
                )
                m = t.blocked_autorange(min_run_time=1.0)
                print(
                    f"  {name:<11} median {m.median * 1e3:7.3f} ms   "
                    f"per node {m.median / b * 1e6:6.2f} us"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<11} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
