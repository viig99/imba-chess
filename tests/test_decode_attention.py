"""Fused decode attention: semantics, accuracy against float64, and exactness
(determinism, padding invariance, compiled/graph replay)."""

import pytest
import torch

from imba_chess.model.decode_attention import (
    _decode_attention_reference,
    decode_attention,
)
from imba_chess.model.hstu_attention import GroupedDecodeCache, _one_query_attention

M, H, D, CAP = 513, 4, 64, 32


def case(g, p, rows, seed, device, dtype):
    gen = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, device=device)
    lengths = torch.randint(0, p + 1, (g,), generator=gen, device=device)
    lengths[0] = p
    depths = torch.randint(0, CAP + 1, (g,), generator=gen, device=device)
    depths[-1] = 0
    chain = torch.randint(1, rows, (g, CAP), generator=gen, device=device)
    chain = torch.where(torch.arange(CAP, device=device) < depths[:, None], chain, 0)
    return dict(
        q=(r(g, H, 1, D) * 2).to(dtype), k_new=(r(g, H, 1, D) * 2).to(dtype),
        v_new=r(g, H, 1, D).to(dtype),
        prefix_k=(r(g, H, p, D) * 2).to(dtype), prefix_v=r(g, H, p, D).to(dtype),
        arena_k=(r(rows, H, D) * 2).to(dtype), arena_v=r(rows, H, D).to(dtype),
        chain=chain, lengths=lengths, depths=depths,
        ps_w=(r(H, 2 * M - 1) * 0.5).to(dtype),
    )


def run(c, fn=decode_attention):
    return fn(
        c["q"], c["k_new"], c["v_new"], c["prefix_k"], c["prefix_v"], c["arena_k"],
        c["arena_v"], c["chain"], c["lengths"], c["depths"], c["ps_w"], D**-0.5, M,
    )


def float64(c):
    """Per row, only the real keys, in float64."""
    out = []
    for i in range(c["q"].size(0)):
        n, depth = int(c["lengths"][i]), int(c["depths"][i])
        rows = c["chain"][i, :depth]
        keys = torch.cat([c["prefix_k"][i, :, :n], c["arena_k"][rows].transpose(0, 1),
                          c["k_new"][i]], 1).double()
        values = torch.cat([c["prefix_v"][i, :, :n], c["arena_v"][rows].transpose(0, 1),
                            c["v_new"][i]], 1).double()
        pos = n + depth
        key_pos = torch.cat([torch.arange(n), n + torch.arange(depth), torch.tensor([pos])])
        rel = (key_pos.to(keys.device) - pos + M - 1).clamp(0, 2 * M - 2)
        s = (c["q"][i].double() @ keys.transpose(-1, -2)).squeeze(1) * D**-0.5
        s = s + c["ps_w"].double()[:, rel]
        out.append(torch.softmax(s, -1)[:, None, :] @ values)
    return torch.stack(out)


def unfused(c):
    """The previous decoder path: gathered suffix plus _one_query_attention."""
    g, p = c["q"].size(0), c["prefix_k"].size(2)
    device = c["q"].device
    live = torch.arange(CAP, device=device) < c["depths"][:, None]
    gather = lambda a: torch.where(
        live[:, None, :, None], a[c["chain"]].permute(0, 2, 1, 3), 0
    )
    positions = c["lengths"] + c["depths"]
    prefix_pos = torch.arange(p, device=device)[None]
    branch_pos = torch.arange(CAP, device=device)[None]
    sp = torch.where(live, c["lengths"][:, None] + branch_pos, 0)
    cache = GroupedDecodeCache(
        [], [], [], [],
        (prefix_pos - positions[:, None] + M - 1).clamp(0, 2 * M - 2),
        (prefix_pos >= c["lengths"][:, None])[:, None, None, :],
        (sp - positions[:, None] + M - 1).clamp(0, 2 * M - 2),
        (branch_pos >= c["depths"][:, None])[:, None, None, :],
    )
    return _one_query_attention(
        c["q"], c["prefix_k"], c["prefix_v"], c["k_new"], c["v_new"],
        gather(c["arena_k"]), gather(c["arena_v"]), cache, c["ps_w"],
        c["ps_w"][:, M - 1].view(1, -1, 1, 1), D**-0.5,
    )


def padded(c, extra_rows=5, extra_prefix=37, extra_arena=100):
    """Garbage beyond every real row, prefix length and arena row."""
    def junk(x, n, dim):
        shape = list(x.shape)
        shape[dim] = n
        return torch.cat([x, (torch.randn(shape, device=x.device) * 50).to(x.dtype)], dim)

    out = dict(c)
    for key in ("prefix_k", "prefix_v"):
        out[key] = junk(junk(c[key], extra_prefix, 2), extra_rows, 0).contiguous()
    for key in ("q", "k_new", "v_new"):
        out[key] = junk(c[key], extra_rows, 0).contiguous()
    for key in ("arena_k", "arena_v"):
        out[key] = junk(c[key], extra_arena, 0).contiguous()
    out["lengths"] = torch.cat([c["lengths"], c["lengths"].new_full((extra_rows,), 3)])
    out["depths"] = torch.cat([c["depths"], c["depths"].new_full((extra_rows,), 2)])
    out["chain"] = torch.cat([c["chain"], c["chain"].new_full((extra_rows, CAP), 1)])
    return out


SHAPES = [(9, 100), (1, 7), (5, 0), (8, 300)]


@pytest.mark.parametrize("g,p", SHAPES)
def test_reference_matches_float64(g, p):
    c = case(g, p, rows=200, seed=g + p, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(run(c).double(), float64(c), atol=1e-5, rtol=1e-5)


def test_reference_never_reads_padding():
    c = case(9, 100, rows=200, seed=2, device="cpu", dtype=torch.float32)
    expected = run(c)
    nan = dict(c)
    for key in ("prefix_k", "prefix_v"):
        t = c[key].clone()
        padding = torch.arange(t.size(2))[None, :] >= c["lengths"][:, None]
        t[padding[:, None, :, None].expand_as(t)] = float("nan")
        nan[key] = t
    out = run(nan)
    assert not out.isnan().any()
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_reference_matches_the_unfused_path_in_float32():
    c = case(9, 100, rows=200, seed=1, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(run(c), unfused(c), atol=1e-5, rtol=1e-5)


def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


@pytest.mark.extended
@pytest.mark.parametrize("g,p", SHAPES)
def test_kernel_matches_reference_in_float32(g, p):
    cuda()
    c = case(g, p, rows=200, seed=g + p, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        run(c), run(c, _decode_attention_reference), atol=2e-5, rtol=2e-5
    )


@pytest.mark.extended
@pytest.mark.parametrize("g,p", SHAPES)
def test_kernel_is_at_least_as_accurate_as_the_unfused_bf16_path(g, p):
    """Adoption rule: in bf16 the fused kernel must be no further from float64
    than the path it replaced (it keeps scores in float32)."""
    cuda()
    c = case(g, p, rows=200, seed=g + p, device="cuda", dtype=torch.bfloat16)
    truth = float64(c)
    fused = (run(c).double() - truth).abs()
    old = (unfused(c).double() - truth).abs()
    assert fused.max() <= old.max()
    assert fused.mean() <= old.mean()


@pytest.mark.extended
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kernel_is_deterministic_and_padding_invariant(dtype):
    cuda()
    for g, p in SHAPES:
        c = case(g, p, rows=200, seed=g + p, device="cuda", dtype=dtype)
        out = run(c)
        assert torch.equal(out, run(c))
        assert torch.equal(run(padded(c))[:g], out)


@pytest.mark.extended
def test_compiled_and_graph_replayed_kernel_is_bit_identical():
    cuda()
    c = case(8, 300, rows=200, seed=5, device="cuda", dtype=torch.bfloat16)
    expected = run(c)
    compiled = torch.compile(lambda c: run(c) * 1, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(c), expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(c)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = compiled(c)
    for key in ("q", "prefix_k", "arena_v"):
        c[key].copy_(torch.randn_like(c[key]))
    graph.replay()
    torch.testing.assert_close(out, run(c), atol=0, rtol=0)


def raw_projection(c, seed):
    """q/k/v as strided [g, H, 1, d] views of one fused projection row, like the
    decoder's uvqk split (raw values before SiLU)."""
    g = c["q"].size(0)
    gen = torch.Generator(device=c["q"].device).manual_seed(seed)
    raw = torch.randn(g, 1, 4 * H * D, generator=gen, device=c["q"].device).to(c["q"].dtype)
    _, v, q, k = raw.split([H * D] * 4, dim=-1)
    return [t.unflatten(-1, (H, -1)).transpose(1, 2) for t in (q, k, v)]


def run_silu(c, q, k, v, fn=None):
    from imba_chess.model.decode_attention import decode_attention_silu
    return decode_attention_silu(
        q, k, v, c["prefix_k"], c["prefix_v"], c["arena_k"], c["arena_v"],
        c["chain"], c["lengths"], c["depths"], c["ps_w"], D**-0.5, M,
    )


def test_silu_reference_applies_silu_and_rounds_the_new_kv():
    c = case(9, 100, rows=200, seed=4, device="cpu", dtype=torch.bfloat16)
    q, k, v = raw_projection(c, 4)
    out, k_new, v_new = run_silu(c, q, k, v)
    torch.testing.assert_close(k_new, torch.nn.functional.silu(k.float()).to(k.dtype), atol=0, rtol=0)
    torch.testing.assert_close(v_new, torch.nn.functional.silu(v.float()).to(v.dtype), atol=0, rtol=0)
    expected = _decode_attention_reference(
        torch.nn.functional.silu(q.float()), k_new, v_new, c["prefix_k"], c["prefix_v"],
        c["arena_k"], c["arena_v"], c["chain"], c["lengths"], c["depths"], c["ps_w"], D**-0.5, M,
    )
    torch.testing.assert_close(out, expected.to(out.dtype), atol=0, rtol=0)


@pytest.mark.extended
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_silu_kernel_matches_reference_and_is_deterministic(dtype):
    cuda()
    for g, p in SHAPES:
        c = case(g, p, rows=200, seed=g + p, device="cuda", dtype=dtype)
        q, k, v = raw_projection(c, g + p)
        out, k_new, v_new = run_silu(c, q, k, v)
        # New K/V: SiLU in float32, rounded once (exp may differ by an ulp).
        torch.testing.assert_close(k_new, torch.nn.functional.silu(k.float()).to(dtype))
        torch.testing.assert_close(v_new, torch.nn.functional.silu(v.float()).to(dtype))
        expected = _decode_attention_reference(
            torch.nn.functional.silu(q.float()), k_new, v_new, c["prefix_k"], c["prefix_v"],
            c["arena_k"], c["arena_v"], c["chain"], c["lengths"], c["depths"], c["ps_w"],
            D**-0.5, M,
        )
        tol = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else {}
        torch.testing.assert_close(out, expected.to(dtype), **tol)
        again = run_silu(c, q, k, v)
        assert all(torch.equal(a, b) for a, b in zip((out, k_new, v_new), again))
