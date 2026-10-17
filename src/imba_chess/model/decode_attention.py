"""Fused one-query decode attention for the search decoder (forward only).

For each (row, head) this computes, in one CUDA kernel with an online float32
softmax, what the reference computes over three key sources:

    score = q . k * scale + ps_w[h, clamp(key_pos - q_pos + M - 1, 0, 2M - 2)]

over the row's real prefix (t < length), its ancestor suffix (j < depth) read
straight from the K/V arena by chain index, and the new token itself. The
query position is length + depth; suffix key j sits at length + j.

Keys past a row's real length/depth are never read, so batch, prefix and arena
padding cannot change a real row's output. Scores stay in float32 (the unfused
path rounded them to the activation dtype first), which measured closer to a
float64 reference on real decode waves. CPU uses an equivalent float32 torch
implementation.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _decode_attention_kernel(
    Q, KN, VN, PK, PV, AK, AV, CHAIN, LENS, DEPTHS, PSW, OUT, KO, VO,
    sq_g, sq_h, skn_g, skn_h, svn_g, svn_h,
    spk_g, spk_h, spk_t, spv_g, spv_h, spv_t,
    sak_r, sak_h, sav_r, sav_h, sc_g, sps_h, so_g, so_h,
    sko_g, sko_h, svo_g, svo_h,
    SCALE: tl.constexpr, MAXPOS: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr, SILU: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    dv = tl.arange(0, DV)
    q = tl.load(Q + row * sq_g + h * sq_h + d).to(tl.float32)
    k_self = tl.load(KN + row * skn_g + h * skn_h + d)
    v_self = tl.load(VN + row * svn_g + h * svn_h + dv)
    if SILU:
        # Q, K and V arrive as raw projections. The new K/V are rounded to the
        # storage dtype exactly where the unfused graph rounded them, and that
        # stored value is what this token and its descendants attend to.
        q = _silu(q)
        k_self = _silu(k_self.to(tl.float32)).to(KO.dtype.element_ty)
        v_self = _silu(v_self.to(tl.float32)).to(VO.dtype.element_ty)
        tl.store(KO + row * sko_g + h * sko_h + d, k_self)
        tl.store(VO + row * svo_g + h * svo_h + dv, v_self)
    length = tl.load(LENS + row).to(tl.int32)
    depth = tl.load(DEPTHS + row).to(tl.int32)
    pos = length + depth
    psw = PSW + h * sps_h

    # The new token's own key (distance 0) seeds the running softmax.
    m = tl.sum(q * k_self.to(tl.float32), 0) * SCALE + tl.load(psw + MAXPOS - 1).to(tl.float32)
    l = tl.full((), 1.0, tl.float32)
    acc = v_self.to(tl.float32)

    pk = PK + row * spk_g + h * spk_h
    pv = PV + row * spv_g + h * spv_h
    for start in range(0, length, BLOCK_N):
        t = start + tl.arange(0, BLOCK_N)
        live = t < length
        k = tl.load(pk + t[:, None] * spk_t + d[None, :], mask=live[:, None], other=0.0)
        s = tl.sum(k.to(tl.float32) * q[None, :], 1) * SCALE
        rel = tl.minimum(tl.maximum(t - pos + MAXPOS - 1, 0), 2 * MAXPOS - 2)
        s += tl.load(psw + rel, mask=live, other=0.0).to(tl.float32)
        s = tl.where(live, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        v = tl.load(pv + t[:, None] * spv_t + dv[None, :], mask=live[:, None], other=0.0)
        l = l * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m = m_new

    j = tl.arange(0, BLOCK_S)
    live = j < depth
    rows = tl.load(CHAIN + row * sc_g + j, mask=live, other=0)
    k = tl.load(AK + rows[:, None] * sak_r + h * sak_h + d[None, :], mask=live[:, None], other=0.0)
    s = tl.sum(k.to(tl.float32) * q[None, :], 1) * SCALE
    rel = tl.minimum(tl.maximum(length + j - pos + MAXPOS - 1, 0), 2 * MAXPOS - 2)
    s += tl.load(psw + rel, mask=live, other=0.0).to(tl.float32)
    s = tl.where(live, s, -float("inf"))
    m_new = tl.maximum(m, tl.max(s, 0))
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new)
    v = tl.load(AV + rows[:, None] * sav_r + h * sav_h + dv[None, :], mask=live[:, None], other=0.0)
    l = l * alpha + tl.sum(p, 0)
    acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)

    tl.store(OUT + row * so_g + h * so_h + dv, (acc / l).to(OUT.dtype.element_ty))


def _launch(q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths,
            depths, ps_w, scale, max_pos, out, k_out, v_out, silu):
    g, heads, _, dim = q.shape
    wrap_triton(_decode_attention_kernel)[(g, heads)](
        q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths,
        depths, ps_w, out, k_out, v_out,
        q.stride(0), q.stride(1), k_new.stride(0), k_new.stride(1),
        v_new.stride(0), v_new.stride(1),
        prefix_k.stride(0), prefix_k.stride(1), prefix_k.stride(2),
        prefix_v.stride(0), prefix_v.stride(1), prefix_v.stride(2),
        arena_k.stride(0), arena_k.stride(1), arena_v.stride(0), arena_v.stride(1),
        chain.stride(0), ps_w.stride(0), out.stride(0), out.stride(1),
        k_out.stride(0), k_out.stride(1), v_out.stride(0), v_out.stride(1),
        SCALE=scale, MAXPOS=max_pos, D=dim, DV=v_new.size(-1), BLOCK_N=64,
        BLOCK_S=chain.size(1), SILU=silu, num_warps=2,
    )


@triton_op("imba_chess::decode_attention", mutates_args=())
def _decode_attention_cuda(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    arena_k: torch.Tensor,
    arena_v: torch.Tensor,
    chain: torch.Tensor,
    lengths: torch.Tensor,
    depths: torch.Tensor,
    ps_w: torch.Tensor,
    scale: float,
    max_pos: int,
) -> torch.Tensor:
    g, heads = q.shape[:2]
    out = q.new_empty((g, heads, 1, v_new.size(-1)))
    _launch(q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths,
            depths, ps_w, scale, max_pos, out, out, out, False)
    return out


@triton_op("imba_chess::decode_attention_silu", mutates_args=())
def _decode_attention_silu_cuda(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    arena_k: torch.Tensor,
    arena_v: torch.Tensor,
    chain: torch.Tensor,
    lengths: torch.Tensor,
    depths: torch.Tensor,
    ps_w: torch.Tensor,
    scale: float,
    max_pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g, heads, _, dim = q.shape
    dim_v = v_new.size(-1)
    out = q.new_empty((g, heads, 1, dim_v))
    k_out = k_new.new_empty((g, heads, 1, dim))
    v_out = v_new.new_empty((g, heads, 1, dim_v))
    _launch(q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths,
            depths, ps_w, scale, max_pos, out, k_out, v_out, True)
    return out, k_out, v_out


def _decode_attention_reference(
    q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths, depths,
    ps_w, scale, max_pos,
):
    # Read only up to the longest real prefix, as the kernel does: padding then
    # cannot change the result, whatever the buffers' capacity.
    p = int(lengths.max()) if lengths.numel() else 0
    prefix_k, prefix_v = prefix_k[:, :, :p], prefix_v[:, :, :p]
    g, s = q.size(0), chain.size(1)
    t = torch.arange(p, device=q.device)
    j = torch.arange(s, device=q.device)
    pos = lengths + depths
    keys = torch.cat(
        [prefix_k, arena_k[chain].permute(0, 2, 1, 3), k_new], 2
    ).float()
    values = torch.cat(
        [prefix_v, arena_v[chain].permute(0, 2, 1, 3), v_new], 2
    ).float()
    key_pos = torch.cat(
        [t.expand(g, p), lengths[:, None] + j, pos[:, None]], 1
    )
    live = torch.cat(
        [
            t < lengths[:, None],
            j < depths[:, None],
            torch.ones((g, 1), dtype=torch.bool, device=q.device),
        ],
        1,
    )
    rel = (key_pos - pos[:, None] + max_pos - 1).clamp(0, 2 * max_pos - 2)
    scores = (q.float() @ keys.transpose(-1, -2)).squeeze(2) * scale
    scores = scores + ps_w.float()[:, rel].permute(1, 0, 2)
    scores = scores.masked_fill(~live[:, None, :], -torch.inf)
    weights = torch.softmax(scores, -1)
    # Like the kernel, never read padding: it may hold stale or uninitialized data.
    values = values.masked_fill(~live[:, None, :, None], 0)
    return (weights[:, :, None, :] @ values).to(q.dtype)


def decode_attention(
    q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths, depths,
    ps_w, scale, max_pos,
):
    """q/k_new: [g, H, 1, D]; v_new: [g, H, 1, DV]; prefix_k/v: [g, H, P, D/DV];
    arena_k/v: [rows, H, D/DV]; chain: [g, S] arena rows of each row's
    ancestors (only the first depth are read); lengths/depths: [g] integers;
    ps_w: [H, 2 * max_pos - 1]. Returns [g, H, 1, DV] in q's dtype. Any
    strides are accepted for q, k_new and v_new."""
    impl = _decode_attention_cuda if q.is_cuda else _decode_attention_reference
    return impl(
        q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths,
        depths, ps_w, float(scale), int(max_pos),
    )


def decode_attention_silu(
    q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths, depths,
    ps_w, scale, max_pos,
):
    """decode_attention over raw (pre-SiLU) projections, typically strided views
    of the fused uvqk output. SiLU is applied in float32; the new K and V are
    rounded to their input dtype, used in that form, and returned for storage.
    Returns (attention [g, H, 1, DV], k [g, H, 1, D], v [g, H, 1, DV])."""
    if q.is_cuda:
        return _decode_attention_silu_cuda(
            q, k_new, v_new, prefix_k, prefix_v, arena_k, arena_v, chain, lengths,
            depths, ps_w, float(scale), int(max_pos),
        )
    k = torch.nn.functional.silu(k_new.float()).to(k_new.dtype)
    v = torch.nn.functional.silu(v_new.float()).to(v_new.dtype)
    out = _decode_attention_reference(
        torch.nn.functional.silu(q.float()), k, v, prefix_k, prefix_v, arena_k,
        arena_v, chain, lengths, depths, ps_w, float(scale), int(max_pos),
    )
    return out.to(q.dtype), k, v
