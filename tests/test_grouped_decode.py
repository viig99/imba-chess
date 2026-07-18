from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from tests.test_prefix_decode import _full_batch, _random_token_ids, _tiny_model

ATOL = 1e-5
RTOL = 1e-5


def _prefill(model, ids, T):
    prefix_ids = {key: value[:T] for key, value in ids.items()}
    with torch.no_grad():
        out = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)
    return out["kv_caches"]


def _decode_chain(model, ids, T, depth, prefix_kv):
    """Sequentially decode `depth` ancestor tokens after the prefix, growing
    the per-row suffix cache the same way test_prefix_decode.py's
    test_model_decode_matches_full_forward_over_depths does. Returns the
    accumulated (suffix_kv, suffix_positions, suffix_mask) after `depth`
    steps -- None-triples if depth == 0.
    """
    suffix_kv = None
    suffix_positions = None
    suffix_mask = None
    for step in range(depth):
        i = T + step
        step_ids = {key: value[i : i + 1] for key, value in ids.items()}
        with torch.no_grad():
            out = model.forward_decode(
                new_token_batch=step_ids,
                positions=torch.tensor([i]),
                prefix_kv=prefix_kv,
                suffix_kv=suffix_kv,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
        if suffix_kv is None:
            suffix_kv = [(k, v) for k, v in out["kv"]]
        else:
            suffix_kv = [
                (torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2))
                for (sk, sv), (k, v) in zip(suffix_kv, out["kv"])
            ]
        s = suffix_kv[0][0].size(2)
        suffix_positions = torch.arange(T, T + s).view(1, s)
        suffix_mask = torch.ones(1, s, dtype=torch.bool)
    return suffix_kv, suffix_positions, suffix_mask


def _pad_prefix_kv_grouped(prefix_kv_list, num_layers):
    """Stack per-game (unpadded) prefix kv [H, T_g, d] into per-layer
    [G, H, maxP, d], padding the token dim on the right with zeros."""
    lens = [kv[0][0].size(1) for kv in prefix_kv_list]
    max_p = max(lens)
    grouped = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for kv in prefix_kv_list:
            k, v = kv[layer_idx]
            pad = max_p - k.size(1)
            ks.append(F.pad(k, (0, 0, 0, pad)).unsqueeze(0))
            vs.append(F.pad(v, (0, 0, 0, pad)).unsqueeze(0))
        grouped.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
    return grouped, torch.tensor(lens, dtype=torch.long)


def _pad_row_suffix(model, row_suffix_kv, row_depth, s_max, num_layers):
    """Pad one row's real suffix_kv (length row_depth) on the right with
    zeros up to s_max, per layer. row_suffix_kv is None if row_depth == 0."""
    H = model.config.num_heads
    attn_dim = model.config.attention_dim
    linear_dim = model.config.linear_hidden_dim
    padded = []
    for layer_idx in range(num_layers):
        if row_depth == 0:
            k = torch.zeros(1, H, s_max, attn_dim)
            v = torch.zeros(1, H, s_max, linear_dim)
        else:
            k_real, v_real = row_suffix_kv[layer_idx]
            pad = s_max - row_depth
            k = F.pad(k_real, (0, 0, 0, pad))
            v = F.pad(v_real, (0, 0, 0, pad))
        padded.append((k, v))
    return padded


def _pad_row_suffix_meta(row_positions, row_mask, row_depth, s_max, base_pos):
    if row_depth == 0:
        positions = torch.zeros(1, s_max, dtype=torch.long)
        mask = torch.zeros(1, s_max, dtype=torch.bool)
    else:
        pad = s_max - row_depth
        if pad > 0:
            positions = torch.cat(
                [row_positions, torch.zeros(1, pad, dtype=torch.long)], dim=1
            )
            mask = torch.cat([row_mask, torch.zeros(1, pad, dtype=torch.bool)], dim=1)
        else:
            positions, mask = row_positions, row_mask
    return positions, mask


def _assemble_batch_suffix(model, rows, s_max, num_layers):
    """rows: list of dicts with keys depth, suffix_kv, suffix_positions,
    suffix_mask. Returns batch-level suffix_kv / positions / mask, or
    (None, None, None) if s_max == 0 (no row has any suffix)."""
    if s_max == 0:
        return None, None, None
    per_row_padded = [
        _pad_row_suffix(model, r["suffix_kv"], r["depth"], s_max, num_layers)
        for r in rows
    ]
    suffix_kv = []
    for layer_idx in range(num_layers):
        ks = [per_row_padded[i][layer_idx][0] for i in range(len(rows))]
        vs = [per_row_padded[i][layer_idx][1] for i in range(len(rows))]
        suffix_kv.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

    pos_mask = [
        _pad_row_suffix_meta(
            r["suffix_positions"], r["suffix_mask"], r["depth"], s_max, r["T"]
        )
        for r in rows
    ]
    suffix_positions = torch.cat([p for p, _ in pos_mask], dim=0)
    suffix_mask = torch.cat([m for _, m in pos_mask], dim=0)
    return suffix_kv, suffix_positions, suffix_mask


def test_forward_decode_grouped_matches_per_game_forward_decode():
    """G=3, different prefix lengths (padding exercised on prefix_kv_grouped),
    one row at depth 0, one row at depth >= 2 (suffix len 2), one at depth 1.
    """
    model = _tiny_model()
    num_layers = len(model.layers)
    T_list = [8, 5, 6]  # different lengths -> maxP=8 pads games 1 and 2
    depth_list = [0, 2, 1]

    games = []
    for g, (T, depth) in enumerate(zip(T_list, depth_list)):
        ids = _random_token_ids(T + depth + 1, seed=100 + g)
        prefix_kv = _prefill(model, ids, T)
        suffix_kv, suffix_positions, suffix_mask = _decode_chain(
            model, ids, T, depth, prefix_kv
        )
        final_pos = T + depth
        final_token = {
            key: value[final_pos : final_pos + 1] for key, value in ids.items()
        }
        games.append(
            {
                "T": T,
                "depth": depth,
                "prefix_kv": prefix_kv,
                "suffix_kv": suffix_kv,
                "suffix_positions": suffix_positions,
                "suffix_mask": suffix_mask,
                "final_pos": final_pos,
                "final_token": final_token,
            }
        )

    # Reference: per-game forward_decode using that game's own (unpadded)
    # prefix -- already proven == full forward by test_prefix_decode.py.
    references = []
    for game in games:
        with torch.no_grad():
            out = model.forward_decode(
                new_token_batch=game["final_token"],
                positions=torch.tensor([game["final_pos"]]),
                prefix_kv=game["prefix_kv"],
                suffix_kv=game["suffix_kv"],
                suffix_positions=game["suffix_positions"],
                suffix_mask=game["suffix_mask"],
            )
        references.append(out)

    # Merged grouped wave: one row per game, in game order.
    new_token_batch = {
        key: torch.cat([g["final_token"][key] for g in games], dim=0)
        for key in games[0]["final_token"]
    }
    positions = torch.tensor([g["final_pos"] for g in games])
    group_index = torch.tensor([0, 1, 2], dtype=torch.long)

    prefix_kv_grouped, prefix_lens = _pad_prefix_kv_grouped(
        [g["prefix_kv"] for g in games], num_layers
    )
    s_max = max(depth_list)
    suffix_kv, suffix_positions, suffix_mask = _assemble_batch_suffix(
        model, games, s_max, num_layers
    )

    with torch.no_grad():
        grouped_out = model.forward_decode_grouped(
            new_token_batch=new_token_batch,
            positions=positions,
            group_index=group_index,
            prefix_kv_grouped=prefix_kv_grouped,
            prefix_lens=prefix_lens,
            suffix_kv=suffix_kv,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
        )

    for i, ref in enumerate(references):
        torch.testing.assert_close(
            grouped_out["logits"][i], ref["logits"].squeeze(0), atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            grouped_out["value_logits"][i],
            ref["value_logits"].squeeze(0),
            atol=ATOL,
            rtol=RTOL,
        )
        for layer_idx in range(num_layers):
            gk, gv = grouped_out["kv"][layer_idx]
            rk, rv = ref["kv"][layer_idx]
            torch.testing.assert_close(gk[i : i + 1], rk, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(gv[i : i + 1], rv, atol=ATOL, rtol=RTOL)


def test_forward_decode_grouped_g1_matches_plain_forward_decode():
    """Degenerate case: G=1 grouped call must equal plain forward_decode."""
    model = _tiny_model()
    num_layers = len(model.layers)
    T, depth = 7, 2
    ids = _random_token_ids(T + depth + 1, seed=42)
    prefix_kv = _prefill(model, ids, T)
    suffix_kv, suffix_positions, suffix_mask = _decode_chain(
        model, ids, T, depth, prefix_kv
    )
    final_pos = T + depth
    final_token = {key: value[final_pos : final_pos + 1] for key, value in ids.items()}

    with torch.no_grad():
        plain_out = model.forward_decode(
            new_token_batch=final_token,
            positions=torch.tensor([final_pos]),
            prefix_kv=prefix_kv,
            suffix_kv=suffix_kv,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
        )

        prefix_kv_grouped = [(k.unsqueeze(0), v.unsqueeze(0)) for k, v in prefix_kv]
        grouped_out = model.forward_decode_grouped(
            new_token_batch=final_token,
            positions=torch.tensor([final_pos]),
            group_index=torch.tensor([0], dtype=torch.long),
            prefix_kv_grouped=prefix_kv_grouped,
            prefix_lens=torch.tensor([T], dtype=torch.long),
            suffix_kv=suffix_kv,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
        )

    torch.testing.assert_close(
        grouped_out["logits"], plain_out["logits"], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        grouped_out["value_logits"], plain_out["value_logits"], atol=ATOL, rtol=RTOL
    )
    for (gk, gv), (pk, pv) in zip(grouped_out["kv"], plain_out["kv"]):
        torch.testing.assert_close(gk, pk, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(gv, pv, atol=ATOL, rtol=RTOL)


def test_forward_decode_grouped_multiple_rows_per_group():
    """Two rows sharing the same group (same prefix, no suffix) plus one row
    in a second group -- exercises the row-bucketing logic itself (the part
    most at risk of mixing up which prefix a row reads)."""
    model = _tiny_model()
    num_layers = len(model.layers)
    T0, T1 = 6, 4
    ids0 = _random_token_ids(T0 + 2, seed=200)  # two depth-0 candidates
    ids1 = _random_token_ids(T1 + 1, seed=201)

    prefix_kv0 = _prefill(model, ids0, T0)
    prefix_kv1 = _prefill(model, ids1, T1)

    row_a = {key: value[T0 : T0 + 1] for key, value in ids0.items()}
    row_b = {key: value[T0 + 1 : T0 + 2] for key, value in ids0.items()}
    row_c = {key: value[T1 : T1 + 1] for key, value in ids1.items()}

    references = []
    for row in (row_a, row_b):
        with torch.no_grad():
            references.append(
                model.forward_decode(
                    new_token_batch=row,
                    positions=torch.tensor([T0]),
                    prefix_kv=prefix_kv0,
                )
            )
    with torch.no_grad():
        references.append(
            model.forward_decode(
                new_token_batch=row_c,
                positions=torch.tensor([T1]),
                prefix_kv=prefix_kv1,
            )
        )

    new_token_batch = {
        key: torch.cat([row_a[key], row_b[key], row_c[key]], dim=0) for key in row_a
    }
    positions = torch.tensor([T0, T0, T1])
    group_index = torch.tensor([0, 0, 1], dtype=torch.long)
    prefix_kv_grouped, prefix_lens = _pad_prefix_kv_grouped(
        [prefix_kv0, prefix_kv1], num_layers
    )

    with torch.no_grad():
        grouped_out = model.forward_decode_grouped(
            new_token_batch=new_token_batch,
            positions=positions,
            group_index=group_index,
            prefix_kv_grouped=prefix_kv_grouped,
            prefix_lens=prefix_lens,
        )

    for i, ref in enumerate(references):
        torch.testing.assert_close(
            grouped_out["logits"][i], ref["logits"].squeeze(0), atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            grouped_out["value_logits"][i],
            ref["value_logits"].squeeze(0),
            atol=ATOL,
            rtol=RTOL,
        )
        for layer_idx in range(num_layers):
            gk, gv = grouped_out["kv"][layer_idx]
            rk, rv = ref["kv"][layer_idx]
            torch.testing.assert_close(gk[i : i + 1], rk, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(gv[i : i + 1], rv, atol=ATOL, rtol=RTOL)
