"""Merged root-eval / decode-wave executors for the cross-game batch scheduler.

scripts/generate_search_rollouts.py's BatchScheduler drives up to G games
concurrently; each tick, it hands this module's executors the tick's pending
per-game GPU work (root forwards, or search decode waves) and expects one
merged model call back. `_merge_root_batches`/`_split_root_output` ragged-
concatenate G single-game root batches into one `_forward_model` call and
slice the result back apart; `_merge_decode_requests`/`_split_decode_output`
do the same for `forward_decode_grouped` decode waves. Both merge paths fall
back to a trivial single-game passthrough (`len(payloads) == 1`) that is
byte-identical to evaluating each game alone, which is what makes
`--concurrent-games 1` an exact behavioral match to the pre-scheduler code.

This module is script-independent by design: it knows nothing about the
rollout-sampling loop, the sampled-ply logic, or how rows get written to
parquet -- only how to merge/split GPU payloads for the scheduler. Timing
stats are injected via the `stats` parameter (never a script-level global),
so any caller can plug in its own accumulator (or none, via `stats=None`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from imba_chess.eval import search
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _autocast_context,
    _forward_model,
)


class TimingStatsLike(Protocol):
    """Structural type for the timing-stats accumulator these executors
    write into -- matches scripts/generate_search_rollouts.py's
    `_TimingStats` without importing it (that script imports this module,
    so importing back would cycle); any object with these four mutable
    numeric attributes works, including `stats=None` at each call site's
    own None-check."""

    root_eval: float
    search_gpu: float
    search_eval_calls: int
    search_eval_items: int


def _merge_root_batches(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate G per-game single-sequence root batches into one ragged
    multi-document batch, exactly generalizing _SequenceHistory's own
    single-game seq_offsets=[0, T] / num_games=1 batch shape.

    The model's batch format is ragged-native (jagged token dim + seq_offsets
    driving both the position embedding restart and create_batch_block_mask's
    per-document mask -- there is no padding anywhere in HSTUChessModel.
    forward/_build_content/PositionEmbedding), so concatenation is the exact
    generalization of the existing per-game batch, not an approximation of
    it: for G=1 this returns payloads[0] unchanged, and the merged forward
    pass computes bit-for-bit the same per-document attention a separate
    per-game call would (the doc mask already prevents any cross-game
    attention within one call).
    """
    if len(payloads) == 1:
        return payloads[0]
    seq_lens = torch.cat([p["seq_lens"] for p in payloads])
    seq_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(seq_lens, dim=0)]
    )
    merged: dict[str, Any] = {
        "game_id": [gid for p in payloads for gid in p["game_id"]],
        "game_result_white": torch.cat([p["game_result_white"] for p in payloads]),
        "num_games": len(payloads),
        "total_tokens": int(seq_offsets[-1].item()),
        "seq_lens": seq_lens,
        "seq_offsets": seq_offsets,
    }
    for key in (
        "piece_ids",
        "seq_token_id",
        "turn_id",
        "castle_id",
        "ep_file_id",
        "halfmove_bucket_id",
        "fullmove_bucket_id",
        "prev_move_id",
        "target_move_id",
        "played_by_elo",
    ):
        merged[key] = torch.cat([p[key] for p in payloads], dim=0)
    return merged


def _split_root_output(
    output: dict[str, Any], payloads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Slice one merged root-forward output back into G per-game-shaped
    dicts, each identical in shape to what _forward_model would have
    returned for that game alone (kv_caches sliced to that game's own
    token span, becoming a fresh 0-indexed [H, T_g, d] prefix per layer)."""
    if len(payloads) == 1:
        return [output]
    results: list[dict[str, Any]] = []
    start = 0
    for payload in payloads:
        end = start + int(payload["total_tokens"])
        results.append(
            {
                "logits": output["logits"][start:end],
                "value_logits": output["value_logits"][start:end],
                "kv_caches": [
                    (k[:, start:end, :], v[:, start:end, :])
                    for k, v in output["kv_caches"]
                ],
            }
        )
        start = end
    return results


def _make_root_eval_executor(*, model, device, dtype, stats: "TimingStatsLike | None"):
    def executor(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = _merge_root_batches(payloads)
        t0 = time.perf_counter()
        output = _forward_model(
            model=model, batch=merged, device=device, dtype=dtype, return_kv=True
        )
        if stats is not None:
            stats.root_eval += time.perf_counter() - t0
        return _split_root_output(output, payloads)

    return executor


@dataclass
class _MergedDecodeRequest:
    new_token_batch: dict[str, Any]
    positions: torch.Tensor
    group_index: torch.Tensor
    prefix_kv_grouped: list[tuple[torch.Tensor, torch.Tensor]]
    prefix_lens: torch.Tensor
    prefix_lens_list: list[int]
    suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None
    suffix_positions: torch.Tensor | None
    suffix_mask: torch.Tensor | None


def _merge_decode_requests(requests: list[Any]) -> _MergedDecodeRequest:
    """Build forward_decode_grouped's inputs from G games' _DecodeRequest.

    prefix_kv_grouped pads every game's [H, T_g, d] prefix to [G, H, maxP, d]
    on the token dim (prefix_lens carries each game's real length, so the
    grouped attention layer slices back to the unpadded prefix per game --
    same convention _DecodeRequest.prefix_kv already used, just stacked).
    prefix_lens_list is the same G lengths as a plain host-side list[int]
    (requests already carry prefix_len as a Python int -- no tensor round
    trip needed to get it): forward_decode_grouped threads it straight to
    the per-group decode loop so that loop never has to .item() a
    device-resident tensor once per group per layer (see
    SequentialTransductionUnitJagged.forward_decode_grouped's docstring).
    prefix_lens (the tensor) is kept alongside for the model's own
    num_groups / group_index boundary validation.
    Suffix tensors are per-row [B_g, H, s_g, d]; games with no suffix at all
    (root-adjacent nodes) or a shorter suffix than the tick's max get
    zero-padded rows with an all-False mask, mirroring how
    CachedPositionEvaluator._wave_suffixes already pads within one game's
    wave -- here applied across games instead of across nodes.
    """
    num_layers = len(requests[0].prefix_kv)
    group_index = torch.cat(
        [
            torch.full((len(req.nodes),), g, dtype=torch.long)
            for g, req in enumerate(requests)
        ]
    )
    new_token_batch = {
        key: torch.cat([req.new_token_batch[key] for req in requests], dim=0)
        for key in requests[0].new_token_batch
    }
    positions = torch.cat([req.positions for req in requests])
    prefix_lens_list = [req.prefix_len for req in requests]
    prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.long)

    max_prefix = max(req.prefix_len for req in requests)
    prefix_kv_grouped: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(num_layers):
        ks, vs = [], []
        for req in requests:
            k, v = req.prefix_kv[layer]
            pad = max_prefix - k.size(1)
            ks.append(F.pad(k, (0, 0, 0, pad)) if pad else k)
            vs.append(F.pad(v, (0, 0, 0, pad)) if pad else v)
        prefix_kv_grouped.append((torch.stack(ks, dim=0), torch.stack(vs, dim=0)))

    max_suffix = max(
        (req.suffix_kv[0][0].size(2) if req.suffix_kv is not None else 0)
        for req in requests
    )
    suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None
    if max_suffix == 0:
        suffix_kv = suffix_positions = suffix_mask = None
    else:
        suffix_k_rows: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        suffix_v_rows: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        suffix_positions_rows: list[torch.Tensor] = []
        suffix_mask_rows: list[torch.Tensor] = []
        for req in requests:
            wave_size = len(req.nodes)
            if req.suffix_kv is None:
                # Fabricated rows must share the request's own device (the
                # model's device, e.g. cuda:0 -- read off prefix_kv, which is
                # always device-resident kv_caches from the model, never a
                # bare CPU tensor). torch.zeros/torch.full/torch.arange with
                # no explicit device argument silently land on the *default*
                # device (cpu unless torch.set_default_device was called),
                # which crashes torch.cat below whenever another request in
                # the same tick DOES have real (device-resident) suffix rows
                # -- exactly the --concurrent-games > 1 GPU crash this guards
                # against. ref_k.new_zeros(...) already inherits its device
                # from ref_k correctly; only the two device-less
                # torch.zeros(...) calls below needed the explicit device=.
                request_device = req.prefix_kv[0][0].device
                for layer in range(num_layers):
                    ref_k, ref_v = req.prefix_kv[layer]
                    suffix_k_rows[layer].append(
                        ref_k.new_zeros((wave_size, ref_k.size(0), max_suffix, ref_k.size(-1)))
                    )
                    suffix_v_rows[layer].append(
                        ref_v.new_zeros((wave_size, ref_v.size(0), max_suffix, ref_v.size(-1)))
                    )
                suffix_positions_rows.append(
                    torch.zeros(
                        (wave_size, max_suffix),
                        dtype=req.positions.dtype,
                        device=request_device,
                    )
                )
                suffix_mask_rows.append(
                    torch.zeros(
                        (wave_size, max_suffix), dtype=torch.bool, device=request_device
                    )
                )
            else:
                s_g = req.suffix_kv[0][0].size(2)
                pad = max_suffix - s_g
                for layer in range(num_layers):
                    k, v = req.suffix_kv[layer]
                    suffix_k_rows[layer].append(F.pad(k, (0, 0, 0, pad)) if pad else k)
                    suffix_v_rows[layer].append(F.pad(v, (0, 0, 0, pad)) if pad else v)
                suffix_positions_rows.append(
                    F.pad(req.suffix_positions, (0, pad)) if pad else req.suffix_positions
                )
                suffix_mask_rows.append(
                    F.pad(req.suffix_mask, (0, pad), value=False) if pad else req.suffix_mask
                )
        suffix_kv = [
            (torch.cat(suffix_k_rows[layer], dim=0), torch.cat(suffix_v_rows[layer], dim=0))
            for layer in range(num_layers)
        ]
        suffix_positions = torch.cat(suffix_positions_rows, dim=0)
        suffix_mask = torch.cat(suffix_mask_rows, dim=0)

    return _MergedDecodeRequest(
        new_token_batch=new_token_batch,
        positions=positions,
        group_index=group_index,
        prefix_kv_grouped=prefix_kv_grouped,
        prefix_lens=prefix_lens,
        prefix_lens_list=prefix_lens_list,
        suffix_kv=suffix_kv,
        suffix_positions=suffix_positions,
        suffix_mask=suffix_mask,
    )


def _split_decode_output(out: dict[str, Any], lengths: list[int]) -> list[dict[str, Any]]:
    """Slice forward_decode_grouped's full-batch output back into G
    per-game-shaped dicts, in original per-game row order (forward_decode_
    grouped guarantees output rows are in original row order, and rows were
    concatenated game-by-game in _merge_decode_requests, so a straight
    cumulative-length split recovers each game's own rows unpermuted)."""
    results: list[dict[str, Any]] = []
    start = 0
    for length in lengths:
        end = start + length
        entry: dict[str, Any] = {
            "logits": out["logits"][start:end],
            "kv": [(k[start:end], v[start:end]) for k, v in out["kv"]],
        }
        if "value_logits" in out:
            entry["value_logits"] = out["value_logits"][start:end]
        results.append(entry)
        start = end
    return results


def _make_decode_wave_executor(*, model, device, dtype, stats: "TimingStatsLike | None"):
    def executor(
        payloads: list[tuple[CachedPositionEvaluator, list]]
    ) -> list[list[search.PositionEval]]:
        if len(payloads) == 1:
            # Single game in this tick's decode_wave batch: the existing
            # single-prefix evaluate() path, byte-identical to the
            # pre-scheduler code (this is the only path exercised when
            # --concurrent-games 1, since every tick then has exactly one
            # live game).
            evaluator, batch = payloads[0]
            t0 = time.perf_counter()
            result = evaluator.evaluate(batch)
            if stats is not None:
                stats.search_gpu += time.perf_counter() - t0
                stats.search_eval_calls += 1
                stats.search_eval_items += len(batch)
            return [result]

        requests = [evaluator.build_decode_request(batch) for evaluator, batch in payloads]
        merged = _merge_decode_requests(requests)
        t0 = time.perf_counter()
        with torch.inference_mode(), _autocast_context(device, dtype):
            out = model.forward_decode_grouped(
                new_token_batch=merged.new_token_batch,
                positions=merged.positions,
                group_index=merged.group_index,
                prefix_kv_grouped=merged.prefix_kv_grouped,
                prefix_lens=merged.prefix_lens,
                prefix_lens_list=merged.prefix_lens_list,
                suffix_kv=merged.suffix_kv,
                suffix_positions=merged.suffix_positions,
                suffix_mask=merged.suffix_mask,
            )
        if stats is not None:
            elapsed = time.perf_counter() - t0
            stats.search_gpu += elapsed
            stats.search_eval_calls += 1
            stats.search_eval_items += sum(len(req.nodes) for req in requests)

        split_outs = _split_decode_output(out, [len(req.nodes) for req in requests])
        return [
            evaluator.consume_decode_result(req, out_g)
            for (evaluator, _), req, out_g in zip(payloads, requests, split_outs)
        ]

    return executor
