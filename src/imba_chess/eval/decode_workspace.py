"""Executor-owned leaf staging. Owners are weak; no search trees are retained.

One synchronous result readback completes all queued work before pinned input
storage is reused. The executor is deliberately single-stream and non-reentrant.
"""

from collections import Counter
from dataclasses import dataclass, field
import weakref

import torch

from . import cozy_bridge
from .position_evaluator import _batched_value_scalars
from .search import PositionEval


@dataclass
class _Slot:
    owner: weakref.ReferenceType
    rows: list[int] = field(default_factory=list)
    free: list[int] = field(default_factory=list)
    stamp: tuple = ()


def _capacity(n):
    return 1 << max(0, n - 1).bit_length()


def _stamp(request):
    # Inference tensors have no version counter. Their root caches are immutable;
    # replacing a cache or changing its logical length still invalidates the row.
    def version(t):
        return None if t.is_inference() else t._version

    return (
        request.prefix_len,
        tuple(
            (id(t), t.data_ptr(), version(t))
            for pair in request.prefix_kv
            for t in pair
        ),
    )


class DecodeWorkspace:
    capacity = 32

    def __init__(self, runner):
        if runner.sdpa:
            raise ValueError(
                "reusable decode buffers require the custom attention decoder"
            )
        self.runner = runner
        self.device = runner.model.piece_square_embedding.weight.device
        self.counters = Counter()
        self.clear()

    def clear(self):
        if self.device.type == "cuda" and getattr(self, "host", None) is not None:
            # Also safe after cancellation or a decoder exception before readback.
            torch.cuda.current_stream(self.device).synchronize()
        self.numpy = None
        self.slots = {}
        self.free_rows = []
        self.arena = None
        self.host = self.metadata = self.branch = None
        self.row_owners = []
        self.batch_capacity = self.prefix_capacity = self.legal_capacity = 0
        self.row_capacity = 0
        self.logical_prefix = 0
        self.device_fields = None
        self.prefix_inputs = self.feature_inputs = None
        self.prefix_inference = None

    def _grow_arena(self, reference):
        old = self.row_capacity
        size = max(256, old * 2)
        arena = tuple(
            t.new_zeros((len(reference), size, t.size(0), t.size(-1)))
            for t in reference[0]
        )
        if self.arena is not None:
            for target, source in zip(arena, self.arena):
                target[:, :old].copy_(source)
                self.counters["arena_growth_copy_bytes"] += (
                    source.numel() * source.element_size()
                )
        self.arena, self.row_capacity = arena, size
        # Row zero is permanent zero padding, never assigned to a node.
        self.free_rows.extend(range(size - 1, max(1, old) - 1, -1))
        self.counters["arena_growths"] += 1

    def _history_stamp(self, evaluator, request):
        if evaluator.immutable_prefix:
            self.counters["fast_validation_calls"] += 1
            return (evaluator.history_revision, request.prefix_len)
        self.counters["fallback_validation_calls"] += 1
        return (evaluator.history_revision, _stamp(request))

    def _prefix_views(self, reference, g, p):
        prefix = []
        for buffers, refs in zip(self.prefix_inputs, reference):
            row = []
            for target, ref in zip(buffers, refs):
                shape = (ref.size(0), p, ref.size(-1))
                # Preserve the reference stack's single-row logical base shape,
                # independent storage, contiguous strides and inference metadata.
                target.resize_(shape if g == 1 else (g, *shape))
                row.append(target.unsqueeze(0) if g == 1 else target)
            prefix.append(tuple(row))
        return prefix

    def _refresh_direct(self, prefix, requests, dirty):
        targets, sources, padding = [], [], []
        for row in dirty:
            req = requests[row]
            for buffers, refs in zip(prefix, req.prefix_kv):
                for target, source in zip(buffers, refs):
                    if req.prefix_len:
                        targets.append(target[row, :, : req.prefix_len])
                        sources.append(source[:, : req.prefix_len])
                    if req.prefix_len < target.size(2):
                        padding.append(target[row, :, req.prefix_len :])
        if targets:
            torch._foreach_copy_(targets, sources)
            self.counters["history_copy_submissions"] += 1
        if padding:
            torch._foreach_zero_(padding)
            self.counters["history_zero_submissions"] += 1
        copied = sum(t.numel() * t.element_size() for t in targets)
        self.counters["decoder_prefix_copy_bytes"] += copied
        self.counters["history_copied_bytes"] += copied
        self.counters["history_zeroed_bytes"] += sum(
            t.numel() * t.element_size() for t in padding
        )

    def _slot(self, evaluator, request):
        key = id(evaluator)
        slot = self.slots.get(key)
        if slot is None:
            if request.nodes[0].parent is not None:
                raise RuntimeError(
                    "Missing workspace ownership for a non-root decode wave"
                )
            slot = self.slots[key] = _Slot(weakref.ref(evaluator))
        stamp = self._history_stamp(evaluator, request)
        if stamp != slot.stamp:
            slot.stamp = stamp
            self.counters["history_refreshes"] += 1
        if not slot.free:
            if not self.free_rows:
                self._grow_arena(request.prefix_kv)
            # Reserve in chunks, reclaim every row when this evaluator dies.
            count = min(128, len(self.free_rows))
            slot.free = self.free_rows[-count:]
            del self.free_rows[-count:]
            slot.rows.extend(slot.free)
        return slot, slot.free.pop()

    def prepare(self, payloads, *, prefix_inference=False):
        if any(len(batch) != 1 for _, batch in payloads):
            raise ValueError("reusable decode buffers require one query per game")
        if len({id(e) for e, _ in payloads}) != len(payloads):
            raise ValueError("duplicate evaluator in decode batch")
        for key, slot in list(self.slots.items()):
            if slot.owner() is None:
                self.free_rows.extend(slot.rows)
                del self.slots[key]
        requests = [
            e.build_decode_request(batch, defer_tensors=True, defer_suffix=True)
            for e, batch in payloads
        ]
        chains = []
        for req in requests:
            node = req.nodes[0]
            chain = [] if node.parent is None else node.parent.arena_chain
            if chain is None:
                raise RuntimeError("Cannot decode a child before evaluating its parent")
            if len(chain) != node.depth or len(chain) > self.capacity:
                raise ValueError(
                    "search suffix exceeds capacity or has invalid ancestry"
                )
            chains.append(chain)
        per_node = [
            cozy_bridge.project_legal_moves(req.boards[0], e._move_vocab)
            for (e, _), req in zip(payloads, requests)
        ]
        slots_rows = [self._slot(e, req) for (e, _), req in zip(payloads, requests)]
        g, p = len(requests), max(r.prefix_len for r in requests)
        width = max(len(row[0]) for row in per_node)
        full_refresh = False
        if (
            g > self.batch_capacity
            or p > self.prefix_capacity
            or prefix_inference != self.prefix_inference
        ):
            self.prefix_inference = prefix_inference
            self.batch_capacity = max(self.batch_capacity, _capacity(g))
            self.prefix_capacity = max(self.prefix_capacity, _capacity(p))
            ref = requests[0].prefix_kv
            # Match the reference's independent input storages. Shared layer
            # views can select numerically different cold-compiled graphs.
            self.branch = [
                tuple(
                    t.new_empty(
                        (self.batch_capacity, t.size(0), self.capacity, t.size(-1))
                    )
                    for t in pair
                )
                for pair in ref
            ]
            # The reference packs prefixes outside inference_mode. Preserve
            # that dispatch metadata too: inference tensors can select a
            # numerically different compiled graph after a weight update.
            with torch.inference_mode(prefix_inference):
                self.prefix_inputs = [
                    tuple(
                        t.new_empty(
                            (
                                self.batch_capacity,
                                t.size(0),
                                self.prefix_capacity,
                                t.size(-1),
                            )
                        )
                        for t in pair
                    )
                    for pair in ref
                ]
            self.feature_inputs = [
                torch.empty(
                    (self.batch_capacity, 64), dtype=torch.long, device=self.device
                )
            ] + [
                torch.empty(self.batch_capacity, dtype=torch.long, device=self.device)
                for _ in range(8)
            ]
            self.row_owners = [None] * self.batch_capacity
            full_refresh = True
            self.host = None
        if self.host is None or width > self.legal_capacity:
            self.legal_capacity = max(1, self.legal_capacity, _capacity(width))
            # 64 squares, seven scalar features, position/prefix/depth/destination,
            # 32 ancestor rows and legal IDs. A single H2D copy supplies all fields.
            self.host = torch.empty(
                (self._packed_size(self.batch_capacity),),
                dtype=torch.long,
                pin_memory=self.device.type == "cuda",
            )
            self.metadata = torch.empty_like(self.host, device=self.device)
            self.numpy = self.host.numpy()
        count = self._packed_size(g)
        self.numpy[:count].fill(0)
        host_fields = self._fields(self.numpy, g)
        if p != self.logical_prefix:
            self.row_owners = [None] * self.batch_capacity
            self.logical_prefix = p
            full_refresh = True
        self.counters["history_full_refreshes"] += int(full_refresh)
        # Logical rows are contiguous just like the compiled reference. Capacity
        # padding lives after the used range, outside each layer's active view.
        keys = tuple(k for k in requests[0].new_token_batch if k != "piece_ids")
        dirty = []
        for row, (req, chain, (slot, dest), legal) in enumerate(
            zip(requests, chains, slots_rows, per_node)
        ):
            host_fields[0][row] = req.new_token_batch["piece_ids"][0]
            for col, key in enumerate(keys):
                host_fields[1][col, row] = req.new_token_batch[key][0]
            host_fields[2][:, row] = req.positions[0], req.prefix_len, len(chain), dest
            host_fields[3][row, : len(chain)] = chain
            host_fields[4][row, : len(legal[0])] = legal[0]
            owner = (slot.owner, slot.stamp)
            if self.row_owners[row] != owner:
                dirty.append(row)
                self.row_owners[row] = owner
        metadata = self.metadata[:count]
        metadata.copy_(self.host[:count], non_blocking=True)
        self.device_fields = self._fields(metadata, g)
        self.counters["h2d_bytes"] += metadata.numel() * metadata.element_size()
        self.counters["h2d_copies"] += 1
        suffix = self._gather_ancestors(g)
        prefix = self._prefix_views(requests[0].prefix_kv, g, p)
        self.counters["history_dirty_rows"] += len(dirty)
        if dirty:
            self._refresh_direct(prefix, requests, dirty)
        feature_targets = [t[:g] for t in self.feature_inputs]
        feature_sources = [
            self.device_fields[0],
            *self.device_fields[1].unbind(0),
            self.device_fields[2][0],
        ]
        torch._foreach_copy_(feature_targets, feature_sources)
        self.counters["decoder_feature_copy_bytes"] += sum(
            t.numel() * t.element_size() for t in feature_targets
        )
        _, lengths, depths, _ = self.device_fields[2].unbind(0)
        positions = feature_targets[-1]
        prefix_rel, prefix_fill, suffix_rel, suffix_fill = self._prepare_attention(
            p, positions, lengths, depths
        )
        batch = {"piece_ids": feature_targets[0]}
        batch.update({key: feature_targets[col + 1] for col, key in enumerate(keys)})
        args = (
            batch,
            positions,
            prefix,
            suffix,
            prefix_rel,
            prefix_fill,
            suffix_rel,
            suffix_fill,
            (),
        )
        return args, (requests, chains, slots_rows, per_node, width)

    def _gather_ancestors(self, g):
        indices = self.device_fields[3]
        suffix = []
        for layer, pair in enumerate(self.branch):
            targets = []
            for arena, buffer in zip(self.arena, pair):
                target = buffer[:g]
                source = arena[layer].permute(1, 0, 2)[None].expand(g, -1, -1, -1)
                gather_indices = indices[:, None, :, None].expand_as(target)
                torch.gather(source, 2, gather_indices, out=target)
                self.counters["gather_bytes"] += target.numel() * target.element_size()
                targets.append(target)
            suffix.append(tuple(targets))
        return suffix

    def _prepare_attention(self, p, positions, lengths, depths):
        max_pos = self.runner.model.layers[0]._max_seq_len
        prefix_pos = torch.arange(p, device=self.device)[None, :]
        branch_pos = torch.arange(self.capacity, device=self.device)[None, :]
        prefix_rel = (prefix_pos - positions[:, None] + max_pos - 1).clamp(
            0, 2 * max_pos - 2
        )
        suffix_positions = torch.where(
            branch_pos < depths[:, None], lengths[:, None] + branch_pos, 0
        )
        suffix_rel = (suffix_positions - positions[:, None] + max_pos - 1).clamp(
            0, 2 * max_pos - 2
        )
        prefix_fill = (prefix_pos >= lengths[:, None])[:, None, None, :]
        suffix_fill = (branch_pos >= depths[:, None])[:, None, None, :]
        return prefix_rel, prefix_fill, suffix_rel, suffix_fill

    def _packed_size(self, g):
        return (96 + self.legal_capacity) * g + 11 * ((g + 31) // 32 * 32)

    def _fields(self, data, g):
        # Align each field as the CUDA allocator aligns the reference tensors.
        pitch = (g + 31) // 32 * 32
        scalars = 64 * g
        positions = scalars + 7 * pitch
        ancestors = positions + 4 * pitch
        legal = ancestors + 32 * g
        return (
            data[:scalars].reshape(g, 64),
            data[scalars:positions].reshape(7, pitch)[:, :g],
            data[positions:ancestors].reshape(4, pitch)[:, :g],
            data[ancestors:legal].reshape(g, 32),
            data[legal : legal + self.legal_capacity * g].reshape(
                g, self.legal_capacity
            ),
        )

    def consume(self, state, out):
        requests, chains, slots_rows, per_node, width = state
        for req, (slot, _) in zip(requests, slots_rows):
            owner = slot.owner()
            if owner is not None:
                owner._validate_handles(req.nodes)
        for i, arena in enumerate(self.arena):
            rows = torch.stack([pair[i].squeeze(2) for pair in out["kv"]])
            arena.index_copy_(1, self.device_fields[2][3], rows)
            self.counters["scatter_bytes"] += rows.numel() * rows.element_size()
        for req, chain, (_, dest) in zip(requests, chains, slots_rows):
            req.nodes[0].arena_chain = chain + [dest]
        picked = out["logits"].float().gather(1, self.device_fields[4][:, :width])
        # Blocking readback also protects pinned metadata from CPU overwrite.
        packed = torch.cat((picked, out["value_logits"].float()), dim=1).cpu()
        self.counters["readbacks"] += 1
        self.counters["d2h_bytes"] += packed.numel() * packed.element_size()
        values = _batched_value_scalars(packed[:, width:].contiguous())
        lens = [len(row[0]) for row in per_node]
        # Match the reference's logical width and contiguous CPU normalization.
        priors = packed[:, :width].contiguous()
        for row, length in enumerate(lens):
            priors[row, length:] = -torch.inf
        prior_rows = (
            torch.log_softmax(priors, dim=1).tolist() if width else [[] for _ in lens]
        )
        return [
            [
                PositionEval(
                    value_stm=values[row],
                    legal_moves=moves,
                    legal_ucis=ucis,
                    legal_log_priors=prior_rows[row][: len(ids)],
                    legal_forcing=forcing,
                    legal_ids=ids,
                )
            ]
            for row, (ids, moves, ucis, forcing, _) in enumerate(per_node)
        ]
