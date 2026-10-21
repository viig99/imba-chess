"""Executor-owned leaf staging. Owners are weak; no search trees are retained.

One synchronous result readback completes all queued work before pinned input
storage is reused. The executor is deliberately single-stream and non-reentrant.
"""

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import weakref

import imba_chess_native as cc
import torch

from . import cozy_bridge
from imba_chess.data.event_builder import EVENT_TOKEN_ID


FEATURE_KEYS = (
    "seq_token_id",
    "turn_id",
    "castle_id",
    "ep_file_id",
    "halfmove_bucket_id",
    "fullmove_bucket_id",
    "prev_move_id",
)


@dataclass(slots=True)
class _StagedRequest:
    nodes: tuple
    boards: tuple
    prefix_kv: object
    prefix_len: int


def _stage_request(evaluator, batch):
    node, board = batch[0]
    # Inline evaluator._validate_handles: this runs for every staged node.
    if node.history_token is not evaluator._history_token:
        raise RuntimeError("stale or foreign history handle")
    return _StagedRequest(
        (node,), (board,), evaluator._prefix_kv, evaluator._prefix_len
    )


def _wave_stager(encoder):
    """Native board-feature writer configured like encoder.encode_cozy."""
    cfg = encoder.config
    return cc.WaveStager(
        encoder._ep_mode_code,
        cfg.halfmove_max,
        cfg.halfmove_bucket_size,
        cfg.fullmove_max,
        cfg.fullmove_bucket_size,
        EVENT_TOKEN_ID,
    )


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
    # Exact-shape decoder CUDA graphs kept per workspace (LRU).
    graph_capacity = 32

    def __init__(self, runner):
        self.runner = runner
        self.device = runner.model.piece_square_embedding.weight.device
        self.counters = Counter()
        self._stager = None  # (encoder, WaveStager)
        self._graphs = OrderedDict()
        self._graph_pool = None
        self._capture_stream = None
        self._generation = 0
        self.clear()

    def clear(self):
        if self.device.type == "cuda" and getattr(self, "host", None) is not None:
            # Also safe after cancellation or a decoder exception before readback.
            torch.cuda.current_stream(self.device).synchronize()
        self._drop_graphs()
        self.numpy = None
        self.slots = {}
        self.free_rows = []
        # K/V of every evaluated node: one [layers, 2 (K, V), rows, heads, dim]
        # tensor. A wave scatters its new rows in one operation, and the decoder
        # reads each row's ancestors from it by chain index.
        self.arena_stack = self.arena = None
        self.host = self.metadata = None
        self.row_owners = []
        self.batch_capacity = self.prefix_capacity = self.legal_capacity = 0
        self.row_capacity = 0
        self.device_fields = None
        # Per batch size: prefix views, metadata fields and decoder feature
        # inputs. Views of fixed buffers, rebuilt only when a buffer is replaced.
        self._wave_views = {}
        self.prefix_inputs = self.feature_inputs = None
        self.prefix_inference = None

    def placement_order(self, payloads):
        """Keep live owners in their active rows; callers restore result order."""
        count = len(payloads)
        by_owner = {id(owner): index for index, (owner, _) in enumerate(payloads)}
        order = [None] * count
        used = set()
        for row, entry in enumerate(self.row_owners[:count]):
            owner = None if entry is None else entry[0]()
            if owner is not None:
                index = by_owner.get(id(owner))
                if index is not None and index not in used:
                    order[row] = index
                    used.add(index)
        remaining = iter(i for i in range(count) if i not in used)
        return [next(remaining) if i is None else i for i in order]

    def _grow_arena(self, reference):
        old = self.row_capacity
        size = max(256, old * 2)
        k = reference[0][0]
        if any(t.shape != k.shape or t.dtype != k.dtype for pair in reference for t in pair):
            raise ValueError("decoder layers must share one K/V shape and dtype")
        stack = k.new_zeros((len(reference), 2, size, k.size(0), k.size(-1)))
        if self.arena_stack is not None:
            stack[:, :, :old].copy_(self.arena_stack)
            self.counters["arena_growth_copy_bytes"] += (
                self.arena_stack.numel() * self.arena_stack.element_size()
            )
        self.arena_stack, self.row_capacity = stack, size
        # Per-kind views ([layers, rows, heads, dim]) for inspection and tests.
        self.arena = (stack[:, 0], stack[:, 1])
        self._drop_graphs()  # Graphs read the arena by address.
        # Row zero is permanent zero padding, never assigned to a node.
        self.free_rows.extend(range(size - 1, max(1, old) - 1, -1))
        self.counters["arena_growths"] += 1

    def _history_stamp(self, evaluator, request):
        if evaluator.immutable_prefix:
            self.counters["fast_validation_calls"] += 1
            return (evaluator.history_revision, request.prefix_len)
        self.counters["fallback_validation_calls"] += 1
        return (evaluator.history_revision, _stamp(request))

    def _views(self, g):
        """Fixed-layout views for a g-row wave. Prefix inputs keep their full
        capacity: the decoder reads each row only up to its own length, so the
        wave's longest prefix never changes the layout, the rows' contents or
        the captured graph."""
        views = self._wave_views.get(g)
        if views is None:
            fields = self._fields(self.metadata, g)
            features = [t[:g] for t in self.feature_inputs]
            views = self._wave_views[g] = (
                [tuple(t[:g] for t in pair) for pair in self.prefix_inputs],
                fields,
                features,
                [fields[0], *fields[1].unbind(0), fields[2][0]],
                fields[2].unbind(0),
            )
        return views

    def _refresh_direct(self, prefix, requests, dirty):
        # Only each row's real history is copied: the decoder never reads past a
        # row's length, so padding is left as is.
        targets, sources = [], []
        for row in dirty:
            req = requests[row]
            for buffers, refs in zip(prefix, req.prefix_kv):
                for target, source in zip(buffers, refs):
                    if req.prefix_len:
                        targets.append(target[row, :, : req.prefix_len])
                        sources.append(source[:, : req.prefix_len])
        if targets:
            torch._foreach_copy_(targets, sources)
            self.counters["history_copy_submissions"] += 1
        copied = sum(t.numel() * t.element_size() for t in targets)
        self.counters["decoder_prefix_copy_bytes"] += copied
        self.counters["history_copied_bytes"] += copied

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
        """Stage one wave; returns the decoder arguments and consume() state."""
        if any(len(batch) != 1 for _, batch in payloads):
            raise ValueError("reusable decode buffers require one query per game")
        if len({id(e) for e, _ in payloads}) != len(payloads):
            raise ValueError("duplicate evaluator in decode batch")
        for key, slot in list(self.slots.items()):
            if slot.owner() is None:
                self.free_rows.extend(slot.rows)
                del self.slots[key]
        requests = [_stage_request(e, batch) for e, batch in payloads]
        nodes = [req.nodes[0] for req in requests]
        boards = [req.boards[0] for req in requests]
        encoder = payloads[0][0]._board_state_encoder
        vocab = payloads[0][0]._move_vocab
        if any(e._board_state_encoder is not encoder for e, _ in payloads):
            raise ValueError("decode wave mixes board-state encoders")
        if any(e._move_vocab is not vocab for e, _ in payloads):
            raise ValueError("decode wave mixes move vocabularies")
        # Native: every row's legal set in one call, and the widest row.
        per_node, width = cozy_bridge.project_wave(boards, vocab)
        if self._stager is None or self._stager[0] is not encoder:
            self._stager = (encoder, _wave_stager(encoder))
        slots_rows = [self._slot(e, req) for (e, _), req in zip(payloads, requests)]
        prefix_lens = [r.prefix_len for r in requests]
        g, p = len(requests), max(prefix_lens)
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
            self._wave_views = {}
            self._drop_graphs()  # Decoder input buffers were reallocated.
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
            self._wave_views = {}
            self._drop_graphs()  # Graphs read lengths/depths from metadata.
            self.numpy = self.host.numpy()
        count = self._packed_size(g)
        self.numpy[:count].fill(0)
        host_fields = self._fields(self.numpy, g)
        self.counters["history_full_refreshes"] += int(full_refresh)
        # Logical rows are contiguous just like the compiled reference. Capacity
        # padding lives after the used range, outside each layer's active view.
        keys = FEATURE_KEYS
        # Native, in _fields' layout: piece ids, the event token, five board
        # scalars and the move id; position, prefix length, ancestor count and
        # destination row; the parent's arena chain (validated against the
        # node's depth); and the legal ids.
        pitch = (g + 31) // 32 * 32
        dests = [dest for _, dest in slots_rows]
        self._stager[1].stage_wave(
            self.numpy,
            nodes,
            boards,
            per_node,
            prefix_lens,
            dests,
            dict(
                pitch=pitch,
                scalars=64 * g,
                positions=64 * g + 7 * pitch,
                ancestors=64 * g + 11 * pitch,
                ancestor_capacity=self.capacity,
                legal=64 * g + 11 * pitch + self.capacity * g,
                legal_capacity=self.legal_capacity,
            ),
        )
        dirty = []
        for row, (slot, _) in enumerate(slots_rows):
            owner = (slot.owner, slot.stamp)
            if self.row_owners[row] != owner:
                dirty.append(row)
                self.row_owners[row] = owner
        metadata = self.metadata[:count]
        metadata.copy_(self.host[:count], non_blocking=True)
        prefix, self.device_fields, feature_targets, feature_sources, scalars = self._views(g)
        self.counters["h2d_bytes"] += metadata.numel() * metadata.element_size()
        self.counters["h2d_copies"] += 1
        self.counters["history_dirty_rows"] += len(dirty)
        if dirty:
            self._refresh_direct(prefix, requests, dirty)
        torch._foreach_copy_(feature_targets, feature_sources)
        self.counters["decoder_feature_copy_bytes"] += sum(
            t.numel() * t.element_size() for t in feature_targets
        )
        _, lengths, depths, _ = scalars
        positions = feature_targets[-1]
        batch = {"piece_ids": feature_targets[0]}
        batch.update({key: feature_targets[col + 1] for col, key in enumerate(keys)})
        args = (
            batch,
            positions,
            prefix,
            self.arena_stack,
            self.device_fields[3],
            lengths,
            depths,
        )
        return args, (requests, nodes, slots_rows, per_node, width)

    def decode(self, args):
        """Run the decoder on prepare() arguments.

        On CUDA each (buffer generation, batch size) gets one captured
        graph of the compiled decoder: the same kernels on the same static
        buffers (prefix inputs, arena, metadata), replayed without Python
        launches. The first wave of a shape runs normally (its result is used),
        then the shape is captured. Graph outputs are overwritten by the next
        replay, so callers consume them before decoding again (consume() does)."""

        def run():
            return self.runner.decode(*args)

        if self.device.type != "cuda":
            return run()
        # Prefix inputs keep their capacity, so a batch size fixes every shape.
        key = (self._generation, args[1].size(0))
        entry = self._graphs.get(key)
        if entry is not None:
            self._graphs.move_to_end(key)
            entry[0].replay()
            self.counters["graph_replays"] += 1
            return entry[1]
        out = run()
        self._graphs[key] = self._capture(run)
        self.counters["graph_captures"] += 1
        while len(self._graphs) > self.graph_capacity:
            self._graphs.popitem(last=False)
            self.counters["graph_evictions"] += 1
        return out

    def _capture(self, run):
        current = torch.cuda.current_stream(self.device)
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(self.device)
        stream = self._capture_stream
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            # The low-level API skips torch.cuda.graph's gc.collect/empty_cache.
            graph.capture_begin(pool=self._graph_pool, capture_error_mode="thread_local")
            try:
                out = run()
            finally:
                graph.capture_end()
        current.wait_stream(stream)
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        return graph, out

    def _drop_graphs(self):
        if getattr(self, "_graphs", None):
            self.counters["graph_invalidations"] += 1
        self._graphs = OrderedDict()
        # The pool is released with its last graph; never capture into it again.
        self._graph_pool = None
        self._generation = getattr(self, "_generation", 0) + 1

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
        requests, nodes, slots_rows, per_node, width = state
        for req, (slot, _) in zip(requests, slots_rows):
            owner = slot.owner()
            if owner is not None:
                token = owner._history_token
                for node in req.nodes:
                    if node.history_token is not token:
                        raise RuntimeError("stale or foreign history handle")
        # Every layer's new K and V rows in one scatter: [layers, 2, g, heads, dim].
        layers = len(out["kv"])
        rows = torch.stack([t.squeeze(2) for pair in out["kv"] for t in pair])
        rows = rows.view(layers, 2, *rows.shape[1:])
        self.arena_stack.index_copy_(2, self.device_fields[2][3], rows)
        self.counters["scatter_bytes"] += rows.numel() * rows.element_size()
        # Each node's chain: its parent's plus the row its K/V was written to.
        cc.extend_arena_chains(nodes, [dest for _, dest in slots_rows])
        picked = out["logits"].float().gather(1, self.device_fields[4][:, :width])
        # Blocking readback also protects pinned metadata from CPU overwrite.
        packed = torch.cat((picked, out["value_logits"].float()), dim=1).cpu()
        self.counters["readbacks"] += 1
        self.counters["d2h_bytes"] += packed.numel() * packed.element_size()
        # The same CPU float32 operations as before; native code only widens
        # the results to f64 (as .tolist() did) and builds one object per row.
        wdls = torch.softmax(packed[:, width:].contiguous(), -1)
        values = (wdls[:, 2] - wdls[:, 0]).contiguous()
        lens = torch.tensor([len(legal) for legal in per_node])
        # Match the reference's logical width and contiguous CPU normalization.
        priors = packed[:, :width].contiguous()
        priors.masked_fill_(torch.arange(width) >= lens[:, None], -torch.inf)
        if width:
            priors = torch.log_softmax(priors, dim=1)
        evaluations = cc.make_evaluations(
            per_node, values.numpy(), wdls.numpy(), priors.numpy(), width
        )
        return [[evaluation] for evaluation in evaluations]
