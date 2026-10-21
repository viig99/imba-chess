"""Cross-game workspace equivalence and ownership/lifetime regressions."""

import gc
import math
import weakref

import chess
import imba_chess_native as cc
import pytest
import torch

from imba_chess.eval import cozy_bridge
from imba_chess.eval.decode_workspace import DecodeWorkspace
from imba_chess.eval.merged_executors import _merge_decode_requests
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    consume_batched_decode_results,
)
from tests.search_references import DecoderRunner
from tests.test_prefix_decode import _tiny_model
from tests.test_self_play import VOCAB, ENCODER


def evaluator(model, length, prefix=None, *, immutable=False):
    return CachedPositionEvaluator(
        model=model,
        move_vocab=VOCAB,
        board_state_encoder=ENCODER,
        device=model.piece_square_embedding.weight.device,
        dtype=torch.float32,
        prefix_len=length,
        immutable_prefix=immutable,
        prefix_kv=prefix
        or [
            (
                torch.randn(
                    2, length, 8, device=model.piece_square_embedding.weight.device
                ),
                torch.randn(
                    2, length, 8, device=model.piece_square_embedding.weight.device
                ),
            )
            for _ in model.layers
        ],
    )


def payload(e, parent=None):
    board = chess.Board()
    board.push_uci("e2e4")
    return e, [(e.extend(parent, "e2e4"), cozy_bridge.board_to_cozy(board))]


def compare_results(a, b, *, prior_atol=1e-6):
    for x, y in zip(a, b):
        x, y = x[0], y[0]
        assert x.legal_ids == y.legal_ids and x.legal_ucis == y.legal_ucis
        assert x.legal_forcing == y.legal_forcing
        assert x.value_stm == pytest.approx(y.value_stm, abs=1e-6)
        assert x.wdl == pytest.approx(y.wdl, abs=1e-6)
        assert sum(x.wdl) == pytest.approx(1)
        assert x.wdl[2] - x.wdl[0] == pytest.approx(x.value_stm, abs=1e-6)
        torch.testing.assert_close(
            torch.tensor(x.legal_log_priors),
            torch.tensor(y.legal_log_priors),
            atol=prior_atol,
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)]
)
@pytest.mark.parametrize(
    "mode", ["tensor", pytest.param("compiled", marks=pytest.mark.extended)]
)
def test_workspace_mixed_depth_owner_changes_and_kv_lifetime(device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cpu" and mode == "compiled":
        pytest.skip("compiled integration runs on CUDA")
    if mode == "compiled":
        # Isolate this model/stride comparison from unrelated SDPA test graphs.
        torch.compiler.reset()
    torch.set_num_threads(4)
    torch.manual_seed(789)
    model = _tiny_model(len(VOCAB)).to(device).eval()
    reference = DecoderRunner(model, mode)
    ws = DecodeWorkspace(DecoderRunner(model, mode))
    a = [evaluator(model, n) for n in (0, 7, 19)]
    b = [evaluator(model, e._prefix_len, e._prefix_kv, immutable=True) for e in a]
    pa, pb = [None] * 3, [None] * 3
    saved = None
    with torch.inference_mode():
        for step in range(38):
            order = [0, 1, 2] if step < 33 else [2, 0] if step < 36 else [0]
            # Game 0 reaches maximum branch depth; other games mix shallow siblings.
            if step == 33:
                a[0] = evaluator(model, 3)
                b[0] = evaluator(model, 3, a[0]._prefix_kv, immutable=True)
                pa[0] = pb[0] = None
            aa = [payload(a[i], pa[i] if i == 0 and step < 33 else None) for i in order]
            bb = [payload(b[i], pb[i] if i == 0 and step < 33 else None) for i in order]
            reqs = [
                e.build_decode_request(batch, defer_tensors=True, stack_suffix=True)
                for e, batch in aa
            ]
            with torch.inference_mode(False):
                merged = _merge_decode_requests(reqs, batch_suffix=True)
            expected = reference(merged)
            args, state = ws.prepare(bb)
            assert all(t.is_contiguous() for t in args[0].values())
            assert args[1].is_contiguous()
            assert not any(t.is_inference() for pair in args[2] for t in pair)
            inputs = [*args[0].values(), args[1]] + [
                t for pair in args[2] for t in pair
            ]
            assert len({t.untyped_storage().data_ptr() for t in inputs}) == len(inputs)
            assert all(t.is_contiguous() for pair in args[2] for t in pair)
            for key in merged.new_token_batch:
                torch.testing.assert_close(
                    args[0][key].cpu(), merged.new_token_batch[key], atol=0, rtol=0
                )
            torch.testing.assert_close(args[1].cpu(), merged.positions, atol=0, rtol=0)
            for got, ref in zip(args[2], merged.prefix_kv_grouped):
                for x, y in zip(got, ref):
                    # Fixed-capacity rows; only real history is staged and read.
                    assert x.shape[:2] == y.shape[:2] and x.size(3) == y.size(3)
                    assert x.size(2) >= y.size(2) and x.is_contiguous()
                    for row, n in enumerate(merged.prefix_lens_list):
                        torch.testing.assert_close(
                            x[row, :, :n], y[row, :, :n], atol=0, rtol=0
                        )
            torch.testing.assert_close(
                args[5].cpu(), merged.prefix_lens, atol=0, rtol=0
            )
            torch.testing.assert_close(
                (args[5] + args[6]).cpu(), merged.positions, atol=0, rtol=0
            )
            if merged.suffix_kv is not None:
                # The decoder reads each row's ancestors from the arena by chain.
                width = merged.suffix_positions.size(1)
                chain = args[4][:, :width]
                for layer, ref in enumerate(merged.suffix_kv):
                    for kind, y in enumerate(ref):
                        x = args[3][layer, kind][chain].permute(0, 2, 1, 3)
                        mask = merged.suffix_mask[:, None, :, None].expand_as(y)
                        torch.testing.assert_close(
                            x[mask], y[mask], atol=1e-5, rtol=1e-5
                        )
            if mode == "compiled" and step in (0, 36):
                # Cold compilation must also match after entering a single-row tail.
                torch.compiler.reset()
            actual = ws.runner.decode(*args)
            for key in ("logits", "value_logits"):
                torch.testing.assert_close(
                    actual[key], expected[key], atol=1e-5, rtol=1e-5
                )
            compare_results(
                consume_batched_decode_results(aa, reqs, expected),
                ws.consume(state, actual),
            )
            # Arena scatter copies every layer exactly, independent of returned tensors.
            for row, (_, dest) in enumerate(state[2]):
                for kind in range(2):
                    for layer, pair in enumerate(actual["kv"]):
                        torch.testing.assert_close(
                            ws.arena[kind][layer, dest],
                            pair[kind][row, :, 0],
                            atol=0,
                            rtol=0,
                        )
            if saved:
                torch.testing.assert_close(saved[0], saved[1], atol=0, rtol=0)
            saved = (actual["kv"][0][0], actual["kv"][0][0].clone())
            for i, x, y in zip(order, aa, bb):
                pa[i], pb[i] = x[1][0][0], y[1][0][0]
        assert ws.counters["readbacks"] == 38
        ws.clear()
        assert not ws.slots and ws.arena is None and ws.host is None
        model.prediction_head.weight.mul_(1.1)
        a = evaluator(model, 4)
        b = evaluator(model, 4, a._prefix_kv, immutable=True)
        aa, bb = [payload(a)], [payload(b)]
        reqs = [a.build_decode_request(aa[0][1])]
        expected = reference(_merge_decode_requests(reqs))
        args, state = ws.prepare(bb)
        actual = ws.runner.decode(*args)
        torch.testing.assert_close(
            actual["logits"], expected["logits"], atol=1e-5, rtol=1e-5
        )
        ws.consume(state, actual)

    if mode == "compiled":
        torch.compiler.reset()


def test_arena_growth_slot_reclamation_and_empty_projection(monkeypatch):
    torch.set_num_threads(4)
    model = _tiny_model(len(VOCAB)).eval()
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"))
    owner = evaluator(model, 3)
    original_projection = cozy_bridge.project_wave
    # A projector over an empty vocabulary maps no legal move.
    empty = cc.MoveProjector({})
    monkeypatch.setattr(
        cozy_bridge, "project_wave", lambda boards, vocab: empty.project_wave(boards)
    )
    with torch.inference_mode():
        saved = None
        for i in range(260):
            args, state = ws.prepare([payload(owner)])
            out = ws.runner.decode(*args)
            result = ws.consume(state, out)
            assert result[0][0].legal_ids == [] and result[0][0].legal_log_priors == []
            if saved is None:
                dest = state[2][0][1]
                saved = ws.arena[0][:, dest].clone()
        assert ws.row_capacity >= 512
        torch.testing.assert_close(ws.arena[0][:, dest], saved, atol=0, rtol=0)
        pointer = ws.arena[0].data_ptr()
        ref = weakref.ref(owner)
        del owner
        gc.collect()
        assert ref() is None
        owner = evaluator(model, 2)
        monkeypatch.setattr(cozy_bridge, "project_wave", original_projection)
        args, state = ws.prepare([payload(owner)])
        ws.consume(state, ws.runner.decode(*args))
        assert len(ws.slots) == 1 and ws.arena[0].data_ptr() == pointer
        assert ws.counters["history_refreshes"] == 2
        ws.clear()


class FixedStarts:
    """Start sampler double: every launch replays one seed under a fresh game ID."""

    durable_launches = True  # Nothing to journal before replay flushes.

    def __init__(self, seed):
        self.seed, self.launches = seed, 0

    def begin_phase(self, iteration, actor_id, completed):
        pass

    def next_launch(self, iteration, actor_id):
        self.launches += 1
        return self.seed, f"{actor_id}:{iteration}:{self.launches}"

    def launch_metadata(self, gid):
        return {}

    def retire(self, gid, reason):
        pass

    def reconcile(self, completed):
        pass


def compare_targets(reference, candidate, path="games"):
    """Exact search decisions/counters, tight tolerances for stored float targets."""
    if isinstance(reference, dict):
        assert reference.keys() == candidate.keys(), path
        for key in reference:
            compare_targets(reference[key], candidate[key], f"{path}.{key}")
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(candidate), path
        for i, (a, b) in enumerate(zip(reference, candidate)):
            compare_targets(a, b, f"{path}[{i}]")
    elif isinstance(reference, float):
        assert math.isclose(
            reference, candidate, rel_tol=1e-06, abs_tol=1e-06
        ), f"{path}: {reference!r} != {candidate!r}"
    else:
        assert reference == candidate, path


def test_scratch_collect_update_collect_and_cancellation(tmp_path):
    from dataclasses import replace
    from imba_chess.data.self_play_store import SelfPlayStore
    from tests.search_references import InferenceRuntime
    from imba_chess.self_play.collector import collect
    from imba_chess.self_play.config import SelfPlayConfig
    from imba_chess.self_play.seeds import Seed
    from imba_chess.self_play.trainer import Stage2Trainer
    from tests.test_self_play import tiny_model

    torch.set_num_threads(4)
    model = tiny_model().eval()
    # A decisive continuation that completes reliably with an untrained model.
    # Decoder K/V remain board-dependent; only final logits strongly favor mate.
    with torch.no_grad():
        model.final_norm.weight.zero_()
        model.final_norm.bias.fill_(1)
        model.prediction_head.weight.zero_()
        model.prediction_head.weight[VOCAB.encode("d8h4")].fill_(10)
    runtimes = [
        InferenceRuntime(
            model=model,
            move_vocab=VOCAB,
            encoder=ENCODER,
            device=torch.device("cpu"),
            one_query_per_game=True,
            cache_prefixes=True,
            decoder_mode="tensor",
            batch_projection=True,
            batch_inputs=True,
            batch_suffix=True,
            reuse_decode_buffers=reuse,
            native_gumbel=True,
        )
        for reuse in (False, True)
    ]
    cfg = SelfPlayConfig()
    cfg = replace(
        cfg,
        collection=replace(cfg.collection, concurrent_games=3),
        learning=replace(cfg.learning, auxiliary_value_weight=0.0),
    )
    seed = Seed("mate", "source", ["f2f3", "e7e5", "g2g4"], 3, "train", "test")
    for cycle in range(2):
        games = []
        for variant, runtime in enumerate(runtimes):
            store = SelfPlayStore(tmp_path / f"{cycle}-{variant}", flush_games=1)
            result = []
            collect(
                runtime=runtime,
                config=cfg,
                actor_id="scratch",
                store=store,
                max_positions=128,
                game_count=4,
                on_game=result.append,
                start_sampler=FixedStarts(seed),
            )
            assert all(
                g["status"] == "completed" and g["moves"] == ["d8h4"] for g in result
            )
            games.append(sorted(result, key=lambda g: g["game_id"]))
        for a, b in zip(*games):
            assert a["moves"] == b["moves"] and a["outcome_white"] == b["outcome_white"]
            compare_targets(a["targets"], b["targets"])
        if cycle == 0:
            trainer = Stage2Trainer(
                model=model,
                config=cfg.learning,
                move_vocab=VOCAB,
                encoder=ENCODER,
                device=torch.device("cpu"),
                max_positions=128,
            )
            trainer.begin_phase(store)
            trainer.train(store, exposure_budget=4)
            assert trainer.steps > 0
            model.eval()
    # Cancellation still invokes executor cleanup even with live pending searches.
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 12

    interrupted = []
    collect(
        runtime=runtimes[1],
        config=cfg,
        actor_id="scratch",
        store=SelfPlayStore(tmp_path / "cancel"),
        max_positions=128,
        game_count=4,
        should_stop=stop,
        on_game=interrupted.append,
        start_sampler=FixedStarts(seed),
    )
    assert any(g["status"] == "unfinished" and not g["targets"] for g in interrupted)


def test_history_refresh_and_padding_shrink_without_owner_change():
    model = _tiny_model(len(VOCAB)).eval()
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"))
    owner = evaluator(model, 9)
    other = evaluator(model, 13)
    with torch.inference_mode():
        args, state = ws.prepare([payload(owner), payload(other)])
        ws.consume(state, ws.runner.decode(*args))
    # Model updates clear the entire workspace; mutable, non-inference root
    # caches also invalidate through their tensor version counters.
    owner._prefix_kv[0][0].add_(7)
    owner._prefix_len = 3
    owner._prefix_kv = [(k[:, :3], v[:, :3]) for k, v in owner._prefix_kv]
    with torch.inference_mode():
        args, state = ws.prepare([payload(owner), payload(other)])
        for layer, pair in enumerate(args[2]):
            for kind, tensor in enumerate(pair):
                torch.testing.assert_close(
                    tensor[0, :, :3], owner._prefix_kv[layer][kind], atol=0, rtol=0
                )
        ws.consume(state, ws.runner.decode(*args))
        assert ws.counters["history_refreshes"] == 3
        assert ws.counters["history_dirty_rows"] == 3
        parent = state[0][0].nodes[0]
        ws.clear()
        with pytest.raises(RuntimeError, match="ownership"):
            ws.prepare([payload(owner, parent)])


def test_immutable_revision_and_stale_handles(monkeypatch):
    model = _tiny_model(len(VOCAB)).eval()
    owner = evaluator(model, 3, immutable=True)
    other = evaluator(model, 3, immutable=True)
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"))

    def no_fingerprint(request):
        raise AssertionError("immutable cache fingerprinted")

    monkeypatch.setattr("imba_chess.eval.decode_workspace._stamp", no_fingerprint)
    with torch.inference_mode():
        args, state = ws.prepare([payload(owner)])
        out = ws.runner.decode(*args)
        ws.consume(state, out)
        parent = state[0][0].nodes[0]
        child = payload(owner, parent)
        with pytest.raises(RuntimeError, match="foreign"):
            other.extend(parent, "e2e4")
        revision = owner.history_revision
        owner.replace_history(other._prefix_kv, 3)
        assert owner.history_revision == revision + 1
        with pytest.raises(RuntimeError, match="stale"):
            owner.extend(parent, "e2e4")
        with pytest.raises(RuntimeError, match="stale"):
            ws.prepare([child])
        with pytest.raises(RuntimeError, match="stale"):
            ws.consume(state, out)
        args, state = ws.prepare([payload(owner)])
        for pair, source in zip(args[2], other._prefix_kv):
            for t, ref in zip(pair, source):
                torch.testing.assert_close(t[0, :, :3], ref[:, :3], atol=0, rtol=0)
        assert ws.counters["fast_validation_calls"] == 2
        assert ws.counters["fallback_validation_calls"] == 0


@pytest.mark.parametrize("immutable", [False, True])
def test_direct_dirty_rows_batch_resize_and_width_changes(immutable):
    model = _tiny_model(len(VOCAB)).eval()
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"))
    owners = [evaluator(model, n, immutable=immutable) for n in (7, 0, 3, 7, 7)]

    def prepare(order, dirty):
        before = ws.counters.copy()
        with torch.inference_mode():
            args, _ = ws.prepare([payload(owners[i]) for i in order])
        assert ws.counters["history_dirty_rows"] - before["history_dirty_rows"] == dirty
        for row, i in enumerate(order):
            for pair, refs in zip(args[2], owners[i]._prefix_kv):
                for t, ref in zip(pair, refs):
                    n = owners[i]._prefix_len
                    torch.testing.assert_close(t[row, :, :n], ref, atol=0, rtol=0)
        if not dirty:
            for key in (
                "history_copied_bytes",
                "history_copy_submissions",
            ):
                assert ws.counters[key] == before[key]

    prepare([0, 1, 2], 3)
    prepare([0, 1, 2], 0)
    prepare([0], 0)
    prepare([0, 1, 2], 0)
    prepare([0, 1, 2, 3], 1)  # geometric capacity already four
    prepare([0, 2, 1, 3], 2)
    prepare([0, 2, 1, 4], 1)
    prepare([0, 1, 2, 3, 4], 5)  # allocation replacement
    owners[2].replace_history(owners[1]._prefix_kv, 0)
    prepare([0, 1, 2, 3, 4], 1)
    if not immutable:
        owners[0]._prefix_kv[0][0].add_(10)
        prepare([0, 1, 2, 3, 4], 1)
    prepare([1, 2], 2)  # zero-width prefix
    prepare([1], 0)
    prepare([0, 1], 2)  # both rows change owner; the longest prefix grows
    owners[0].replace_history(evaluator(model, 19)._prefix_kv, 19)
    prepare([0, 1], 2)  # capacity growth reallocates every row
    owners[0].replace_history(evaluator(model, 2)._prefix_kv, 2)
    # The longest prefix shrinks, but row 1's history is untouched.
    prepare([0, 1], 1)
    # First fill and two reallocations; prefix length changes alone never refresh.
    assert ws.counters["history_full_refreshes"] == 3


def test_workspace_executor_error_cleans_buffers(monkeypatch):
    from imba_chess.eval.merged_executors import _make_decode_wave_executor

    monkeypatch.setattr("imba_chess.model.tensor_decoder.DecoderRunner", DecoderRunner)
    model = _tiny_model(len(VOCAB)).eval()
    executor = _make_decode_wave_executor(
        model=model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        stats=None,
        algorithm="gumbel",
    )

    def fail(*args):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(executor.workspace.runner, "decode", fail)
    with pytest.raises(RuntimeError, match="decoder failed"):
        executor([payload(evaluator(model, 3))])
    assert executor.workspace.arena is None
    assert executor.workspace.host is None
    assert not executor.workspace.slots


def test_mutable_external_tensor_replacement_is_fingerprinted():
    model = _tiny_model(len(VOCAB)).eval()
    original = evaluator(model, 3)
    prefix = list(original._prefix_kv)
    owner = evaluator(model, 3, prefix)
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"))
    with torch.inference_mode():
        ws.prepare([payload(owner)])
    prefix[0] = tuple(t + 4 for t in prefix[0])
    with torch.inference_mode():
        args, _ = ws.prepare([payload(owner)])
    for t, expected in zip(args[2][0], prefix[0]):
        torch.testing.assert_close(t[0, :, :3], expected, atol=0, rtol=0)
    assert ws.counters["fallback_validation_calls"] == 2
    assert ws.counters["history_dirty_rows"] == 2


@pytest.mark.parametrize(
    "device,mode",
    [("cpu", "tensor"), pytest.param("cuda", "compiled", marks=pytest.mark.extended)],
)
def test_stable_placement_and_ancestor_chains(device, mode, monkeypatch):
    """Compare reordered execution to the independent legacy decoder path."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from imba_chess.eval.merged_executors import _make_decode_wave_executor

    torch.set_num_threads(4)
    torch.manual_seed(789)
    model = _tiny_model(len(VOCAB)).to(device).eval()
    if mode == "tensor":
        monkeypatch.setattr(
            "imba_chess.model.tensor_decoder.DecoderRunner", DecoderRunner
        )
    kwargs = dict(
        model=model, device=torch.device(device), dtype=torch.float32, stats=None
    )
    reference = _make_decode_wave_executor(**kwargs, algorithm="value_search_halving")
    candidate = _make_decode_wave_executor(**kwargs, algorithm="gumbel")
    ws = candidate.workspace
    prepare = ws.prepare
    depths = set()

    def checked_prepare(*args, **kwargs):
        decoder_args, state = prepare(*args, **kwargs)
        # Each row's chain names exactly its node's ancestor arena rows.
        for row, node in enumerate(state[1]):
            chain = [] if node.parent is None else node.parent.arena_chain
            assert decoder_args[4][row, : len(chain)].tolist() == chain
            assert int(decoder_args[6][row]) == len(chain)
        depths.update(decoder_args[6].tolist())
        return decoder_args, state

    monkeypatch.setattr(ws, "prepare", checked_prepare)
    a = [evaluator(model, n, immutable=True) for n in (0, 7, 19)]
    b = [evaluator(model, e._prefix_len, e._prefix_kv, immutable=True) for e in a]
    pa, pb = [None] * 3, [None] * 3
    with torch.inference_mode():
        for step in range(40):
            order = [0, 1, 2] if step % 2 == 0 else [2, 0, 1]
            if step == 34:
                order = [2, 0]
            elif step == 36:
                order = [0]
            if step == 37:
                a[1] = evaluator(model, 2, immutable=True)
                b[1] = evaluator(model, 2, a[1]._prefix_kv, immutable=True)
            aa = [payload(a[i], pa[i] if i == 0 and step < 33 else None) for i in order]
            bb = [payload(b[i], pb[i] if i == 0 and step < 33 else None) for i in order]
            copied = ws.counters["history_copied_bytes"]
            # Compiled FP32 and the legacy grouped path also differ by ~2e-6
            # on the committed implementation. Match the decoder tensor test's
            # absolute tolerance; ownership/gather checks stay exact.
            compare_results(reference(aa), candidate(bb),
                            prior_atol=1e-5 if mode == "compiled" else 1e-6)
            if 0 < step < 33:
                assert ws.counters["history_copied_bytes"] == copied
            for i, x, y in zip(order, aa, bb):
                pa[i], pb[i] = x[1][0][0], y[1][0][0]
        assert 0 in depths and 32 in depths
        reference.clear_cache()
        candidate.clear_cache()
        assert not ws.slots and ws.arena is None


@pytest.mark.extended
@pytest.mark.parametrize("mode", ["tensor", "compiled"])
def test_decoder_graphs_replay_exactly_what_eager_decoding_computes(mode, monkeypatch):
    """Two identical workspaces stage the same waves; one decodes through
    exact-shape CUDA graphs, the other eagerly. Every output must match bit for
    bit, including after shape changes, graph reuse and invalidation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from imba_chess.model.tensor_decoder import DecoderRunner as CompiledRunner

    torch.manual_seed(321)
    model = _tiny_model(len(VOCAB)).cuda().eval()
    runner_class = DecoderRunner if mode == "tensor" else CompiledRunner
    graphed, eager = DecodeWorkspace(runner_class(model)), DecodeWorkspace(runner_class(model))
    prefixes = [evaluator(model, n, immutable=True) for n in (3, 9, 17, 5)]
    owners = {
        "graphed": prefixes,
        "eager": [evaluator(model, e._prefix_len, e._prefix_kv, immutable=True) for e in prefixes],
    }
    parents = {"graphed": [None] * 4, "eager": [None] * 4}
    with torch.inference_mode():
        for step in range(30):
            rows = [0, 1, 2, 3] if step % 3 else [1, 3]
            if step == 20:  # A longer history changes the prefix shape and refreshes rows.
                owners["graphed"][0] = evaluator(model, 23, immutable=True)
                owners["eager"][0] = evaluator(
                    model, 23, owners["graphed"][0]._prefix_kv, immutable=True
                )
                parents = {"graphed": [None] * 4, "eager": [None] * 4}
            outs = {}
            for name, ws in (("graphed", graphed), ("eager", eager)):
                batch = [payload(owners[name][i], parents[name][i]) for i in rows]
                args, state = ws.prepare(batch)
                out = ws.decode(args) if name == "graphed" else ws.runner.decode(*args)
                outs[name] = (
                    out["logits"].clone(), out["value_logits"].clone(),
                    [t.clone() for pair in out["kv"] for t in pair],
                )
                ws.consume(state, out)
                for i, (_, entry) in zip(rows, batch):
                    parents[name][i] = entry[0][0] if step % 4 else None
            (lg, vg, kg), (le, ve, ke) = outs["graphed"], outs["eager"]
            assert torch.equal(lg, le) and torch.equal(vg, ve)
            assert all(torch.equal(a, b) for a, b in zip(kg, ke))
    assert graphed.counters["graph_captures"] > 0
    assert graphed.counters["graph_replays"] > 0
    graphed.clear()
    assert not graphed._graphs
