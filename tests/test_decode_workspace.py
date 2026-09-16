"""Cross-game workspace equivalence and ownership/lifetime regressions."""

import gc
import weakref

import chess
import pytest
import torch

from imba_chess.eval import cozy_bridge
from imba_chess.eval.decode_workspace import DecodeWorkspace
from imba_chess.eval.merged_executors import _merge_decode_requests
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    consume_batched_decode_results,
)
from imba_chess.model.tensor_decoder import DecoderRunner
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


def compare_results(a, b):
    for x, y in zip(a, b):
        x, y = x[0], y[0]
        assert x.legal_ids == y.legal_ids and x.legal_ucis == y.legal_ucis
        assert x.legal_forcing == y.legal_forcing
        assert x.value_stm == pytest.approx(y.value_stm, abs=1e-6)
        torch.testing.assert_close(
            torch.tensor(x.legal_log_priors),
            torch.tensor(y.legal_log_priors),
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)]
)
@pytest.mark.parametrize(
    "mode", ["tensor", pytest.param("compiled", marks=pytest.mark.extended)]
)
@pytest.mark.parametrize("cache_mode", ["current", "revision", "direct"])
def test_workspace_mixed_depth_owner_changes_and_kv_lifetime(device, mode, cache_mode):
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
    ws = DecodeWorkspace(DecoderRunner(model, mode), history_cache_mode=cache_mode)
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
                t for pairs in args[2:4] for pair in pairs for t in pair
            ]
            assert len({t.untyped_storage().data_ptr() for t in inputs}) == len(inputs)
            assert all(
                t.is_contiguous() for pairs in args[2:4] for pair in pairs for t in pair
            )
            for key in merged.new_token_batch:
                torch.testing.assert_close(
                    args[0][key].cpu(), merged.new_token_batch[key], atol=0, rtol=0
                )
            torch.testing.assert_close(args[1].cpu(), merged.positions, atol=0, rtol=0)
            for got, ref in zip(args[2], merged.prefix_kv_grouped):
                for x, y in zip(got, ref):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
                    assert x._is_view() == y._is_view()
                    if x._base is not None:
                        assert x._base.shape == y._base.shape
                        assert x._base.stride() == y._base.stride()
            if merged.suffix_kv is not None:
                width = merged.suffix_positions.size(1)
                for got, ref in zip(args[3], merged.suffix_kv):
                    for x, y in zip(got, ref):
                        mask = merged.suffix_mask[:, None, :, None].expand_as(y)
                        torch.testing.assert_close(
                            x[:, :, :width][mask], y[mask], atol=1e-5, rtol=1e-5
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


@pytest.mark.parametrize("cache_mode", ["current", "revision", "direct"])
def test_arena_growth_slot_reclamation_and_empty_projection(monkeypatch, cache_mode):
    torch.set_num_threads(4)
    model = _tiny_model(len(VOCAB)).eval()
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"), history_cache_mode=cache_mode)
    owner = evaluator(model, 3)
    original_projection = cozy_bridge.project_legal_moves
    monkeypatch.setattr(
        cozy_bridge, "project_legal_moves", lambda *a: ([], [], [], [], 0)
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
        monkeypatch.setattr(cozy_bridge, "project_legal_moves", original_projection)
        args, state = ws.prepare([payload(owner)])
        ws.consume(state, ws.runner.decode(*args))
        assert len(ws.slots) == 1 and ws.arena[0].data_ptr() == pointer
        assert ws.counters["history_refreshes"] == 2
        ws.clear()


@pytest.mark.parametrize("cache_mode", ["current", "revision", "direct"])
def test_scratch_collect_update_collect_and_cancellation(tmp_path, cache_mode):
    from dataclasses import replace
    from imba_chess.data.self_play_store import SelfPlayStore
    from imba_chess.self_play.collector import InferenceRuntime, collect
    from imba_chess.self_play.config import SelfPlayConfig
    from imba_chess.self_play.seeds import Seed
    from imba_chess.self_play.trainer import Stage2Trainer
    from scripts.profile_gumbel_pipeline import compare_targets
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
            history_cache_mode=cache_mode,
            native_gumbel=True,
        )
        for reuse in (False, True)
    ]
    cfg = SelfPlayConfig()
    cfg = replace(cfg, collection=replace(cfg.collection, concurrent_games=3))
    seeds = [Seed("mate", "source", ["f2f3", "e7e5", "g2g4"], 3, "train", "test")]
    for cycle in range(2):
        games = []
        for variant, runtime in enumerate(runtimes):
            store = SelfPlayStore(tmp_path / f"{cycle}-{variant}", flush_games=1)
            result = []
            collect(
                seeds=seeds,
                runtime=runtime,
                config=cfg,
                actor_id="scratch",
                store=store,
                max_positions=128,
                game_count=4,
                on_game=result.append,
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
        seeds=seeds,
        runtime=runtimes[1],
        config=cfg,
        actor_id="scratch",
        store=SelfPlayStore(tmp_path / "cancel"),
        max_positions=128,
        game_count=4,
        should_stop=stop,
        on_game=interrupted.append,
    )
    assert any(g["status"] == "unfinished" and not g["targets"] for g in interrupted)


@pytest.mark.parametrize("cache_mode", ["current", "revision", "direct"])
def test_history_refresh_and_padding_shrink_without_owner_change(cache_mode):
    model = _tiny_model(len(VOCAB)).eval()
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"), history_cache_mode=cache_mode)
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
                assert torch.count_nonzero(tensor[0, :, 3:]) == 0
        ws.consume(state, ws.runner.decode(*args))
        assert ws.counters["history_refreshes"] == 3
        assert ws.counters["history_dirty_rows"] == 3
        parent = state[0][0].nodes[0]
        ws.clear()
        with pytest.raises(RuntimeError, match="ownership"):
            ws.prepare([payload(owner, parent)])


def test_profile_compares_targets_only_within_identical_collection_sizes():
    from scripts.profile_gumbel_pipeline import compare_workload_targets

    references = {}
    warmup = [{"id": "a", "targets": [{"value": 0.5, "visits": [1, 0]}]}]
    measured = [
        {"id": "a", "targets": [{"value": 0.50001, "visits": [0, 1]}]},
        {"id": "b", "targets": []},
    ]
    compare_workload_targets(references, 1, warmup)
    compare_workload_targets(references, 2, measured)
    compare_workload_targets(references, 2, measured)
    with pytest.raises(AssertionError):
        compare_workload_targets(references, 2, warmup + measured[1:])


@pytest.mark.parametrize("cache_mode", ["revision", "direct"])
def test_immutable_revision_and_stale_handles(cache_mode, monkeypatch):
    model = _tiny_model(len(VOCAB)).eval()
    owner = evaluator(model, 3, immutable=True)
    other = evaluator(model, 3, immutable=True)
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"), history_cache_mode=cache_mode)

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
                torch.testing.assert_close(t[0], ref, atol=0, rtol=0)
        assert ws.counters["fast_validation_calls"] == 2
        assert ws.counters["fallback_validation_calls"] == 0


@pytest.mark.parametrize("immutable", [False, True])
def test_direct_dirty_rows_batch_resize_and_width_changes(immutable):
    model = _tiny_model(len(VOCAB)).eval()
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"), history_cache_mode="direct")
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
                    assert torch.count_nonzero(t[row, :, n:]) == 0
        if not dirty:
            for key in (
                "history_copied_bytes",
                "history_zeroed_bytes",
                "history_copy_submissions",
                "history_zero_submissions",
            ):
                assert ws.counters[key] == before[key]
        assert ws.prefix is None and all(not slot.history for slot in ws.slots.values())

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
    prepare([0, 1], 2)  # logical width changes within storage capacity
    owners[0].replace_history(evaluator(model, 19)._prefix_kv, 19)
    prepare([0, 1], 2)
    owners[0].replace_history(evaluator(model, 2)._prefix_kv, 2)
    prepare([0, 1], 2)
    assert ws.counters["history_full_refreshes"] == 6


def test_workspace_executor_error_cleans_buffers(monkeypatch):
    from imba_chess.eval.merged_executors import _make_decode_wave_executor

    model = _tiny_model(len(VOCAB)).eval()
    executor = _make_decode_wave_executor(
        model=model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        stats=None,
        one_query_per_game=True,
        decoder_mode="tensor",
        reuse_decode_buffers=True,
        history_cache_mode="direct",
    )

    def fail(*args):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(executor.workspace.runner, "decode", fail)
    with pytest.raises(RuntimeError, match="decoder failed"):
        executor([payload(evaluator(model, 3))])
    assert executor.workspace.arena is None
    assert executor.workspace.host is None
    assert not executor.workspace.slots


@pytest.mark.parametrize("cache_mode", ["current", "revision", "direct"])
def test_mutable_external_tensor_replacement_is_fingerprinted(cache_mode):
    model = _tiny_model(len(VOCAB)).eval()
    original = evaluator(model, 3)
    prefix = list(original._prefix_kv)
    owner = evaluator(model, 3, prefix)
    ws = DecodeWorkspace(DecoderRunner(model, "tensor"), history_cache_mode=cache_mode)
    with torch.inference_mode():
        ws.prepare([payload(owner)])
    prefix[0] = tuple(t + 4 for t in prefix[0])
    with torch.inference_mode():
        args, _ = ws.prepare([payload(owner)])
    for t, expected in zip(args[2][0], prefix[0]):
        torch.testing.assert_close(t[0], expected, atol=0, rtol=0)
    assert ws.counters["fallback_validation_calls"] == 2
    assert ws.counters["history_dirty_rows"] == 2
