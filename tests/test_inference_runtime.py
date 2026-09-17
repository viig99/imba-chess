"""Shared runtime protocol checks; tensor math is covered by decoder integration."""

from types import SimpleNamespace

import chess
import pytest
import torch

from imba_chess.eval import inference_runtime as runtime
from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.eval.search import HalvingConfig
from tests.test_self_play import VOCAB, ENCODER


def make_runtime(monkeypatch, algorithm):
    options = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def factory(**kwargs):
        options.append(kwargs)
        return lambda payloads: payloads

    monkeypatch.setattr(runtime, "_make_root_eval_executor", factory)
    monkeypatch.setattr(runtime, "_make_decode_wave_executor", factory)
    model = SimpleNamespace(training=False, parameters=lambda: iter(()))
    result = runtime.InferenceRuntime(
        model=model,
        move_vocab=VOCAB,
        encoder=ENCODER,
        device="cuda",
        algorithm=algorithm,
    )
    return result, options


@pytest.mark.parametrize(
    "algorithm,single", [("gumbel", True), ("value_search_halving", False)]
)
def test_algorithm_owns_execution_choices(monkeypatch, algorithm, single):
    instance, options = make_runtime(monkeypatch, algorithm)
    leaf = options[1]
    assert leaf["algorithm"] == algorithm
    assert (
        not {"decoder_mode", "one_query_per_game", "reuse_decode_buffers"} & leaf.keys()
    )
    assert instance.options["algorithm"] == algorithm


def test_identified_results_and_duplicate_owners(monkeypatch):
    instance, _ = make_runtime(monkeypatch, "gumbel")
    execute = instance._identified(
        "root_eval", lambda payloads: list(reversed(payloads))
    )
    assert execute([("a", 1), ("b", 2)]) == [("a", 2), ("b", 1)]
    with pytest.raises(ValueError, match="outstanding"):
        execute([("a", 1), ("a", 2)])
    execute = instance._identified("root_eval", lambda payloads: [])
    with pytest.raises(RuntimeError, match="count mismatch"):
        execute([("a", 1)])


def test_config_and_owner_checks_precede_leaf_execution(monkeypatch):
    instance, _ = make_runtime(monkeypatch, "value_search_halving")
    args = dict(board=chess.Board(), batch={}, owner=("model", "game"))
    with pytest.raises(ValueError, match="does not match"):
        next(instance.search_batch(**args, config=GumbelConfig()))
    gen = instance.search_batch(**args, config=HalvingConfig())
    request = next(gen)
    assert request.kind == "root_eval"
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        gen.send((("different-model", "game"), {}))


def test_cancel_before_root_and_context_cleanup(monkeypatch):
    instance, _ = make_runtime(monkeypatch, "gumbel")
    with pytest.raises(InterruptedError):
        next(
            instance.search_batch(
                board=chess.Board(),
                batch={},
                owner=("a", "b"),
                config=GumbelConfig(),
                should_stop=lambda: True,
            )
        )
    cleared = []
    for kind, executor in instance.executors.items():
        executor.clear_cache = lambda kind=kind: cleared.append(kind)
    with pytest.raises(RuntimeError):
        with instance:
            raise RuntimeError("failed evaluation")
    assert sorted(cleared) == ["decode_wave", "root_eval"]


def test_cpu_is_rejected_before_decoder_construction():
    with pytest.raises(ValueError, match="requires CUDA"):
        runtime.InferenceRuntime(
            model=None, move_vocab=None, encoder=None, device="cpu"
        )


@pytest.mark.parametrize(
    "algorithm,config",
    [
        ("gumbel", GumbelConfig(simulations=16, top_m=4)),
        ("value_search_halving", HalvingConfig(budget=32, top_m=4, max_depth=3)),
    ],
)
def test_shared_dispatch_preserves_search_results(monkeypatch, algorithm, config):
    import random
    from imba_chess.eval.position_evaluator import _project_legal_logits
    from imba_chess.eval.gumbel_search import select_gumbel
    from imba_chess.eval.search import select_value_search_halving
    from imba_chess.self_play.benchmarks import UniformEvaluator

    instance, _ = make_runtime(monkeypatch, algorithm)
    evaluator = UniformEvaluator(VOCAB)
    monkeypatch.setattr(runtime, "CachedPositionEvaluator", lambda **kwargs: evaluator)
    board = chess.Board()
    logits = torch.zeros(len(VOCAB))
    root = dict(logits=logits[None], value_logits=torch.zeros(1, 3), kv_caches=[])
    owner = ("actor", "game")
    noise = 0.0 if algorithm == "gumbel" else None
    gen = instance.search_batch(
        board=board,
        batch={"total_tokens": 1},
        owner=owner,
        config=config,
        rng=random.Random(42),
        noise=noise,
    )
    request = next(gen)
    wave_sizes = []
    try:
        while True:
            if request.kind == "root_eval":
                answer = root
            else:
                requested_evaluator, batch = request.payload[1]
                assert requested_evaluator is evaluator
                wave_sizes.append(len(batch))
                answer = evaluator.evaluate(batch)
            request = gen.send((owner, answer))
    except StopIteration as stop:
        actual = stop.value
    if algorithm == "gumbel":
        expected = select_gumbel(
            evaluator=evaluator,
            board=board,
            config=config,
            noise=[0.0] * board.legal_moves.count(),
        )
        assert actual.move_uci == expected.move_uci
        assert actual.visits == expected.visits
        assert actual.policy == pytest.approx(expected.policy, abs=1e-6)
        assert max(wave_sizes) == 1
    else:
        legal_logits, moves, _, _ = _project_legal_logits(
            logits=logits, board=board, move_vocab=VOCAB
        )
        index, rows = select_value_search_halving(
            evaluator=evaluator,
            root_handle=None,
            board=board,
            legal_moves=moves,
            legal_log_priors=torch.log_softmax(legal_logits, 0).tolist(),
            config=config,
        )
        assert actual.move_uci == moves[index].uci()
        assert actual.candidates == rows
        assert max(wave_sizes) > 1


def test_refresh_rejects_pending_root(monkeypatch):
    instance, _ = make_runtime(monkeypatch, "gumbel")
    gen = instance.search_batch(
        board=chess.Board(), batch={}, owner=("a", "b"), config=GumbelConfig()
    )
    next(gen)
    instance.clear_caches()
    with pytest.raises(RuntimeError, match="invalidated"):
        gen.send((("a", "b"), {}))


def test_leaf_cache_rejects_foreign_and_stale_owners(monkeypatch):
    first, _ = make_runtime(monkeypatch, "value_search_halving")
    second, _ = make_runtime(monkeypatch, "value_search_halving")
    evaluator = SimpleNamespace(_runtime_token=first._cache_token)
    payload = [("game", (evaluator, []))]
    execute = first._identified("decode_wave", lambda items: [[] for item in items])
    assert execute(payload) == [("game", [])]
    with pytest.raises(RuntimeError, match="foreign"):
        second.executors["decode_wave"](payload)
    first.clear_caches()
    with pytest.raises(RuntimeError, match="stale"):
        execute(payload)
