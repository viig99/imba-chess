"""Policy/value composition: lockstep, provenance and noise hoisting."""

import random

import chess
import pytest

from imba_chess.eval.batch_scheduler import WorkRequest
from imba_chess.eval.composed_runtime import (
    ComposedRuntime,
    compose_nodes,
    gumbel_noise,
)
from imba_chess.eval.gumbel_search import GumbelResult
from imba_chess.eval.search import PositionEval


def node(value, priors, ucis=("e2e4", "d2d4")):
    return PositionEval(value, list(range(len(ucis))), list(ucis), list(priors),
                        [False] * len(ucis), list(range(10, 10 + len(ucis))))


def result(uci="e2e4"):
    return GumbelResult(uci, 0, [0, 1], [0.5, 0.5], 0.0, (0.0, 1.0, 0.0),
                        [1, 0], [0.0, 0.0], 1, 1, 0, 0, 1, [0.0, 0.0])


class FakeRuntime:
    """Mirrors the real request protocol without a model."""

    algorithm = "gumbel"

    def __init__(self, name, *, value, prior, move="e2e4", waves=1):
        self.name, self.value, self.prior, self.move, self.waves = (
            name, value, prior, move, waves)
        self.move_vocab, self.encoder = object(), object()
        self.options = {"name": name}
        self.received, self.noises, self.cleared = [], [], 0
        self.executors = {
            "root_eval": self._root,
            "decode_wave": self._wave,
        }

    def _root(self, payloads):
        return [(p[0], {"logits": self.name, "value_logits": self.name,
                        "kv_caches": self.name}) for p in payloads]

    def _wave(self, payloads):
        return [(p[0], [node(self.value, [self.prior, self.prior])
                        for _ in p[1][1]]) for p in payloads]

    def clear_caches(self):
        self.cleared += 1

    def search(self, *, board, actor_id="a", game_id="g", noise=None, **kwargs):
        self.noises.append(noise)
        owner = (actor_id, game_id)
        _, root = yield WorkRequest("root_eval", (owner, {"tokens": 1}))
        self.received.append(root)
        for _ in range(self.waves):
            _, nodes = yield WorkRequest(
                "decode_wave", (owner, (object(), [(0, board)]))
            )
            self.received.append(nodes)
        return result(self.move)


def drive(runtime, board, **kwargs):
    """Run a composed search, routing requests to the named sub-runtime."""
    gen = runtime.search(board=board, **kwargs)
    response = None
    try:
        while True:
            request = gen.send(response)
            role, _, kind = request.kind.partition("_")
            target = runtime.policy if role == "policy" else runtime.value
            response = target.executors[kind]([request.payload])[0]
    except StopIteration as stop:
        return stop.value


def make(**kwargs):
    policy = FakeRuntime("policy", value=0.1, prior=-0.5, **kwargs)
    value = FakeRuntime("value", value=0.9, prior=-2.0, **kwargs)
    return ComposedRuntime(policy, value), policy, value


def test_compose_takes_priors_from_policy_and_value_from_value():
    composed = compose_nodes([node(0.1, [-0.5, -0.5])], [node(0.9, [-2.0, -2.0])])
    assert composed[0].value_stm == 0.9
    assert composed[0].legal_log_priors == [-0.5, -0.5]


@pytest.mark.parametrize(
    "value_node",
    [node(0.9, [-2.0, -2.0], ucis=("e2e4",)), node(0.9, [-2.0, -2.0], ucis=("a2a3", "d2d4"))],
)
def test_compose_rejects_legal_misalignment(value_node):
    with pytest.raises(ValueError, match="alignment|count"):
        compose_nodes([node(0.1, [-0.5, -0.5])], [value_node])


def test_lockstep_search_returns_one_result_and_composes_every_wave():
    runtime, policy, value = make(waves=3)
    assert drive(runtime, chess.Board()).move_uci == "e2e4"
    # Both sides saw identical composed nodes on every decode wave.
    waves = [r for r in policy.received if isinstance(r, list)]
    assert len(waves) == 3
    for wave in waves:
        assert wave[0].value_stm == 0.9 and wave[0].legal_log_priors == [-0.5, -0.5]
    assert [r for r in value.received if isinstance(r, list)] == waves


def test_root_keeps_each_networks_own_kv_cache():
    runtime, policy, value = make()
    drive(runtime, chess.Board())
    policy_root, value_root = policy.received[0], value.received[0]
    assert policy_root["kv_caches"] == "policy"
    assert value_root["kv_caches"] == "value"
    # Policy logits and frozen value logits cross; representations do not.
    assert policy_root["logits"] == "policy" and policy_root["value_logits"] == "value"
    assert value_root["logits"] == "policy" and value_root["value_logits"] == "value"


def test_diverging_results_are_rejected():
    policy = FakeRuntime("policy", value=0.1, prior=-0.5, move="e2e4")
    value = FakeRuntime("value", value=0.9, prior=-2.0, move="d2d4")
    with pytest.raises(ValueError, match="diverged"):
        drive(ComposedRuntime(policy, value), chess.Board())


def test_diverging_budgets_are_rejected():
    policy = FakeRuntime("policy", value=0.1, prior=-0.5, waves=1)
    value = FakeRuntime("value", value=0.9, prior=-2.0, waves=2)
    with pytest.raises(ValueError, match="diverged"):
        drive(ComposedRuntime(policy, value), chess.Board())


def test_noise_is_drawn_once_and_shared_by_both_sides():
    runtime, policy, value = make()
    board = chess.Board()
    drive(runtime, board, rng=random.Random(7))
    expected = gumbel_noise(board.legal_moves.count(), random.Random(7))
    assert policy.noises == value.noises == [expected]
    assert len(expected) == 20


def test_hoisted_noise_matches_what_a_single_runtime_would_draw():
    """The composition must consume the caller's generator identically."""
    board = chess.Board()
    shared, solo = random.Random(11), random.Random(11)
    runtime, policy, _ = make()
    drive(runtime, board, rng=shared)
    # gumbel_stepwise draws exactly one value per legal action, then stops.
    assert policy.noises[0] == gumbel_noise(board.legal_moves.count(), solo)
    assert shared.random() == solo.random()


@pytest.mark.parametrize("noise", [0.0, [0.25, 0.5]])
def test_explicit_noise_passes_through_untouched(noise):
    runtime, policy, value = make()
    drive(runtime, chess.Board(), noise=noise, rng=random.Random(3))
    assert policy.noises == value.noises == [noise]


def test_clear_caches_reaches_both_networks():
    runtime, policy, value = make()
    runtime.clear_caches()
    assert policy.cleared == value.cleared == 1


def test_composition_requires_two_distinct_gumbel_runtimes():
    policy = FakeRuntime("policy", value=0.1, prior=-0.5)
    with pytest.raises(ValueError, match="distinct"):
        ComposedRuntime(policy, policy)
    halving = FakeRuntime("halving", value=0.1, prior=-0.5)
    halving.algorithm = "value_search_halving"
    with pytest.raises(ValueError, match="Gumbel"):
        ComposedRuntime(policy, halving)


def test_executors_are_namespaced_by_role_and_options_record_both():
    runtime, policy, value = make()
    assert set(runtime.executors) == {
        "policy_root_eval", "policy_decode_wave",
        "value_root_eval", "value_decode_wave",
    }
    assert runtime.options["policy"] == {"name": "policy"}
    assert runtime.options["value"] == {"name": "value"}
    assert runtime.options["algorithm"] == "gumbel-composed"
