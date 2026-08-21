"""The decode executor must account for its own CPU pre/post work.

Instrumentation bug this pins: `_make_decode_wave_executor`'s multi-game path
timed ONLY `model.forward_decode_grouped`. Everything around it was timed by
nothing --

    requests = [evaluator.build_decode_request(b) for ...]   # untimed
    merged   = _merge_decode_requests(requests)               # untimed
    t0 = perf_counter(); forward_decode_grouped(...)          # search_gpu
    ...
    evaluator.consume_decode_result(...)                      # untimed

-- yet cProfile attributes ~18.8s of a 122.9s profiled run to exactly that
region (`_project_legal_logits_cozy` 8.50s, `torch.cat` 3.92s, `pad` 3.22s,
`encode_cozy` 3.15s self-time). So the `--profile` buckets silently omitted
real work, and two optimizations landing in that region measured as noise.

`decode_prep` closes the gap: board encoding, jagged->padded merge, and the
legal-move projection that turns raw logits back into PositionEvals.
"""

import pytest

torch = pytest.importorskip("torch")

from imba_chess.eval.merged_executors import _make_decode_wave_executor


class _Stats:
    """Minimal TimingStatsLike, including the new bucket."""

    def __init__(self) -> None:
        self.root_eval = 0.0
        self.search_gpu = 0.0
        self.decode_prep = 0.0
        self.decode_project = 0.0
        self.search_eval_calls = 0
        self.search_eval_items = 0


class _FakeRequest:
    def __init__(self, n: int) -> None:
        self.nodes = list(range(n))


class _FakeEvaluator:
    """Stands in for CachedPositionEvaluator with measurable CPU cost."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.built = 0
        self.consumed = 0

    def build_decode_request(self, batch):
        self.built += 1
        # Busy work standing in for encode_cozy + suffix cat/pad.
        sum(i * i for i in range(20000))
        return _FakeRequest(self.n)

    def consume_decode_result(self, request, out):
        self.consumed += 1
        # Busy work standing in for _project_legal_logits_cozy.
        sum(i * i for i in range(20000))
        return [f"eval{i}" for i in request.nodes]

    def evaluate(self, batch):
        return [f"eval{i}" for i in range(len(batch))]


def _install_fakes(monkeypatch, stats):
    import imba_chess.eval.merged_executors as me

    class _Merged:
        new_token_batch = {}
        positions = None
        group_index = None
        prefix_kv_grouped = []
        prefix_lens = None
        prefix_lens_list = [1]
        suffix_kv = None
        suffix_positions = None
        suffix_mask = None

    monkeypatch.setattr(me, "_merge_decode_requests", lambda reqs: _Merged())
    monkeypatch.setattr(
        me, "_split_decode_output", lambda out, counts: [None for _ in counts]
    )
    monkeypatch.setattr(me, "_autocast_context", lambda device, dtype: _noop())

    class _Model:
        def forward_decode_grouped(self, **kwargs):
            return None

    return _Model()


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_multi_game_path_accounts_for_prep_and_consume(monkeypatch):
    """decode_prep must capture build_decode_request + merge + consume."""
    stats = _Stats()
    model = _install_fakes(monkeypatch, stats)
    executor = _make_decode_wave_executor(
        model=model, device=torch.device("cpu"), dtype=torch.float32, stats=stats
    )

    payloads = [(_FakeEvaluator(3), [1, 2, 3]), (_FakeEvaluator(2), [4, 5])]
    out = executor(payloads)

    assert [len(o) for o in out] == [3, 2]
    assert all(ev.built == 1 and ev.consumed == 1 for ev, _ in payloads)
    assert stats.decode_prep > 0.0 and stats.decode_project > 0.0, (
        "build_decode_request/_merge/consume time was not accounted for"
    )
    assert stats.search_eval_items == 5


def test_single_game_path_still_only_charges_search_gpu(monkeypatch):
    """G=1 short-circuits to evaluate(); it has no separate prep phase."""
    stats = _Stats()
    _install_fakes(monkeypatch, stats)

    class _Model:
        pass

    executor = _make_decode_wave_executor(
        model=_Model(), device=torch.device("cpu"), dtype=torch.float32, stats=stats
    )
    out = executor([(_FakeEvaluator(4), [1, 2, 3, 4])])

    assert [len(o) for o in out] == [4]
    assert stats.search_gpu > 0.0
    assert stats.decode_prep == 0.0 and stats.decode_project == 0.0
    assert stats.search_eval_items == 4


def test_stats_none_is_still_supported(monkeypatch):
    """Timing is opt-in; stats=None must not crash either path."""
    model = _install_fakes(monkeypatch, None)
    executor = _make_decode_wave_executor(
        model=model, device=torch.device("cpu"), dtype=torch.float32, stats=None
    )
    assert [len(o) for o in executor([(_FakeEvaluator(2), [1, 2]), (_FakeEvaluator(1), [3])])] == [2, 1]
