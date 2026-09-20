import json
import math
from pathlib import Path
import chess
import pytest
import torch

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.self_play_store import SelfPlayStore
from imba_chess.eval.gumbel_search import GumbelConfig, GumbelResult
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.self_play.collector import play_game
from imba_chess.self_play.dataset import reconstruct, collate_self_play
from imba_chess.self_play.losses import self_play_loss
from imba_chess.self_play.seeds import Seed, source_split
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.trainer import Stage2Trainer
from imba_chess.model import HSTUChessConfig, HSTUChessModel


VOCAB = MoveVocab.load("artifacts/move_vocab_static_uci.json")
ENCODER = BoardStateEncoder()


class ScriptRuntime:
    move_vocab = VOCAB
    encoder = ENCODER

    def search(self, *, board, **kwargs):
        yield WorkRequest("tick", None)
        uci = ["f2f3", "e7e5", "g2g4", "d8h4"][len(board.move_stack)]
        ids = [VOCAB.encode(m.uci()) for m in board.legal_moves]
        return GumbelResult(
            uci,
            VOCAB.encode(uci),
            ids,
            [1 / len(ids)] * len(ids),
            0.0,
            (0.0, 1.0, 0.0),
            [0] * len(ids),
            [0.0] * len(ids),
            1,
            1,
            0,
            0,
            1,
        )


def mate_game(gid="g", prefix=None):
    prefix = [] if prefix is None else prefix
    source = "train-source"
    while source_split(source) != "train":
        source += "x"
    seed = Seed("s", source, prefix, len(prefix), "train", "c")
    gen = play_game(
        seed=seed,
        game_id=gid,
        actor_id="a",
        runtime=ScriptRuntime(),
        search_config=GumbelConfig(simulations=1, max_depth=1),
        max_positions=128,
    )
    try:
        next(gen)
        while True:
            gen.send(None)
    except StopIteration as stop:
        return stop.value


def test_full_trajectory_alignment_and_prefix_mask():
    game = mate_game(prefix=["f2f3", "e7e5"])
    assert game["status"] == "completed" and game["outcome_white"] == -1
    sample = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128)
    assert sample["supervised_indices"] == [3, 4]
    assert sample["has_value_target"] == [False, False, False, True, True]
    assert sample["value_target"][3:] == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert sample["target_move_id"] == [-100] * 5
    batch = collate_self_play([sample, sample])
    assert batch["supervised_indices"].tolist() == [3, 4, 8, 9]
    assert (
        len(sample["seq_token_id"]) == 5
    )  # BOS + 4 pre-move boards, no terminal board


def test_limits_do_not_fabricate_outcomes():
    seed = Seed("s", "train-source", [], 0, source_split("train-source"), "c")
    gen = play_game(
        seed=seed,
        game_id="x",
        actor_id="a",
        runtime=ScriptRuntime(),
        search_config=GumbelConfig(),
        max_positions=2,
    )
    with pytest.raises(StopIteration) as exc:
        next(gen)
    assert exc.value.value["status"] == "unfinished"
    assert exc.value.value["outcome_white"] is None
    assert exc.value.value["targets"] == []
    gen = play_game(
        seed=seed,
        game_id="x",
        actor_id="a",
        runtime=ScriptRuntime(),
        search_config=GumbelConfig(max_depth=1),
        max_positions=128,
        max_game_plies=4,
    )
    try:
        next(gen)
        while True:
            gen.send(None)
    except StopIteration as exc:
        assert exc.value["status"] == "completed"  # mate before administrative cap


def test_replay_recovery_dedup_eviction(tmp_path):
    store = SelfPlayStore(tmp_path, window_positions=8, flush_games=1)
    for i in range(3):
        store.add(mate_game(str(i)))
    assert store.game_ids() == ["1", "2"]
    assert not store.add(mate_game("0"))
    assert store.read_game("1")["moves"] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    (tmp_path / "orphan.tmp").write_text("bad")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["shards"].pop()
    manifest["active"] = ["1"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    recovered = SelfPlayStore(tmp_path, window_positions=8)
    assert recovered.game_ids() == ["1", "2"]
    assert not recovered.add(mate_game("2"))
    recovered.collect_garbage(pinned_shards=[])
    assert len(list(tmp_path.glob("shard-*.parquet"))) == 2
    assert not SelfPlayStore(tmp_path, window_positions=8).add(mate_game("0"))


def test_completion_order_option():
    def game(n):
        for _ in range(n):
            yield WorkRequest("tick", n)
        return [n]

    output = []
    BatchScheduler(
        game_factory=iter([("slow", game(5)), ("fast", game(1)), ("refill", game(1))]),
        executors={"tick": lambda ps: ps},
        concurrent_games=2,
        completion_order=True,
        on_game_done=lambda gid, r: output.append(gid),
        on_game_error=lambda *a: None,
    ).run()
    assert output == ["fast", "refill", "slow"]


def test_sparse_loss_by_hand_and_finite_gradients():
    logits = torch.tensor(
        [[900.0, 0.0, 0.0], [900.0, math.log(3), 0.0]], requires_grad=True
    )
    value = torch.zeros((2, 3), requires_grad=True)
    batch = dict(
        supervised_indices=torch.tensor([0, 1]),
        legal_ids=torch.tensor([[1, 2], [1, 0]]),
        legal_mask=torch.tensor([[True, True], [True, False]]),
        policy=torch.tensor([[0.25, 0.75], [1.0, 0.0]]),
        value_target=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    loss = self_play_loss(dict(logits=logits, value_logits=value), batch)
    assert loss["policy_loss"].item() == pytest.approx(math.log(2) / 2)
    assert loss["value_loss"].item() == pytest.approx(math.log(3))
    assert loss["model_policy_entropy"].item() == pytest.approx(math.log(2) / 2)
    assert loss["policy_entropy"].item() == pytest.approx(
        -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / 2
    )
    loss["loss"].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(value.grad).all()
    assert logits.grad[:, 0].tolist() == [0, 0]
    torch.testing.assert_close(
        logits.grad, torch.tensor([[0.0, 0.125, -0.125], [0.0, 0.0, 0.0]])
    )
    # Draw logits must not affect conditional win/loss discrimination.
    for draw_logit in (0.0, 100.0):
        values = torch.tensor([[0.0, draw_logit, math.log(3)]]).repeat(2, 1)
        metrics = self_play_loss(dict(logits=logits, value_logits=values), batch)
        assert metrics["decisive_positions"] == 1
        assert metrics["conditional_wl_accuracy"] == 1
        assert metrics["conditional_wl_loss"].item() == pytest.approx(-math.log(.75))
    batch["value_target"][:] = torch.tensor([0.0, 1.0, 0.0])
    metrics = self_play_loss(dict(logits=logits, value_logits=value), batch)
    assert metrics["decisive_positions"] == 0
    assert metrics["conditional_wl_loss"] == 0  # No decisive samples; ignore this batch.


def tiny_model():
    return HSTUChessModel(
        HSTUChessConfig(
            move_vocab_size=len(VOCAB),
            model_dim=16,
            linear_hidden_dim=4,
            attention_dim=4,
            num_heads=2,
            num_layers=1,
            max_position_embeddings=128,
            enable_value_head=True,
        )
    )


def test_trainer_overfit_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    store = SelfPlayStore(tmp_path / "replay", flush_games=1)
    store.add(mate_game())
    model = tiny_model()
    config = LearningConfig(lr=0.01)

    def trainer(model):
        return Stage2Trainer(
            model=model,
            config=config,
            move_vocab=VOCAB,
            encoder=ENCODER,
            device=torch.device("cpu"),
            max_positions=128,
        )

    a = trainer(model)
    a.begin_phase(store)
    records = []
    a.train(store, exposure_budget=80, on_step=records.append)
    assert records[-1]["loss"] < records[0]["loss"] * 0.8
    a.checkpoint(
        tmp_path / "state.pt", progress={"phase": "train"}, store=store, config_id="cfg"
    )
    a.train(store, exposure_budget=88)
    b = trainer(tiny_model())
    assert b.resume(tmp_path / "state.pt", store=store, config_id="cfg") == {
        "phase": "train"
    }
    b.train(store, exposure_budget=88)
    assert a.exposures == b.exposures == 88
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_paired_evaluation_gates():
    from imba_chess.self_play.evaluation import paired_interval, decision

    rows = [
        dict(
            pair=i,
            candidate_white=white,
            status="completed",
            outcome_white=1 if white else -1,
        )
        for i in range(5)
        for white in (True, False)
    ]
    interval = paired_interval(rows, pairs=5, samples=100)
    assert interval["score"] == interval["lower"] == interval["upper"] == 1
    assert decision(interval, interval) == "promote"
    assert decision({"upper": 0.44}) == "rollback_stop"
    assert decision({"upper": 0.6}) == "retain_best"
    with pytest.raises(ValueError, match="every planned"):
        paired_interval(rows[:-1], pairs=5)


def test_root_executor_token_bound(monkeypatch):
    from imba_chess.eval import merged_executors as module
    from imba_chess.eval.position_evaluator import _SequenceHistory

    batches = []
    for moves in [[], ["e2e4"], ["e2e4", "e7e5"]]:
        history = _SequenceHistory(move_vocab=VOCAB, board_state_encoder=ENCODER)
        board = chess.Board()
        for uci in moves:
            history.append_observed_position(board)
            history.record_played_move(uci)
            board.push_uci(uci)
        batches.append(history.build_batch_for_current_position(board))
    sizes = []

    def forward(**kwargs):
        size = kwargs["batch"]["total_tokens"]
        sizes.append(size)
        return dict(
            logits=torch.zeros(size, 2),
            value_logits=torch.zeros(size, 3),
            kv_caches=[(torch.zeros(1, size, 1), torch.zeros(1, size, 1))],
        )

    monkeypatch.setattr(module, "_forward_model", forward)
    out = module._make_root_eval_executor(
        model=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        stats=None,
        max_tokens=5,
    )(batches)
    assert sizes == [5, 4]
    assert [x["logits"].shape[0] for x in out] == [2, 3, 4]
    with pytest.raises(ValueError, match="token limit"):
        module._make_root_eval_executor(
            model=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            stats=None,
            max_tokens=3,
        )(batches)


def test_run_lock_and_stop_budget(tmp_path):
    from imba_chess.self_play.runtime import run_lock, StopBudget

    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with run_lock(tmp_path):
                pass
    with StopBudget(seconds=100, reserve_seconds=20, drain_seconds=10) as budget:
        assert budget.launch() and not budget.stop_collection()
        budget._signal(None, None)
        assert not budget.launch() and budget.stop() and not budget.stop_collection()
        budget._signal(None, None)
        assert budget.stop_collection()



def test_continuous_stop_budget_remains_interruptible():
    from imba_chess.self_play.runtime import StopBudget

    with StopBudget(seconds=float("inf"), drain_seconds=10) as budget:
        assert budget.timer is None
        assert budget.launch() and not budget.stop()
        budget._signal(None, None)
        assert not budget.launch() and budget.stop()
        assert not budget.stop_collection()
        budget._signal(None, None)
        assert budget.stop_collection()


def test_corrupt_outcome_rejected():
    game = mate_game()
    game["outcome_white"] = 1
    with pytest.raises(ValueError, match="actual terminal"):
        reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128)


@pytest.mark.parametrize("failure_stage", ["screen", "confirmation"])
@pytest.mark.parametrize("timed_screen", [False, True])
def test_multiple_iterations_and_runner_resume(tmp_path, monkeypatch, failure_stage, timed_screen):
    import sys
    import scripts.run_self_play as runner
    from imba_chess.self_play.config import (
        SelfPlayConfig,
        RunConfig,
        CollectionConfig,
        ReplayConfig,
    )

    class IdentifiedRuntime(ScriptRuntime):
        def __init__(self, model):
            self.model = model
            self.device = torch.device("cpu")
            self.executors = {"tick": lambda payloads: payloads}

        def search(self, *, board, actor_id, game_id, **kwargs):
            owner = (actor_id, game_id)
            identity, _ = yield WorkRequest("tick", (owner, None))
            assert identity == owner
            uci = ["f2f3", "e7e5", "g2g4", "d8h4"][len(board.move_stack)]
            ids = [VOCAB.encode(m.uci()) for m in board.legal_moves]
            return GumbelResult(
                uci,
                VOCAB.encode(uci),
                ids,
                [1 / len(ids)] * len(ids),
                0.0,
                (0.0, 1.0, 0.0),
                [0] * len(ids),
                [0.0] * len(ids),
                1,
                1,
                0,
                0,
                1,
            )

    def runtime(cfg, checkpoint, device):
        model = tiny_model()
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
        return IdentifiedRuntime(model), 128

    cfg = SelfPlayConfig(
        search=GumbelConfig(simulations=1, max_depth=1),
        collection=CollectionConfig(concurrent_games=2, fresh_positions=2),
        replay=ReplayConfig(window_positions=20, flush_games=2),
        run=RunConfig(screen_pairs=1, confirmation_pairs=1),
    )
    source = "monitor"
    while source_split(source) != "monitor":
        source += "x"
    seeds = [
        Seed(
            "train",
            "train-source",
            ["f2f3", "e7e5", "g2g4"],
            3,
            source_split("train-source"),
            "c",
        ),
        Seed("monitor", source, ["f2f3", "e7e5", "g2g4"], 3, "monitor", "c"),
    ]
    assert seeds[0].split == "train"
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text("test")
    monkeypatch.setattr(runner, "load_config", lambda path: cfg)
    monkeypatch.setattr(runner, "load_runtime", runtime)
    monkeypatch.setattr(runner, "load_seeds", lambda path: seeds)
    output = tmp_path / "run"
    base = [
        "run_self_play.py",
        "--config",
        "unused",
        "--seeds",
        str(seed_path),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        base + ["--initialize", str(tmp_path / "weights.pt"), "--max-iterations", "2"],
    )
    runner.main()
    state = json.loads((output / "state.json").read_text())
    assert state["iteration"] == 2 and state["phase"] == "collect"
    checkpoint = torch.load(state["checkpoint"], weights_only=False)
    assert checkpoint["exposures"] >= 8
    assert len(list(output.glob("*.pt"))) <= 3
    monkeypatch.setattr(sys, "argv", base + ["--resume", "--max-iterations", "3"])
    runner.main()
    state = json.loads((output / "state.json").read_text())
    assert state["iteration"] == 3 and not state["halted"]
    replay = SelfPlayStore(output / "replay", window_positions=20)
    assert sum(replay.index[g][1]["positions"] for g in replay.game_ids(None)) <= 20

    # Operational screen cadence changes must preserve optimizer/config resume.
    original_evaluate = runner.evaluate_pair_checkpoints
    calls = []

    def deferred_confirmation(**kwargs):
        calls.append(kwargs["output"].name)
        assert kwargs["output"].name.startswith("screen-")
        return dict(score=0.6, lower=0.45, upper=0.75)

    monkeypatch.setattr(runner, "evaluate_pair_checkpoints", deferred_confirmation)
    monkeypatch.setattr(
        sys,
        "argv",
        base
        + [
            "--resume",
            "--max-iterations",
            "6",
            "--screen-every",
            "3",
            "--checkpoint-seconds",
            "3600",
            "--keep-recovery-checkpoints",
            "2",
            "--defer-confirmation",
        ] + (["--screen-seconds", "10800"] if timed_screen else []),
    )
    runner.main()
    state = json.loads((output / "state.json").read_text())
    assert state["iteration"] == 6 and not state["halted"]
    assert calls == ([] if timed_screen else ["screen-000005.json"])
    assert len(list(output.glob("state-*.pt"))) == 2
    for key in ("actor", "best", "checkpoint"):
        assert Path(state[key]).exists()
    assert state["best"].endswith("actor-000000.pt")
    # Research screens remain observable without halting or promoting best.
    for upper, limit in ((0.40, 7), (0.80, 8)):
        monkeypatch.setattr(
            runner, "evaluate_pair_checkpoints",
            lambda **kw: dict(score=upper - .05, lower=upper - .1, upper=upper),
        )
        monkeypatch.setattr(sys, "argv", base + [
            "--resume", "--max-iterations", str(limit), "--observe-only-screen",
        ])
        runner.main()
        state = json.loads((output / "state.json").read_text())
        assert state["iteration"] == limit and not state["halted"]
        assert state["best"].endswith("actor-000000.pt")
        assert state["actor"] != state["best"]
    assert '"recommended_decision": "rollback_stop"' in (output / "metrics.jsonl").read_text()
    monkeypatch.setattr(runner, "evaluate_pair_checkpoints", original_evaluate)

    def failed_evaluation(**kwargs):
        if failure_stage == "confirmation" and kwargs["output"].name.startswith(
            "screen-"
        ):
            return dict(score=0.75, lower=0.5, upper=1.0)
        raise runner.EvaluationProtocolError("game_limit")

    monkeypatch.setattr(runner, "evaluate_pair_checkpoints", failed_evaluation)
    monkeypatch.setattr(sys, "argv", base + ["--resume", "--max-iterations", "9"])
    runner.main()
    state = json.loads((output / "state.json").read_text())
    assert state["halted"] and state["halt_reason"] == "evaluation_protocol"
    assert state["phase"] == "evaluate" and state["iteration"] == 8
    assert state["actor"] != state["best"]
    assert all(Path(state[key]).exists() for key in ("actor", "best", "checkpoint"))
    assert "protocol_stop" in (output / "metrics.jsonl").read_text()
    with pytest.raises(RuntimeError, match="automatic learning was halted"):
        runner.main()


def test_hard_budget_exits_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from imba_chess.self_play.runtime import StopBudget; import time\nwith StopBudget(seconds=.1,hard_exit=True): time.sleep(10)",
        ],
        timeout=10,
    )
    assert result.returncode == 124


def test_collection_resume_and_executor_failure(tmp_path):
    from dataclasses import replace
    from imba_chess.self_play.collector import collect, CollectionMetrics
    from imba_chess.self_play.config import SelfPlayConfig

    cfg = SelfPlayConfig()
    cfg = replace(
        cfg,
        search=GumbelConfig(simulations=1, max_depth=1),
        collection=replace(cfg.collection, concurrent_games=1),
    )
    seed = Seed("s", "train-source", ["f2f3", "e7e5"], 2, "train", "c")
    runtime = ScriptRuntime()
    runtime.executors = {"tick": lambda ps: ps}
    store = SelfPlayStore(tmp_path / "partial", flush_games=16)
    stopped = []
    common = dict(
        seeds=[seed],
        runtime=runtime,
        config=cfg,
        actor_id="actor",
        max_positions=128,
        game_count=3,
    )
    collect(
        **common,
        store=store,
        should_launch=lambda: not stopped,
        on_game=lambda game: stopped.append(game["game_id"]),
    )
    assert len(store.game_ids()) == 1
    collect(**common, store=store, skip_ids=store.seen)
    whole = SelfPlayStore(tmp_path / "whole")
    collect(**common, store=whole, concurrent_games=3)
    assert store.game_ids() == whole.game_ids()
    assert [store.read_game(g) for g in store.game_ids()] == [
        whole.read_game(g) for g in whole.game_ids()
    ]

    def fail(payloads):
        raise RuntimeError("inference failed")

    runtime.executors = {"tick": fail}
    metrics = CollectionMetrics()
    unfinished = []
    with pytest.raises(RuntimeError, match="inference failed"):
        collect(
            **common,
            store=SelfPlayStore(tmp_path / "failed"),
            metrics=metrics,
            on_game=unfinished.append,
        )
    assert metrics.counts["unfinished_games"] == 1
    assert unfinished[0]["seed_id"] == "s" and unfinished[0]["targets"] == []
    assert "inference failed" in unfinished[0]["error"]


def test_read_only_replay_preserves_published_snapshot(tmp_path):
    writer = SelfPlayStore(tmp_path, flush_games=1)
    writer.add(mate_game("first"))
    manifest = tmp_path / "manifest.json"
    before = manifest.read_bytes()
    stamp = manifest.stat().st_mtime_ns
    # Opening with different defaults must not trim or republish a writer's window.
    reader = SelfPlayStore(tmp_path, read_only=True, window_positions=1)
    assert reader.game_ids() == ["first"]
    assert manifest.read_bytes() == before
    assert manifest.stat().st_mtime_ns == stamp
    writer.add(mate_game("second"))
    updated = manifest.read_bytes()
    assert reader.read_game("first")["game_id"] == "first"
    for operation in (
        lambda: reader.add(mate_game("third")),
        reader.flush,
        lambda: reader.collect_garbage(pinned_shards=[]),
        reader._publish,
    ):
        with pytest.raises(PermissionError, match="read-only"):
            operation()
    assert manifest.read_bytes() == updated
    assert SelfPlayStore(tmp_path, read_only=True).game_ids() == ["first", "second"]
    # A reader also leaves recovery to the writer; unindexed shards stay invisible.
    manifest.write_bytes(before)
    assert SelfPlayStore(tmp_path, read_only=True).game_ids() == ["first"]
    assert manifest.read_bytes() == before
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        SelfPlayStore(missing, read_only=True)
    assert not missing.exists()


@pytest.mark.parametrize("reason", ["game_limit", "context_limit"])
def test_evaluation_permanent_caps_do_not_retry(tmp_path, monkeypatch, reason):
    from types import SimpleNamespace
    import imba_chess.self_play.evaluation as evaluation
    from imba_chess.self_play.config import SelfPlayConfig

    calls = []

    def capped(**kwargs):
        calls.append(kwargs["game_id"])
        yield from ()
        return dict(status="unfinished", termination=reason, outcome_white=None)

    monkeypatch.setattr(evaluation, "play_game", capped)
    runtime = SimpleNamespace(executors={})
    output = tmp_path / "eval.json"
    args = dict(
        candidate=runtime,
        best=runtime,
        candidate_id="a",
        best_id="b",
        seeds=[Seed("s", "source", [], 0, "monitor", "c")],
        config=SelfPlayConfig(),
        max_positions=128,
        output=output,
        pairs=1,
    )
    with pytest.raises(evaluation.EvaluationProtocolError, match=reason):
        evaluation.evaluate_pair_checkpoints(**args)
    state = json.loads(output.read_text())
    assert state["protocol_failure"] and "interval" not in state
    assert all(r["outcome_white"] is None for r in state["results"].values())
    count = len(calls)
    # Also exercise recovery of old progress without the new failure marker.
    del state["protocol_failure"]
    output.write_text(json.dumps(state))
    with pytest.raises(evaluation.EvaluationProtocolError, match=reason):
        evaluation.evaluate_pair_checkpoints(**args)
    assert len(calls) == count


def test_evaluation_interruption_remains_resumable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import imba_chess.self_play.evaluation as evaluation
    from imba_chess.self_play.config import SelfPlayConfig

    interrupted = True
    calls = []

    def game(**kwargs):
        calls.append(kwargs["game_id"])
        yield from ()
        return dict(
            status="unfinished" if interrupted else "completed",
            termination="interrupted" if interrupted else "checkmate",
            outcome_white=None if interrupted else 1,
        )

    monkeypatch.setattr(evaluation, "play_game", game)
    runtime = SimpleNamespace(executors={})
    output = tmp_path / "eval.json"
    args = dict(
        candidate=runtime,
        best=runtime,
        candidate_id="a",
        best_id="b",
        seeds=[Seed("s", "source", [], 0, "monitor", "c")],
        config=SelfPlayConfig(),
        max_positions=128,
        output=output,
        pairs=1,
    )
    assert evaluation.evaluate_pair_checkpoints(**args) is None
    interrupted = False
    assert evaluation.evaluate_pair_checkpoints(**args)["score"] == 0.5
    assert calls[:2] == calls[2:]
    assert "protocol_failure" not in json.loads(output.read_text())


def test_zero_value_weight_freezes_head_across_training_and_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    store = SelfPlayStore(tmp_path / 'replay', flush_games=1)
    store.add(mate_game())
    config = LearningConfig(lr=0.01, value_weight=0, weight_decay=0.1)

    def make_trainer():
        return Stage2Trainer(model=tiny_model(), config=config, move_vocab=VOCAB,
                             encoder=ENCODER, device=torch.device('cpu'), max_positions=128)

    a = make_trainer()
    initial = {k: v.clone() for k, v in a.model.state_dict().items()}
    head_ids = {id(p) for p in a.model.value_head.parameters()}
    assert all(not p.requires_grad for p in a.model.value_head.parameters())
    assert not head_ids.intersection(id(p) for g in a.optimizer.param_groups for p in g['params'])
    a.begin_phase(store)
    a.train(store, exposure_budget=16)
    a.checkpoint(tmp_path / 'state.pt', progress={'phase': 'train'}, store=store, config_id='zero')
    a.train(store, exposure_budget=24)
    b = make_trainer()
    b.resume(tmp_path / 'state.pt', store=store, config_id='zero')
    b.train(store, exposure_budget=24)
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k])
        if k.startswith('value_head.'):
            assert torch.equal(v, initial[k])
    assert any(not torch.equal(v, initial[k]) for k, v in a.model.state_dict().items()
               if not k.startswith('value_head.'))
    assert all(p.grad is None for p in b.model.value_head.parameters())


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf')])
def test_invalid_value_weight(weight):
    with pytest.raises(ValueError, match='value_weight'):
        LearningConfig(value_weight=weight)


def test_policy_only_default_configs():
    from imba_chess.self_play.config import load_config
    assert LearningConfig().value_weight == 1
    for path in Path('config').glob('self_play*.toml'):
        assert load_config(path).learning.value_weight == 1


def test_joint_value_gradients_and_independent_clipping(tmp_path):
    from imba_chess.self_play.trainer import _training_loss
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = tiny_model()
    trainer = Stage2Trainer(model=model, config=LearningConfig(value_weight=1),
                            move_vocab=VOCAB, encoder=ENCODER,
                            device=torch.device('cpu'), max_positions=128)
    sample = reconstruct(mate_game(), move_vocab=VOCAB, encoder=ENCODER, max_positions=128)
    batch = collate_self_play([sample])
    model.eval()
    losses = _training_loss(model, batch, 1)
    losses['value_loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.policy_parameters)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.value_parameters)
    model.zero_grad(set_to_none=True)
    _training_loss(model, batch, 1)['weighted_policy_loss'].backward()
    assert all(p.grad is None for p in trainer.value_parameters)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.policy_parameters)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    _training_loss(model, batch, 1)['loss'].backward()
    trainer._clip_gradients()
    trainer.optimizer.step()
    assert any(not torch.equal(v, before[k]) for k,v in model.state_dict().items() if k.startswith('value_head.'))
    assert any(not torch.equal(v, before[k]) for k,v in model.state_dict().items() if not k.startswith('value_head.'))
    clipped = []
    for magnitude in (1., 1000000.):
        for p in trainer.policy_parameters: p.grad = torch.ones_like(p)
        for p in trainer.value_parameters: p.grad = torch.full_like(p, magnitude)
        trainer._clip_gradients()
        clipped.append([p.grad.clone() for p in trainer.policy_parameters])
    assert all(torch.equal(a,b) for a,b in zip(*clipped))
