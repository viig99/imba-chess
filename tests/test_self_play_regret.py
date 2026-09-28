from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import random

import pytest
import torch

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.self_play.collector import collect
from imba_chess.self_play.config import (
    LearningConfig,
    RegretConfig,
    SelfPlayConfig,
    StreamingConfig,
    load_config,
)
from imba_chess.self_play.dataset import collate_self_play, reconstruct
from imba_chess.self_play.losses import self_play_loss
from imba_chess.self_play.regret import (
    RegretBuffer,
    RegretProtocolError,
    empty_buffer,
    history_id,
    suffix_regrets,
)
from imba_chess.self_play.seeds import Seed
from imba_chess.self_play.streaming import StreamingStarts, dataset_settings, BUCKETS
from imba_chess.self_play.trainer import Stage2Trainer
from tests.test_self_play import ScriptRuntime, mate_game, tiny_model, VOCAB, ENCODER
from tests.test_self_play_streaming import source_id, long_game


MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]


def set_q(game, qs):
    for target, q in zip(game["targets"], qs):
        target["qvalues"] = [0.99] * len(target["legal_ids"])
        target["qvalues"][target["legal_ids"].index(target["move_id"])] = q
    return game


def buffer(capacity=256, max_positions=128, max_game_plies=512):
    return RegretBuffer(
        empty_buffer(),
        RegretConfig(capacity=capacity),
        max_positions=max_positions,
        max_game_plies=max_game_plies,
        max_depth=1,
    )


def seed(prefix, source=None):
    return Seed(
        history_id(prefix),
        source or source_id(0),
        prefix,
        len(prefix),
        "train",
        "corpus",
    )


def admit(buf, prefix, priority):
    return buf.admit(
        seed(prefix), priority, parent_game_id="parent", parent_ply=len(prefix)
    )


def sampler(tmp_path, *, capacity=256, short=False):
    cfg = SelfPlayConfig(
        search=GumbelConfig(simulations=1, max_depth=1),
        learning=LearningConfig(auxiliary_value_weight=0),
        streaming=StreamingConfig(),
        regret=RegretConfig(capacity=capacity),
    )
    starts = StreamingStarts(tmp_path / "stream", cfg, start_worker=False)
    moves = MATE if short else long_game()
    groups = []
    for b in range(1, 4):
        ply = b if short else BUCKETS[b][0]
        groups.append(
            [asdict(seed(moves[:ply], source_id(i + b * 100))) for i in range(100)]
        )
    atomic_json(
        starts.directory / "block-00000000.json",
        dict(identity=starts.identity, number=0, exhausted=True, groups=groups),
    )
    store = SelfPlayStore(tmp_path / "replay", flush_games=1)
    starts.begin_phase(0, "a", store, max_positions=128)
    return starts, store, cfg


def next_bucket(starts, bucket):
    # Keep the genuine deterministic launch cycle, retiring intervening launches.
    for _ in range(10):
        s, gid = starts.next_launch(0, "a")
        if starts.launch_metadata(gid)["starting_bucket"] == bucket:
            return s, gid
        starts.retire(gid, "test_skip")
    raise AssertionError("bucket was not available")


def game_for(starts, s, gid, qs=None):
    game = mate_game(gid, s.prefix_moves)
    for key, value in asdict(s).items():
        game[key] = value
    game.update(iteration=0, **starts.launch_metadata(gid))
    if qs is not None:
        set_q(game, qs)
    return game


def test_suffix_regret_alignment_colors_draw_and_single_position():
    game = set_q(mate_game(), [0.5, 0.25, -0.5, 1])
    # Outcome is -1: errors are 2.25, .5625, .25, 0.
    assert suffix_regrets(game) == pytest.approx([3.0625 / 4, 0.8125 / 3, 0.125, 0])
    game["outcome_white"] = 0
    assert suffix_regrets(game) == pytest.approx([1.5625 / 4, 1.3125 / 3, 0.625, 1])
    odd = set_q(mate_game(prefix=MATE[:3]), [-0.5])
    assert suffix_regrets(odd) == [2.25]
    assert suffix_regrets(dict(status="unfinished")) is None
    for target in game["targets"]:
        target["root_value"] = 999
        target["root_wdl"] = target["search_wdl"] = [1, 0, 0]
    assert suffix_regrets(game)[-1] == 1


@pytest.mark.parametrize(
    "corrupt", ["missing", "length", "move_id", "duplicate", "nan", "range", "move_uci"]
)
def test_invalid_selected_q_is_protocol_error(corrupt):
    game = mate_game()
    t = game["targets"][0]
    if corrupt == "missing":
        del t["qvalues"]
    elif corrupt == "length":
        t["qvalues"].pop()
    elif corrupt == "move_id":
        t["move_id"] = -999
    elif corrupt == "duplicate":
        t["legal_ids"][1] = t["legal_ids"][0]
    elif corrupt == "move_uci":
        t["move_uci"] = "a2a3"
    else:
        t["qvalues"][t["legal_ids"].index(t["move_id"])] = (
            float("nan") if corrupt == "nan" else 2
        )
    with pytest.raises(RegretProtocolError, match="selected raw Q"):
        suffix_regrets(game)


def test_capacity_duplicates_ties_and_history_identity():
    buf = buffer(2)
    assert admit(buf, ["e2e4"], 1)
    assert not admit(buf, ["e2e4"], 4)
    assert admit(buf, ["d2d4"], 1)
    assert not admit(buf, ["c2c4"], 1)
    assert admit(buf, ["c2c4"], 2)
    assert [e["seed"]["prefix_moves"] for e in buf.state["entries"]] == [
        ["d2d4"],
        ["c2c4"],
    ]
    assert buf.state["admissions"] == 3 and buf.state["replacements"] == 1
    # Same position, different complete histories: both must survive.
    a = ["g1f3", "g8f6", "b1c3", "b8c6"]
    b = ["b1c3", "b8c6", "g1f3", "g8f6"]
    assert seed(a).board().fen() == seed(b).board().fen()
    buf = buffer()
    assert admit(buf, a, 1) and admit(buf, b, 1)


def test_sampling_power_ten_zero_mass_and_stability():
    buf = buffer()
    for prefix, p in [(["e2e4"], 1), (["d2d4"], 2), (["c2c4"], 0)]:
        admit(buf, prefix, p)

    class Inspect:
        def choices(self, entries, *, weights, k):
            assert len(entries) == 2 and k == 1
            assert weights == pytest.approx([1 / 1024, 1])
            return [entries[1]]

    assert buf.sample(Inspect())["priority"] == 2
    for e, p in zip(buf.state["entries"], [1e-300, 2e-300, 0]):
        e["priority"] = p
    assert buf.sample(Inspect())["priority"] == 2e-300
    for e in buf.state["entries"]:
        e["priority"] = 0
    assert buf.sample(random.Random(1)) is None
    assert len(buf.state["entries"]) == 3  # No expiry or decay of zero priorities.


def test_observation_admission_eligibility_ema_and_no_descendants():
    buf = buffer()
    game = mate_game()
    buf.observe(game, {})
    entry = buf.state["entries"][0]
    assert (
        entry["seed"]["prefix_moves"] == MATE[:1]
    )  # Earliest eligible tie; no empty prefix.
    assert entry["parent_ply"] == 1 and entry["parent_game_id"] == "g"
    assert entry["seed"]["source_id"] == game["source_id"]
    assert entry["seed"]["corpus_id"] == game["corpus_id"]
    restart = set_q(mate_game(prefix=MATE[:1]), [-0.5] * 3)
    parent = dict(history_id=entry["history_id"], admission_id=entry["admission_id"])
    buf.observe(restart, dict(restart_parent=parent))
    assert len(buf.state["entries"]) == 1
    assert entry["priority"] == pytest.approx(0.5 + 0.5 * (4.75 / 3))
    assert entry["refresh_count"] == 1
    buf.observe(game, {})  # Skip buffered history, admit next-earliest tie.
    assert buf.state["entries"][1]["seed"]["prefix_moves"] == MATE[:2]
    for limit in (buffer(max_positions=3), buffer(max_game_plies=1)):
        limit.observe(game, {})
        assert not limit.state["entries"]
    before = deepcopy(buf.state)
    buf.observe(dict(status="unfinished"), {})
    buf.observe(dict(game, split="monitor"), {})
    assert buf.state == before
    admit(buf, MATE, 2)  # Terminal and over-limit entries can be retired on sampling.
    admit(buf, ["a1a8"], 2)
    buf.sample(random.Random(1))
    assert buf.state["unusable_retirements"] == 2


def test_evicted_inflight_restart_cannot_refresh_readmitted_history():
    buf = buffer(1)
    admit(buf, MATE[:1], 1)
    old = deepcopy(buf.state["entries"][0])
    admit(buf, MATE[:2], 2)
    admit(buf, MATE[:1], 3)
    buf.observe(mate_game(prefix=MATE[:1]), dict(restart_parent=old))
    assert buf.state["entries"][0]["priority"] == 3
    assert buf.state["entries"][0]["refresh_count"] == 0
    assert buf.state["stale_refreshes"] == 1


def test_five_bucket_cycles_and_balanced_persisted_fallback(tmp_path):
    starts, store, cfg = sampler(tmp_path)
    launches = [starts.next_launch(0, "a") for _ in range(20)]
    assert starts.state["requested"] == [4] * 5
    assert starts.state["launched"] == [5, 5, 5, 5, 0]
    assert starts.state["fallbacks"] == [1] * 4
    for group in range(4):
        records = [
            starts.launch_metadata(gid)
            for _, gid in launches[group * 5 : group * 5 + 5]
        ]
        assert sorted(r["requested_bucket"] for r in records) == list(range(5))
    restored = StreamingStarts(starts.directory, cfg, start_worker=False)
    restored.begin_phase(0, "a", store, max_positions=128)
    assert [restored.next_launch(0, "a") for _ in range(20)] == launches
    assert restored.state == starts.state
    assert (
        len({starts.launch_metadata(gid)["exploration_seed"] for _, gid in launches})
        == 20
    )
    admit(restored.regret_buffer, ["e2e4"], 1)
    for _ in range(20):
        s, gid = restored.next_launch(0, "a")
        s.board()
        assert s.split == "train"
    assert restored.state["requested"] == [8] * 5
    assert restored.state["launched"] == [9, 9, 9, 9, 4]
    assert restored.state["fallbacks"] == [1] * 4
    assert restored.report()["regret"]["size"] == 1


def test_resume_mid_fallback_cycle_matches_uninterrupted_launches(tmp_path):
    starts, store, cfg = sampler(tmp_path)
    for _ in range(7):
        _, gid = starts.next_launch(0, "a")
        starts.retire(gid, "test_skip")
    restored = StreamingStarts(starts.directory, cfg, start_worker=False)
    restored.begin_phase(0, "a", store, max_positions=128)
    expected = [starts.next_launch(0, "a") for _ in range(13)]
    actual = [restored.next_launch(0, "a") for _ in range(13)]
    assert expected == actual
    assert restored.state == starts.state
    assert restored.state["fallbacks"] == [1] * 4


@pytest.mark.parametrize("restart", [False, True])
def test_invalid_history_or_outcome_cannot_supply_observation(restart):
    buf = buffer()
    admit(buf, MATE[:1], 1)
    launch = dict(restart_parent=buf.state["entries"][0]) if restart else {}
    game = mate_game(prefix=MATE[:1])
    game["outcome_white"] = 1
    before = deepcopy(buf.state)
    with pytest.raises(RegretProtocolError, match="full regret trajectory"):
        buf.observe(game, launch)
    assert buf.state == before
    game = mate_game(prefix=MATE[:1])
    game["prefix_moves"] = ["0000"]
    with pytest.raises(RegretProtocolError, match="full regret trajectory"):
        buf.observe(game, launch)
    assert buf.state == before


def test_legacy_sampler_identity_format_and_exact_launch_order(tmp_path):
    from tests.test_self_play_streaming import sampler_with_blocks

    starts, cfg = sampler_with_blocks(tmp_path, counts=(10, 10, 10))
    settings = dict(
        dataset=dataset_settings(cfg.base_config),
        **asdict(cfg.streaming),
        run_seed=cfg.run.seed,
        buckets=BUCKETS,
        schema_version=1,
    )
    assert (
        starts.identity
        == hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    )
    assert set(starts.state) == {
        "identity",
        "block",
        "bucket_blocks",
        "offsets",
        "sequence",
        "pending",
        "launched",
        "retired",
    }
    actual = []
    for _ in range(12):
        s, gid = starts.next_launch(0, "a")
        actual.append(
            next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= s.takeover_ply <= hi)
        )
        assert set(starts.state["pending"][gid]) == {
            "seed",
            "iteration",
            "actor_id",
            "sequence",
        }
        assert starts.launch_metadata(gid) == {}
    expected = []
    for cycle in range(3):
        order = list(range(4))
        random.Random(f"{cfg.run.seed}:mixture:{cycle}").shuffle(order)
        expected.extend(order)
    assert actual == expected
    with pytest.raises(ValueError, match="settings changed"):
        StreamingStarts(
            tmp_path, replace(cfg, regret=RegretConfig()), start_worker=False
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacity", 0),
        ("capacity", 1.5),
        ("capacity", True),
        ("temperature", 0),
        ("temperature", math.nan),
        ("ema_alpha", -1),
        ("ema_alpha", 1.1),
    ],
)
def test_config_validation(field, value):
    with pytest.raises(ValueError):
        RegretConfig(**{field: value})


def test_enabled_configuration_and_resume_identity(tmp_path):
    with pytest.raises(ValueError, match="require streaming"):
        SelfPlayConfig(regret=RegretConfig())
    path = tmp_path / "config.toml"
    path.write_text(
        "[streaming]\n[regret]\ncapacity=256\ntemperature=0.1\nema_alpha=0.5\n"
    )
    cfg = load_config(path)
    assert cfg.regret == RegretConfig()
    assert cfg.identifier != replace(cfg, regret=None).identifier
    for field, value in (("capacity", 128), ("temperature", 0.2), ("ema_alpha", 0.2)):
        changed = replace(cfg, regret=replace(cfg.regret, **{field: value}))
        assert cfg.identifier != changed.identifier
    starts, store, cfg = sampler(tmp_path / "run")
    with pytest.raises(ValueError, match="context limit changed"):
        starts.begin_phase(0, "a", store, max_positions=129)
    with pytest.raises(ValueError, match="settings changed"):
        StreamingStarts(
            starts.directory, replace(cfg, regret=RegretConfig(128)), start_worker=False
        )


@pytest.mark.parametrize(
    "stage", ["before_publication", "orphan_shard", "before_ack", "after_ack"]
)
@pytest.mark.parametrize("restart", [False, True])
def test_recovery_applies_observations_once(tmp_path, stage, restart):
    starts, store, cfg = sampler(tmp_path, short=True)
    if restart:
        admit(starts.regret_buffer, MATE[:1], 2)
    s, gid = next_bucket(starts, 4 if restart else 0)
    original_record = deepcopy(starts.state["pending"][gid])
    game = game_for(starts, s, gid)
    store.flush_games = 100
    before = deepcopy(store.manifest)
    store.add(game)
    if stage != "before_publication":
        store.flush()
    if stage == "orphan_shard":
        atomic_json(store.directory / "manifest.json", before)
    if stage == "after_ack":
        starts.reconcile(store)
    restored_store = SelfPlayStore(store.directory, flush_games=1)
    restored = StreamingStarts(starts.directory, cfg, start_worker=False)
    restored.begin_phase(0, "a", restored_store, max_positions=128)
    if stage == "before_publication":
        assert restored.state["pending"][gid] == original_record
        assert restored.next_launch(0, "a") == (s, gid)
        restored_store.add(game_for(restored, s, gid))
        restored.reconcile(restored_store)
    assert not restored.state["pending"]
    entry = restored.state["regret"]["entries"][0]
    assert entry["priority"] == (1.5 if restart else 1)
    assert entry["refresh_count"] == int(restart)
    assert restored.state["regret"]["admissions"] == 1
    expected = deepcopy(restored.state)
    restored.reconcile(restored_store)
    restored.begin_phase(0, "a", restored_store, max_positions=128)
    assert restored.state == expected


def test_reconcile_launch_order_atomic_rollback_and_gc_survival(tmp_path, monkeypatch):
    starts, store, cfg = sampler(tmp_path, capacity=1, short=True)
    launches = [next_bucket(starts, 0) for _ in range(2)]
    games = [game_for(starts, s, gid) for s, gid in launches]
    # Publish opposite completion order. Earliest launch wins equal priorities.
    for game in reversed(games):
        store.add(game)
    old_state = deepcopy(starts.state)
    saved = starts.save

    def fail_save():
        raise OSError("crash before acknowledgement")

    monkeypatch.setattr(starts, "save", fail_save)
    with pytest.raises(OSError):
        starts.reconcile(store)
    assert starts.state == old_state
    monkeypatch.setattr(starts, "save", saved)
    starts.reconcile(store)
    entry = deepcopy(starts.state["regret"]["entries"][0])
    assert entry["parent_game_id"] == launches[0][1]
    assert not starts.state["pending"]
    # Force all parent shards out of the active window and remove their files.
    store.window_positions = 4
    store.add(mate_game("unrelated"))
    store.collect_garbage(pinned_shards=[])
    with pytest.raises(FileNotFoundError):
        store.read_game(entry["parent_game_id"])
    restored = StreamingStarts(starts.directory, cfg, start_worker=False)
    restored.begin_phase(0, "a", store, max_positions=128)
    s, gid = next_bucket(restored, 4)
    assert asdict(s) == entry["seed"]
    s.board()
    store.add(game_for(restored, s, gid))
    restored.reconcile(store)
    assert restored.state["regret"]["entries"][0]["refresh_count"] == 1


def test_protocol_error_leaves_durable_launch_unacknowledged(tmp_path):
    starts, store, cfg = sampler(tmp_path, short=True)
    s, gid = next_bucket(starts, 0)
    game = game_for(starts, s, gid)
    del game["targets"][-1]["qvalues"]
    store.add(game)  # Replay stays backward compatible; regret validates raw Q.
    before = deepcopy(starts.state)
    with pytest.raises(RegretProtocolError):
        starts.reconcile(store)
    assert starts.state == before
    assert json.loads(starts.path.read_text()) == before


@pytest.mark.parametrize(
    "termination", ["context_limit", "game_limit", "interrupted", "error"]
)
def test_incomplete_collection_never_updates_regret(tmp_path, termination):
    starts, store, cfg = sampler(tmp_path, short=True)
    admit(starts.regret_buffer, MATE[:1], 2)
    s, gid = next_bucket(starts, 4)
    starts.save()
    runtime = ScriptRuntime()
    runtime.executors = {"tick": lambda ps: ps}
    positions = 128
    if termination == "context_limit":
        positions = 3
        # Begin a run with this limit to exercise the collector's exact check.
        starts.state["regret_max_positions"] = positions
        starts.save()
    elif termination == "game_limit":
        cfg = replace(cfg, collection=replace(cfg.collection, max_game_plies=1))
    elif termination == "error":

        def fail(**kwargs):
            raise RuntimeError("search failure")

        runtime.search = fail
    before = deepcopy(starts.state["regret"])
    games = []
    collect(
        seeds=[],
        runtime=runtime,
        config=cfg,
        actor_id="a",
        store=store,
        max_positions=positions,
        game_count=1,
        concurrent_games=1,
        start_sampler=starts,
        should_stop=lambda: termination == "interrupted",
        on_game=games.append,
    )
    assert games[0]["termination"] == termination
    assert not store.seen and starts.state["regret"] == before
    assert (gid in starts.state["pending"]) == (termination in ("interrupted", "error"))
    report = starts.report()
    assert report["regret"]["refreshes"] == 0


def test_collector_observes_completion_flush_before_next_launch(tmp_path):
    starts, store, cfg = sampler(tmp_path, short=True)
    # Queue one ordinary launch first, then collect through the next five-bucket
    # cycle. The flush must make its admission available inside this collect().
    next_bucket(starts, 0)
    runtime = ScriptRuntime()
    runtime.executors = {"tick": lambda ps: ps}
    metrics = collect(
        seeds=[],
        runtime=runtime,
        config=cfg,
        actor_id="a",
        store=store,
        max_positions=128,
        game_count=7,
        concurrent_games=1,
        start_sampler=starts,
    )
    games = [store.read_game(gid) for gid in store.game_ids()]
    assert any(g["starting_bucket"] == 4 for g in games)
    assert not starts.state["pending"]
    assert starts.state["regret"]["refreshes"] >= 1
    report = metrics.report()
    assert report["distinct_restarted_positions"] >= 1
    assert (
        sum(b["searched_positions"] for b in report["starting_buckets"].values())
        == report["searched_positions"]
    )
    assert sum(
        b["neural_evaluation_share"] for b in report["starting_buckets"].values()
    ) == pytest.approx(1)
    for b in report["starting_buckets"].values():
        assert b["completion_rate"] == 1
        assert b["limit_rate"] == 0
        assert b["mean_continuation_plies"] > 0


def test_end_to_end_admit_restart_train_refresh_resume_and_unchanged_targets(tmp_path):
    torch.set_num_threads(1)
    starts, store, cfg = sampler(tmp_path, short=True)
    runtime = ScriptRuntime()
    runtime.executors = {"tick": lambda ps: ps}
    next_bucket(starts, 0)
    common = dict(
        seeds=[],
        runtime=runtime,
        config=cfg,
        actor_id="a",
        store=store,
        max_positions=128,
        game_count=1,
        concurrent_games=1,
        start_sampler=starts,
    )
    collect(**common)
    assert starts.state["regret"]["admissions"] == 1
    entry = starts.state["regret"]["entries"][0]
    assert entry["seed"]["prefix_moves"] == MATE[:1]
    entry["priority"] = 2
    s, gid = next_bucket(starts, 4)
    collect(**common)
    assert starts.state["regret"]["entries"][0]["priority"] == 1.5
    game = store.read_game(gid)
    assert game["source_id"] == s.source_id
    assert game["restart_parent"]["admission_id"] == entry["admission_id"]
    # Trajectory metadata cannot change targets, masks or losses.
    plain = json.loads(json.dumps(mate_game(gid, MATE[:1])))
    for key in ("targets", "moves", "outcome_white", "termination"):
        assert plain[key] == game[key]
    kw = dict(
        move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=cfg.learning
    )
    actual, expected = reconstruct(game, **kw), reconstruct(plain, **kw)
    assert actual == expected
    assert actual["supervised_indices"] == [2, 3, 4]
    assert actual["has_value_target"] == [False, False, True, True, True]
    batch_a, batch_b = collate_self_play([actual]), collate_self_play([expected])
    logits = torch.randn(5, len(VOCAB))
    value_logits = torch.randn(5, 3)
    output = dict(logits=logits, value_logits=value_logits)
    loss_a = self_play_loss(output, batch_a, auxiliary_value_weight=0)
    loss_b = self_play_loss(output, batch_b, auxiliary_value_weight=0)
    torch.testing.assert_close(loss_a["loss"], loss_b["loss"], rtol=0, atol=0)

    def trainer():
        return Stage2Trainer(
            model=tiny_model(),
            config=cfg.learning,
            move_vocab=VOCAB,
            encoder=ENCODER,
            device=torch.device("cpu"),
            max_positions=128,
        )

    a = trainer()
    a.begin_phase(store)
    records = []
    a.train(store, exposure_budget=7, on_step=records.append)
    assert records and all(math.isfinite(r["loss"]) for r in records)
    a.checkpoint(
        tmp_path / "trainer.pt",
        progress={"phase": "train"},
        store=store,
        config_id=cfg.identifier,
    )
    a.train(store, exposure_budget=14)
    b = trainer()
    b.resume(tmp_path / "trainer.pt", store=store, config_id=cfg.identifier)
    b.train(store, exposure_budget=14)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match="config"):
        trainer().resume(
            tmp_path / "trainer.pt",
            store=store,
            config_id=replace(cfg, regret=None).identifier,
        )
    restored = StreamingStarts(starts.directory, cfg, start_worker=False)
    restored.begin_phase(0, "a", store, max_positions=128)
    assert restored.state == starts.state
