from dataclasses import asdict, replace
import itertools
import json
import random
import subprocess
import sys
import time

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.config import SelfPlayConfig, StreamingConfig, load_config
from imba_chess.self_play.seeds import Seed, source_split
from imba_chess.self_play.streaming import (
    BUCKETS,
    StreamingStarts,
    dataset_settings,
    make_block,
    open_source,
)


def source_id(i, split="train"):
    value = f"stream-test:{i}"
    while source_split(value) != split:
        value += "x"
    return value


def long_game():
    rng = random.Random(17)
    while True:
        board = chess.Board()
        moves = []
        for _ in range(140):
            if board.is_game_over(claim_draw=True):
                break
            move = rng.choice(list(board.legal_moves))
            moves.append(move.uci())
            board.push(move)
        if len(moves) == 140:
            return moves


@pytest.fixture
def rows():
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    for uci in long_game():
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    text = game.accept(
        chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    )
    return [
        dict(
            Site=source_id(i),
            WhiteElo=2200,
            BlackElo=2200,
            WhiteTitle="",
            BlackTitle="",
            TimeControl="600+0",
            Result="1-0",
            Termination="Normal",
            movetext=text,
        )
        for i in range(36)
    ]


def test_shared_filter_cursor_resume_and_buckets(tmp_path, rows):
    rows += [
        dict(rows[0], Site="bot", WhiteTitle="BOT"),
        dict(rows[0], Site="low", WhiteElo=1000, BlackElo=1000),
        dict(rows[0], Site="fast", TimeControl="30+0"),
        dict(rows[0], Site=source_id(999, "monitor")),
    ]
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=7)
    settings = dataset_settings("config/imba_chess_v4.toml")
    settings["local_corpus_path"] = str(path)
    dataset, stream = open_source(settings)
    iterator = iter(stream)
    first = list(itertools.islice(iterator, 11))
    cursor = stream.state_dict()
    rest = list(iterator)
    _, restored = open_source(settings)
    restored.load_state_dict(cursor)
    assert list(restored) == rest
    assert len(first + rest) == 37
    groups = make_block(dataset, first + rest, identity="id", number=0, run_seed=42)
    assert list(map(len, groups)) == [12, 12, 12]
    seeds = [Seed(**s) for group in groups for s in group]
    assert len({s.source_id for s in seeds}) == 36
    for group, (lower, upper) in zip(groups, BUCKETS[1:]):
        for raw in group:
            seed = Seed(**raw)
            assert lower <= seed.takeover_ply <= upper
            assert seed.split == source_split(seed.source_id) == "train"
            seed.board()
    assert groups == make_block(
        dataset, first + rest, identity="id", number=0, run_seed=42
    )


def sampler_with_blocks(tmp_path, counts=(3, 2, 2)):
    cfg = replace(SelfPlayConfig(), streaming=StreamingConfig())
    sampler = StreamingStarts(tmp_path, cfg, start_worker=False)
    moves = long_game()
    for number in range(3):
        groups = []
        for bucket, count in enumerate(counts, 1):
            ply = BUCKETS[bucket][0]
            groups.append(
                [
                    asdict(
                        Seed(
                            f"seed-{number}-{bucket}-{i}",
                            source_id(number * 100 + bucket * 10 + i),
                            moves[:ply],
                            ply,
                            "train",
                            sampler.identity,
                        )
                    )
                    for i in range(count)
                ]
            )
        atomic_json(
            tmp_path / f"block-{number:08d}.json",
            dict(
                identity=sampler.identity, number=number, exhausted=False, groups=groups
            ),
        )
    return sampler, cfg


def test_exact_mixture_and_crash_reissue_without_losing_prefetch(tmp_path):
    sampler, cfg = sampler_with_blocks(tmp_path)
    sampler.begin_phase(0, "actor", set())
    issued = [sampler.next_launch(0, "actor") for _ in range(8)]
    assert sampler.state["launched"] == [2, 2, 2, 2]
    # Pretend only three launches reached durable replay before a crash.
    completed = {gid for _, gid in issued[:3]}
    restored = StreamingStarts(tmp_path, cfg, start_worker=False)
    restored.begin_phase(0, "actor", completed)
    assert [restored.next_launch(0, "actor") for _ in range(5)] == issued[3:]
    assert restored.state["sequence"] == 8
    restored.reconcile({gid for _, gid in issued})
    further = [restored.next_launch(0, "actor") for _ in range(12)]
    assert restored.state["launched"] == [5, 5, 5, 5]
    # Uneven blocks retain the third early seed even while other buckets advance.
    assert any(s.seed_id == "seed-0-1-2" for s, _ in further)
    assert len({gid for _, gid in issued + further}) == 20
    human = [s.seed_id for s, _ in issued + further if s.takeover_ply]
    assert len(human) == len(set(human))
    for seed, _ in issued + further:
        seed.board()


def test_stop_during_queue_wait_and_bounded_shutdown(tmp_path):
    cfg = replace(SelfPlayConfig(), streaming=StreamingConfig(shutdown_timeout=0.1))
    sampler = StreamingStarts(
        tmp_path, cfg, start_worker=False, should_stop=lambda: True
    )
    with pytest.raises(InterruptedError):
        sampler.warm()
    sampler.worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ]
    )
    child = sampler.worker
    time.sleep(0.1)
    start = time.monotonic()
    sampler.close()
    assert child.poll() is not None
    assert time.monotonic() - start < 2


def test_stream_settings_and_pending_actor_mismatch_rejected(tmp_path):
    sampler, cfg = sampler_with_blocks(tmp_path)
    sampler.next_launch(0, "actor")
    with pytest.raises(ValueError, match="actor/phase"):
        sampler.begin_phase(1, "new", set())
    changed = replace(cfg, streaming=replace(cfg.streaming, block_rows=20000))
    with pytest.raises(ValueError, match="settings changed"):
        StreamingStarts(tmp_path, changed, start_worker=False)


def test_real_producer_restart_preserves_next_block(tmp_path, rows):
    from pathlib import Path

    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), corpus, row_group_size=7)
    base = tmp_path / "base.toml"
    base.write_text(
        Path("config/imba_chess_v4.toml")
        .read_text()
        .replace("[dataset]", f'[dataset]\nlocal_corpus_path = "{corpus}"')
    )
    cfg = replace(
        SelfPlayConfig(),
        base_config=str(base),
        streaming=StreamingConfig(block_rows=12, startup_timeout=20),
    )
    directory = tmp_path / "stream"
    with StreamingStarts(directory, cfg) as sampler:
        sampler._block(1)
        sampler.state.update(block=1, bucket_blocks=[1, 1, 1])
        sampler.save()
    with StreamingStarts(directory, cfg) as restored:
        actual = restored._block(2)
        dataset, source = open_source(dataset_settings(base))
        expected = make_block(
            dataset,
            list(source)[24:36],
            identity=restored.identity,
            number=2,
            run_seed=cfg.run.seed,
        )
        assert actual["groups"] == expected
        assert actual["source_rows"] == 12


def test_legacy_configuration_identity_is_preserved():
    import hashlib
    from imba_chess.self_play.config import _base_config_identity

    cfg = load_config("config/self_play_laptop_pilot.toml")
    old_settings = asdict(cfg)
    del old_settings["streaming"]
    previous = hashlib.sha256(
        json.dumps(
            dict(
                settings=old_settings,
                base_sha256=_base_config_identity(cfg.base_config),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert cfg.identifier == previous


def test_nightly_deadline_and_initialize_resume(tmp_path):
    from argparse import Namespace
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from scripts.run_streaming_self_play_nightly import command

    args = Namespace(
        config="config/self_play_streaming.toml",
        run=tmp_path,
        seeds="monitor.json",
        initialize="ckpt34.pt",
        until=None,
    )
    now = datetime(2026, 9, 17, 21, tzinfo=ZoneInfo("America/Toronto"))
    argv = command(args, now)
    assert argv[argv.index("--until") + 1] == "2026-09-18T08:00:00-04:00"
    assert argv[argv.index("--initialize") + 1] == "ckpt34.pt"
    (tmp_path / "state.json").write_text("{}")
    assert "--resume" in command(args, now)
    args.until = datetime(2026, 9, 18, 8)
    with pytest.raises(ValueError, match="timezone"):
        command(args, now)
