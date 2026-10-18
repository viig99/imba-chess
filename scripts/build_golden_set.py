"""Build or resume a frozen all-actions Stockfish reference set.

Sampling and labelling are separate commands; a label manifest binds every
append to the exact positions, engine binary, and analysis budget.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.golden import (close_engines, digest, engine_identity, file_sha256, human_games, label_position,
                                   pgn_games, read_labels, replay_games, sample_games,
                                   start_engine, validate_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--human", type=Path, required=True)
    sample.add_argument("--replay", type=Path, required=True)
    sample.add_argument("--pgn-root", type=Path, default=Path("artifacts/eval"))
    sample.add_argument("--human-games", type=int, default=750)
    sample.add_argument("--self-play-games", type=int, default=200)
    sample.add_argument("--stockfish-games", type=int, default=50)
    sample.add_argument("--per-game", type=int, default=5)
    sample.add_argument("--seed", type=int, default=1042)
    sample.add_argument("--out", type=Path, required=True)
    label = sub.add_parser("label")
    label.add_argument("--positions", type=Path, required=True)
    label.add_argument("--out", type=Path, required=True)
    label.add_argument("--engine", type=Path, default=Path("/usr/bin/stockfish"))
    label.add_argument("--nodes", type=int, default=200000, help="nodes per legal action")
    label.add_argument("--root-nodes", type=int, default=1000000)
    label.add_argument("--workers", type=int, default=6)
    label.add_argument("--hash-mb", type=int, default=128)
    label.add_argument("--limit", type=int, help="pilot prefix; stored in the identity")
    args = parser.parse_args()
    if args.command == "sample":
        if args.out.exists():
            raise FileExistsError("positions already frozen; use a new version")
        sources = [
            (human_games(args.human), args.human_games),
            (replay_games(args.replay, args.seed), args.self_play_games),
            (pgn_games(p for p in args.pgn_root.glob("sf*/**/*.pgn")), args.stockfish_games),
        ]
        manifest = sample_games([(it, count) for it, count in sources if count],
                                per_game=args.per_game, seed=args.seed)
        manifest["source_files"] = dict(human=str(args.human), replay=str(args.replay),
                                        pgn_root=str(args.pgn_root))
        atomic_json(args.out, manifest)
        print(json.dumps(dict(positions=len(manifest["positions"]), games=manifest["game_counts"],
                              splits=dict(Counter(p["split"] for p in manifest["positions"])),
                              phases=dict(Counter(p["phase"] for p in manifest["positions"])))), flush=True)
        return
    if min(args.nodes, args.root_nodes, args.workers, args.hash_mb) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("analysis sizes must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(args.positions.read_text())
        if digest(manifest["positions"]) != manifest["positions_sha256"]:
            raise ValueError("frozen position digest mismatch")
        positions = manifest["positions"][:args.limit]
        identity = dict(schema_version=1, positions_sha256=digest(positions),
                        source_positions=str(args.positions.resolve()), positions=len(positions),
                        engine=engine_identity(args.engine), nodes_per_action=args.nodes,
                        root_nodes=args.root_nodes, threads=1, hash_mb=args.hash_mb,
                        bound_handling="last completed exact score report; retry at 2x, 4x, 8x only if none",
                        clear_hash="before every root and forced-action analysis", wdl="native engine UCI_ShowWDL; loss/draw/win; root side to move")
        identity_path = args.out / "manifest.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("analysis identity changed; use a new output directory")
        atomic_json(identity_path, identity)
        labels_path = args.out / "labels.jsonl"
        rows = read_labels(labels_path, recover_partial=True)
        if set(rows) - {p["position_id"] for p in positions}:
            raise ValueError("foreign position in reference labels")
        todo = [(p, args.nodes, args.root_nodes) for p in positions if p["position_id"] not in rows]
        started, initial = time.monotonic(), len(rows)
        try:
            # Each thread owns an independent external Stockfish process. The
            # CPU search occurs in those processes; Python only dispatches I/O.
            with ThreadPoolExecutor(max_workers=args.workers, initializer=start_engine,
                                    initargs=(args.engine, args.hash_mb)) as pool, labels_path.open("a") as stream:
                futures = [pool.submit(label_position, job) for job in todo]
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except BaseException as error:
                        for pending in futures:
                            pending.cancel()
                        print(f"Reference worker failed: {error!r}", flush=True)
                        raise
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows[row["position_id"]] = row
                    if (len(rows) - initial) % 10 == 0 or len(rows) == len(positions):
                        elapsed = time.monotonic() - started
                        atomic_json(args.out / "progress.json", dict(completed=len(rows), total=len(positions),
                                    elapsed_seconds=elapsed, positions_per_hour=3600 * (len(rows) - initial) / max(elapsed, 1e-9)))
                        print(json.dumps(dict(completed=len(rows), total=len(positions), seconds=round(elapsed))), flush=True)
        finally:
            close_engines()
        validate_labels(dict(positions=positions), rows)
        spreads = [max(s["expectation"] for s in r["moves"].values()) - min(s["expectation"] for s in r["moves"].values()) for r in rows.values()]
        atomic_json(args.out / "summary.json", dict(complete=True, positions=len(rows),
                    moves=sum(len(r["moves"]) for r in rows.values()), flat_positions=sum(x == 0 for x in spreads),
                    informative_positions=sum(x >= 0.02 for x in spreads), labels_sha256=file_sha256(labels_path)))


if __name__ == "__main__":
    main()
