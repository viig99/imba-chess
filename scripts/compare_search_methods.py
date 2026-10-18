"""Paired, resumable frozen-checkpoint searches on a Stockfish reference set.

This measures search and target quality. It does not claim neural learning.
Use the production collection benchmark separately for multi-worker throughput.
"""

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.golden import digest, file_sha256, read_labels
from imba_chess.eval.search_study import summarize, paired_regret_interval
from imba_chess.self_play.config import search_config


def source_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("src/**/*.py"), *root.glob("native/imba_chess_native/src/*.rs"),
                    root / "scripts/compare_search_methods.py"])
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run_arm(args, arm, positions, checkpoint_hash, reference_hash):
    import chess
    import torch
    from imba_chess.config import load_repo_config
    from imba_chess.eval.inference_runtime import load_runtime, INFERENCE_DTYPES
    from imba_chess.eval.batch_scheduler import BatchScheduler
    from imba_chess.eval.position_evaluator import _SequenceHistory

    arm = dict(arm)
    name = arm.pop("name")
    algorithm = arm.get("algorithm", "gumbel")
    config = search_config(simulations=args.simulations, **arm)
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(schema_version=1, checkpoint_sha256=checkpoint_hash,
                    reference_sha256=reference_hash, positions_sha256=digest(positions),
                    search=asdict(config), dtype=args.dtype, concurrent_games=args.concurrent_games,
                    root_batching="fixed planned chunks; same root grouping for every method",
                    seeds=args.seeds, code_sha256=source_digest(), base_config_sha256=file_sha256(args.base_config))
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != identity:
        raise ValueError(f"{name}: study inputs changed; use a fresh output directory")
    atomic_json(out / "manifest.json", identity)
    path = out / "searches.jsonl"
    # Searches have two-key identities, unlike one-label-per-position references.
    rows = []
    if path.exists():
        with path.open("rb+") as stream:
            while True:
                offset, line = stream.tell(), stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                rows.append(json.loads(line))
    done = {(r["position_id"], r["variant"]) for r in rows}
    if len(done) != len(rows):
        raise ValueError("duplicate search observation")
    variants = ["eval", *[f"train{k}" for k in range(args.seeds)]]
    planned = [(p, v) for p in positions for v in variants]
    chunks = [planned[i:i + args.concurrent_games] for i in range(0, len(planned), args.concurrent_games)]
    chunks = [chunk for chunk in chunks if any((p["position_id"], v) not in done for p, v in chunk)]
    if not chunks:
        return rows
    runtime, _ = load_runtime(repo_config=load_repo_config(args.base_config), checkpoint=args.checkpoint,
                              device="cuda", algorithm=algorithm, dtype=INFERENCE_DTYPES[args.dtype])

    def play(position, variant, cfg=config):
        board = chess.Board()
        history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
        for uci in position["prefix_moves"]:
            history.append_observed_position(board)
            history.record_played_move(uci)
            board.push_uci(uci)
        key = f"{position['position_id']}:{variant}"
        batch = history.build_batch_for_current_position(board)
        batch["game_id"] = [key]
        # A fixed root has no next turn: avoid allocating a played-root carry.
        result = yield from runtime.search_batch(board=board,
            batch=batch, owner=(name, key), config=cfg,
            rng=random.Random(f"search-study:{key}"), **({"noise": 0.0} if variant == "eval" else {}))
        return dict(position_id=position["position_id"], variant=variant,
                    legal_ucis=[runtime.move_vocab.decode(i) for i in result.legal_ids],
                    **{k: getattr(result, k) for k in ("move_uci", "policy", "visits", "qvalues",
                       "root_log_priors", "root_value", "root_wdl", "search_wdl", "improved_wdl", "neural_evaluations",
                       "terminal_hits", "depth_cutoffs", "maximum_depth")})

    def fail(key, exc):
        raise exc

    with runtime:
        warm_config = replace(config, simulations=min(32, config.simulations), max_depth=min(8, config.max_depth))
        warm = [(p, "eval") for p in positions[:args.concurrent_games]]
        BatchScheduler(game_factory=iter((p["position_id"], play(p, v, warm_config)) for p, v in warm),
                       executors=runtime.executors, concurrent_games=args.concurrent_games,
                       on_game_done=lambda *unused: None, on_game_error=fail, completion_order=True).run()
        runtime.clear_caches()
        before = dict(runtime.inference_rows)
        calls_before = {k: sum(v.values()) for k, v in runtime.waves.items()}
        started = time.perf_counter()
        with path.open("a") as stream:
            def finished(key, row):
                identity = row["position_id"], row["variant"]
                if identity in done:
                    return
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                done.add(identity)
                if len(rows) % 100 == 0:
                    print(json.dumps(dict(arm=name, completed=len(rows), total=len(positions) * len(variants))), flush=True)
            for chunk in chunks:
                BatchScheduler(game_factory=iter((f"{p['position_id']}:{v}", play(p, v)) for p, v in chunk),
                               executors=runtime.executors, concurrent_games=args.concurrent_games,
                               on_game_done=finished, on_game_error=fail, completion_order=True).run()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        counts = {k: n - before.get(k, 0) for k, n in runtime.inference_rows.items()}
        calls = {k: sum(v.values()) - calls_before.get(k, 0) for k, v in runtime.waves.items()}
        searched = sum(map(len, chunks))
        atomic_json(out / "timing-last-segment.json", dict(seconds=elapsed, searches=searched,
                    searches_per_second=searched / elapsed, actual_neural_rows=counts,
                    actual_neural_rows_per_second=sum(counts.values()) / elapsed,
                    decode_batch_mean=counts.get("decode_wave", 0) / max(1, calls.get("decode_wave", 0)),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    warmup_excluded=True, model_load_excluded=True,
                    note="Fixed-position single-process throughput; use bench_collection for completed self-play throughput."))
    del runtime
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("config/imba_chess_v4_aux3.toml"))
    parser.add_argument("--arms", type=Path, default=Path("config/search_study.toml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("tune", "test", "all"), default="tune")
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", type=int, default=0, help="additional training-exploration seeds")
    parser.add_argument("--concurrent-games", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--allow-partial-reference", action="store_true", help="pilot only; report actual coverage")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    import torch
    import tomllib

    torch.set_num_threads(args.cpu_threads)
    if min(args.simulations, args.concurrent_games) < 1 or args.seeds < 0 or (args.limit is not None and args.limit < 1):
        parser.error("invalid study sizes")
    if args.split == "all" and args.allow_partial_reference:
        parser.error("full-set comparisons require a complete reference")
    manifest = json.loads(args.positions.read_text())
    if digest(manifest["positions"]) != manifest["positions_sha256"]:
        raise ValueError("position digest mismatch")
    labels = read_labels(args.reference / "labels.jsonl")
    identity = json.loads((args.reference / "manifest.json").read_text())
    if identity["positions_sha256"] != manifest["positions_sha256"] and not args.allow_partial_reference:
        raise ValueError("reference belongs to different positions or is a pilot")
    expected = [p for p in manifest["positions"] if args.split == "all" or p["split"] == args.split]
    if not args.allow_partial_reference and any(p["position_id"] not in labels for p in expected):
        raise ValueError("reference labelling is incomplete")
    selected = expected[:args.limit]
    positions = [p for p in selected if p["position_id"] in labels]
    # A pilot can report partial coverage, but it must not substitute later
    # positions as reference labels arrive in a different completion order.
    if not positions:
        raise ValueError("no labelled positions in selected split")
    arms = tomllib.loads(args.arms.read_text())["arms"]
    if len({a["name"] for a in arms}) != len(arms) or any(not a["name"].replace("-", "").replace("_", "").isalnum() for a in arms):
        raise ValueError("arm names must be unique safe identifiers")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint_hash, reference_hash = file_sha256(args.checkpoint), file_sha256(args.reference / "manifest.json")
        results = {}
        report = dict(split=args.split, positions=len(positions), planned_split_positions=len(expected),
                      requested_positions=len(selected),
                      partial_reference=args.allow_partial_reference,
                      checkpoint_sha256=checkpoint_hash, comparisons={}, by_split={})
        partitions = {s: [p for p in positions if p["split"] == s]
                      for s in sorted({p["split"] for p in positions})}
        for split, subset in partitions.items():
            report["by_split"][split] = dict(positions=len(subset), comparisons={})
        for arm in arms:
            name = arm["name"]
            rows = run_arm(args, arm, positions, checkpoint_hash, reference_hash)
            results[name] = rows
            report[name] = summarize(rows, positions, labels)
            if "gumbel" in results and name != "gumbel":
                report["comparisons"][name] = {metric: paired_regret_interval(results["gumbel"], rows, labels, positions, metric=metric)
                                                 for metric in ("target_regret", "move_regret")}
            # The combined 5,000-root report is descriptive. Keep the held-out
            # test comparison explicit even when all roots share one GPU run.
            for split, subset in partitions.items():
                ids = {p["position_id"] for p in subset}
                selected_rows = [r for r in rows if r["position_id"] in ids]
                split_report = report["by_split"][split]
                split_report[name] = summarize(selected_rows, subset, labels)
                if "gumbel" in results and name != "gumbel":
                    baseline = [r for r in results["gumbel"] if r["position_id"] in ids]
                    split_report["comparisons"][name] = {
                        metric: paired_regret_interval(baseline, selected_rows, labels, subset, metric=metric)
                        for metric in ("target_regret", "move_regret")}
            atomic_json(args.out / "summary.json", report)
            print(json.dumps(dict(arm=name, metrics={v: summary["all"] for v, summary in report[name].items()})), flush=True)


if __name__ == "__main__":
    main()
