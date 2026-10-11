"""How much of the Gumbel policy target is seed noise, and what zero noise would lose.

Read-only study on positions from a self-play run's own recent games. Every
position is searched with K training-noise seeds and once with zero noise, per
checkpoint, at the run's search config. Stockfish (multipv over every legal
move) scores each move as an expected score for the side to move.

Decomposition, exact for any K: mean_k KL(t_k || p) = mean_k KL(t_k || T) + KL(T || p),
where T is the mean target over seeds and p the network prior. The first term
is seed noise around the mean target, the second what search agrees on across
seeds. With finite K the noise term is biased low, so pairwise KL is reported too.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json

EPS = 1e-12


def kl(p, q):
    return math.fsum(a * (math.log(max(a, EPS)) - math.log(max(b, EPS))) for a, b in zip(p, q) if a > 0)


def tv(p, q):
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(p, q))


def sample_positions(run, *, count, seed, max_positions, max_depth):
    """One random searched ply from each of the newest completed training games."""
    store = SelfPlayStore(run / "replay", read_only=True)
    ids = [g for g in store.manifest["active"] if store.index[g][1]["split"] == "train"]
    newest = max(store.index[g][1]["iteration"] for g in ids)
    ids = sorted(ids, key=lambda g: (-store.index[g][1]["iteration"], g))
    rng = random.Random(seed)
    rows = []
    for gid in ids:
        if len(rows) == count:
            break
        game = store.read_game(gid)
        if game["status"] != "completed" or not game["moves"]:
            continue
        ply = rng.randrange(len(game["moves"]))
        prefix = list(game["prefix_moves"]) + list(game["moves"][:ply])
        if len(prefix) + 2 + max_depth > max_positions:
            continue
        rows.append(dict(
            position_id=hashlib.sha256(f"{gid}:{ply}".encode()).hexdigest()[:16],
            game_id=gid, iteration=store.index[gid][1]["iteration"],
            ply_after_takeover=ply, prefix_moves=prefix))
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} usable positions in the replay window")
    return rows, newest


def stockfish_scores(position, *, depth, threads, hash_mb):
    import chess
    import chess.engine
    board = chess.Board()
    for uci in position["prefix_moves"]:
        board.push_uci(uci)
    legal = list(board.legal_moves)
    with chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") as engine:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=len(legal))
    scores = {}
    for info in infos:
        move = info["pv"][0]
        scores[move.uci()] = info["score"].pov(board.turn).wdl(model="sf", ply=board.ply()).expectation()
    if set(scores) != {m.uci() for m in legal}:
        raise RuntimeError(f"stockfish did not score every legal move at {position['position_id']}")
    return dict(position_id=position["position_id"], scores=scores, depth=depth)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(start)
                break
            rows.append(json.loads(line))
    return rows


def append(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_searches(*, checkpoint, label, positions, search_config, cfg, seeds, out, concurrency, dtype):
    import chess
    import torch
    from imba_chess.config import load_repo_config
    from imba_chess.eval import inference_runtime
    from imba_chess.eval.batch_scheduler import BatchScheduler
    from imba_chess.eval.position_evaluator import _SequenceHistory

    path = out / f"search-{label}.jsonl"
    rows = read_jsonl(path)
    done = {(r["position_id"], r["variant"]) for r in rows}
    planned = [(p, v) for p in positions for v in [f"seed{k}" for k in range(seeds)] + ["zero"]]
    todo = [(p, v) for p, v in planned if (p["position_id"], v) not in done]
    if not todo:
        return
    runtime, _ = inference_runtime.load_runtime(
        repo_config=load_repo_config(Path(cfg.base_config)), checkpoint=checkpoint, device="cuda",
        root_batch_tokens=cfg.collection.root_batch_tokens,
        dtype=inference_runtime.INFERENCE_DTYPES[dtype])
    began = time.perf_counter()
    with runtime, path.open("a") as stream:
        def play(position, variant):
            board = chess.Board()
            history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
            for uci in position["prefix_moves"]:
                history.append_observed_position(board)
                history.record_played_move(uci)
                board.push_uci(uci)
            key = f"{position['position_id']}:{variant}"
            result = yield from runtime.search(
                board=board, history=history, actor_id=label, game_id=key, config=search_config,
                rng=random.Random(f"target-noise:{key}"),
                **({"noise": 0.0} if variant == "zero" else {}))
            return dict(position_id=position["position_id"], variant=variant,
                        legal_ucis=[runtime.move_vocab.decode(i) for i in result.legal_ids],
                        policy=list(result.policy), visits=list(result.visits),
                        root_log_priors=list(result.root_log_priors), move_uci=result.move_uci,
                        neural_evaluations=result.neural_evaluations)

        def factory():
            for position, variant in todo:
                yield f"{position['position_id']}:{variant}", play(position, variant)

        def finished(key, row):
            append(stream, row)
            rows.append(row)
            if len(rows) % 200 == 0:
                print(json.dumps(dict(event="progress", checkpoint=label, searches=len(rows),
                                      planned=len(planned), seconds=round(time.perf_counter() - began))), flush=True)

        def fail(key, exc):
            raise exc

        BatchScheduler(game_factory=iter(factory()), executors=runtime.executors,
                       concurrent_games=concurrency, on_game_done=finished, on_game_error=fail,
                       completion_order=True).run()
    del runtime
    torch.cuda.empty_cache()


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def summarize_checkpoint(rows, sf, *, seeds, top_m, blind_margins=(0.02, 0.05, 0.10)):
    by_position = {}
    for row in rows:
        by_position.setdefault(row["position_id"], {})[row["variant"]] = row
    per = []
    for pid, variants in sorted(by_position.items()):
        noisy = [variants[f"seed{k}"] for k in range(seeds)]
        zero = variants["zero"]
        moves = zero["legal_ucis"]
        if any(r["legal_ucis"] != moves for r in noisy):
            raise AssertionError("legal move order differs between searches of one position")
        logp = zero["root_log_priors"]
        prior = [math.exp(x) for x in logp]
        total = math.fsum(prior)
        prior = [x / total for x in prior]
        targets = [r["policy"] for r in noisy]
        avg = [math.fsum(t[i] for t in targets) / seeds for i in range(len(moves))]
        kl_actor = [kl(t, prior) for t in targets]
        noise = [kl(t, avg) for t in targets]
        signal = kl(avg, prior)
        if abs(mean(kl_actor) - mean(noise) - signal) > 1e-6:
            raise AssertionError("KL decomposition identity failed")
        pairs = [(i, j) for i in range(seeds) for j in range(seeds) if i != j]
        row = dict(
            position_id=pid, legal=len(moves),
            kl_actor=mean(kl_actor), noise=mean(noise), signal=signal,
            pairwise_kl=mean(kl(targets[i], targets[j]) for i, j in pairs),
            pairwise_tv=mean(tv(targets[i], targets[j]) for i, j in pairs if i < j),
            move_flip=mean(noisy[i]["move_uci"] != noisy[j]["move_uci"] for i, j in pairs if i < j),
            zero_kl_actor=kl(zero["policy"], prior),
            zero_vs_mean_kl=kl(avg, zero["policy"]), zero_vs_mean_tv=tv(avg, zero["policy"]),
            zero_move_is_mean_argmax=zero["move_uci"] == moves[max(range(len(moves)), key=avg.__getitem__)],
            visited_noisy=mean(sum(v > 0 for v in r["visits"]) for r in noisy),
            visited_union=sum(any(r["visits"][i] > 0 for r in noisy) for i in range(len(moves))),
            visited_zero=sum(v > 0 for v in zero["visits"]),
            evals_noisy=mean(r["neural_evaluations"] for r in noisy),
            evals_zero=zero["neural_evaluations"])
        if pid in sf:
            score = [sf[pid][m] for m in moves]
            best = max(score)
            loss = [best - s for s in score]
            order = sorted(range(len(moves)), key=lambda i: -prior[i])
            rank = {i: r for r, i in enumerate(order)}
            top = set(order[:top_m])
            best_move = max(range(len(moves)), key=score.__getitem__)
            best_inside = max(score[i] for i in top)
            argmax = lambda dist: max(range(len(moves)), key=dist.__getitem__)
            row.update(
                sf_best_prior_rank=rank[best_move] + 1,
                sf_best_outside_top_m=best_move not in top,
                **{f"blind_spot_{m:.2f}": best - best_inside >= m for m in blind_margins},
                sf_best_visited_noisy=mean(r["visits"][best_move] > 0 for r in noisy),
                sf_best_visited_zero=zero["visits"][best_move] > 0,
                loss_prior_argmax=loss[argmax(prior)],
                loss_noisy_move=mean(loss[moves.index(r["move_uci"])] for r in noisy),
                loss_zero_move=loss[moves.index(zero["move_uci"])],
                loss_mean_target_argmax=loss[argmax(avg)],
                expected_loss_prior=math.fsum(p * l for p, l in zip(prior, loss)),
                expected_loss_noisy_target=mean(math.fsum(p * l for p, l in zip(t, loss)) for t in targets),
                expected_loss_mean_target=math.fsum(p * l for p, l in zip(avg, loss)),
                expected_loss_zero_target=math.fsum(p * l for p, l in zip(zero["policy"], loss)))
        per.append(row)
    keys = [k for k in per[0] if k != "position_id"]
    summary = {k: mean(float(r[k]) for r in per if k in r) for k in keys}
    summary["positions"] = len(per)
    summary["positions_with_stockfish"] = sum("loss_zero_move" in r for r in per)
    summary["noise_share_of_kl_actor"] = summary["noise"] / summary["kl_actor"]
    return summary, per


def main():
    import torch
    from imba_chess.self_play.config import load_config
    from imba_chess.eval import inference_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path, repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--stockfish-depth", type=int, default=14)
    parser.add_argument("--stockfish-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seeds < 2 or args.positions < 1:
        parser.error("need at least two seeds and one position")
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    cfg = load_config(args.config)
    if inference_runtime.RUNTIME_REVISION != "shared-search-v2":
        raise RuntimeError("study requires shared-search-v2")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "positions.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        positions = saved["positions"]
    else:
        positions, newest = sample_positions(args.run, count=args.positions, seed=args.seed,
                                             max_positions=512, max_depth=cfg.search.max_depth)
        saved = dict(run=str(args.run), newest_iteration=newest, search=asdict(cfg.search),
                     seeds=args.seeds, dtype=args.dtype, positions=positions)
        atomic_json(manifest_path, saved)
    if saved["seeds"] != args.seeds or saved["search"] != asdict(cfg.search) or saved["dtype"] != args.dtype:
        raise RuntimeError("study inputs changed; use a fresh output directory")
    print(json.dumps(dict(event="positions", count=len(positions), newest_iteration=saved["newest_iteration"])), flush=True)

    sf_path = args.out / "stockfish.jsonl"
    sf_rows = read_jsonl(sf_path)
    sf_done = {r["position_id"] for r in sf_rows}
    pending = [p for p in positions if p["position_id"] not in sf_done]
    pool = ThreadPoolExecutor(max_workers=args.stockfish_workers)
    sf_futures = [pool.submit(stockfish_scores, p, depth=args.stockfish_depth, threads=1, hash_mb=128) for p in pending]

    for label, path in checkpoints.items():
        run_searches(checkpoint=path, label=label, positions=positions, search_config=cfg.search, cfg=cfg,
                     seeds=args.seeds, out=args.out, concurrency=args.concurrency, dtype=args.dtype)
        print(json.dumps(dict(event="searched", checkpoint=label)), flush=True)

    with sf_path.open("a") as stream:
        for future in sf_futures:
            row = future.result()
            append(stream, row)
            sf_rows.append(row)
    pool.shutdown()
    sf = {r["position_id"]: r["scores"] for r in sf_rows}

    report = dict(search=asdict(cfg.search), seeds=args.seeds, dtype=args.dtype,
                  stockfish_depth=args.stockfish_depth, newest_iteration=saved["newest_iteration"],
                  expected_score="Stockfish WDL expectation for the side to move; loss = best move minus this move",
                  checkpoints={})
    for label in checkpoints:
        rows = read_jsonl(args.out / f"search-{label}.jsonl")
        summary, per = summarize_checkpoint(rows, sf, seeds=args.seeds, top_m=cfg.search.top_m)
        report["checkpoints"][label] = summary
        with (args.out / f"per-position-{label}.jsonl").open("w") as stream:
            for row in per:
                stream.write(json.dumps(row) + "\n")
    atomic_json(args.out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
