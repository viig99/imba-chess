"""Can the policy absorb its search targets given far more training?

Trains a copy of a self-play actor (with its optimizer state) on a frozen replay
snapshot, using the run's own trainer, loss and settings, with the held-out
games of a measure_target_noise study removed. At each milestone the raw policy
prior is scored on the held-out positions against Stockfish and against the
study's mean search target. Training data KL comes from the trainer's metrics.

Held-out prior quality improving means the loop under-trains the policy; training
KL falling while held-out quality stays flat means the targets are fit but do
not generalize.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import statistics
import time

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json

EPS = 1e-12


def kl(p, q):
    return math.fsum(a * (math.log(max(a, EPS)) - math.log(max(b, EPS))) for a, b in zip(p, q) if a > 0)


def load_study(directory):
    positions = json.loads((directory / "positions.json").read_text())["positions"]
    sf = {r["position_id"]: r["scores"] for r in map(json.loads, (directory / "stockfish.jsonl").open())}
    searches = {}
    for row in map(json.loads, (directory / "search-now283.jsonl").open()):
        searches.setdefault(row["position_id"], []).append(row)
    targets = {}
    for pid, rows in searches.items():
        noisy = [r for r in rows if r["variant"].startswith("seed")]
        moves = noisy[0]["legal_ucis"]
        if any(r["legal_ucis"] != moves for r in noisy):
            raise AssertionError("legal move order differs within a position")
        targets[pid] = (moves, [statistics.fmean(r["policy"][i] for r in noisy) for i in range(len(moves))])
    if set(targets) != {p["position_id"] for p in positions} or set(sf) != set(targets):
        raise AssertionError("study files cover different positions")
    return positions, sf, targets


def prior_logs(runtime, positions, *, concurrency):
    import chess
    from imba_chess.eval.batch_scheduler import BatchScheduler
    from imba_chess.eval.gumbel_search import GumbelConfig
    from imba_chess.eval.position_evaluator import _SequenceHistory

    config = GumbelConfig(simulations=2)
    out = {}
    runtime.clear_caches()

    def play(position):
        board = chess.Board()
        history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
        for uci in position["prefix_moves"]:
            history.append_observed_position(board)
            history.record_played_move(uci)
            board.push_uci(uci)
        result = yield from runtime.search(board=board, history=history, actor_id="distill",
                                           game_id=position["position_id"], config=config,
                                           rng=random.Random(0), noise=0.0)
        return [runtime.move_vocab.decode(i) for i in result.legal_ids], list(result.root_log_priors)

    def done(key, value):
        out[key] = value

    def fail(key, exc):
        raise exc

    BatchScheduler(game_factory=iter((p["position_id"], play(p)) for p in positions),
                   executors=runtime.executors, concurrent_games=concurrency,
                   on_game_done=done, on_game_error=fail, completion_order=True).run()
    runtime.clear_caches()
    return out


def score_prior(priors, sf, targets, subset=None):
    rows = []
    for pid, (moves, target) in targets.items():
        if subset is not None and pid not in subset:
            continue
        legal, logs = priors[pid]
        if legal != moves:
            raise AssertionError("prior legal order differs from the study's")
        p = [math.exp(x) for x in logs]
        total = math.fsum(p)
        p = [x / total for x in p]
        score = [sf[pid][m] for m in moves]
        best = max(score)
        loss = [best - s for s in score]
        top = max(range(len(moves)), key=p.__getitem__)
        rows.append(dict(expected_loss=math.fsum(a * b for a, b in zip(p, loss)),
                         argmax_loss=loss[top], argmax_is_best=float(score[top] == best),
                         good_mass=math.fsum(a for a, l in zip(p, loss) if l <= 0.02),
                         kl_to_target=kl(target, p)))
    return {k: statistics.fmean(r[k] for r in rows) for k in rows[0]} | dict(positions=len(rows))


def main():
    import torch
    from imba_chess.self_play.config import load_config
    from imba_chess.self_play.runtime import load_runtime
    from imba_chess.self_play.trainer import Stage2Trainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="actor checkpoint with optimizer state")
    parser.add_argument("--replay", type=Path, required=True, help="frozen replay snapshot directory")
    parser.add_argument("--study", type=Path, required=True, help="measure_target_noise output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--milestones", default="0,250,500,1000,2000")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--microbatch-tokens", type=int, default=None,
                        help="smaller microbatches with more accumulation, same tokens per step")
    parser.add_argument("--save-weights", default="", help="comma-separated milestones to save model weights at")
    args = parser.parse_args()
    milestones = sorted({int(x) for x in args.milestones.split(",")})
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    cfg = load_config(args.config)
    learning = replace(cfg.learning, ema_decay=0.0)
    if args.microbatch_tokens is not None:
        factor, remainder = divmod(cfg.learning.microbatch_tokens * cfg.learning.gradient_accumulation, args.microbatch_tokens)
        if remainder:
            raise ValueError("microbatch tokens must divide the configured tokens per step")
        learning = replace(learning, microbatch_tokens=args.microbatch_tokens, gradient_accumulation=factor)
    save_at = {int(x) for x in args.save_weights.split(",") if x}
    positions, sf, targets = load_study(args.study)
    heldout_games = {p["game_id"] for p in positions}

    store = SelfPlayStore(args.replay, read_only=True, **{k: getattr(cfg.replay, k) for k in
                                                         ("window_positions", "flush_games", "flush_positions")})
    # The snapshot may already have evicted the study's games; drop any that remain.
    active = [g for g in store.manifest["active"] if g not in heldout_games]
    excluded = len(store.manifest["active"]) - len(active)
    store.manifest["active"] = active
    runtime, max_positions = load_runtime(cfg, args.checkpoint, "cuda")
    trainer = Stage2Trainer(model=runtime.model, config=learning, move_vocab=runtime.move_vocab,
                            encoder=runtime.encoder, device=runtime.device,
                            max_positions=max_positions, run_seed=cfg.run.seed)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer.optimizer.load_state_dict(saved["optimizer"])
    del saved
    trainer.begin_phase(store)
    train_positions = sum(store.index[g][1]["positions"] for g in trainer.sample_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train.jsonl").open("a")
    recent = []

    def on_step(metrics):
        log.write(json.dumps(metrics) + "\n")
        recent.append(metrics)
        del recent[:-50]
        if metrics["steps"] % 50 == 0:
            log.flush()

    start_priors = None
    report = dict(checkpoint=str(args.checkpoint), replay=str(args.replay), lr=learning.lr,
                  train_games=len(trainer.sample_ids), train_positions=train_positions,
                  heldout_games_excluded_from_replay=excluded,
                  train_iterations=[min(store.index[g][1]["iteration"] for g in trainer.sample_ids),
                                    max(store.index[g][1]["iteration"] for g in trainer.sample_ids)],
                  heldout_positions=len(positions), milestones=[])
    began = time.perf_counter()
    for milestone in milestones:
        if milestone > trainer.steps:
            trainer.train(store, exposure_budget=float("inf"),
                          should_stop=lambda: trainer.steps >= milestone, on_step=on_step)
        torch.cuda.empty_cache()
        if trainer.steps in save_at:
            torch.save(dict(model=runtime.model.state_dict()), args.out / f"distilled-{trainer.steps:06d}.pt")
        priors = prior_logs(runtime, positions, concurrency=args.concurrency)
        torch.cuda.empty_cache()
        if start_priors is None:
            start_priors = priors
            gains = {}
            for pid, (moves, target) in targets.items():
                p = [math.exp(x) for x in priors[pid][1]]
                total = math.fsum(p)
                score = [sf[pid][m] for m in moves]
                best = max(score)
                gains[pid] = math.fsum((a / total - b) * (best - s) for a, b, s in zip(p, target, score))
            ranked = sorted(gains, key=gains.get, reverse=True)
            high_gain = set(ranked[: len(ranked) // 4])
        train_metrics = ({k: statistics.fmean(m[k] for m in recent)
                          for k in ("policy_loss", "policy_target_kl", "policy_entropy", "value_loss")}
                         if recent else None)
        entry = dict(steps=trainer.steps, exposures=trainer.exposures,
                     epochs=trainer.exposures / train_positions,
                     seconds=round(time.perf_counter() - began),
                     heldout=score_prior(priors, sf, targets),
                     heldout_high_gain=score_prior(priors, sf, targets, subset=high_gain),
                     train_last_50_steps=train_metrics)
        report["milestones"].append(entry)
        atomic_json(args.out / "summary.json", report)
        print(json.dumps(entry), flush=True)
    log.close()


if __name__ == "__main__":
    main()
