"""Fixed-position screen of zero-noise Gumbel inference settings (c_visit x top_m).

Positions are model-to-move states sampled from saved evaluation games, with full
move history. Every legal move is scored once by unrestricted Stockfish (fresh hash,
equal nodes per move, WDL expectation from the side to move). Each setting's played
move is then scored against those labels, paired by position with a baseline
setting. Engine expectation is a proxy for move quality, not a match win rate.
No training or remote state is modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import threading
import time

import chess
import chess.engine
import chess.pgn
import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval import cozy_bridge
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import file_hash
from imba_chess.eval.search import terminal_value_for_color
from scripts.audit_search_scales import metrics, score_move


def sample_positions(pgn_paths, per_game, min_ply):
    """Model-to-move prefixes spread evenly over each game, deduplicated by EPD."""
    seen, positions = set(), []
    for path in pgn_paths:
        game = chess.pgn.read_game(io.StringIO(Path(path).read_text()))
        color = chess.WHITE if game.headers["White"] == "imba-chess" else chess.BLACK
        if game.headers["Black" if color == chess.WHITE else "White"] == "imba-chess":
            raise ValueError(f"cannot identify model colour in {path}")
        board, moves, candidates = game.board(), [], []
        for move in game.mainline_moves():
            if (board.turn == color and len(moves) >= min_ply
                    and terminal_value_for_color(board, color=board.turn) is None):
                candidates.append(list(moves))
            moves.append(move.uci())
            board.push(move)
        if not candidates:
            continue
        count = min(per_game, len(candidates))
        picks = [candidates[round(i * (len(candidates) - 1) / max(count - 1, 1))]
                 for i in range(count)]
        for prefix in picks:
            b = chess.Board()
            for uci in prefix:
                b.push_uci(uci)
            key = b.epd()
            if key in seen:
                continue
            seen.add(key)
            positions.append(dict(
                position_id=hashlib.sha256(" ".join(prefix).encode()).hexdigest()[:16],
                source=Path(path).name, prefix=prefix, fen=b.fen()))
    return positions


def paired_summary(rows, baseline, rng_seed=42):
    """Per-setting means plus position-bootstrap CI of the paired difference to baseline."""
    by_setting = {}
    for r in rows:
        by_setting.setdefault(r["setting"], {})[r["position_id"]] = r["metrics"]
    base = by_setting[baseline]
    summary = {}
    for setting, per_pos in sorted(by_setting.items()):
        ids = sorted(set(per_pos) & set(base))
        out = dict(positions=len(ids))
        for key in ("selected_expectation", "selected_regret", "changed_move", "target_gain"):
            out[key] = statistics.mean(per_pos[i][key] for i in ids)
        diffs = [per_pos[i]["selected_expectation"] - base[i]["selected_expectation"] for i in ids]
        rng = random.Random(rng_seed)
        means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
        out["diff_vs_baseline"] = statistics.mean(diffs)
        out["diff_vs_baseline_95ci"] = [means[50], means[1949]]
        out["same_move_as_baseline"] = statistics.mean(
            float(per_pos[i]["selected"] == base[i]["selected"]) for i in ids)
        summary[setting] = out
    return summary


def setting_key(c_visit, top_m):
    return f"c_visit={c_visit:g},top_m={top_m}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--games", type=Path, nargs="+", required=True, help="Directories of saved PGNs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-game", type=int, default=4)
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--c-visit", default="0,5,10,20,50")
    p.add_argument("--top-m", default="2,5,12,16,20")
    p.add_argument("--baseline", default="50,16")
    p.add_argument("--value-scale", type=float, default=0.1)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--stockfish", default="/usr/bin/stockfish")
    p.add_argument("--nodes-per-move", type=int, default=100000)
    p.add_argument("--engines", type=int, default=8)
    p.add_argument("--concurrent", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    c_visits = [float(x) for x in args.c_visit.split(",")]
    top_ms = [int(x) for x in args.top_m.split(",")]
    b_visit, b_top = args.baseline.split(",")
    baseline = setting_key(float(b_visit), int(b_top))
    settings = [(c, m) for c in c_visits for m in top_ms]
    if baseline not in {setting_key(c, m) for c, m in settings}:
        p.error("baseline must be part of the grid")
    if min(c_visits) < 0 or min(top_ms) < 1 or min(args.per_game, args.engines, args.concurrent) < 1:
        p.error("invalid bounds")
    cfg = load_config(args.config)
    pgns = sorted(str(f) for d in args.games for f in Path(d).glob("*.pgn"))
    if not pgns:
        p.error("no PGNs found")
    positions = sample_positions(pgns, args.per_game, args.min_ply)
    identity = dict(checkpoint=file_hash(args.checkpoint), config=cfg.identifier,
                    positions=[x["position_id"] for x in positions], settings=[setting_key(*s) for s in settings],
                    baseline=baseline, value_scale=args.value_scale, simulations=args.simulations,
                    top_depth=cfg.search.max_depth, nodes=args.nodes_per_move,
                    stockfish=file_hash(args.stockfish), noise="zero", device=args.device)
    with run_lock(args.output):
        path = args.output / "results.json"
        data = json.loads(path.read_text()) if path.exists() else dict(identity=identity, stockfish={}, search=[], timings={})
        if data["identity"] != identity:
            raise ValueError("different experiment identity")
        lock = threading.Lock()

        def save():
            with lock:
                atomic_json(path, data)

        start = time.monotonic()
        pending = [x for x in positions if x["position_id"] not in data["stockfish"]]
        local = threading.local()
        engines = []

        def engine():
            if not hasattr(local, "engine"):
                e = chess.engine.SimpleEngine.popen_uci(args.stockfish)
                e.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False, "UCI_ShowWDL": True})
                local.engine = e
                with lock:
                    engines.append(e)
            return local.engine

        def label(pos):
            board = chess.Board()  # full history so repetition draws are visible
            for uci in pos["prefix"]:
                board.push_uci(uci)
            scores ={m.uci(): score_move(engine(), board, m, args.nodes_per_move)
                      for m in sorted(board.legal_moves, key=lambda m: m.uci())}
            with lock:
                data["stockfish"][pos["position_id"]] = dict(pos, scores=scores)
                done = len(data["stockfish"])
            if done % 25 == 0:
                save()
                print("Stockfish", done, len(positions), flush=True)

        try:
            with ThreadPoolExecutor(args.engines) as pool:
                list(pool.map(label, pending))
        finally:
            for e in engines:
                e.quit()
        save()
        data["timings"]["stockfish_this_invocation"] = time.monotonic() - start

        torch.set_num_threads(4)
        runtime, max_positions = load_runtime(cfg, args.checkpoint, args.device)
        done = {(r["position_id"], r["setting"]) for r in data["search"]}
        tasks = {}

        def factory():
            for c_visit, top_m in settings:
                key_setting = setting_key(c_visit, top_m)
                config = replace(cfg.search, simulations=args.simulations, value_scale=args.value_scale,
                                 maxvisit_init=c_visit, top_m=top_m)
                for pos in positions:
                    if (pos["position_id"], key_setting) in done:
                        continue
                    history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
                    board = chess.Board()
                    for uci in pos["prefix"]:
                        history.append_observed_position(board)
                        history.record_played_move(uci)
                        board.push_uci(uci)
                    if len(history.seq_token_id) + 1 + config.max_depth > max_positions:
                        raise ValueError(f"context limit at {pos['position_id']}")
                    ids, _, ucis, _, _ = cozy_bridge.project_legal_moves(cozy_bridge.board_to_cozy(board), runtime.move_vocab)
                    key = json.dumps([pos["position_id"], key_setting])
                    tasks[key] = (pos, key_setting, dict(zip(ids, ucis)))
                    yield key, runtime.search(board=board, history=history, actor_id="screen", game_id=key,
                                              config=config, noise=0.0)

        def complete(key, result):
            pos, key_setting, mapping = tasks.pop(key)
            raw = asdict(result)
            moves = [mapping[i] for i in raw["legal_ids"]]
            scores = data["stockfish"][pos["position_id"]]["scores"]
            m = metrics(raw, moves, scores)
            m.update(selected=raw["move_uci"], selected_expectation=scores[raw["move_uci"]]["expectation"])
            data["search"].append(dict(position_id=pos["position_id"], setting=key_setting, metrics=m,
                                       visits=raw["visits"], depth_cutoffs=raw["depth_cutoffs"]))
            if len(data["search"]) % 250 == 0:
                save()
                print("Search", len(data["search"]), len(positions) * len(settings), flush=True)

        def error(key, exc):
            raise RuntimeError(key) from exc

        start = time.monotonic()
        BatchScheduler(game_factory=iter(factory()), executors=runtime.executors, concurrent_games=args.concurrent,
                       on_game_done=complete, on_game_error=error, completion_order=True).run()
        data["timings"]["search_this_invocation"] = time.monotonic() - start
        data["summary"] = paired_summary(data["search"], baseline)
        save()
        atomic_json(args.output / "summary.json", data["summary"])
        print(json.dumps(data["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
