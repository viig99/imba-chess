"""Frozen value recognition along engine continuations; never trains or writes replay."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import statistics
import time

import chess
import chess.engine

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.seeds import file_hash


PROTOCOL = dict(
    version=1, continuation_plies=16, nodes=100_000, verification_nodes=1_000_000,
    seed=42, engine_threads=1, engine_hash_mib=64, fresh_hash=True,
    engine_limit_strength=False, dtype="float32", tf32=False,
    material_values={"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9},
    material_loss_threshold=3, quiet_following_plies=2,
    stable_expectation_tolerance=.10, strongly_losing_threshold=.20,
    prediction_advantage_threshold=.50, recognition_error_tolerance=.15,
    bootstrap_replicates=10000,
)


def board_for(prefix):
    board = chess.Board()
    for uci in prefix:
        board.push_uci(uci)
    return board


def material(board, color):
    return sum(v * (len(board.pieces(t, color)) - len(board.pieces(t, not color)))
               for t, v in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                            (chess.ROOK, 5), (chess.QUEEN, 9)])


def expected_score(wdl, turn, root_color):
    score = wdl[2] + .5 * wdl[1]  # network order: loss, draw, win
    return score if turn == root_color else 1 - score


def terminal_record(board, root_color):
    # Match the repository's repetition/fifty-move adjudication, without prospective claims.
    outcome = board.outcome(claim_draw=False)
    if outcome is None and (board.is_repetition(3) or board.is_fifty_moves()):
        return dict(expectation=.5, termination="threefold_or_fifty_moves")
    if outcome is None:
        return None
    return dict(expectation=.5 if outcome.winner is None else float(outcome.winner == root_color),
                termination=outcome.termination.name)


def select_positions(screen, audit):
    if screen["identity"]["checkpoint"] != audit["checkpoint"]:
        raise ValueError("audit/screen checkpoint mismatch")
    result = []
    for rec in sorted(audit["records"], key=lambda r: r["position_id"]):
        if rec["group"] != "blunder" or rec["cause"] != "value_misorder":
            continue
        pos = screen["stockfish"][rec["position_id"]]
        board = board_for(pos["prefix"])
        if board.fen() != pos["fen"]:
            raise ValueError("history/FEN mismatch")
        for move in (rec["chosen"], rec["best_good_move"]):
            if chess.Move.from_uci(move) not in board.legal_moves:
                raise ValueError("illegal audited move")
        result.append(dict(position_id=rec["position_id"], source=pos["source"],
                           prefix=pos["prefix"], fen=pos["fen"], root_white=board.turn,
                           initial_material=material(board, board.turn),
                           bad=rec["chosen"], good=rec["best_good_move"],
                           historical_bad=pos["scores"][rec["chosen"]],
                           historical_good=pos["scores"][rec["best_good_move"]]))
    return result


def ensure_manifest(path, identity):
    if path.exists():
        old = json.loads(path.read_text())
        if old != identity:
            raise ValueError("incompatible tactical diagnostic resume")
    else:
        atomic_json(path, identity)


def prepare(args):
    from imba_chess.self_play.config import load_config
    cfg = load_config(args.config)
    screen = json.loads(args.screen.read_text())
    audit = json.loads(args.blunders.read_text())
    positions = select_positions(screen, audit)
    sources = sorted(Path("src/imba_chess").rglob("*.py")) + [Path(__file__)]
    identity = dict(protocol=PROTOCOL, positions=positions,
                    checkpoints={"actor300": str(args.actor), "start": str(args.start)},
                    checkpoint_hashes={"actor300": file_hash(args.actor), "start": file_hash(args.start)},
                    config=str(args.config), config_hash=file_hash(args.config),
                    base_config_hash=file_hash(cfg.base_config),
                    screen_hash=file_hash(args.screen), audit_hash=file_hash(args.blunders),
                    stockfish=str(args.stockfish.resolve()), engine_hash=file_hash(args.stockfish),
                    source_hashes={str(p): file_hash(p) for p in sources})
    ensure_manifest(args.output / "manifest.json", identity)
    return identity


def engine_score(engine, board, root_color, nodes):
    engine.configure({"Clear Hash": None})
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    score, wdl = info["score"].pov(root_color), info["wdl"].pov(root_color)
    return dict(cp=score.score(), mate=score.mate(), expectation=wdl.expectation(),
                wdl_wdl=list(wdl), depth=info.get("depth"), nodes=info.get("nodes"),
                pv=[m.uci() for m in info.get("pv", [])])


def mark_resolution(states, initial_material):
    """Uses only engine/board evidence, never network predictions."""
    for i, state in enumerate(states):
        state["resolution"] = []
        if state["terminal"] or state["ply"] < 1:
            continue
        a, b = state["engine"], state["verified"]
        state["stable"] = abs(a["expectation"] - b["expectation"]) <= .10
        state["stable_strong_loss"] = (state["stable"] and
                                        max(a["expectation"], b["expectation"]) <= .20)
        if not state["stable_strong_loss"]:
            continue
        if a["mate"] is not None and b["mate"] is not None and a["mate"] < 0 and b["mate"] < 0:
            state["resolution"].append("forced_mate")
        # Require two subsequent quiet played plies with the deficit still present.
        tail = states[i + 1:i + 3]
        quiet = len(tail) == 2 and not state["in_check"] and all(
            not s["capture"] and not s["promotion"] and not s["in_check"] and
            not s["terminal"] for s in tail)
        if quiet and all(s["material"] <= initial_material - 3 for s in [state] + tail):
            state["resolution"].append("settled_material_loss")
        opponent_promoted = any(s["promotion"] and s["mover_white"] != state["root_white"]
                                for s in states[:i + 1])
        if quiet and opponent_promoted:
            state["resolution"].append("opponent_promotion")
    return states


def generate_line(pos, arm, args):
    path = args.output / "lines" / f'{pos["position_id"]}-{arm}.json'
    if path.exists():
        line = json.loads(path.read_text())
        if line.get("complete"):
            return line
    else:
        line = dict(position_id=pos["position_id"], source=pos["source"], arm=arm,
                    root_white=pos["root_white"], prefix=pos["prefix"], states=[], complete=False)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False, "UCI_ShowWDL": True})
        line["engine_identity"] = engine.id
        board = board_for(pos["prefix"])
        moves = [s["move"] for s in line["states"] if s["move"] is not None]
        for move in moves:
            board.push_uci(move)
        # State 0 is before the forced move; state 1 is immediately after it.
        for ply in range(len(line["states"]), PROTOCOL["continuation_plies"] + 2):
            move, capture, promotion, mover = None, False, None, None
            if ply:
                previous = line["states"][-1]
                if previous["terminal"]:
                    break
                move = pos[arm] if ply == 1 else previous["verified"]["pv"][0]
                m = chess.Move.from_uci(move)
                if m not in board.legal_moves:
                    raise ValueError("illegal engine continuation")
                capture, promotion, mover = board.is_capture(m), m.promotion, board.turn
                board.push(m)
                moves.append(move)
            terminal = terminal_record(board, pos["root_white"])
            row = dict(ply=ply, move=move, capture=capture, promotion=promotion,
                       mover_white=mover, root_white=pos["root_white"], turn_white=board.turn,
                       fen=board.fen(), material=material(board, pos["root_white"]),
                       in_check=board.is_check(), terminal=terminal)
            if terminal is None:
                row["engine"] = engine_score(engine, board, pos["root_white"], PROTOCOL["nodes"])
                row["verified"] = engine_score(engine, board, pos["root_white"], PROTOCOL["verification_nodes"])
            line["states"].append(row)
            atomic_json(path, line)
            if terminal:
                break
        line["states"] = mark_resolution(line["states"], pos["initial_material"])
        line["complete"] = True
        atomic_json(path, line)
        return line
    finally:
        engine.quit()


def generate(args, manifest):
    tasks = [(p, arm) for p in manifest["positions"] for arm in ("bad", "good")]
    if args.limit:
        tasks = tasks[:args.limit]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_line, pos, arm, args) for pos, arm in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            line = future.result()
            print(json.dumps(dict(phase="engine", complete=n, total=len(tasks),
                                  line=f'{line["position_id"]}-{line["arm"]}',
                                  states=len(line["states"]), seconds=time.monotonic() - started)), flush=True)


def make_batch(runtime, prefix):
    from imba_chess.eval.position_evaluator import _SequenceHistory
    history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
    board = chess.Board()
    for uci in prefix:
        history.append_observed_position(board)
        history.record_played_move(uci)
        board.push_uci(uci)
    batch = history.build_batch_for_current_position(board)
    batch["game_id"] = ["tactical-audit"]
    return board, batch


def full_prediction(runtime, prefix, captured):
    import torch
    from imba_chess.eval.position_evaluator import _forward_model
    board, batch = make_batch(runtime, prefix)
    out = _forward_model(model=runtime.model, batch=batch, device=runtime.device, dtype=torch.float32)
    with torch.inference_mode():
        main = out["value_logits"][-1].float().softmax(-1).tolist()
        aux = runtime.model.auxiliary_value_head(captured["features"])[-1].float().softmax(-1).tolist()
    return board, batch, out, main, aux


def verify_cache(runtime, lines, captured):
    import torch
    from imba_chess.eval import cozy_bridge
    from imba_chess.eval.position_evaluator import CachedPositionEvaluator
    checks = []
    # Distinct original colors, bad and good branches, multiple successive suffix nodes.
    selected = []
    for color in (True, False):
        for arm in ("bad", "good"):
            candidates = [l for l in lines if l["root_white"] == color and l["arm"] == arm]
            if candidates:
                selected.append(max(candidates, key=lambda l: len(l["states"])))
    for li, line in enumerate(selected):
        prefix = list(line["prefix"])
        board, batch, out, main, aux = full_prediction(runtime, prefix, captured)
        owner = ("cache-verification", str(li))
        result = runtime.executors["root_eval"]([(owner, batch)])[0][1]
        root_wdl = result["value_logits"][-1].float().softmax(-1).tolist()
        root_err = max(abs(a - b) for a, b in zip(root_wdl, main))
        if root_err > 2e-4:
            raise AssertionError("root executor differs from full forward")
        evaluator = CachedPositionEvaluator(model=runtime.model, move_vocab=runtime.move_vocab,
            board_state_encoder=runtime.encoder, device=runtime.device, dtype=torch.float32,
            prefix_kv=result["kv_caches"], prefix_len=batch["total_tokens"], immutable_prefix=True)
        evaluator._runtime_token = runtime._cache_token
        handle = None
        for state in line["states"][1:9]:
            if state["terminal"]:
                break
            prefix.append(state["move"])
            board, batch, out, main, aux = full_prediction(runtime, prefix, captured)
            handle = evaluator.extend(handle, state["move"])
            prediction = runtime.executors["decode_wave"]([
                (owner, (evaluator, [(handle, cozy_bridge.board_to_cozy(board))]))])[0][1][0]
            # Decode workspace may bypass Python module hooks: main WDL is authoritative here.
            error = max(abs(a - b) for a, b in zip(prediction.wdl, main))
            legal = sorted(m.uci() for m in board.legal_moves)
            if prediction.legal_ucis != legal or error > 2e-4:
                raise AssertionError(f"cache/legal mismatch: {error}")
            checks.append(dict(position_id=line["position_id"], arm=line["arm"],
                               ply=state["ply"], turn_white=board.turn, max_wdl_error=error,
                               root_error=root_err, legal_moves=len(legal)))
    runtime.clear_caches()
    return checks


def infer(args, manifest):
    import torch
    from imba_chess.self_play.config import load_config
    from imba_chess.self_play.runtime import load_runtime
    torch.set_num_threads(4)
    lines = [json.loads(p.read_text()) for p in sorted((args.output / "lines").glob("*.json"))]
    lines = [l for l in lines if l["complete"]]
    if args.limit:
        lines = lines[:args.limit]
    if not lines:
        raise ValueError("no completed engine lines")
    cfg = load_config(args.config)
    for name, checkpoint in manifest["checkpoints"].items():
        runtime, max_positions = load_runtime(cfg, Path(checkpoint), "cuda")
        captured = {}
        hook = runtime.model.value_head[-1].register_forward_hook(
            lambda m, i, o: captured.__setitem__("features", i[0]))
        try:
            checks = verify_cache(runtime, lines, captured)
            atomic_json(args.output / f"verification-{name}.json", dict(checks=checks,
                max_wdl_error=max(c["max_wdl_error"] for c in checks), dtype="float32", tf32=False))
            for n, line in enumerate(lines, 1):
                path = args.output / name / f'{line["position_id"]}-{line["arm"]}.json'
                if path.exists():
                    continue
                prefix, rows = list(line["prefix"]), []
                for state in line["states"]:
                    if state["move"]:
                        prefix.append(state["move"])
                    if len(prefix) + 1 > max_positions:
                        raise ValueError("history exceeds model context")
                    if state["terminal"]:
                        rows.append(dict(ply=state["ply"], terminal=state["terminal"], main=None, aux=None))
                        continue
                    board, _, _, main, aux = full_prediction(runtime, prefix, captured)
                    if board.fen() != state["fen"]:
                        raise ValueError("inference history mismatch")
                    rows.append(dict(ply=state["ply"], main_wdl_ldw=main, aux_wdl_ldw=aux,
                        main=expected_score(main, board.turn, line["root_white"]),
                        aux=expected_score(aux, board.turn, line["root_white"])))
                atomic_json(path, dict(position_id=line["position_id"], arm=line["arm"],
                                       checkpoint_sha256=manifest["checkpoint_hashes"][name], states=rows))
                print(json.dumps(dict(phase="network", checkpoint=name, complete=n, total=len(lines))), flush=True)
        finally:
            hook.remove()
            runtime.clear_caches()
            del runtime
            captured.clear()
            torch.cuda.empty_cache()


def cluster_summary(rows, key, seed=42):
    groups = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            groups[row["source"]].append(row[key])
    if not groups:
        return dict(n=0, games=0, mean=None, ci95=None)
    pairs = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    rng = random.Random(seed)
    draws = []
    for _ in range(PROTOCOL["bootstrap_replicates"]):
        sample = rng.choices(pairs, k=len(pairs))
        draws.append(sum(p[0] for p in sample) / sum(p[1] for p in sample))
    draws.sort()
    return dict(n=sum(p[1] for p in pairs), games=len(pairs),
                mean=sum(p[0] for p in pairs) / sum(p[1] for p in pairs),
                ci95=[draws[249], draws[9749]])


def summarize(args, manifest):
    # Preserve one observation per original position and endpoint; bootstrap source games.
    rows = []
    for pos in manifest["positions"]:
        linepaths = {a: args.output / "lines" / f'{pos["position_id"]}-{a}.json' for a in ("bad", "good")}
        if not all(p.exists() for p in linepaths.values()):
            continue
        lines = {a: json.loads(p.read_text()) for a, p in linepaths.items()}
        pred = {}
        for name in manifest["checkpoints"]:
            for arm in lines:
                path = args.output / name / f'{pos["position_id"]}-{arm}.json'
                if path.exists():
                    pred[name, arm] = json.loads(path.read_text())["states"]
        if len(pred) != 4:
            continue
        row = dict(position_id=pos["position_id"], source=pos["source"], root_white=pos["root_white"])
        states = lines["bad"]["states"]
        endpoints = {"after_blunder": 1}
        for label in ("settled_material_loss", "opponent_promotion", "forced_mate"):
            endpoints[label] = next((s["ply"] for s in states if label in s.get("resolution", [])), None)
        endpoints["physical"] = min((x for x in [endpoints["settled_material_loss"],
                                    endpoints["opponent_promotion"]] if x is not None), default=None)
        endpoints["last_nonterminal"] = max(s["ply"] for s in states if not s["terminal"])
        row["endpoints"] = endpoints
        row["terminal"] = states[-1]["terminal"]
        row["resolved"] = any(endpoints[k] is not None for k in ("physical", "forced_mate"))
        for name in manifest["checkpoints"]:
            bad0, good0 = pred[name, "bad"][1], pred[name, "good"][1]
            row[f"{name}_misorders"] = (bad0["main"] >= good0["main"]
                                         if bad0["main"] is not None and good0["main"] is not None else None)
            for head in ("main", "aux"):
                for label, index in endpoints.items():
                    if index is None or states[index]["terminal"]:
                        continue
                    state, prediction = states[index], pred[name, "bad"][index][head]
                    ref = state["verified"]["expectation"]
                    key = f"{name}_{head}_{label}"
                    row[key + "_prediction"] = prediction
                    row[key + "_reference"] = ref
                    row[key + "_error"] = abs(prediction - ref)
                    row[key + "_bias"] = prediction - ref
                    row[key + "_wrong_advantage"] = float(prediction > .5)
                physical = endpoints["physical"]
                if physical is not None:
                    before = abs(pred[name, "bad"][1][head] - states[1]["verified"]["expectation"])
                    row[f"{name}_{head}_physical_error_change"] = row[f"{name}_{head}_physical_error"] - before
                    # Recognition must persist through all subsequent stable, strongly losing nonterminal states.
                    eligible = [s for s in states[physical:] if s.get("stable_strong_loss")]
                    onset = next((s["ply"] for s in eligible if all(
                        abs(pred[name, "bad"][t["ply"]][head] - t["verified"]["expectation"]) <= .15
                        for t in eligible if t["ply"] >= s["ply"])), None)
                    row[f"{name}_{head}_recognition_delay"] = None if onset is None else onset - physical
                    row[f"{name}_{head}_unrecognized_at_end"] = float(onset is None)
        for label in endpoints:
            key = f"main_{label}_error"
            if f"actor300_{key}" in row:
                row[f"actor_minus_start_{key}"] = row[f"actor300_{key}"] - row[f"start_{key}"]
        # Paired good branch at the same ply as the physical endpoint, where available.
        physical = endpoints["physical"]
        if physical is not None and physical < len(lines["good"]["states"]):
            good_state = lines["good"]["states"][physical]
            if not good_state["terminal"]:
                for name in manifest["checkpoints"]:
                    row[f"{name}_main_good_control_error"] = abs(
                        pred[name, "good"][physical]["main"] - good_state["verified"]["expectation"])
        rows.append(row)
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (float, int)) and k != "root_white"})
    summary = dict(positions=len(rows), expected_positions=len(manifest["positions"]),
                   source_games=len({r["source"] for r in rows}),
                   endpoints={k: sum(r["endpoints"].get(k) is not None for r in rows)
                              for k in ("physical", "settled_material_loss", "opponent_promotion", "forced_mate")},
                   unresolved=sum(not r["resolved"] for r in rows),
                   metrics={k: cluster_summary(rows, k) for k in keys},
                   actor_misordered_metrics={k: cluster_summary([r for r in rows if r["actor300_misorders"]], k)
                                            for k in keys}, positions_detail=rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("metrics", "actor_misordered_metrics", "positions_detail")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "infer", "summarize"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/tactical-recognition-2026-09-24"))
    parser.add_argument("--screen", type=Path, default=Path("artifacts/eval/search-settings-screen-2026-09-24/results.json"))
    parser.add_argument("--blunders", type=Path, default=Path("artifacts/eval/blunder-study-2026-09-24/records.json"))
    parser.add_argument("--config", type=Path, default=Path("artifacts/self_play/flatten-53250-auxiliary-2026-09-21/config.toml"))
    parser.add_argument("--actor", type=Path, default=Path("artifacts/eval/flatten-auxiliary-actor300/actor-000300.pt"))
    parser.add_argument("--start", type=Path, default=Path("artifacts/eval/flatten-auxiliary-actor103-2026-09-22/starting-checkpoint.pt"))
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/bin/stockfish"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="smoke only: first N lines, not part of frozen protocol")
    args = parser.parse_args()
    from imba_chess.self_play.runtime import run_lock
    with run_lock(args.output):
        manifest = prepare(args)
        if args.phase == "generate":
            generate(args, manifest)
        elif args.phase == "infer":
            infer(args, manifest)
        elif args.phase == "summarize":
            summarize(args, manifest)


if __name__ == "__main__":
    main()
