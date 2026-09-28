"""Compare the main (search) and auxiliary (training-only) value readouts as evaluators.

Uses positions and full-strength Stockfish labels for every legal move from a
finished `audit_search_settings` run. For each position, every legal child is
evaluated by the network (full history, one forward pass shared by both readouts
through the same value features); terminal children use the rules. Child values
are converted to the root player's expected score and compared with Stockfish's
per-move expectation: greedy one-ply move quality, rank correlation, blunder-pair
ordering, and root calibration. Engine labels are proxies. Eval only.
"""
import argparse
import json
from pathlib import Path
import statistics

import chess
import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.merged_executors import _merge_root_batches
from imba_chess.eval.position_evaluator import _SequenceHistory, _forward_model
from imba_chess.eval.search import terminal_value_for_color
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import file_hash

GOOD_MARGIN = 0.02


def expectation(wdl):
    """[loss, draw, win] from the side to move -> expected score for that side."""
    return wdl[2] + 0.5 * wdl[1]


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def score_readout(positions, values, root_values, blunder_pairs):
    """values[pid][move] -> root-player expectation; returns summary metrics."""
    greedy_good, greedy_regret, rhos, root_err, root_bias = [], [], [], [], []
    for pid, pos in positions.items():
        exp = {m: s["expectation"] for m, s in pos["scores"].items()}
        best = max(exp.values())
        v = values[pid]
        pick = max(v, key=v.get)
        greedy_good.append(float(exp[pick] >= best - GOOD_MARGIN))
        greedy_regret.append(best - exp[pick])
        moves = sorted(exp)
        if len(moves) > 2:
            rho = spearman([v[m] for m in moves], [exp[m] for m in moves])
            if rho is not None:
                rhos.append(rho)
        root_err.append(abs(root_values[pid] - best))
        root_bias.append(root_values[pid] - best)
    prefers_good = [float(values[pid][good] > values[pid][bad]) for pid, good, bad in blunder_pairs]
    return dict(
        positions=len(positions),
        greedy_good_move_rate=statistics.mean(greedy_good),
        greedy_regret=statistics.mean(greedy_regret),
        mean_spearman=statistics.mean(rhos),
        root_abs_error=statistics.mean(root_err),
        root_bias=statistics.mean(root_bias),
        blunder_pairs=len(blunder_pairs),
        blunder_pairs_prefers_good=statistics.mean(prefers_good) if prefers_good else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--blunders", type=Path, required=True, help="audit_blunders records.json")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=1024)  # dense-mask path
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    screen = json.loads((args.screen / "results.json").read_text())
    positions = screen["stockfish"]
    blunder_pairs = [(r["position_id"], r["best_good_move"], r["chosen"])
                     for r in json.loads(args.blunders.read_text())["records"] if r["group"] == "blunder"]
    cfg = load_config(args.config)
    runtime, _ = load_runtime(cfg, args.checkpoint, args.device)
    model = runtime.model
    if model.auxiliary_value_head is None:
        raise ValueError("checkpoint has no auxiliary value head")
    captured = {}
    hook = model.value_head[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("features", i[0]))

    # One request per (position, child or root). Root is keyed with move None.
    requests, readouts = [], {"main": {}, "aux": {}}
    roots = {"main": {}, "aux": {}}
    for pid, pos in positions.items():
        for key in ("main", "aux"):
            readouts[key][pid] = {}
        board = chess.Board()
        for uci in pos["prefix"]:
            board.push_uci(uci)
        root_color = board.turn
        for move in [None] + sorted(pos["scores"]):
            b = board.copy()
            if move is not None:
                b.push_uci(move)
                terminal = terminal_value_for_color(b, color=root_color)
                if terminal is not None:
                    for key in ("main", "aux"):
                        readouts[key][pid][move] = (terminal + 1) / 2
                    continue
            history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
            replay = chess.Board()
            for uci in pos["prefix"] + ([move] if move else []):
                history.append_observed_position(replay)
                history.record_played_move(uci)
                replay.push_uci(uci)
            batch = history.build_batch_for_current_position(replay)
            batch["game_id"] = [f"{pid}:{move}"]
            requests.append((pid, move, batch))

    def flush(group):
        merged = _merge_root_batches([b for _, _, b in group])
        out = _forward_model(model=model, batch=merged, device=runtime.device, dtype=torch.float32)
        last = merged["seq_offsets"][1:].to(out["value_logits"].device) - 1
        with torch.inference_mode():
            main = torch.softmax(out["value_logits"].float()[last], -1).tolist()
            aux = torch.softmax(model.auxiliary_value_head(captured["features"]).float()[last], -1).tolist()
        for (pid, move, _), m, a in zip(group, main, aux):
            for key, wdl in (("main", m), ("aux", a)):
                if move is None:
                    roots[key][pid] = expectation(wdl)  # root side to move
                else:
                    readouts[key][pid][move] = 1 - expectation(wdl)  # child side is the opponent

    group, tokens = [], 0
    for req in requests:
        size = int(req[2]["total_tokens"])
        if group and tokens + size > args.max_tokens:
            flush(group)
            group, tokens = [], 0
        group.append(req)
        tokens += size
    if group:
        flush(group)
    hook.remove()

    summary = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                   screen=str(args.screen), evaluations=len(requests))
    for key in ("main", "aux"):
        summary[key] = score_readout(positions, readouts[key], roots[key], blunder_pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "values.json", dict(readouts=readouts, roots=roots))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
