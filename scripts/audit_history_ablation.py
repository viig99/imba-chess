"""Fixed-weight history ablation of the main value head on the tactical-recognition states.

Re-evaluates every nonterminal state from `audit_frozen_value_probes` rows with the
input history truncated to the last K positions (K=1 is the current board alone;
the first kept position has no previous-move token, like a game start). Compares
with 1M-node Stockfish expected score for the original player: lag after the
audited move, absolute error per branch, and good-vs-bad ordering right after the
paired moves. Truncated sequences are out of distribution for a model trained on
games from move one, so the good branch is the matched control. Eval only.
"""
import argparse
import json
from pathlib import Path
import statistics
from collections import defaultdict

import chess
import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.merged_executors import _merge_root_batches
from imba_chess.eval.position_evaluator import _SequenceHistory, _forward_model
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import file_hash


def truncated_batch(runtime, prefix, keep):
    """History holding only the last `keep` positions (None keeps everything)."""
    start = 0 if keep is None else max(0, len(prefix) - (keep - 1))
    board = chess.Board()
    for uci in prefix[:start]:
        board.push_uci(uci)
    history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
    for uci in prefix[start:]:
        history.append_observed_position(board)
        history.record_played_move(uci)
        board.push_uci(uci)
    return board, history.build_batch_for_current_position(board)


def original_player_expectation(ldw, turn_white, root_white):
    own = ldw[2] + 0.5 * ldw[1]
    return own if turn_white == root_white else 1 - own


def summarize(rows, values):
    lines = defaultdict(dict)
    for r, v in zip(rows, values):
        lines[(r["position_id"], r["arm"])][r["ply"]] = (r["reference"], v)
    out = {}
    for arm in ("bad", "good"):
        errors = [abs(v - ref) for (pid, a), d in lines.items() if a == arm for ref, v in d.values()]
        out[f"{arm}_mae"] = statistics.mean(errors)
    for k in (1, 2, 4, 8, 16):
        e, m = [], []
        for (pid, arm), d in lines.items():
            if arm == "bad" and 0 in d and k in d:
                e.append(d[0][0] - d[k][0])
                m.append(d[0][1] - d[k][1])
        out[f"lag_registered_ply{k}"] = statistics.mean(m) / statistics.mean(e)
    prefers, pids = [], {pid for pid, _ in lines}
    for pid in pids:
        bad, good = lines.get((pid, "bad"), {}).get(1), lines.get((pid, "good"), {}).get(1)
        if bad and good:
            prefers.append(float(good[1] > bad[1]))
    out["good_preferred_after_paired_moves"] = f"{int(sum(prefers))}/{len(prefers)}"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, required=True, help="frozen-value-probes rows.json")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, action="append", required=True)
    p.add_argument("--keep", default="full,16,8,4,2,1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args()
    rows = json.loads(args.rows.read_text())
    for r in rows:
        r["prefix"] = r["prefix"] if isinstance(r["prefix"], list) else json.loads(r["prefix"].replace("'", '"'))
        r["ply"], r["reference"] = int(r["ply"]), float(r["reference"])
        for key in ("root_white", "turn_white"):
            r[key] = r[key] in (True, "True")
    keeps = [None if k == "full" else int(k) for k in args.keep.split(",")]
    cfg = load_config(args.config)
    result = dict(rows=str(args.rows), rows_sha256=file_hash(args.rows), checkpoints={})
    for checkpoint in args.checkpoint:
        runtime, _ = load_runtime(cfg, checkpoint, "cuda")
        per_keep = {}
        for keep in keeps:
            requests = []
            for r in rows:
                board, batch = truncated_batch(runtime, r["prefix"], keep)
                if board.fen() != r["fen"]:
                    raise ValueError(f"history mismatch at {r['position_id']} ply {r['ply']}")
                batch["game_id"] = [f"{r['position_id']}:{r['arm']}:{r['ply']}"]
                requests.append(batch)
            values, group, tokens = [], [], 0

            def flush(group):
                merged = _merge_root_batches(group)
                out = _forward_model(model=runtime.model, batch=merged, device=runtime.device, dtype=torch.float32)
                last = merged["seq_offsets"][1:].to(out["value_logits"].device) - 1
                return torch.softmax(out["value_logits"].float()[last], -1).tolist()

            for batch in requests:
                size = int(batch["total_tokens"])
                if group and tokens + size > args.max_tokens:
                    values += flush(group)
                    group, tokens = [], 0
                group.append(batch)
                tokens += size
            if group:
                values += flush(group)
            expectations = [original_player_expectation(v, r["turn_white"], r["root_white"])
                            for v, r in zip(values, rows)]
            if keep is None:
                baseline = [json.loads(r["baseline_wdl"]) if isinstance(r["baseline_wdl"], str) else r["baseline_wdl"]
                            for r in rows]
                if "actor-000300" in str(checkpoint):
                    worst = max(max(abs(a - b) for a, b in zip(v, w)) for v, w in zip(values, baseline))
                    if worst > 2e-4:
                        raise AssertionError(f"full-history values differ from the audit by {worst}")
            per_keep["full" if keep is None else str(keep)] = dict(summary=summarize(rows, expectations),
                                                                   expectations=expectations)
            print(checkpoint.name, keep, json.dumps(per_keep["full" if keep is None else str(keep)]["summary"]),
                  flush=True)
        result["checkpoints"][str(checkpoint)] = dict(sha256=file_hash(checkpoint), keeps=per_keep)
        del runtime
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "results.json", result)
    atomic_json(args.output / "summary.json", {c: {k: v["summary"] for k, v in d["keeps"].items()}
                                               for c, d in result["checkpoints"].items()})


if __name__ == "__main__":
    main()
