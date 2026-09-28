"""Frozen-feature value probes at scale: every legal child of screened positions.

Rows are all nonterminal children (position + one legal move) of the positions in one
or more finished `audit_search_settings` runs, labelled with that move's full-strength
Stockfish WDL (100k nodes, fresh hash). Features come from one frozen checkpoint:
the trunk output feeding the private value branch (1,024), the input to the final
value projection (512) and the original logits (3). Probes are refit on source-game
folds (a game's positions and all their children stay together; same-named PGNs from
different directories are merged, which is conservative). Held-out metrics focus on
what search needs: ordering sibling moves by value. Engine labels are proxies. Eval only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import time

import chess
import torch
import torch.nn.functional as F

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.merged_executors import _merge_root_batches
from imba_chess.eval.position_evaluator import _SequenceHistory, _forward_model
from imba_chess.eval.search import terminal_value_for_color
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import file_hash
from scripts.audit_frozen_value_probes import fit_linear, predict, soft_ce

PENALTIES = [0.001, 0.01, 0.1, 1.0, 10.0]
GOOD_MARGIN = 0.02


def build_rows(screens):
    rows = []
    for screen in screens:
        data = json.loads((screen / "results.json").read_text())
        for pid, pos in data["stockfish"].items():
            board = chess.Board()
            for uci in pos["prefix"]:
                board.push_uci(uci)
            for move, score in sorted(pos["scores"].items()):
                child = board.copy()
                child.push_uci(move)
                if terminal_value_for_color(child, color=board.turn) is not None:
                    continue
                wdl = score["wdl"]  # [win, draw, loss] for the parent's side to move
                total = sum(wdl)
                # The child's side to move is the opponent: its [loss, draw, win] = parent's [win, draw, loss].
                rows.append(dict(position=f"{screen.name}:{pid}", position_id=pid, group=pos["source"],
                                 move=move, prefix=pos["prefix"] + [move],
                                 target=[wdl[0] / total, wdl[1] / total, wdl[2] / total],
                                 parent_expectation=score["expectation"]))
    return rows


def extract(runtime, rows, max_tokens):
    captured = {}
    hooks = [runtime.model.value_head[0].register_forward_pre_hook(lambda m, i: captured.__setitem__("shared1024", i[0])),
             runtime.model.value_head[-1].register_forward_pre_hook(lambda m, i: captured.__setitem__("value512", i[0]))]
    out = {k: [] for k in ("shared1024", "value512", "output_only")}

    def flush(group):
        merged = _merge_root_batches(group)
        result = _forward_model(model=runtime.model, batch=merged, device=runtime.device, dtype=torch.float32)
        last = merged["seq_offsets"][1:].to(result["value_logits"].device) - 1
        out["shared1024"].append(captured["shared1024"][last].float().cpu())
        out["value512"].append(captured["value512"][last].float().cpu())
        out["output_only"].append(result["value_logits"][last].float().cpu())

    try:
        group, tokens = [], 0
        for i, row in enumerate(rows):
            history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
            board = chess.Board()
            for uci in row["prefix"]:
                history.append_observed_position(board)
                history.record_played_move(uci)
                board.push_uci(uci)
            batch = history.build_batch_for_current_position(board)
            batch["game_id"] = [str(i)]
            size = int(batch["total_tokens"])
            if group and tokens + size > max_tokens:
                flush(group)
                group, tokens = [], 0
            group.append(batch)
            tokens += size
            if (i + 1) % 5000 == 0:
                print(json.dumps(dict(phase="extract", done=i + 1, total=len(rows))), flush=True)
        if group:
            flush(group)
    finally:
        for h in hooks:
            h.remove()
    return {k: torch.cat(v) for k, v in out.items()}


def fit_mlp(x, y, xv, yv, seed):
    """Nonlinear probe: 1024 -> 512 -> 3, early-stopped on validation soft CE."""
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0).clamp_min(0.01)
    norm = lambda t: (t - mean) / scale
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 512), torch.nn.SiLU(), torch.nn.Linear(512, 3)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    xs, ys, xvs, yvs = norm(x).cuda(), y.float().cuda(), norm(xv).cuda(), yv.float().cuda()
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(60):
        perm = torch.randperm(len(xs), device="cuda")
        model.train()
        for i in range(0, len(xs), 1024):
            idx = perm[i:i + 1024]
            loss = soft_ce(model(xs[idx]), ys[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(soft_ce(model(xvs), yvs).mean())
        if val < best - 1e-4:
            best, best_state, stale = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return lambda t: model(norm(t).cuda()).double().cpu(), dict(validation_ce=best, epochs=epoch + 1)


def metrics(rows, idx, logits):
    """Held-out sibling-ordering and calibration metrics for rows `idx`."""
    probs = logits.softmax(-1)
    # Child [loss, draw, win] for the child's mover -> expectation for the parent's mover.
    parent_exp = (probs[:, 0] + 0.5 * probs[:, 1]).tolist()
    by_pos = defaultdict(list)
    for j, i in enumerate(idx):
        by_pos[rows[i]["position"]].append((parent_exp[j], rows[i]["parent_expectation"], rows[i]))
    rhos, good, regret = [], [], []
    for siblings in by_pos.values():
        if len(siblings) < 3:
            continue
        pred = [s[0] for s in siblings]
        ref = [s[1] for s in siblings]
        rho = spearman(pred, ref)
        if rho is not None:
            rhos.append(rho)
        best = max(ref)
        pick = max(range(len(pred)), key=pred.__getitem__)
        good.append(float(ref[pick] >= best - GOOD_MARGIN))
        regret.append(best - ref[pick])
    targets = torch.tensor([rows[i]["target"] for i in idx], dtype=torch.float64)
    ref_exp = [rows[i]["parent_expectation"] for i in idx]
    return dict(positions=len(rhos), mean_spearman=statistics.mean(rhos) if rhos else None, greedy_good=statistics.mean(good),
                greedy_regret=statistics.mean(regret),
                expectation_mae=statistics.mean(abs(a - b) for a, b in zip(parent_exp, ref_exp)),
                wdl_ce=float(soft_ce(logits.double(), targets).mean())), {p: s for p, s in by_pos.items()}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r, i = [0.0] * len(v), 0
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", type=Path, action="append", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.screen)
    groups = sorted({r["group"] for r in rows})
    rng = random.Random(42)
    rng.shuffle(groups)
    fold_of = {g: i % args.folds for i, g in enumerate(groups)}
    print(json.dumps(dict(rows=len(rows), positions=len({r["position"] for r in rows}), groups=len(groups))), flush=True)

    cache = args.output / "features.pt"
    identity = dict(checkpoint=file_hash(args.checkpoint), screens=[file_hash(s / "results.json") for s in args.screen],
                    rows=len(rows))
    if cache.exists():
        feats = torch.load(cache, weights_only=False)
        if feats["identity"] != identity:
            raise ValueError("incompatible feature cache")
    else:
        started = time.monotonic()
        runtime, _ = load_runtime(load_config(args.config), args.checkpoint, "cuda")
        feats = extract(runtime, rows, args.max_tokens)
        # The captured 512 features must reproduce the original output exactly.
        with torch.inference_mode():
            recon = runtime.model.value_head[-1](feats["value512"][:4096].cuda()).float().cpu()
        err = float((recon.softmax(-1) - feats["output_only"][:4096].softmax(-1)).abs().max())
        if err > 2e-4:
            raise AssertionError(f"wrong value features captured ({err})")
        feats["identity"] = identity
        feats["extract_seconds"] = time.monotonic() - started
        torch.save(feats, cache)
        del runtime
        torch.cuda.empty_cache()
    target = torch.tensor([r["target"] for r in rows], dtype=torch.float64)

    held = {name: torch.zeros(len(rows), 3, dtype=torch.float64)
            for name in ("original", "constant", "output_only", "value512", "shared1024", "mlp1024")}
    held["original"] = feats["output_only"].double()
    diagnostics = []
    for fold in range(args.folds):
        test = [i for i, r in enumerate(rows) if fold_of[r["group"]] == fold]
        dev_groups = [g for g in groups if fold_of[g] != fold]
        val_groups = set(dev_groups[: max(1, len(dev_groups) // 5)])
        train = [i for i, r in enumerate(rows) if fold_of[r["group"]] != fold and r["group"] not in val_groups]
        val = [i for i, r in enumerate(rows) if r["group"] in val_groups]
        dev = train + val
        prior = target[dev].mean(0)
        held["constant"][test] = prior.log().expand(len(test), 3)
        for name in ("output_only", "value512", "shared1024"):
            x = feats[name].double()
            w_train = torch.full((len(train),), 1 / len(train), dtype=torch.float64)
            scores = []
            for penalty in PENALTIES:
                fitted = fit_linear(x[train], target[train], w_train, penalty)
                scores.append((float(soft_ce(predict(fitted, x[val]), target[val]).mean()), penalty))
            best_ce, best_pen = min(scores)
            fitted = fit_linear(x[dev], target[dev], torch.full((len(dev),), 1 / len(dev), dtype=torch.float64), best_pen)
            held[name][test] = predict(fitted, x[test])
            diagnostics.append(dict(fold=fold, probe=name, penalty=best_pen, validation_ce=best_ce, **fitted["diagnostics"]))
        x = feats["shared1024"]
        predictor, diag = fit_mlp(x[train], target[train], x[val], target[val], seed=fold)
        with torch.no_grad():
            held["mlp1024"][test] = predictor(x[test])
        diagnostics.append(dict(fold=fold, probe="mlp1024", **diag))
        print(json.dumps(dict(phase="fit", fold=fold, train=len(train), val=len(val), test=len(test))), flush=True)

    everything = list(range(len(rows)))
    summary, per_position = {}, {}
    for name, logits in held.items():
        summary[name], per_position[name] = metrics(rows, everything, logits)
    # Paired, group-bootstrapped differences in sibling ordering versus the original readout.
    pos_group = {r["position"]: r["group"] for r in rows}
    def per_pos_stat(name, stat):
        out = {}
        for pos, sib in per_position[name].items():
            if len(sib) < 3:
                continue
            pred, ref = [s[0] for s in sib], [s[1] for s in sib]
            if stat == "rho":
                v = spearman(pred, ref)
            else:
                v = max(ref) - ref[max(range(len(pred)), key=pred.__getitem__)]
            if v is not None:
                out[pos] = v
        return out
    for name in held:
        if name == "original":
            continue
        for stat in ("rho", "regret"):
            a, b = per_pos_stat(name, stat), per_pos_stat("original", stat)
            by_group = defaultdict(list)
            for pos in set(a) & set(b):
                by_group[pos_group[pos]].append(a[pos] - b[pos])
            keys = list(by_group)
            if not keys:  # e.g. constant predictions have no defined rank correlation
                continue
            brng = random.Random(42)
            means = []
            for _ in range(2000):
                sample = [d for g in brng.choices(keys, k=len(keys)) for d in by_group[g]]
                means.append(statistics.mean(sample))
            means.sort()
            diffs = [d for g in keys for d in by_group[g]]
            summary[name][f"{stat}_minus_original"] = statistics.mean(diffs)
            summary[name][f"{stat}_minus_original_95ci"] = [means[50], means[1949]]
    atomic_json(args.output / "summary.json", dict(checkpoint=str(args.checkpoint), identity=identity,
                                                   folds=args.folds, groups=len(groups), probes=summary))
    atomic_json(args.output / "diagnostics.json", diagnostics)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
