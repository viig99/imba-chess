"""Run a recoverable checkpoint-vs-Stockfish match in fixed-seed batches.

Plays --games in batches of --batch-games through scripts/eval_vs_stockfish.py
with seed = --seed-base + batch index, so arms that share seeds see the same
openings. Finished batches are kept (re-running resumes); progress.json and,
at the end, results.json aggregate them. Any failed batch stops the run with
its exit code. Search options after "--" are passed through unchanged, e.g.

  python scripts/run_stockfish_eval.py --checkpoint actor.pt \
      --config config/imba_chess_v4_aux3.toml --out artifacts/eval/NAME -- \
      --gumbel-simulations 512 --gumbel-root-forcing --gumbel-forcing-floor \
      --gumbel-minimax-weight 0.5

Defaults are the SF2600 protocol used for the self-play evals: UCI_Elo 2600,
40k nodes (5 s cap), 1 thread, 64 MiB hash, Gumbel with raw Q at value_scale 0.5,
8 concurrent games, no random opening plies.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from imba_chess.eval.inference_runtime import RUNTIME_REVISION

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def eval_arguments(args, batch, out, extra):
    """eval_vs_stockfish.py arguments for one batch (the SF protocol plus search options)."""
    return ["--config", str(args.config), "--checkpoint", str(args.checkpoint),
            "--games", str(args.batch_games), "--ladder-elos", str(args.elo),
            "--ladder-games-per-segment", str(args.batch_games), "--no-include-full-strength-segment",
            "--stockfish-limit-strength", "--stockfish-elo", str(args.elo),
            "--stockfish-path", args.stockfish, "--stockfish-time-sec", "5",
            "--stockfish-nodes", str(args.nodes), "--stockfish-threads", "1", "--stockfish-hash-mb", "64",
            "--device", "cuda", "--model-move-policy", "gumbel", "--gumbel-value-scale", "0.5",
            "--concurrent-games", str(args.concurrent_games), "--max-plies", "512",
            "--opening-random-plies", "0", "--seed", str(args.seed_base + batch),
            "--save-games", "--save-games-dir", str(out / "games"),
            "--output-json", str(out / "results.json"), *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="model/repo config for the checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--batch-games", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1042)
    parser.add_argument("--elo", type=int, default=2600)
    parser.add_argument("--stockfish", default="/usr/bin/stockfish")
    parser.add_argument("--nodes", type=int, default=40000)
    parser.add_argument("--concurrent-games", type=int, default=8)
    parser.add_argument("search_args", nargs=argparse.REMAINDER, help='eval_vs_stockfish search options after "--"')
    args = parser.parse_args()
    extra = args.search_args[1:] if args.search_args[:1] == ["--"] else args.search_args
    if args.games % args.batch_games:
        parser.error("--games must be a multiple of --batch-games")
    if not args.checkpoint.is_file() or not args.config.is_file():
        parser.error("checkpoint and config must exist")
    batches = args.games // args.batch_games
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint=str(args.checkpoint.resolve()), config=str(args.config.resolve()),
                    games=args.games, batch_games=args.batch_games, seed_base=args.seed_base,
                    stockfish_elo=args.elo, stockfish_nodes=args.nodes,
                    concurrent_games=args.concurrent_games, search_args=extra,
                    runtime_revision=RUNTIME_REVISION,
                    git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip())
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        changed = {k for k in manifest if k != "git_head" and saved.get(k) != manifest[k]}
        if changed:
            raise SystemExit(f"refusing to resume with different settings: {sorted(changed)}")
    else:
        atomic_json(manifest_path, manifest)

    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def report():
        rows = []
        for i in range(batches):
            path = args.out / f"batch-{i:02d}/results.json"
            if path.exists():
                data = json.loads(path.read_text())
                aggregate = data["aggregate"]
                payloads = [aggregate, *(s.get("results", {}) for s in data["segments"])]
                if any(p.get("run_config", {}).get("runtime_revision") != RUNTIME_REVISION
                       for p in payloads):
                    raise SystemExit(f"refusing to reuse batch from a different runtime: {path}")
                assert data["segments"][0]["stockfish"]["elo"] == args.elo
                assert aggregate["games"] == args.batch_games and aggregate["incomplete_games"] == 0
                rows.append(aggregate)
        totals = {k: sum(r[k] for r in rows) for k in ("games", "wins", "draws", "losses")}
        totals.update(batches=len(rows), updated=datetime.now().astimezone().isoformat(),
                      score=(totals["wins"] + 0.5 * totals["draws"]) / totals["games"] if totals["games"] else None)
        atomic_json(args.out / "progress.json", totals)
        return totals

    report()
    for i in range(batches):
        if stopping:
            break
        out = args.out / f"batch-{i:02d}"
        if (out / "results.json").exists():
            continue
        out.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(ROOT / "scripts/eval_vs_stockfish.py"),
                   *eval_arguments(args, i, out, extra)]
        print(json.dumps(dict(event="batch_start", batch=i, command=command,
                              time=datetime.now().astimezone().isoformat())), flush=True)
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, cwd=ROOT)
        code = child.wait()
        if stopping:
            break
        if code:
            sys.exit(code)
        print(json.dumps(dict(event="batch_completed", batch=i, **report())), flush=True)
    if not stopping:
        final = report()
        assert final["games"] == args.games
        atomic_json(args.out / "results.json", dict(manifest=manifest, aggregate=final))
        print(json.dumps(dict(event="evaluation_completed", **final)), flush=True)


if __name__ == "__main__":
    main()
