"""Sequential GPU job queue for the authorized isolated search-method study.

CPU reference labelling runs separately. Children run one at a time, so learning
and throughput measurements never contend with another study job on the GPU.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.golden import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/eval/search-study-2026-10-18"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--golden-positions", type=Path, required=True)
    p.add_argument("--skip-benchmarks", action="store_true")
    p.add_argument("--skip-learning", action="store_true")
    p.add_argument("--tuning-reference", type=Path,
                   help="run the expanded 500-position tuning checks before other GPU jobs")
    p.add_argument("--evaluation-reference", type=Path,
                   help="complete reference for locked full-set search and raw learning evaluation")
    p.add_argument("--evaluation-arms", type=Path,
                   help="frozen search configurations for the locked full-set comparison")
    p.add_argument("--exploration-seeds", type=int, default=0,
                   help="additional noise seeds for a 500-root finalist target check")
    args = p.parse_args()
    if bool(args.evaluation_reference) != bool(args.evaluation_arms):
        p.error("provide both --evaluation-reference and --evaluation-arms")
    if args.exploration_seeds < 0 or (args.exploration_seeds and not args.evaluation_reference):
        p.error("exploration seeds need frozen evaluation inputs")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.evaluation_reference:
        selection = json.loads((args.root / "frozen-selection.json").read_text())
        if (file_sha256(args.evaluation_arms) != selection["finalists_sha256"] or
                file_sha256(args.checkpoint) != selection["checkpoint_sha256"] or
                any(file_sha256(c["path"]) != c["sha256"] for c in selection["configs"].values())):
            raise ValueError("frozen study configuration changed")
    jobs = []
    configs = dict(gumbel="config/self_play.toml", puct="config/self_play_puct.toml",
                   puct_variance="config/self_play_puct_variance.toml", puct_rpo="config/self_play_puct_rpo.toml",
                   puct_variance_rpo="config/self_play_puct_variance_rpo.toml")
    if args.tuning_reference:
        for name, arms in (("grid", "config/search_study_tune.toml"),
                           ("targets", "config/search_study_targets.toml")):
            out = args.root / f"tune-500-{name}-s256"
            # The comparer itself checks every arm's identity and resumes any
            # partial output. A summary may exist before all arms have finished.
            jobs.append((f"tune-500-{name}", None, ["scripts/compare_search_methods.py",
                "--positions", str(args.golden_positions), "--reference", str(args.tuning_reference),
                "--checkpoint", str(args.checkpoint), "--arms", arms, "--out", str(out),
                "--simulations", "256", "--limit", "500", "--concurrent-games", "32",
                "--allow-partial-reference"]))
    if args.evaluation_reference:
        if args.exploration_seeds:
            jobs.append(("finalist-noise-tune", None, ["scripts/compare_search_methods.py",
                "--positions", str(args.golden_positions), "--reference", str(args.evaluation_reference),
                "--checkpoint", str(args.checkpoint), "--arms", str(args.evaluation_arms),
                "--out", str(args.root / "finalist-noise-tune-s256"), "--split", "tune",
                "--simulations", "256", "--limit", "500", "--concurrent-games", "32",
                "--seeds", str(args.exploration_seeds)]))
        jobs.append(("locked-search-all", None, ["scripts/compare_search_methods.py",
            "--positions", str(args.golden_positions), "--reference", str(args.evaluation_reference),
            "--checkpoint", str(args.checkpoint), "--arms", str(args.evaluation_arms),
            "--out", str(args.root / "locked-search-all-s256"), "--split", "all",
            "--simulations", "256", "--concurrent-games", "32"]))
    if not args.skip_benchmarks:
        for name, config in {**configs, "puct_v": "config/self_play_puct_v.toml"}.items():
            out = args.root / "throughput" / name
            jobs.append((f"throughput-{name}", out / "result.json", ["scripts/bench_collection.py",
                "--config", config, "--checkpoint", str(args.checkpoint),
                "--seeds", "artifacts/corpus/v4_self_play_seeds_4096.json", "--output", str(out),
                "--workers", "5", "--concurrent-games", "160", "--warmup-seconds", "30",
                "--fresh-positions", "20000", "--max-seconds", "600", "--phases", "2", "--carry-over"]))
    if not args.skip_learning:
        for name, config in configs.items():
            out = args.root / "learning" / name
            jobs.append((f"learning-{name}", out / "summary.json", ["scripts/train_search_study.py",
                "--config", config, "--initialize", str(args.checkpoint),
                "--golden-positions", str(args.golden_positions), "--out", str(out),
                "--phases", "3", "--games-per-phase", "320", "--exposures-per-phase", "50000",
                "--workers", "5", "--concurrent-games", "160", "--max-seconds", "5400"]))
        if args.evaluation_reference:
            jobs.append(("raw-learning-test", None, ["scripts/evaluate_search_learning.py",
                "--positions", str(args.golden_positions), "--reference", str(args.evaluation_reference),
                "--initial", str(args.checkpoint), "--learning", str(args.root / "learning"),
                "--out", str(args.root / "raw-learning-test"), "--split", "test"]))
    progress_path = args.root / "gpu-queue.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for name, result, command in jobs:
        if result is not None and result.exists():
            saved = json.loads(result.read_text())
            if name.startswith("throughput-") or saved.get("complete"):
                # A previous queue parent may have stopped after the child
                # published its result but before recording job completion.
                progress.setdefault(name, dict(command=command))
                progress[name].update(status="complete", returncode=0)
                atomic_json(progress_path, progress)
                continue
        # Benchmark output directories are exclusive; an interrupted benchmark
        # needs a fresh directory rather than overwriting partial evidence.
        if name.startswith("throughput-") and result.parent.exists():
            raise FileExistsError(f"incomplete benchmark at {result.parent}; inspect before rerunning")
        started = time.time()
        progress[name] = dict(status="running", started_unix=started, command=command)
        atomic_json(progress_path, progress)
        print(json.dumps(dict(event="job_start", job=name)), flush=True)
        with (args.root / f"{name}.log").open("a") as log:
            child = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
        progress[name].update(status="complete" if child.returncode == 0 else "failed",
                              returncode=child.returncode, seconds=time.time() - started)
        atomic_json(progress_path, progress)
        print(json.dumps(dict(event="job_finish", job=name, **progress[name])), flush=True)
        if child.returncode:
            raise SystemExit(f"{name} failed; see {args.root / (name + '.log')}")
        if name.startswith("learning-") and not json.loads(result.read_text()).get("complete"):
            progress[name]["status"] = "partial"
            atomic_json(progress_path, progress)
            raise SystemExit(f"{name} reached its time limit; resume before evaluating final learning")


if __name__ == "__main__":
    main()
