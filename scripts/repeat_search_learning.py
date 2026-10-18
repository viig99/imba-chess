"""Repeat the fixed search-learning screen with another paired training seed.

Only run.seed changes. Search settings, initial model/optimizer, data exclusions,
and budgets stay fixed. Evaluate available milestones after every completed arm
so the Gumbel versus variance-PUCT-plus-RPO comparison appears first.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.golden import file_sha256
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import run_lock
from compare_search_methods import source_digest


ORDER = ("gumbel", "puct_variance_rpo", "puct_variance", "puct_rpo", "puct")


def prepare_configs(study, output, seed):
    selection = json.loads((study / "frozen-selection.json").read_text())
    configs = {}
    for name in ORDER:
        if name not in selection["learning_arms"]:
            continue
        original = Path(selection["configs"][name]["path"])
        if file_sha256(original) != selection["configs"][name]["sha256"]:
            raise ValueError(f"frozen configuration changed: {original}")
        before = load_config(original)
        text = original.read_text()
        text, count = re.subn(r"(?m)^(seed\s*=\s*)\d+\s*$", rf"\g<1>{seed}", text)
        if count != 1:
            raise ValueError(f"expected one run seed in {original}")
        path = output / "configs" / f"{name}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != text:
            raise ValueError(f"repeat configuration changed: {path}")
        path.write_text(text)
        after = load_config(path)
        expected = asdict(before)
        expected["run"]["seed"] = seed
        if asdict(after) != expected:
            raise ValueError("only the training seed may change")
        configs[name] = path
    return selection, configs


def combine_reports(study, output):
    manifests = [json.loads((p / "raw-learning-test/initial/manifest.json").read_text())
                 for p in (study, output)]
    if manifests[0] != manifests[1]:
        raise ValueError("the two seeds must use identical held-out evaluation inputs")
    first = json.loads((study / "raw-learning-test/summary.json").read_text())
    second = json.loads((output / "raw-learning-test/summary.json").read_text())
    metrics = ("target_regret", "move_regret", "root_wdl_mae", "root_wdl_mse")
    report = dict(interpretation="Two training seeds are a consistency check, not a reliable estimate of training-seed confidence intervals.", arms={})
    for name in ORDER:
        a, b = first["arms"][name][-1], second["arms"][name][-1]
        entries = [a["metrics"]["all"], b["metrics"]["all"]]
        report["arms"][name] = dict(
            per_seed=[{k: e[k] for k in metrics} for e in entries],
            mean={k: sum(e[k] for e in entries) / 2 for k in metrics})
        if name != "gumbel":
            report["arms"][name]["policy_regret_difference_vs_gumbel"] = [
                r["final_vs_gumbel"][name]["target_regret"]["candidate_minus_baseline"]
                for r in (first, second)]
    atomic_json(output / "two-seed-summary.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=Path("artifacts/eval/search-study-2026-10-18"))
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if not 0 <= args.seed < 2**32:
        p.error("seed must be a nonnegative 32-bit integer")
    output = args.out or args.study / f"seed-{args.seed}"
    golden = Path("artifacts/eval/golden-v1-2026-10-18/positions.json")
    reference = golden.parent / "reference-200k"
    first_queue = json.loads((args.study / "gpu-queue.json").read_text())
    if first_queue.get("raw-learning-test", {}).get("status") != "complete":
        raise ValueError("finish the first seed before starting its repeat")
    idle_queue = args.study / "idle-benchmark-queue.json"
    if idle_queue.exists() and json.loads(idle_queue.read_text()).get("status") != "complete":
        raise ValueError("wait for the matched throughput queue to release the GPU")
    with run_lock(output):
        selection, configs = prepare_configs(args.study, output, args.seed)
        checkpoint = Path(first_queue["learning-gumbel"]["command"][
            first_queue["learning-gumbel"]["command"].index("--initialize") + 1])
        if file_sha256(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError("initial checkpoint changed")
        original = load_config(selection["configs"]["gumbel"]["path"])
        if args.seed == original.run.seed:
            raise ValueError("repeat needs a different seed")
        identity = dict(seed=args.seed, first_seed=original.run.seed,
            checkpoint_sha256=selection["checkpoint_sha256"],
            frozen_selection_sha256=file_sha256(args.study / "frozen-selection.json"),
            training_script_sha256=file_sha256("scripts/train_search_study.py"),
            evaluator_sha256=file_sha256("scripts/evaluate_search_learning.py"),
            search_code_sha256=source_digest(),
            configs={name: file_sha256(path) for name, path in configs.items()},
            phases=3, games_per_phase=320, exposures_per_phase=50000, workers=5,
            concurrent_games=160, order=list(configs),
            note="Only run.seed differs from the first seed; methods within this seed share source cohorts and launch identities.")
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("repeat inputs changed")
        atomic_json(manifest, identity)
        progress_path = output / "gpu-queue.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}

        def run(name, command):
            started = time.time()
            progress[name] = dict(status="running", command=command, started_unix=started)
            atomic_json(progress_path, progress)
            print(json.dumps(dict(event="job_start", job=name, seed=args.seed)), flush=True)
            with (output / f"{name}.log").open("a") as log:
                result = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
            progress[name].update(status="complete" if result.returncode == 0 else "failed",
                                  returncode=result.returncode, seconds=time.time() - started)
            atomic_json(progress_path, progress)
            if result.returncode:
                raise SystemExit(f"{name} failed; inspect its log")
            print(json.dumps(dict(event="job_finish", job=name, seed=args.seed,
                                  seconds=time.time() - started)), flush=True)

        for name, config in configs.items():
            arm = output / "learning" / name
            summary = arm / "summary.json"
            if not summary.exists() or not json.loads(summary.read_text()).get("complete"):
                run(f"learning-{name}", ["scripts/train_search_study.py", "--config", str(config),
                    "--initialize", str(checkpoint), "--golden-positions", str(golden), "--out", str(arm),
                    "--phases", "3", "--games-per-phase", "320", "--exposures-per-phase", "50000",
                    "--workers", "5", "--concurrent-games", "160", "--max-seconds", "5400"])
                if not json.loads(summary.read_text()).get("complete"):
                    progress[f"learning-{name}"]["status"] = "partial"
                    atomic_json(progress_path, progress)
                    raise SystemExit("resume the partial learning arm before continuing")
            run(f"held-out-after-{name}", ["scripts/evaluate_search_learning.py", "--positions", str(golden),
                "--reference", str(reference), "--initial", str(checkpoint), "--learning", str(output / "learning"),
                "--out", str(output / "raw-learning-test"), "--split", "test"])
        combine_reports(args.study, output)
        print(json.dumps(dict(event="repeat_complete", seed=args.seed, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
