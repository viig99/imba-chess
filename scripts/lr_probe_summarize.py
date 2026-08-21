"""Summarize scripts/lr_probe.sh runs from their TensorBoard event files.

Reports, per learning rate, whether a short continuation from a converged
checkpoint IMPROVES the model or merely perturbs it:

  * train policy_loss, first vs last window (averaged over several logged
    points, since single points are noisy)
  * held-out fast-val hr@10 (top10_acc) across the run

A run is "improving" if train policy_loss falls and held-out hr@10 does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

WINDOW = 5


def _load(tb_dir: Path):
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _mean(values):
    return sum(values) / len(values)


def _summarize_run(ckpt_dir: Path) -> dict | None:
    tb_dir = ckpt_dir / "tb"
    if not tb_dir.is_dir():
        return None
    ea = _load(tb_dir)
    tags = ea.Tags()["scalars"]
    if "train/policy_loss" not in tags:
        return None

    pl = [e.value for e in ea.Scalars("train/policy_loss")]
    if len(pl) < 2 * WINDOW:
        return None
    first, last = _mean(pl[:WINDOW]), _mean(pl[-WINDOW:])

    # Only present when the KL term actually fired (train.py logs it
    # conditionally on has_policy_kl_loss), so absence means "KL never
    # applied", not "metric missing".
    kl_first = kl_last = kl_n = None
    if "train/policy_kl_loss" in tags:
        kl = [e.value for e in ea.Scalars("train/policy_kl_loss")]
        kl_n = len(kl)
        if len(kl) >= 2 * WINDOW:
            kl_first, kl_last = _mean(kl[:WINDOW]), _mean(kl[-WINDOW:])

    hr10 = None
    if "val_fast/top10_acc" in tags:
        hr10 = [(e.step, e.value) for e in ea.Scalars("val_fast/top10_acc")]

    lr = None
    if "train/lr" in tags:
        lr = ea.Scalars("train/lr")[-1].value

    return {
        "policy_loss_first": first,
        "policy_loss_last": last,
        "delta": last - first,
        "hr10": hr10,
        "lr": lr,
        "kl_first": kl_first,
        "kl_last": kl_last,
        "kl_n": kl_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted(args.probe_dir.glob("ckpt_lr*"))
    if not run_dirs:
        raise SystemExit(f"no ckpt_lr* directories under {args.probe_dir}")

    rows = []
    for d in run_dirs:
        summary = _summarize_run(d)
        if summary is None:
            print(f"WARNING: no usable TB scalars for {d.name}")
            continue
        rows.append((d.name.replace("ckpt_", ""), summary))

    print()
    print(f"{'run':<12} {'lr':>10} {'ploss_i':>9} {'ploss_f':>9} {'delta':>9}  verdict")
    print("-" * 66)
    for name, s in rows:
        lr_txt = f"{s['lr']:.2e}" if s["lr"] is not None else "?"
        verdict = "IMPROVING" if s["delta"] < 0 else "degrading"
        print(
            f"{name:<12} {lr_txt:>10} {s['policy_loss_first']:>9.4f} "
            f"{s['policy_loss_last']:>9.4f} {s['delta']:>+9.4f}  {verdict}"
        )

    if any(s["kl_n"] for _, s in rows):
        print()
        print("policy_kl_loss (only logged on batches where the KL term fired):")
        print(f"  {'run':<12} {'batches':>8} {'kl_i':>9} {'kl_f':>9} {'delta':>9}")
        for name, s in rows:
            if not s["kl_n"]:
                print(f"  {name:<12} {'0':>8}   (KL never fired)")
                continue
            if s["kl_first"] is None:
                print(f"  {name:<12} {s['kl_n']:>8}   (too few points to average)")
                continue
            print(
                f"  {name:<12} {s['kl_n']:>8} {s['kl_first']:>9.4f} "
                f"{s['kl_last']:>9.4f} {s['kl_last'] - s['kl_first']:>+9.4f}"
            )

    print()
    print("held-out fast-val hr@10 (top10_acc) by step:")
    for name, s in rows:
        if not s["hr10"]:
            print(f"  {name:<12} (none logged)")
            continue
        pts = ", ".join(f"{step}:{val:.4f}" for step, val in s["hr10"])
        print(f"  {name:<12} {pts}")

    print()
    print(
        "Read: a negative delta with flat/rising hr@10 means the lr is low enough\n"
        "that continuation actually trains rather than random-walks the checkpoint.\n"
        "Pick the largest such lr for the Phase 1b distillation runs."
    )


if __name__ == "__main__":
    main()
