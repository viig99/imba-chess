"""Sample where self-play's main thread spends time; works without ptrace.

Wraps any run_self_play.py (including a run's frozen source/ snapshot) in the
same interpreter: a daemon thread samples the main thread's Python stack at a
fixed rate while another polls nvidia-smi. After --start-after seconds of
warm-up it records for --duration seconds, writes the results, then sends
SIGTERM so self-play drains normally. Use it where py-spy cannot attach (for
example unprivileged containers without CAP_SYS_PTRACE).

Outputs in --out:
  stacks.collapsed  root;...;leaf count lines (speedscope / flamegraph.pl)
  summary.json      top functions by self and inclusive samples, GPU stats
  gpu.csv           per-poll GPU utilization and memory

Native and CUDA work appears under the Python frame that called it. A thread
holding the GIL in C code delays samples, so sampling favours GIL-releasing
points; read results as a breakdown of the Python loop, not exact CPU time.

  python scripts/profile_self_play.py --out DIR --start-after 240 --duration 180 \
      -- RUN/source/scripts/run_self_play.py --config ... --output ... [args]
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading
import time


def frame_label(frame):
    code = frame.f_code
    return f"{Path(code.co_filename).name}:{getattr(code, 'co_qualname', code.co_name)}"


class StackSampler(threading.Thread):
    def __init__(self, target_ident, interval):
        super().__init__(name="stack-sampler", daemon=True)
        self.target_ident, self.interval = target_ident, interval
        self.stacks = Counter()
        self.samples = 0
        self.recording = threading.Event()
        self.done = threading.Event()

    def run(self):
        while not self.done.is_set():
            time.sleep(self.interval)
            if not self.recording.is_set():
                continue
            frame = sys._current_frames().get(self.target_ident)
            labels = []
            while frame is not None:
                labels.append(frame_label(frame))
                frame = frame.f_back
            if labels:
                self.stacks[";".join(reversed(labels))] += 1
                self.samples += 1


class GpuPoller(threading.Thread):
    def __init__(self, interval):
        super().__init__(name="gpu-poller", daemon=True)
        self.interval = interval
        self.rows = []
        self.recording = threading.Event()
        self.done = threading.Event()

    def run(self):
        query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"]
        while not self.done.is_set():
            time.sleep(self.interval)
            if not self.recording.is_set():
                continue
            out = subprocess.run(query, capture_output=True, text=True, check=True).stdout
            util, memory = (int(v) for v in out.splitlines()[0].split(","))
            self.rows.append((round(time.monotonic(), 3), util, memory))


def summarize(stacks, samples, gpu_rows, top):
    own, inclusive = Counter(), Counter()
    for stack, count in stacks.items():
        frames = stack.split(";")
        own[frames[-1]] += count
        for label in set(frames):
            inclusive[label] += count
    share = lambda c: round(c / samples, 4) if samples else 0.0
    utils = [u for _, u, _ in gpu_rows]
    return dict(
        samples=samples,
        top_self=[dict(frame=f, share=share(c)) for f, c in own.most_common(top)],
        top_inclusive=[dict(frame=f, share=share(c)) for f, c in inclusive.most_common(top)],
        gpu=dict(
            polls=len(utils),
            mean_utilization=round(sum(utils) / len(utils), 1) if utils else None,
            idle_fraction=round(sum(u < 5 for u in utils) / len(utils), 3) if utils else None,
            max_memory_mib=max((m for _, _, m in gpu_rows), default=None),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-after", type=float, default=240.0, help="warm-up seconds before recording")
    parser.add_argument("--duration", type=float, default=180.0, help="recording seconds")
    parser.add_argument("--hz", type=float, default=200.0, help="stack samples per second")
    parser.add_argument("--gpu-interval", type=float, default=0.25)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("script", type=Path, help="run_self_play.py to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.script_args[:1] == ["--"]:
        args.script_args = args.script_args[1:]
    args.out.mkdir(parents=True, exist_ok=True)

    sampler = StackSampler(threading.get_ident(), 1.0 / args.hz)
    gpu = GpuPoller(args.gpu_interval)
    sampler.start()
    gpu.start()

    def finish():
        sampler.recording.clear()
        gpu.recording.clear()
        (args.out / "stacks.collapsed").write_text(
            "".join(f"{stack} {count}\n" for stack, count in sampler.stacks.most_common()))
        (args.out / "gpu.csv").write_text(
            "monotonic,utilization,memory_mib\n" + "".join(f"{t},{u},{m}\n" for t, u, m in gpu.rows))
        summary = summarize(sampler.stacks, sampler.samples, gpu.rows, args.top)
        summary.update(finished=datetime.now().astimezone().isoformat(), hz=args.hz,
                       start_after=args.start_after, duration=args.duration,
                       command=[str(args.script), *args.script_args])
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(dict(event="profile_written", out=str(args.out), samples=sampler.samples,
                              gpu=summary["gpu"])), flush=True)

    def controller():
        time.sleep(args.start_after)
        sampler.recording.set()
        gpu.recording.set()
        print(json.dumps(dict(event="profile_recording", duration=args.duration)), flush=True)
        time.sleep(args.duration)
        finish()
        os.kill(os.getpid(), signal.SIGTERM)  # self-play drains and exits normally

    threading.Thread(target=controller, name="profile-controller", daemon=True).start()
    sys.argv = [str(args.script), *args.script_args]
    sys.path.insert(0, str(args.script.resolve().parent))
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
