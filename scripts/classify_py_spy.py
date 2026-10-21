"""Attribute py-spy raw (collapsed) samples of collection workers to stages.

Each sample is assigned to the innermost rule that matches, scanning its
frames from the leaf upwards. Prints stage shares of all worker samples and
the hottest innermost Python frames for unclassified time.
"""

import argparse
from collections import Counter
from pathlib import Path
import re

# (category, predicate over the frame list), innermost match wins; checked in
# order for each frame from the leaf upwards.
FRAME = re.compile(r"^(?P<func>.*?) \((?P<file>[^:()]+)(?::(?P<line>\d+))?\)$")
# Inductor's generated output-code modules are named by a content hash.
GENERATED = re.compile(r"^(?:.*/)?[a-z0-9]{40,}\.py$")


def parse(line):
    stack, _, count = line.rpartition(" ")
    frames = []
    for raw in stack.split(";"):
        match = FRAME.match(raw)
        frames.append(
            (match["func"], match["file"], match["line"]) if match else (raw, "", None)
        )
    return frames, int(count)


def native_has(frames, *needles):
    return any(any(n in f[0] for n in needles) for f in frames)


def classify(frames):
    files = [f[1] for f in frames]
    funcs = [f[0] for f in frames]

    def inside(func, file_part):
        return any(fn == func and file_part in fl for fn, fl, _ in frames)

    def below(func, file_part):
        for i, (fn, fl, _) in enumerate(frames):
            if fn == func and file_part in fl:
                return frames[i + 1 :]
        return []

    if inside("consume", "decode_workspace.py"):
        sub = below("consume", "decode_workspace.py")
        if native_has(sub, "THPVariable_cpu", "_local_scalar_dense", "cudaMemcpy", "Synchronize"):
            return "decode_consume: blocking readback (GPU wait)"
        if native_has(sub, "index_copy", "THPVariable_stack", "stack"):
            return "decode_consume: KV scatter launch"
        if any("PositionEval" in f or "<listcomp>" in f for f in funcs):
            return "decode_consume: PositionEval/list building"
        return "decode_consume: CPU normalize/tolist/other"
    if inside("prepare", "decode_workspace.py"):
        for func, label in (
            ("encode_cozy", "decode_prep: board encoding"),
            ("_stage_request", "decode_prep: board encoding"),
            ("project_legal_moves", "decode_prep: legal projection"),
            ("_gather_ancestors", "decode_prep: ancestor gather (16 gathers)"),
            ("_prepare_attention", "decode_prep: attention index/mask ops"),
            ("_refresh_direct", "decode_prep: prefix refresh"),
            ("_prefix_views", "decode_prep: prefix views"),
            ("_slot", "decode_prep: slot bookkeeping"),
        ):
            if func in funcs:
                return label
        if native_has(below("prepare", "decode_workspace.py"), "copy_", "foreach"):
            return "decode_prep: H2D + feature copies"
        return "decode_prep: staging loop/other"
    if inside("placement_order", "decode_workspace.py"):
        return "decode_prep: placement order"
    if inside("executor", "merged_executors.py") and any(
        f[0] == "decode" or "torchinductor" in f[1] or "_dynamo" in f[1] or "_inductor" in f[1]
        for f in frames
    ):
        sub = frames
        generated = [f for f in sub if GENERATED.match(f[1])]
        if any("triton_heuristics" in f[1] or "triton/" in f[1] for f in sub):
            return "decode_launch: triton kernel launcher"
        if generated:
            if native_has(sub, "empty_strided", "empty", "alloc"):
                return "decode_launch: inductor wrapper allocations"
            if native_has(sub, "mm", "gemm", "cublas", "addmm", "bmm"):
                return "decode_launch: extern GEMM launches"
            return "decode_launch: inductor wrapper (python)"
        if native_has(sub, "guard", "Guard"):
            return "decode_launch: dynamo guard evaluation"
        if any("_dynamo" in f[1] for f in sub):
            return "decode_launch: dynamo frame/dispatch"
        if any("_inductor" in f[1] for f in sub):
            return "decode_launch: inductor runtime (python)"
        return "decode_launch: other"
    if any(f[0] == "_forward_model" for f in frames) or inside("root_executor", "merged_executors.py") or any(
        "root" in f[0] and "merged_executors" in f[1] for f in frames
    ):
        return "root_eval executor"
    if inside("gumbel_stepwise", "gumbel_search.py"):
        sub = below("gumbel_stepwise", "gumbel_search.py")
        names = " ".join(f[0] for f in sub)
        # The tree and its simulations are native (gumbel_tree.rs); Rust v0
        # symbols are mangled, so match distinctive substrings, innermost first.
        for needle, label in (
            ("drop_glue", "search: native tree teardown"),
            ("8interior", "search: native interior selection"),
            ("NodeStats4root", "search: native root selection"),
            ("push_classify", "search: native push/classify (child board+terminal)"),
            ("backup_edge", "search: native backup"),
            ("attach", "search: native leaf init (attach)"),
            ("next_leaf", "search: native tree walk (next_leaf)"),
            ("extend", "search: extend handle"),
            ("completed_q", "search: final completed_q/softmax"),
            ("softmax", "search: final completed_q/softmax"),
            ("finish", "search: final snapshot"),
        ):
            if needle in names:
                return label
        return "search: python driver (self)"
    if inside("search_batch", "inference_runtime.py"):
        sub = below("search_batch", "inference_runtime.py")
        if native_has(sub, "drop_glue"):
            return "search: native tree teardown"
        if any("_project_legal_logits" in f[0] for f in sub):
            return "root setup: python-chess legal projection"
        if any("py_move_to_cozy" in f[0] or "gives_check" in f[0] or "is_capture" in f[0] for f in sub):
            return "root setup: root PositionEval (python-chess)"
        return "root setup: other"
    if inside("search", "inference_runtime.py"):
        return "root setup: history batch build"
    if inside("play_game", "collector.py"):
        sub = below("play_game", "collector.py")
        if any("asdict" in f[0] for f in sub):
            return "game: asdict(result) targets"
        if any("terminal_outcome" in f[0] or "terminal_value" in f[0] for f in sub):
            return "game: terminal check (python-chess)"
        if any(f[0] in ("on_position", "send") for f in sub):
            return "game: IPC position report"
        return "game: other bookkeeping"
    if inside("factory", "workers.py"):
        return "IPC: launch request/recv"
    if any("send" == f[0] for f in frames):
        return "IPC: send"
    if inside("run", "inference_runtime.py"):
        return "executor wrapper"
    if inside("_run", "batch_scheduler.py") or inside("_advance", "batch_scheduler.py"):
        return "scheduler/generator plumbing"
    if any("gc" in f[0] and "collect" in f[0] for f in frames):
        return "gc"
    return "other"


def innermost_python(frames):
    for func, file, line in reversed(frames):
        if file.endswith(".py"):
            return f"{func} ({file}:{line})"
    return frames[-1][0] if frames else "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--detail", help="print innermost frames for this category prefix")
    args = parser.parse_args()
    totals, inner, detail = Counter(), Counter(), Counter()
    for path in args.files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            frames, count = parse(line)
            # Only the worker main thread runs the collection loop.
            if not any(f[0] == "_worker_main" for f in frames):
                continue
            category = classify(frames)
            totals[category] += count
            inner[innermost_python(frames)] += count
            if args.detail and category.startswith(args.detail):
                leaf = [f[0] for f in frames if not f[1].endswith(".py")][-3:]
                detail[innermost_python(frames) + " | " + " < ".join(reversed(leaf))[:160]] += count
    total = sum(totals.values())
    print(f"samples: {total}")
    groups = Counter()
    for category, count in totals.items():
        groups[category.split(":")[0]] += count
    print("\n== stage groups ==")
    for group, count in groups.most_common():
        print(f"{100 * count / total:6.1f}%  {group}")
    print("\n== categories ==")
    for category, count in totals.most_common():
        print(f"{100 * count / total:6.1f}%  {category}")
    print(f"\n== top {args.top} innermost python frames ==")
    for frame, count in inner.most_common(args.top):
        print(f"{100 * count / total:6.1f}%  {frame}")
    if args.detail:
        print("\n== detail ==")
        for frame, count in detail.most_common(args.top):
            print(f"{100 * count / total:6.1f}%  {frame}")


if __name__ == "__main__":
    main()
