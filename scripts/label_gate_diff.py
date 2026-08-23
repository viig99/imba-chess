"""Exact label gate: diff a candidate rollout parquet against a baseline.

The §7 gate in docs/GENERATION_PERF_HANDOFF.md, as a script so every change
in the decode-wave campaign is judged the same way. Stages that claim to be
bit-identical must print `EXACT` -- anything else means the change moved a
label and needs explaining before it is accepted.

`checkpoint` is excluded from the comparison: it records the path the run was
given, and an absolute-vs-relative spelling of the same file is not a label
difference (see §13c).
"""

import sys

import numpy as np
import pandas as pd

KEY = ["game_id", "ply"]
IGNORED = {"checkpoint"}


def main(baseline_path: str, candidate_path: str) -> int:
    a = pd.read_parquet(baseline_path).sort_values(KEY).reset_index(drop=True)
    b = pd.read_parquet(candidate_path).sort_values(KEY).reset_index(drop=True)
    print(f"rows: baseline={len(a)} candidate={len(b)}")
    if len(a) != len(b) or not a[KEY].equals(b[KEY]):
        print("FAIL: (game_id, ply) index differs -- the games diverged structurally")
        return 1

    columns = [c for c in a.columns if c not in IGNORED]
    if list(a.columns) != list(b.columns):
        print(f"FAIL: column sets differ\n  {list(a.columns)}\n  {list(b.columns)}")
        return 1

    differing = []
    for col in columns:
        x, y = a[col], b[col]
        if x.dtype == object:
            # Object columns here are either strings (uci) or numpy arrays
            # (arm_evals_spent, arm_log_prior, ...). Compare arrays
            # element-for-element -- that is the check that proves the search
            # trajectory is unchanged node-for-node, which is much stronger
            # than the final labels happening to match.
            same = all(
                np.array_equal(np.asarray(p), np.asarray(q))
                if isinstance(p, (list, np.ndarray))
                else p == q
                for p, q in zip(x, y)
            )
        else:
            # `.equals` rather than `==`: two NaNs in the same slot are the
            # same label (human_move_backed_value is NaN where the human
            # move was not searched), but `NaN == NaN` is False and would
            # report a phantom difference with max|d| of exactly 0.
            same = x.equals(y)
        if not same:
            differing.append(col)

    if not differing:
        print(f"EXACT: all {len(columns)} compared columns bit-identical "
              f"(ignored: {sorted(IGNORED)})")
        return 0

    print(f"DIFFER: {differing}")
    for col in differing:
        if col == "best_arm_move_uci":
            print(f"  move agreement: {(a[col] == b[col]).mean():.4%}")
        elif a[col].dtype != object:
            d = (a[col].astype(float) - b[col].astype(float)).abs()
            print(f"  |d {col}| p99={np.percentile(d, 99):.3e} max={d.max():.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
