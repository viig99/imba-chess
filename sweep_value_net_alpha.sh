#!/usr/bin/env bash
# Sweep value_net_alpha from pure model value head (0) to pure distilled
# ValueNet (1) with the halving search, then print a summary table.
#
# Usage:
#   ./sweep_value_net_alpha.sh                       # alphas 0 0.25 0.5 1.0 @ SF1800, 1024/d6
#   ALPHAS="0.25 0.75" ./sweep_value_net_alpha.sh    # custom grid
#   ELO=2000 BUDGET=512 DEPTH=4 GAMES=50 ./sweep_value_net_alpha.sh
#   VNET=artifacts/value_net/value_net_last.pt ./sweep_value_net_alpha.sh
#
# Each alpha runs under TAG=vneta<alpha>, so outputs never collide and a
# re-run skips alphas whose output JSON already exists (delete the JSON to
# re-measure). alpha=0 re-measures the pure-model-head baseline with the
# same seed — a like-for-like anchor for the rest of the grid.
set -euo pipefail

ALPHAS="${ALPHAS:-0 0.25 0.5 1.0}"
VNET="${VNET:-artifacts/value_net/value_net_best.pt}"
ELO="${ELO:-1800}"
BUDGET="${BUDGET:-1024}"
DEPTH="${DEPTH:-6}"
OUT_DIR="${OUT_DIR:-artifacts/eval}"

for alpha in ${ALPHAS}; do
  tag="vneta${alpha//./}_${ELO}_${BUDGET}_${DEPTH}"
  echo "== value_net_alpha=${alpha} (TAG=${tag}) =="
  POLICIES="value_search_halving" ELO="${ELO}" TAG="${tag}" OUT_DIR="${OUT_DIR}" \
    ./eval_best_checkpoint.sh \
    --search-budget "${BUDGET}" --search-max-depth "${DEPTH}" \
    --value-net-checkpoint "${VNET}" --value-net-alpha "${alpha}" "$@"
done

echo
echo "== alpha sweep summary (SF${ELO}, budget ${BUDGET}, depth ${DEPTH}) =="
OUT_DIR="${OUT_DIR}" ALPHAS="${ALPHAS}" .venv/bin/python - <<'PY'
import glob, json, os

out_dir = os.environ["OUT_DIR"]
print(f"{'alpha':>6}  {'W':>3} {'D':>3} {'L':>3}  {'score':>6}  file")
for alpha in os.environ["ALPHAS"].split():
    tag = "vneta" + alpha.replace(".", "") + f"_{os.environ['ELO']}_{os.environ['BUDGET']}_{os.environ['DEPTH']}"
    paths = sorted(glob.glob(os.path.join(out_dir, f"*_{tag}.json")))
    if not paths:
        print(f"{alpha:>6}  (no output found for TAG={tag})")
        continue
    agg = json.load(open(paths[-1]))["aggregate"]
    print(
        f"{alpha:>6}  {agg['wins']:>3} {agg['draws']:>3} {agg['losses']:>3}"
        f"  {agg['score_rate']:>6.3f}  {os.path.basename(paths[-1])}"
    )
PY
