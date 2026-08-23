#!/usr/bin/env bash
# Phase B4: the verdict. 750 games/arm vs SF2200, same protocol as the anchor.
#
# Matched to artifacts/phase1b_paired/eval_ckpt23_750.json (score 0.6107,
# SE 0.0147) so all three numbers sit on one ruler: seed 42, budget 2048,
# depth 8, top_m 16, lam 0.05, opening_random_plies 0, SF UCI_Elo 2200 at
# 40,000 nodes, concurrent_games 6.
#
# Same seed for both arms => same opening sequence and the same 375/375 colour
# split, so this is paired. Colour matters more than the effect being measured:
# ckpt23 scored 0.6627 as White vs 0.5587 as Black (~76 Elo) over 750 games.
#
# 750 games/arm gives SE ~0.0147 on score rate (~11 Elo), so a two-arm
# difference resolves to roughly +/-30 Elo at 95%.
set -euo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
OUT_DIR="${REPO_DIR}/artifacts/value_distill"
GAMES="${GAMES:-750}"

cd "${REPO_DIR}"
exec >> "${OUT_DIR}/eval.log" 2>&1
echo "=============================================================="
echo "$(date): B4 paired eval starting (games=${GAMES}/arm)"

source .venv/bin/activate

for arm in control distill; do
    ck=$(ls -t "${OUT_DIR}/ckpt_${arm}"/last_checkpoint_*.pt 2>/dev/null | head -1)
    if [[ -z "${ck}" ]]; then echo "FATAL: no checkpoint for ${arm}"; exit 1; fi
    echo "--------------------------------------------------------------"
    echo "$(date): EVAL arm=${arm} ckpt=$(basename "${ck}")"
    python scripts/eval_vs_stockfish.py \
        --config "${OUT_DIR}/config_${arm}.toml" \
        --checkpoint "${ck}" \
        --games "${GAMES}" \
        --ladder-games-per-segment "${GAMES}" \
        --output-json "${OUT_DIR}/eval_${arm}.json"
    echo "$(date): EVAL arm=${arm} done"
done

echo "=============================================================="
echo "$(date): B4 finished"
python - <<'PYEOF'
import json, math
from pathlib import Path
out = Path("artifacts/value_distill")
rows = []
for arm in ("control", "distill"):
    d = json.load(open(out / f"eval_{arm}.json"))
    a = d["aggregate"]
    n = a["completed_games"]; s = a["score_rate"]
    rows.append((arm, n, s, math.sqrt(max(s * (1 - s), 1e-9) / n), a["wins"], a["draws"], a["losses"]))
print(f"{'arm':<10}{'games':>7}{'score':>9}{'SE':>9}   W/D/L")
for arm, n, s, se, w, dr, l in rows:
    print(f"{arm:<10}{n:>7}{s:>9.4f}{se:>9.4f}   {w}/{dr}/{l}")
if len(rows) == 2:
    (_, n0, s0, se0, *_), (_, n1, s1, se1, *_) = rows
    d = s1 - s0; sed = math.sqrt(se0**2 + se1**2)
    elo = lambda x: -400 * math.log10(1 / x - 1)
    print(f"\ndistill - control = {d:+.4f}  (SE {sed:.4f},  z = {d/sed:+.2f})")
    print(f"Elo: control {elo(s0):+.1f}   distill {elo(s1):+.1f}   delta {elo(s1)-elo(s0):+.1f}")
    print(f"ckpt23 anchor 0.6107 -> {elo(0.6107):+.1f} Elo (750 games, SE 0.0147)")
PYEOF
