#!/usr/bin/env bash
# Paired A/B/B/A benchmark for this session's Python hot-path optimizations.
#
#   A = pre-optimization behaviour, restored by monkeypatch:
#         * _cozy_move_id_and_uci -> cozy_move_to_uci + dict lookup
#         * encode_cozy           -> per-call import, cc.* attribute lookups,
#                                    chess.scan_forward generators
#   B = current production code
#
# A/B/B/A because absolute wall time is not comparable across sessions on this
# machine (~±10% drift, and a run can thermally throttle ~50% mid-way). The
# search_gpu bucket is the drift control: this change cannot touch GPU time,
# so it must come out ~1.00x. If it does not, the total number is untrustworthy
# and only the bookkeeping delta should be read.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export BENCH_GAMES="${GAMES:-20}"
export BENCH_BUDGET="${BUDGET:-2048}"
export BENCH_CONCURRENT="${CONCURRENT:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
OUT="${OUT:-/tmp/pyopt_paired}"
mkdir -p "${OUT}"

run() {  # run <A|B> <tag>
  IMBA_BENCH_ARM="$1" BENCH_TAG="$2" .venv/bin/python - > "${OUT}/$2.log" 2>&1 <<'PY'
import os, runpy, sys

if os.environ["IMBA_BENCH_ARM"] == "A":
    import chess
    from imba_chess.data import board_state as bs
    from imba_chess.data.board_state import BoardState, _bucket
    from imba_chess.eval import cozy_bridge, position_evaluator as pe

    def _uncached_move_id(cozy_board, move, move_vocab):
        uci = cozy_bridge.cozy_move_to_uci(cozy_board, move)
        return (move_vocab.token_to_id.get(uci), uci)

    pe._cozy_move_id_and_uci = _uncached_move_id

    def _old_encode_cozy(self, board):
        import cozy_chess as cc

        cfg = self.config
        ids = [0] * 64
        white = int(board.colors(cc.Color.White))
        for offset, piece in (
            (0, cc.Piece.Pawn), (1, cc.Piece.Knight), (2, cc.Piece.Bishop),
            (3, cc.Piece.Rook), (4, cc.Piece.Queen), (5, cc.Piece.King),
        ):
            bb = int(board.pieces(piece))
            for square in chess.scan_forward(bb & white):
                ids[square] = offset + 1
            for square in chess.scan_forward(bb & ~white):
                ids[square] = offset + 7
        rw = board.castle_rights(cc.Color.White)
        rb = board.castle_rights(cc.Color.Black)
        castle_id = (
            (1 if rw.short is not None else 0) | (2 if rw.long is not None else 0)
            | (4 if rb.short is not None else 0) | (8 if rb.long is not None else 0)
        )
        return BoardState(
            piece_ids=ids,
            turn_id=int(board.side_to_move() == cc.Color.Black),
            castle_id=castle_id,
            ep_file_id=self._ep_file_id_cozy(board),
            halfmove_bucket_id=_bucket(board.halfmove_clock, cfg.halfmove_max, cfg.halfmove_bucket_size),
            fullmove_bucket_id=_bucket(board.fullmove_number, cfg.fullmove_max, cfg.fullmove_bucket_size),
        )

    bs.BoardStateEncoder.encode_cozy = _old_encode_cozy

tag = os.environ["BENCH_TAG"]
sys.argv = [
    "generate_search_rollouts.py",
    "--config", "config/imba_chess_exit_seeded_rollout.toml",
    "--checkpoint", "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
    "--output-path", f"/tmp/pyopt_{tag}.parquet",
    "--max-games", os.environ["BENCH_GAMES"],
    "--search-budget", os.environ["BENCH_BUDGET"],
    "--concurrent-games", os.environ["BENCH_CONCURRENT"],
    "--dtype", "float32", "--sample-seed", "42",
    "--profile", "--profile-every-games", os.environ["BENCH_GAMES"],
]
runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
PY
}

# One throwaway run first: the very first process pays CUDA context creation
# and kernel autotune (measured 66.5s vs 49.3s for the identical arm), and
# A/B/B/A cancels linear drift but NOT a one-off warmup spike.
echo "-- warmup (discarded) --"
run B warmup

for spec in A:a1 B:b1 B:b2 A:a2; do
  arm="${spec%%:*}"; tag="${spec##*:}"
  run "${arm}" "${tag}"
  echo "arm=${arm} $(tr '\r' '\n' < "${OUT}/${tag}.log" | grep -oE 'total [0-9.]+s' | tail -1)"
done

echo
.venv/bin/python - "${OUT}" <<'PY'
import re, sys
from pathlib import Path

out = Path(sys.argv[1])


def grab(tag):
    txt = (out / f"{tag}.log").read_text(errors="replace").replace("\r", "\n")
    return (
        float(re.findall(r"total ([0-9.]+)s", txt)[-1]),
        float(re.findall(r"decode_prep[^:]*: ([0-9.]+)s", txt)[-1]),
        float(re.findall(r"search_bookkeeping[^:]*: ([0-9.]+)s", txt)[-1]),
        float(re.findall(r"search_gpu[^:]*: ([0-9.]+)s", txt)[-1]),
        int(re.findall(r"\(([0-9]+) search waves", txt)[-1]),
        int(re.findall(r"([0-9]+) search evals", txt)[-1]),
    )


A, B = [grab("a1"), grab("a2")], [grab("b1"), grab("b2")]
print("=== paired: A = pre-optimization, B = current ===")
for nm, arm in (("A before", A), ("B after ", B)):
    t, d, b, g = (sum(x[i] for x in arm) / 2 for i in (0, 1, 2, 3))
    print(f"  {nm}: total {t:6.2f}s  decode_prep {d:6.2f}s  "
          f"bookkeeping {b:6.2f}s  search_gpu {g:6.2f}s")
ta, tb = (sum(x[0] for x in a) / 2 for a in (A, B))
da, db = (sum(x[1] for x in a) / 2 for a in (A, B))
ba, bb = (sum(x[2] for x in a) / 2 for a in (A, B))
ga, gb = (sum(x[3] for x in a) / 2 for a in (A, B))
print(f"\n  total        {ta/tb:5.3f}x  ({ta - tb:+.2f}s)")
print(f"  decode_prep  {da/db:5.3f}x  ({da - db:+.2f}s)   <- where both changes live")
print(f"  bookkeeping  {ba/bb:5.3f}x  ({ba - bb:+.2f}s)")
print(f"  search_gpu   {ga/gb:5.3f}x  <- drift control, want ~1.00x")
w, e = {x[4] for x in A + B}, {x[5] for x in A + B}
print(f"\n  identical work: waves={w} evals={e}",
      "OK" if len(w) == len(e) == 1 else "*** MISMATCH ***")
PY
