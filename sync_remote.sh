#!/usr/bin/env bash
# Sync code to / artifacts from the rented GPU box.
#
# Usage:
#   ./sync_remote.sh push    # working tree -> remote, respecting .gitignore
#                            # (no .venv, no artifacts, no logs, no caches);
#                            # preserves remote ignored files (venv, artifacts).
#   ./sync_remote.sh pull-self-play # latest run checkpoints + monitoring only
#   ./sync_remote.sh pull    # latest self-play run: 3 checkpoints + monitoring
#
#   ./sync_remote.sh setup   # push, `uv sync`, materialize the fixed v4
#                            # validation corpus if absent, then print the
#                            # exact command to start training.
#
#   REMOTE=gpu_remote REMOTE_DIR=/workspace/imba-chess ./sync_remote.sh push
#
# Note on artifacts/: the .gitignore filter means push sends NOTHING from
# artifacts/, including artifacts/move_vocab_static_uci.json. That is safe --
# load_or_create_static_move_vocab rebuilds it from MoveVocab.build_static(),
# verified byte-identical to the local file (1970 tokens) -- so remote-trained
# checkpoints use the same move ids as local evaluation.
set -euo pipefail

REMOTE="${REMOTE:-gpu_remote}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/imba-chess}"
export RSYNC_RSH="${RSYNC_RSH:-ssh -o ClearAllForwardings=yes}"

case "${1:-}" in
  push)
    ssh -o ClearAllForwardings=yes "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"
    # Respect per-directory ignore rules without deleting remote files.
    rsync -avz \
      --exclude '.git' \
      --filter=':- .gitignore' \
      ./ "${REMOTE}:${REMOTE_DIR}/"
    ;;
  pull|pull-self-play)
    mkdir -p artifacts
    exec 9>artifacts/.remote-self-play-sync.lock
    flock 9
    # Latest published run only: at most three current checkpoints,
    # configuration, metrics and TensorBoard. No replay/history/stream cache.
    # This is an evaluation/monitoring export, not a resumable replay backup.
    manifest=$(mktemp)
    trap 'rm -f "$manifest"' EXIT
    ssh -o ClearAllForwardings=yes "${REMOTE}" "cd '${REMOTE_DIR}' && python3 scripts/self_play_sync_manifest.py" > "$manifest"
    mkdir -p artifacts
    rsync -avz --from0 --files-from="$manifest" \
      "${REMOTE}:${REMOTE_DIR}/artifacts/" artifacts/
    python3 - "$manifest" <<'PYTHON'
from pathlib import Path
import sys
names = [Path(p.decode()) for p in Path(sys.argv[1]).read_bytes().split(b'\0') if p]
checkpoints = {Path('artifacts') / p for p in names if p.suffix == '.pt'}
for state in (p for p in names if p.name == "state.json"):
    print(f"Synced latest published self-play run: {state.parent.name}")
for directory in {p.parent for p in checkpoints}:
    for old in directory.glob('*.pt'):
        if old not in checkpoints:
            old.unlink()
PYTHON
    ;;
  setup)
    "$0" push
    ssh -o ClearAllForwardings=yes "${REMOTE}" "cd '${REMOTE_DIR}' && uv sync --python 3.13 && \
      if [[ ! -f artifacts/corpus/v4_val_125k.parquet ]]; then \
        .venv/bin/python scripts/materialize_corpus.py \
          --config config/imba_chess_v4.toml --split val \
          --output artifacts/corpus/v4_val_125k.parquet --max-rows 125000; \
      fi"
    cat <<EOF

Remote is ready. Start training inside tmux so it survives the ssh session:

  ssh ${REMOTE}
  cd ${REMOTE_DIR}
  tmux new -s train
  ./train_autorestart.sh --config config/imba_chess_v4.toml

Detach with ctrl-b then d; reattach with 'tmux attach -t train'.
The wrapper reads checkpoint_dir from the config and appends to
<checkpoint_dir>/train.log, so progress survives a dropped connection.

Bring checkpoints back with:  ./sync_remote.sh pull
EOF
    ;;
  *)
    echo "usage: $0 {push|pull|pull-self-play|setup}" >&2
    exit 2
    ;;
esac
