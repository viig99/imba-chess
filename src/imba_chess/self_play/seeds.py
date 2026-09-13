"""Offline human prefix manifests. Source outcomes/annotations never enter seeds."""

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import random

import chess
import chess.pgn
import pyarrow.parquet as pq

from imba_chess.eval.search import terminal_value_for_color


def stable_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_split(source_id):
    return "monitor" if int(stable_hash(source_id)[:16], 16) % 10 == 0 else "train"


@dataclass(frozen=True)
class Seed:
    seed_id: str
    source_id: str
    prefix_moves: list[str]
    takeover_ply: int
    split: str
    corpus_id: str

    def board(self):
        if self.takeover_ply != len(self.prefix_moves) or self.split != source_split(
            self.source_id
        ):
            raise ValueError("invalid seed identity/split")
        board = chess.Board()
        for uci in self.prefix_moves:
            board.push_uci(uci)
        if terminal_value_for_color(board, color=board.turn) is not None:
            raise ValueError("terminal or claimable-draw seed")
        return board


def prepare_seeds(
    corpus, output, *, provenance, run_seed=42, min_ply=20, max_ply=120, max_games=10000
):
    if provenance.get("split") != "train":
        raise ValueError("seed corpus must be materialized from the training split")
    if not 0 <= min_ply <= max_ply or max_games < 1:
        raise ValueError("invalid seed bounds")
    corpus_id = file_hash(corpus)
    seeds, seen = [], set()
    for batch in pq.ParquetFile(corpus).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            source_id = str(row.get("Site") or "")
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            game = chess.pgn.read_game(io.StringIO(row.get("movetext", "")))
            if game is None or game.errors or game.board().fen() != chess.STARTING_FEN:
                continue
            board, prefix, eligible = chess.Board(), [], []
            for ply, move in enumerate(game.mainline_moves()):
                if ply > max_ply:
                    break
                if (
                    ply >= min_ply
                    and terminal_value_for_color(board, color=board.turn) is None
                ):
                    eligible.append(list(prefix))
                board.push(move)
                prefix.append(move.uci())
            if not eligible:
                continue
            chosen = random.Random(f"{run_seed}:{source_id}").choice(eligible)
            seeds.append(
                Seed(
                    stable_hash(f"{corpus_id}:{source_id}:{len(chosen)}"),
                    source_id,
                    chosen,
                    len(chosen),
                    source_split(source_id),
                    corpus_id,
                )
            )
            if len(seeds) >= max_games:
                break
        if len(seeds) >= max_games:
            break
    if not seeds:
        raise ValueError("no eligible training seeds")
    from imba_chess.data.self_play_store import atomic_json

    atomic_json(
        Path(output),
        dict(
            schema_version=1,
            corpus_id=corpus_id,
            provenance=provenance,
            run_seed=run_seed,
            seeds=[asdict(s) for s in seeds],
        ),
    )
    return seeds


def load_seeds(path, split=None):
    data = json.loads(Path(path).read_text())
    if data["schema_version"] != 1 or data["provenance"].get("split") != "train":
        raise ValueError("unsupported or non-training seed manifest")
    seeds = [Seed(**s) for s in data["seeds"] if split is None or s["split"] == split]
    for seed in seeds:
        if seed.corpus_id != data["corpus_id"]:
            raise ValueError("seed corpus identity mismatch")
        seed.board()
    if not seeds:
        raise ValueError(f"no seeds for split {split}")
    return seeds
