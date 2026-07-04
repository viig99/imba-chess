from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chess

DEFAULT_STATIC_MOVE_VOCAB_PATH = Path("artifacts/move_vocab_static_uci.json")


@dataclass(frozen=True)
class MoveVocabConfig:
    pad_token: str = "<pad>"
    start_token: str = "<start>"
    unk_token: str = "<unk>"
    include_unk: bool = True


class MoveVocab:
    def __init__(self, token_to_id: Dict[str, int], config: Optional[MoveVocabConfig] = None) -> None:
        self.config = config or MoveVocabConfig()
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

        self.pad_id = self.token_to_id[self.config.pad_token]
        self.start_id = self.token_to_id[self.config.start_token]
        self.unk_id = self.token_to_id.get(self.config.unk_token)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, move_uci: str) -> int:
        token_id = self.token_to_id.get(move_uci)
        if token_id is not None:
            return token_id
        if self.unk_id is None:
            raise KeyError(f"Unknown move token and UNK disabled: {move_uci}")
        return self.unk_id

    def decode(self, token_id: int) -> str:
        if token_id not in self.id_to_token:
            raise KeyError(f"Unknown token id: {token_id}")
        return self.id_to_token[token_id]

    @classmethod
    def build(
        cls,
        move_uci_iterable: Iterable[str],
        config: Optional[MoveVocabConfig] = None,
    ) -> "MoveVocab":
        cfg = config or MoveVocabConfig()
        token_to_id: Dict[str, int] = {cfg.pad_token: 0, cfg.start_token: 1}

        next_id = 2
        if cfg.include_unk:
            token_to_id[cfg.unk_token] = next_id
            next_id += 1

        for move_uci in move_uci_iterable:
            if move_uci not in token_to_id:
                token_to_id[move_uci] = next_id
                next_id += 1

        return cls(token_to_id=token_to_id, config=cfg)

    @classmethod
    def build_from_games(
        cls,
        games: Iterable[object],
        config: Optional[MoveVocabConfig] = None,
    ) -> "MoveVocab":
        def iter_moves() -> Iterable[str]:
            for game in games:
                for play in game["plays"]:
                    yield play["move_uci"]

        return cls.build(iter_moves(), config=config)

    @classmethod
    def build_static(cls, config: Optional[MoveVocabConfig] = None) -> "MoveVocab":
        cfg = config or MoveVocabConfig(include_unk=False)
        return cls.build(all_possible_uci_moves(), config=cfg)

    def save(self, path: str | Path) -> None:
        output = {
            "config": {
                "pad_token": self.config.pad_token,
                "start_token": self.config.start_token,
                "unk_token": self.config.unk_token,
                "include_unk": self.config.include_unk,
            },
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MoveVocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg_data = payload["config"]
        config = MoveVocabConfig(
            pad_token=cfg_data["pad_token"],
            start_token=cfg_data["start_token"],
            unk_token=cfg_data["unk_token"],
            include_unk=cfg_data["include_unk"],
        )
        return cls(token_to_id=payload["token_to_id"], config=config)


def all_possible_uci_moves() -> List[str]:
    """Deterministic superset of UCI moves: geometrically reachable from->to
    pairs plus promotions.

    Queen rays + knight jumps from each square cover every piece's movement
    (king, pawn, castling, and en passant moves are all subsets), so no legal
    standard-chess move is outside this set. Non-reachable pairs (e.g. a1h2)
    can never be played and are excluded to keep the label space compact.
    """
    moves: list[str] = []

    # Non-promotion moves: queen-ray + knight targets per square = 1792.
    board = chess.Board(None)
    for from_square in chess.SQUARES:
        from_name = chess.square_name(from_square)
        targets_mask = 0
        for piece_type in (chess.QUEEN, chess.KNIGHT):
            board.set_piece_at(from_square, chess.Piece(piece_type, chess.WHITE))
            targets_mask |= board.attacks_mask(from_square)
            board.remove_piece_at(from_square)
        for to_square in chess.SquareSet(targets_mask):
            moves.append(f"{from_name}{chess.square_name(to_square)}")

    # Promotions: 44 base pawn destinations * 4 promo pieces = 176
    promo_pieces = ("q", "r", "b", "n")

    def add_promotions(from_rank: int, to_rank: int) -> None:
        for file_idx in range(8):
            from_square = chess.square(file_idx, from_rank)
            from_name = chess.square_name(from_square)
            for delta in (-1, 0, 1):
                to_file = file_idx + delta
                if to_file < 0 or to_file > 7:
                    continue
                to_square = chess.square(to_file, to_rank)
                to_name = chess.square_name(to_square)
                for promo in promo_pieces:
                    moves.append(f"{from_name}{to_name}{promo}")

    add_promotions(from_rank=6, to_rank=7)  # White: rank 7 -> 8
    add_promotions(from_rank=1, to_rank=0)  # Black: rank 2 -> 1

    # Keep deterministic order while deduplicating (there should be no dups).
    return list(dict.fromkeys(moves))


def load_or_create_static_move_vocab(
    path: str | Path = DEFAULT_STATIC_MOVE_VOCAB_PATH,
    *,
    include_unk: bool = False,
) -> MoveVocab:
    vocab_path = Path(path)
    if vocab_path.exists():
        return MoveVocab.load(vocab_path)

    vocab = MoveVocab.build_static(config=MoveVocabConfig(include_unk=include_unk))
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    return vocab
