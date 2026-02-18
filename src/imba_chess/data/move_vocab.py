from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


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
                game_obj = dataclasses.asdict(game) if dataclasses.is_dataclass(game) else game
                plays = game_obj["plays"]
                for play in plays:
                    play_obj = dataclasses.asdict(play) if dataclasses.is_dataclass(play) else play
                    yield play_obj["move_uci"]

        return cls.build(iter_moves(), config=config)

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
