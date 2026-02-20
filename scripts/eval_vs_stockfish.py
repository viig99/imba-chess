#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.event_builder import (
    BOS_TOKEN_ID,
    EVENT_TOKEN_ID,
    TARGET_IGNORE_INDEX,
)
from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab
from imba_chess.model import (
    HSTUChessModel,
    build_hstu_chess_config,
    create_batch_block_mask,
)
from tqdm.auto import tqdm


@dataclass
class EvalSummary:
    games: int = 0
    completed_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_as_white: int = 0
    games_as_black: int = 0
    wins_as_white: int = 0
    losses_as_white: int = 0
    draws_as_white: int = 0
    wins_as_black: int = 0
    losses_as_black: int = 0
    draws_as_black: int = 0
    incomplete_games: int = 0
    total_plies: int = 0

    @property
    def avg_plies(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_plies / self.games

    @property
    def avg_full_moves(self) -> float:
        return self.avg_plies / 2.0


class _SequenceHistory:
    """Incrementally builds the BOS+event sequence used for model inference."""

    def __init__(
        self, *, move_vocab: MoveVocab, board_state_encoder: BoardStateEncoder
    ) -> None:
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder

        self.seq_token_id: list[int] = [BOS_TOKEN_ID]
        self.piece_ids: list[list[int]] = [[0] * 64]
        self.turn_id: list[int] = [0]
        self.castle_id: list[int] = [0]
        self.ep_file_id: list[int] = [0]
        self.halfmove_bucket_id: list[int] = [0]
        self.fullmove_bucket_id: list[int] = [0]
        self.prev_move_id: list[int] = [self._move_vocab.start_id]
        self.target_move_id: list[int] = [TARGET_IGNORE_INDEX]
        self.played_by_elo: list[int] = [0]

        self._prev_move_id_for_next_token = self._move_vocab.start_id

    def append_observed_position(self, board: chess.Board) -> None:
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)

    def record_played_move(self, move_uci: str) -> None:
        self._prev_move_id_for_next_token = int(self._move_vocab.encode(move_uci))

    def _append_from_state(self, state) -> None:
        self.seq_token_id.append(EVENT_TOKEN_ID)
        self.piece_ids.append(list(state.piece_ids))
        self.turn_id.append(int(state.turn_id))
        self.castle_id.append(int(state.castle_id))
        self.ep_file_id.append(int(state.ep_file_id))
        self.halfmove_bucket_id.append(int(state.halfmove_bucket_id))
        self.fullmove_bucket_id.append(int(state.fullmove_bucket_id))
        self.prev_move_id.append(int(self._prev_move_id_for_next_token))
        self.target_move_id.append(TARGET_IGNORE_INDEX)
        self.played_by_elo.append(0)

    def _pop_last(self) -> None:
        self.seq_token_id.pop()
        self.piece_ids.pop()
        self.turn_id.pop()
        self.castle_id.pop()
        self.ep_file_id.pop()
        self.halfmove_bucket_id.pop()
        self.fullmove_bucket_id.pop()
        self.prev_move_id.pop()
        self.target_move_id.pop()
        self.played_by_elo.pop()

    def _build_single_batch(self) -> dict[str, Any]:
        # Single-sequence jagged batch; avoids collate list-copy overhead.
        total_tokens = len(self.seq_token_id)
        return {
            "game_id": ["stockfish_eval"],
            "num_games": 1,
            "total_tokens": total_tokens,
            "seq_lens": torch.tensor([total_tokens], dtype=torch.long),
            "seq_offsets": torch.tensor([0, total_tokens], dtype=torch.long),
            "piece_ids": torch.tensor(self.piece_ids, dtype=torch.long),
            "seq_token_id": torch.tensor(self.seq_token_id, dtype=torch.long),
            "turn_id": torch.tensor(self.turn_id, dtype=torch.long),
            "castle_id": torch.tensor(self.castle_id, dtype=torch.long),
            "ep_file_id": torch.tensor(self.ep_file_id, dtype=torch.long),
            "halfmove_bucket_id": torch.tensor(
                self.halfmove_bucket_id, dtype=torch.long
            ),
            "fullmove_bucket_id": torch.tensor(
                self.fullmove_bucket_id, dtype=torch.long
            ),
            "prev_move_id": torch.tensor(self.prev_move_id, dtype=torch.long),
            "target_move_id": torch.tensor(self.target_move_id, dtype=torch.long),
            "played_by_elo": torch.tensor(self.played_by_elo, dtype=torch.long),
        }

    def build_batch_for_current_position(self, board: chess.Board) -> dict[str, Any]:
        # Add transient current-position token for next-move prediction only.
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)
        try:
            return self._build_single_batch()
        finally:
            self._pop_last()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained imba-chess model against Stockfish via UCI."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--stockfish-path", type=Path, default=Path("/usr/bin/stockfish")
    )
    parser.add_argument("--stockfish-time-sec", type=float, default=0.05)
    parser.add_argument("--stockfish-nodes", type=int, default=None)
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--stockfish-hash-mb", type=int, default=64)
    parser.add_argument("--stockfish-limit-strength", action="store_true")
    parser.add_argument("--stockfish-elo", type=int, default=None)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_arg]


def _load_model(
    *,
    checkpoint_path: Path,
    repo_config,
    move_vocab: MoveVocab,
    device: torch.device,
    compile_model: bool,
) -> torch.nn.Module:
    model_cfg = build_hstu_chess_config(
        repo_config.model,
        move_vocab_size=len(move_vocab),
    )
    model: torch.nn.Module = HSTUChessModel(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint must be a model state_dict or Ignite checkpoint containing key 'model'."
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if compile_model:
        model = torch.compile(model, dynamic=True, fullgraph=False)
    return model


def _build_engine_limit(args: argparse.Namespace) -> chess.engine.Limit:
    kwargs: dict[str, float | int] = {}
    if args.stockfish_time_sec is not None:
        kwargs["time"] = float(args.stockfish_time_sec)
    if args.stockfish_nodes is not None:
        kwargs["nodes"] = int(args.stockfish_nodes)
    if args.stockfish_depth is not None:
        kwargs["depth"] = int(args.stockfish_depth)
    if not kwargs:
        kwargs["time"] = 0.05
    return chess.engine.Limit(**kwargs)


def _select_model_move(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    board: chess.Board,
    move_vocab: MoveVocab,
    device: torch.device,
    dtype: torch.dtype,
) -> chess.Move:
    seq_offsets = batch["seq_offsets"].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    block_mask = create_batch_block_mask(
        seq_offsets=seq_offsets,
        total_tokens=int(batch["total_tokens"]),
        device=device,
    )
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_amp
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        output = model(batch, block_mask=block_mask, return_loss=False)

    logits = output["logits"][-1]
    legal_moves = list(board.legal_moves)
    legal_move_ids: list[int] = []
    legal_moves_with_ids: list[chess.Move] = []
    for move in legal_moves:
        move_id = move_vocab.token_to_id.get(move.uci())
        if move_id is not None:
            legal_move_ids.append(int(move_id))
            legal_moves_with_ids.append(move)
    if not legal_move_ids:
        raise RuntimeError("No legal moves mapped to vocab ids for current board.")

    legal_ids_tensor = torch.tensor(
        legal_move_ids, device=logits.device, dtype=torch.long
    )
    legal_logits = logits.index_select(0, legal_ids_tensor)
    best_index = int(torch.argmax(legal_logits).item())
    return legal_moves_with_ids[best_index]


def _update_summary(
    summary: EvalSummary,
    *,
    result: str,
    model_color: chess.Color,
    completed: bool,
    plies: int,
) -> None:
    summary.games += 1
    summary.total_plies += int(plies)
    if model_color == chess.WHITE:
        summary.games_as_white += 1
    else:
        summary.games_as_black += 1

    if not completed:
        summary.incomplete_games += 1
        return

    summary.completed_games += 1
    if result == "1/2-1/2":
        summary.draws += 1
        if model_color == chess.WHITE:
            summary.draws_as_white += 1
        else:
            summary.draws_as_black += 1
        return

    model_won = (model_color == chess.WHITE and result == "1-0") or (
        model_color == chess.BLACK and result == "0-1"
    )
    if model_won:
        summary.wins += 1
        if model_color == chess.WHITE:
            summary.wins_as_white += 1
        else:
            summary.wins_as_black += 1
    else:
        summary.losses += 1
        if model_color == chess.WHITE:
            summary.losses_as_white += 1
        else:
            summary.losses_as_black += 1


def main() -> None:
    args = _parse_args()
    if args.games < 1:
        raise ValueError("--games must be >= 1")
    if args.max_plies < 1:
        raise ValueError("--max-plies must be >= 1")
    if args.stockfish_limit_strength and args.stockfish_elo is None:
        raise ValueError(
            "--stockfish-elo is required when --stockfish-limit-strength is set"
        )
    if args.stockfish_threads < 1:
        raise ValueError("--stockfish-threads must be >= 1")
    if args.stockfish_hash_mb < 1:
        raise ValueError("--stockfish-hash-mb must be >= 1")
    if not args.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish binary not found: {args.stockfish_path}")

    random.seed(args.seed)
    repo_config = load_repo_config(args.config)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path,
        include_unk=repo_config.vocab.include_unk,
    )
    board_state_encoder = BoardStateEncoder(repo_config.board_state)

    model = _load_model(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=bool(args.compile),
    )
    engine_limit = _build_engine_limit(args)

    summary = EvalSummary()
    stockfish_options: dict[str, Any] = {
        "Threads": int(args.stockfish_threads),
        "Hash": int(args.stockfish_hash_mb),
        "UCI_LimitStrength": bool(args.stockfish_limit_strength),
    }
    if args.stockfish_limit_strength and args.stockfish_elo is not None:
        stockfish_options["UCI_Elo"] = int(args.stockfish_elo)

    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish_path)) as engine:
        engine.configure(stockfish_options)

        print("Running model vs Stockfish")
        print(f"  games={args.games}")
        print(f"  stockfish={args.stockfish_path}")
        print(f"  limit={engine_limit}")
        print(f"  stockfish_options={stockfish_options}")
        print(f"  device={device}, dtype={dtype}, compile={bool(args.compile)}")

        with tqdm(
            total=args.games, desc="stockfish-eval", unit="game", dynamic_ncols=True
        ) as progress:
            for game_idx in range(args.games):
                board = chess.Board()
                history = _SequenceHistory(
                    move_vocab=move_vocab,
                    board_state_encoder=board_state_encoder,
                )
                model_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
                completed = True
                plies = 0

                while not board.is_game_over(claim_draw=True):
                    if plies >= args.max_plies:
                        completed = False
                        break
                    if board.turn == model_color:
                        batch = history.build_batch_for_current_position(board)
                        move = _select_model_move(
                            model=model,
                            batch=batch,
                            board=board,
                            move_vocab=move_vocab,
                            device=device,
                            dtype=dtype,
                        )
                    else:
                        result = engine.play(board, engine_limit)
                        if result.move is None:
                            raise RuntimeError("Stockfish returned no move.")
                        move = result.move

                    history.append_observed_position(board)
                    history.record_played_move(move.uci())
                    board.push(move)
                    plies += 1

                result = board.result(claim_draw=True) if completed else "*"
                _update_summary(
                    summary,
                    result=result,
                    model_color=model_color,
                    completed=completed,
                    plies=plies,
                )

                progress.update(1)
                progress.set_postfix(
                    {
                        "W": summary.wins,
                        "D": summary.draws,
                        "L": summary.losses,
                        "inc": summary.incomplete_games,
                        "avg_plies": f"{summary.avg_plies:.1f}",
                        "avg_moves": f"{summary.avg_full_moves:.1f}",
                    }
                )

    if summary.completed_games > 0:
        win_rate = summary.wins / summary.completed_games
        draw_rate = summary.draws / summary.completed_games
        loss_rate = summary.losses / summary.completed_games
        score_rate_completed = (
            summary.wins + 0.5 * summary.draws
        ) / summary.completed_games
    else:
        win_rate = float("nan")
        draw_rate = float("nan")
        loss_rate = float("nan")
        score_rate_completed = float("nan")
    score_rate_all_games = (
        (summary.wins + 0.5 * summary.draws) / summary.games
        if summary.games > 0
        else float("nan")
    )

    payload = {
        "games": summary.games,
        "completed_games": summary.completed_games,
        "wins": summary.wins,
        "draws": summary.draws,
        "losses": summary.losses,
        "incomplete_games": summary.incomplete_games,
        "average_plies_per_game": summary.avg_plies,
        "average_full_moves_per_game": summary.avg_full_moves,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "score_rate": score_rate_completed,
        "score_rate_all_games": score_rate_all_games,
        "rate_denominator_games": summary.completed_games,
        "by_color": {
            "white": {
                "games": summary.games_as_white,
                "wins": summary.wins_as_white,
                "draws": summary.draws_as_white,
                "losses": summary.losses_as_white,
            },
            "black": {
                "games": summary.games_as_black,
                "wins": summary.wins_as_black,
                "draws": summary.draws_as_black,
                "losses": summary.losses_as_black,
            },
        },
        "run_config": {
            "checkpoint": str(args.checkpoint),
            "stockfish_path": str(args.stockfish_path),
            "stockfish_limit": str(engine_limit),
            "stockfish_options": stockfish_options,
            "device": str(device),
            "dtype": str(dtype),
            "compile": bool(args.compile),
            "seed": int(args.seed),
            "max_plies": int(args.max_plies),
        },
    }

    print("\nModel vs Stockfish summary")
    print(f"  games: {payload['games']}")
    print(
        f"  wins/draws/losses: {payload['wins']} / {payload['draws']} / {payload['losses']}"
    )
    print(
        f"  completed_games: {payload['completed_games']} "
        f"(incomplete={payload['incomplete_games']})"
    )
    print(
        f"  average plies/game: {payload['average_plies_per_game']:.2f} "
        f"(avg full moves: {payload['average_full_moves_per_game']:.2f})"
    )
    print(
        f"  score_rate (completed games): {payload['score_rate']:.4f} "
        f"(denominator={payload['rate_denominator_games']})"
    )
    print(f"  score_rate (all games): {payload['score_rate_all_games']:.4f}")
    print(
        "  as_white (W/D/L): "
        f"{summary.wins_as_white}/{summary.draws_as_white}/{summary.losses_as_white}"
    )
    print(
        "  as_black (W/D/L): "
        f"{summary.wins_as_black}/{summary.draws_as_black}/{summary.losses_as_black}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote: {args.output_json}")


if __name__ == "__main__":
    main()
