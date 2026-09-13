"""Full accepted histories, sparse continuation policy targets and outcome WDL."""

import chess
import torch
from imba_chess.data.collate import collate_jagged_batch
from imba_chess.data.self_play_store import validate_game
from imba_chess.eval.position_evaluator import _SequenceHistory


def outcome_wdl(outcome_white, white_to_move):
    outcome = outcome_white if white_to_move else -outcome_white
    return [float(outcome == -1), float(outcome == 0), float(outcome == 1)]


def reconstruct(game, *, move_vocab, encoder, max_positions):
    validate_game(game)
    history = _SequenceHistory(move_vocab=move_vocab, board_state_encoder=encoder)
    board = chess.Board()
    values, indices, policies, legal = [[0.0, 0.0, 0.0]], [], [], []
    for uci in game["prefix_moves"]:
        history.append_observed_position(board)
        values.append([0.0, 0.0, 0.0])
        history.record_played_move(uci)
        board.push_uci(uci)
    for uci, target in zip(game["moves"], game["targets"]):
        if target["move_uci"] != uci or target["move_id"] != move_vocab.encode(uci):
            raise ValueError("played move/target mismatch")
        expected = {move_vocab.encode(m.uci()) for m in board.legal_moves}
        if set(target["legal_ids"]) != expected:
            raise ValueError("target must cover every legal move")
        indices.append(len(history.seq_token_id))
        legal.append(target["legal_ids"])
        policies.append(target["policy"])
        history.append_observed_position(board)
        values.append(outcome_wdl(game["outcome_white"], board.turn))
        history.record_played_move(uci)
        board.push_uci(uci)
    from .collector import terminal_outcome

    actual = terminal_outcome(board)
    if actual is None or actual != (game["outcome_white"], game["termination"]):
        raise ValueError("trajectory outcome does not match actual terminal board")
    size = len(history.seq_token_id)
    if size > max_positions:
        raise ValueError("trajectory exceeds positional context")
    sample = {
        key: getattr(history, key)
        for key in (
            "seq_token_id",
            "piece_ids",
            "turn_id",
            "castle_id",
            "ep_file_id",
            "halfmove_bucket_id",
            "fullmove_bucket_id",
            "prev_move_id",
            "target_move_id",
            "played_by_elo",
        )
    }
    sample.update(
        game_id=game["game_id"],
        game_result_white=game["outcome_white"],
        value_target=values,
        has_value_target=[False] * (game["takeover_ply"] + 1) + [True] * len(indices),
        supervised_indices=indices,
        legal_ids=legal,
        policy=policies,
        actor_log_priors=[t.get("root_log_priors") for t in game["targets"]],
    )
    return sample


def collate_self_play(samples):
    batch = collate_jagged_batch(samples)
    indices, legal, policies, actor_priors = [], [], [], []
    offset = 0
    for sample in samples:
        indices.extend(offset + i for i in sample["supervised_indices"])
        legal.extend(sample["legal_ids"])
        policies.extend(sample["policy"])
        actor_priors.extend(sample["actor_log_priors"])
        offset += len(sample["seq_token_id"])
    if not indices:
        raise ValueError("empty supervised batch")
    width = max(map(len, legal))
    batch["supervised_indices"] = torch.tensor(indices, dtype=torch.long)
    batch["legal_ids"] = torch.tensor(
        [row + [0] * (width - len(row)) for row in legal], dtype=torch.long
    )
    batch["policy"] = torch.tensor(
        [row + [0.0] * (width - len(row)) for row in policies], dtype=torch.float32
    )
    batch["legal_mask"] = (
        torch.arange(width)[None, :] < torch.tensor(list(map(len, legal)))[:, None]
    )
    batch["actor_prior_available"] = torch.tensor(
        [p is not None for p in actor_priors], dtype=torch.bool
    )
    batch["actor_log_priors"] = torch.tensor(
        [
            (p + [0.0] * (width - len(p))) if p is not None else [0.0] * width
            for p in actor_priors
        ],
        dtype=torch.float32,
    )
    return batch
