"""Full accepted histories, sparse continuation policy targets and outcome WDL."""

import math
import chess
import torch
from imba_chess.data.collate import collate_jagged_batch
from imba_chess.data.self_play_store import validate_game
from imba_chess.eval.position_evaluator import _SequenceHistory


def outcome_wdl(outcome_white, white_to_move):
    outcome = outcome_white if white_to_move else -outcome_white
    return [float(outcome == -1), float(outcome == 0), float(outcome == 1)]


def _policy_surprise(target, actor_logs):
    """KL(target || actor), removing stored FP32 normalization roundoff.

    Search accepts logits up to an additive constant. Re-normalize the stored
    actor logs in float64 so that this constant cannot become training surprise.
    Normalize the target too, and suppress residual double-precision cancellation
    far below the 1e-8 denominator regularizer used for game-relative weighting.
    """
    maximum = max(actor_logs)
    log_sum = math.log(math.fsum(math.exp(lp - maximum) for lp in actor_logs))
    mass = math.fsum(target)
    divergence = math.fsum(
        (p / mass) * (math.log(p / mass) - ((lp - maximum) - log_sum))
        for p, lp in zip(target, actor_logs) if p > 0
    )
    return divergence if divergence > 1e-12 else 0.0


def policy_weights(targets, *, learning=None):
    """Compute detached weights over one complete continuation, never a batch.

    Missing priors retain multiplier one and do not enter normalization.
    """
    base = [t.get("policy_training_weight", 1.0) for t in targets]
    surprise = [
        _policy_surprise(t["policy"], t["root_log_priors"])
        if t.get("root_log_priors") is not None else 0.0
        for t in targets
    ]
    eligible = [i for i, t in enumerate(targets)
                if base[i] > 0 and t.get("root_log_priors") is not None]
    weights, clipped = [1.0] * len(targets), [False] * len(targets)
    if learning is not None and learning.policy_surprise_enabled and eligible:
        mean = math.fsum(surprise[i] for i in eligible) / len(eligible)
        if mean > 0:
            fraction, cap = learning.policy_surprise_fraction, learning.policy_surprise_cap
            raw = [(1 - fraction) + fraction * surprise[i] / (mean + 1e-8) for i in eligible]
            bounded = [min(cap, u) for u in raw]
            normalizer = math.fsum(bounded) / len(bounded)
            for i, u, r in zip(eligible, bounded, raw):
                weights[i], clipped[i] = u / normalizer, r > cap
    eligible_set = set(eligible)
    return dict(policy_training_weight=[b * w for b, w in zip(base, weights)],
                policy_surprise_weight=weights, policy_surprise=surprise,
                policy_surprise_eligible=[i in eligible_set for i in range(len(targets))],
                policy_eligible=[b > 0 for b in base], policy_surprise_clipped=clipped)


def reconstruct(game, *, move_vocab, encoder, max_positions, learning=None):
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
    sample.update(policy_weights(game["targets"], learning=learning))
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
    for key in ("policy_training_weight", "policy_surprise_weight", "policy_surprise",
                "policy_surprise_eligible", "policy_eligible", "policy_surprise_clipped"):
        batch[key] = torch.tensor([v for sample in samples for v in sample[key]])
    # CPU diagnostics avoid dynamic selection/quantiles in the compiled GPU loss.
    eligible = batch["policy_surprise_eligible"]
    surprises = batch["policy_surprise"][eligible].float()
    policy_eligible = batch["policy_eligible"]
    weights = batch["policy_surprise_weight"][policy_eligible].float()
    effective = batch["policy_training_weight"].float()
    batch["policy_weight_metrics"] = dict(
        eligible_surprise_mean=surprises.mean() if surprises.numel() else torch.tensor(0.),
        eligible_surprise_p95=torch.quantile(surprises, .95) if surprises.numel() else torch.tensor(0.),
        missing_actor_prior_fraction=(policy_eligible & ~batch["actor_prior_available"]).sum() / policy_eligible.sum().clamp_min(1),
        policy_weight_mean=weights.mean() if weights.numel() else torch.tensor(0.),
        policy_weight_max=weights.max() if weights.numel() else torch.tensor(0.),
        policy_weight_clipping_fraction=batch["policy_surprise_clipped"].sum() / eligible.sum().clamp_min(1),
        policy_weight_ess=effective.sum().square() / effective.square().sum().clamp_min(1e-38),
    )
    return batch
