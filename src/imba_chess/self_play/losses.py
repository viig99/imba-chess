"""Float32 soft legal-policy and actual outcome WDL objectives."""

import torch
import torch.nn.functional as F


def self_play_loss(output, batch, *, value_weight=1.0):
    indices = batch["supervised_indices"].to(output["logits"].device)
    logits = output["logits"].index_select(0, indices).float()
    legal = batch["legal_ids"].to(logits.device)
    mask = batch["legal_mask"].to(logits.device)
    target = batch["policy"].to(logits.device).float()
    legal_logits = logits.gather(1, legal).masked_fill(~mask, -torch.inf)
    log_probs = F.log_softmax(legal_logits, -1).masked_fill(~mask, 0.0)
    per_position_ce = -(target * log_probs).sum(-1)
    policy_loss = per_position_ce.mean()
    weighted_policy_loss = policy_loss
    if "policy_training_weight" in batch:
        weights = batch["policy_training_weight"].to(logits.device).float().detach()
        denominator = weights.sum()
        weighted_policy_loss = (weights * per_position_ce).sum() / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
    value_logits = output["value_logits"].index_select(0, indices).float()
    wdl = batch["value_target"].to(logits.device).index_select(0, indices).float()
    value_log_probs = F.log_softmax(value_logits, -1)
    value_loss = -(wdl * value_log_probs).sum(-1).mean()
    probs = value_log_probs.exp()
    # Keep the historical target entropy key; model entropy measures the
    # student's distribution and can move in a different direction.
    entropy = -(target * target.clamp_min(1e-38).log()).sum(-1).mean()
    with torch.no_grad():
        model_entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        decisive = wdl[:, 1] == 0
        decisive_count = decisive.sum()
        wl_logits = value_logits[:, [0, 2]]
        wl_targets = wdl[:, [0, 2]]
        wl_ce = -(wl_targets * F.log_softmax(wl_logits, -1)).sum(-1)
        wl_correct = wl_logits.argmax(-1) == wl_targets.argmax(-1)
    actor_kl = {}
    if "actor_log_priors" in batch:
        available = batch["actor_prior_available"].to(logits.device)
        actor_logs = batch["actor_log_priors"].to(logits.device)
        per_position = (target * (target.clamp_min(1e-38).log() - actor_logs)).sum(-1)
        actor_kl = dict(
            search_prior_kl=(per_position * available).sum()
            / available.sum().clamp_min(1),
            actor_prior_positions=available.sum(),
        )
    return dict(
        loss=weighted_policy_loss + value_weight * value_loss,
        weighted_policy_loss=weighted_policy_loss,
        **batch.get("policy_weight_metrics", {}),
        policy_loss=policy_loss,
        value_loss=value_loss,
        policy_entropy=entropy,
        model_policy_entropy=model_entropy,
        decisive_positions=decisive_count,
        conditional_wl_loss=(wl_ce * decisive).sum() / decisive_count.clamp_min(1),
        conditional_wl_accuracy=(wl_correct * decisive).sum() / decisive_count.clamp_min(1),
        policy_target_kl=policy_loss - entropy,
        **actor_kl,
        brier=(probs - wdl).square().sum(-1).mean(),
        predicted_draw=probs[:, 1].mean(),
        observed_draw=wdl[:, 1].mean(),
        predicted_value=(probs[:, 2] - probs[:, 0]).mean(),
    )
