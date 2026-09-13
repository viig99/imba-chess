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
    policy_loss = -(target * log_probs).sum(-1).mean()
    value_logits = output["value_logits"].index_select(0, indices).float()
    wdl = batch["value_target"].to(logits.device).index_select(0, indices).float()
    value_log_probs = F.log_softmax(value_logits, -1)
    value_loss = -(wdl * value_log_probs).sum(-1).mean()
    probs = value_log_probs.exp()
    entropy = -(target * target.clamp_min(1e-38).log()).sum(-1).mean()
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
        loss=policy_loss + value_weight * value_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        policy_entropy=entropy,
        policy_target_kl=policy_loss - entropy,
        **actor_kl,
        brier=(probs - wdl).square().sum(-1).mean(),
        predicted_draw=probs[:, 1].mean(),
        observed_draw=wdl[:, 1].mean(),
        predicted_value=(probs[:, 2] - probs[:, 0]).mean(),
    )
