"""Float32 soft legal-policy and actual outcome WDL objectives."""

import torch
import torch.nn.functional as F


def self_play_loss(output, batch, *, value_weight=1.0, auxiliary_value_weight=1.0):
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
    loss = weighted_policy_loss if value_weight == 0 else weighted_policy_loss + value_weight * value_loss
    auxiliary_metrics = {}
    if auxiliary_value_weight > 0:
        auxiliary_logits = output["auxiliary_value_logits"].index_select(0, indices).float()
        auxiliary_target = batch["auxiliary_value_target"].to(logits.device).float().detach()
        if auxiliary_target.dim() == 2:
            auxiliary_target = auxiliary_target.unsqueeze(1)
        horizons = auxiliary_target.shape[1]
        if auxiliary_logits.shape[-1] != 3 * horizons:
            raise ValueError("auxiliary head count does not match auxiliary targets")
        auxiliary_logits = auxiliary_logits.view(-1, horizons, 3)
        # Each horizon is a separate loss term with the full auxiliary weight.
        head_losses = [
            -(auxiliary_target[:, h] * F.log_softmax(auxiliary_logits[:, h], -1)).sum(-1).mean()
            for h in range(horizons)
        ]
        for head_loss in head_losses:
            loss = loss + auxiliary_value_weight * head_loss
        auxiliary_metrics = dict(
            auxiliary_value_loss=head_losses[0] if horizons == 1 else sum(head_losses) / horizons,
            auxiliary_target_draw=auxiliary_target[..., 1].mean(),
            auxiliary_predicted_draw=auxiliary_logits.softmax(-1)[..., 1].mean(),
        )
        if horizons > 1:
            auxiliary_metrics.update(
                {f"auxiliary_value_loss_{h}": head_loss for h, head_loss in enumerate(head_losses)}
            )
    return dict(
        loss=loss,
        **auxiliary_metrics,
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
