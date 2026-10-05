"""One fitting epoch over an iteration's self-play games (reference: 1 epoch).

Games are packed whole into jagged batches of at most `batch_tokens` tokens
([BOS, event_1..event_T] per game). Loss, as in the reference:
cross-entropy to the stored improved policy + (Q(s, a_played) - G)^2, plus our
value head's outcome-WDL cross-entropy.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from optimi import StableAdamW

from imba_chess.data.event_builder import BOS_TOKEN_ID, EVENT_TOKEN_ID
from imba_chess.model import create_batch_block_mask, create_batch_dense_mask
from imba_chess.optim import build_decay_param_groups

from .engine import TOKEN_KEYS


def pack(lengths, batch_tokens, rng):
    """Shuffled whole-game packing: lists of game indices, each <= batch_tokens tokens."""
    batches, current, size = [], [], 0
    for index in rng.permutation(len(lengths)):
        tokens = int(lengths[index]) + 1
        if tokens > batch_tokens:
            raise ValueError("a game exceeds batch_tokens")
        if current and size + tokens > batch_tokens:
            batches.append(current)
            current, size = [], 0
        current.append(int(index))
        size += tokens
    if current:
        batches.append(current)
    return batches


def build_batch(games, start_id):
    plies = np.array([len(g["move_id"]) for g in games])
    tokens = plies + 1
    offsets = np.concatenate([[0], np.cumsum(tokens)])
    total = int(offsets[-1])
    bos = offsets[:-1]
    events = np.ones(total, dtype=bool)
    events[bos] = False
    piece_ids = np.zeros((total, 64), dtype=np.int64)
    piece_ids[events] = np.concatenate([g["piece_ids"] for g in games])
    scalars = np.zeros((total, 5), dtype=np.int64)
    scalars[events] = np.concatenate([g["scalars"] for g in games])
    prev = np.full(total, start_id, dtype=np.int64)
    prev[events] = np.concatenate([g["prev_move_id"] for g in games])
    seq_token = np.full(total, EVENT_TOKEN_ID, dtype=np.int64)
    seq_token[bos] = BOS_TOKEN_ID
    batch = dict(
        piece_ids=torch.from_numpy(piece_ids),
        seq_token_id=torch.from_numpy(seq_token),
        prev_move_id=torch.from_numpy(prev),
        seq_offsets=torch.from_numpy(offsets),
        total_tokens=total,
    )
    for column, key in enumerate(TOKEN_KEYS[1:-1]):
        batch[key] = torch.from_numpy(scalars[:, column].copy())

    counts = np.concatenate([np.diff(g["legal_offsets"]) for g in games])
    positions = len(counts)
    width = int(counts.max())
    rows = np.repeat(np.arange(positions), counts)
    cols = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    legal = np.zeros((positions, width), dtype=np.int64)
    legal[rows, cols] = np.concatenate([g["legal_ids"] for g in games])
    policy = np.zeros((positions, width), dtype=np.float32)
    policy[rows, cols] = np.concatenate([g["policy"] for g in games])
    mask = np.zeros((positions, width), dtype=bool)
    mask[rows, cols] = True
    batch.update(
        supervised_indices=torch.from_numpy(np.flatnonzero(events)),
        legal_ids=torch.from_numpy(legal),
        legal_mask=torch.from_numpy(mask),
        policy=torch.from_numpy(policy),
        move_id=torch.from_numpy(np.concatenate([g["move_id"] for g in games]).astype(np.int64)),
        returns=torch.from_numpy(np.concatenate([g["returns"] for g in games])),
        # side-to-move outcome -1/0/+1 -> WDL index loss/draw/win.
        outcome=torch.from_numpy(np.concatenate([g["outcome"] for g in games]).astype(np.int64) + 1),
    )
    return batch


def klent_loss(output, batch, *, policy_weight, value_weight):
    device = output["logits"].device
    indices = batch["supervised_indices"].to(device)
    legal = batch["legal_ids"].to(device)
    mask = batch["legal_mask"].to(device)
    logits = output["logits"].index_select(0, indices).float()
    # Normalize over the entire move vocabulary, as in the reference. Targets
    # have mass only on legal moves, so illegal logits receive downward gradients.
    log_probs = F.log_softmax(logits, -1).gather(1, legal).masked_fill(~mask, 0.0)
    policy_loss = -(batch["policy"].to(device) * log_probs).sum(-1).mean()
    q = output["q"].index_select(0, indices).float()
    q_played = q.gather(1, batch["move_id"].to(device)[:, None]).squeeze(1)
    q_loss = (q_played - batch["returns"].to(device)).square().mean()
    loss = policy_weight * policy_loss + q_loss
    metrics = dict(policy_loss=policy_loss, q_loss=q_loss)
    if value_weight > 0:
        value_logits = output["value_logits"].index_select(0, indices).float()
        value_loss = F.cross_entropy(value_logits, batch["outcome"].to(device))
        loss = loss + value_weight * value_loss
        metrics["value_loss"] = value_loss
    metrics["loss"] = loss
    return metrics


class KlentTrainer:
    """Owns the fp32 training model, its (optionally compiled) forward and optimizer."""

    def __init__(self, model, *, device, start_id, batch_tokens, grad_clip, compile_model):
        self.model, self.device, self.start_id = model, device, start_id
        self.batch_tokens, self.grad_clip = batch_tokens, grad_clip
        compiled = compile_model and device.type == "cuda"
        self.forward = torch.compile(model, dynamic=True, fullgraph=True) if compiled else model
        self.optimizer = None
        self.warmup = None

    def set_phase(self, *, warmup, lr, weight_decay):
        """Warm-up trains only the action-value and value heads (trunk and policy
        frozen); the main phase trains everything.

        A checkpoint's value head may be badly calibrated for self-play (the
        supervised 53250 head never predicts draws), so it is fitted alongside
        the fresh Q head before any gradient reaches the shared trunk.
        """
        if self.warmup == warmup:
            # Runtime overrides must preserve the accumulated optimizer moments.
            groups = build_decay_param_groups(self.model, weight_decay=weight_decay)
            for group, configured in zip(self.optimizer.param_groups, groups):
                group["lr"] = lr
                group["weight_decay"] = configured["weight_decay"]
            return
        self.warmup = warmup
        for parameter in self.model.parameters():
            parameter.requires_grad_(not warmup)
        self.model.action_value_head.requires_grad_(True)
        self.model.value_head.requires_grad_(True)
        self.optimizer = StableAdamW(
            build_decay_param_groups(self.model, weight_decay=weight_decay),
            lr=lr,
            triton=False,
            kahan_sum=True,
        )

    def _mask(self, batch):
        factory = create_batch_block_mask if self.device.type == "cuda" else create_batch_dense_mask
        with torch.no_grad():
            return factory(
                batch["seq_offsets"].to(self.device),
                total_tokens=batch["total_tokens"],
                device=self.device,
            )

    def train_epoch(self, games, rng, *, policy_weight, value_weight):
        lengths = [len(g["move_id"]) for g in games]
        sums, steps, tokens, start = {}, 0, 0, time.perf_counter()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.train()
        try:
            for indices in pack(lengths, self.batch_tokens, rng):
                batch = build_batch([games[i] for i in indices], self.start_id)
                self.optimizer.zero_grad(set_to_none=True)
                output = self.forward(batch, block_mask=self._mask(batch), return_loss=False)
                losses = klent_loss(
                    output, batch, policy_weight=policy_weight, value_weight=value_weight
                )
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("nonfinite KLENT loss")
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.grad_clip, error_if_nonfinite=True
                )
                self.optimizer.step()
                losses["gradient_norm"] = norm
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                steps += 1
                tokens += batch["total_tokens"]
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()
        elapsed = time.perf_counter() - start
        metrics = {f"train/{k}": v / max(steps, 1) for k, v in sums.items()}
        metrics.update({
            "train/steps": steps,
            "train/tokens": tokens,
            "time/train": elapsed,
            "train/tokens_per_second": tokens / elapsed,
        })
        return metrics

    def state_dict(self):
        return dict(warmup=self.warmup, optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state, *, lr, weight_decay):
        self.set_phase(warmup=state["warmup"], lr=lr, weight_decay=weight_decay)
        self.optimizer.load_state_dict(state["optimizer"])
        # load_state_dict also restores old group settings; apply current overrides.
        self.set_phase(warmup=state["warmup"], lr=lr, weight_decay=weight_decay)
