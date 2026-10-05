"""One fitting epoch over an iteration's self-play games (reference: 1 epoch).

Games are packed whole into jagged batches of at most `batch_tokens` tokens
([BOS, event_1..event_T] per game). Loss, as in the reference:
cross-entropy to the stored improved policy + (Q(s, a_played) - G)^2, plus our
value head's outcome-WDL cross-entropy.
"""

from concurrent.futures import ThreadPoolExecutor
import contextlib
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
    # Token fields cover every ply (human prefix included); targets exist only
    # for supervised plies, so they map to a subset of the event tokens.
    plies = np.array([len(g["prev_move_id"]) for g in games])
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
        supervised_indices=torch.from_numpy(np.concatenate([
            start + 1 + np.flatnonzero(g.get("supervised", np.ones(n, dtype=bool)))
            for start, n, g in zip(offsets[:-1], plies, games)
        ])),
        legal_ids=torch.from_numpy(legal),
        legal_mask=torch.from_numpy(mask),
        policy=torch.from_numpy(policy),
        move_id=torch.from_numpy(np.concatenate([g["move_id"] for g in games]).astype(np.int64)),
        returns=torch.from_numpy(np.concatenate([g["returns"] for g in games])),
        # side-to-move outcome -1/0/+1 -> WDL index loss/draw/win.
        outcome=torch.from_numpy(np.concatenate([g["outcome"] for g in games]).astype(np.int64) + 1),
    )
    return batch


def klent_loss(output, batch, *, policy_weight, value_weight, advantage=False):
    """advantage=True reads the action-value head as A(s, a) with
    Q(s, a) = V(s) + A(s, a), V the (frozen) value head's W-L, and fits the
    played move's A to G - V(s): the position value is subtracted from the
    target, so the head only has to learn how moves differ."""
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
    target = batch["returns"].to(device)
    if advantage:
        wdl = torch.softmax(output["value_logits"].index_select(0, indices).float(), -1)
        target = target - (wdl[:, 2] - wdl[:, 0]).detach()
    q_loss = (q_played - target).square().mean()
    loss = policy_weight * policy_loss + q_loss
    metrics = dict(policy_loss=policy_loss, q_loss=q_loss)
    if value_weight > 0:
        value_logits = output["value_logits"].index_select(0, indices).float()
        value_loss = F.cross_entropy(value_logits, batch["outcome"].to(device))
        loss = loss + value_weight * value_loss
        metrics["value_loss"] = value_loss
    metrics["loss"] = loss
    return metrics


def _forward_loss(model, batch, block_mask, policy_weight, value_weight, advantage):
    output = model(batch, block_mask=block_mask, return_loss=False)
    return klent_loss(output, batch, policy_weight=policy_weight, value_weight=value_weight,
                      advantage=advantage)


def _prepare(games, start_id, pin):
    """Background-thread half of a step: build the jagged batch on the CPU."""
    batch = build_batch(games, start_id)
    if pin:
        batch = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in batch.items()}
    return batch


def trunk_parameters(model):
    """Trainable parameters shared by every head (excludes the head modules and
    the policy matrix tied to the previous-move embedding)."""
    heads = {
        id(p)
        for head in (model.prediction_head, model.value_head, model.action_value_head,
                     model.moves_left_head, getattr(model, "auxiliary_value_head", None))
        if head is not None
        for p in head.parameters()
    }
    return [p for p in model.parameters() if p.requires_grad and id(p) not in heads]


def trunk_gradient_probe(losses, weights, trunk):
    """Each weighted loss's gradient norm on the shared trunk, plus the policy/Q
    cosine: how strongly each objective shapes the shared features."""
    grads = {}
    for name, weight in weights.items():
        if weight > 0:
            grads[name] = torch.autograd.grad(
                weight * losses[name], trunk, retain_graph=True, allow_unused=True
            )
    sq = lambda g: sum((x.float() ** 2).sum() for x in g if x is not None)
    norms = {name: sq(g).sqrt() for name, g in grads.items()}
    probe = {f"grad_trunk_{name.removesuffix('_loss')}": n for name, n in norms.items()}
    if "policy_loss" in grads and "q_loss" in grads:
        dot = sum((a.float() * b.float()).sum() for a, b in zip(grads["policy_loss"], grads["q_loss"])
                  if a is not None and b is not None)
        probe["grad_trunk_cos_policy_q"] = dot / (norms["policy_loss"] * norms["q_loss"]).clamp_min(1e-30)
        probe["grad_trunk_ratio_policy_q"] = norms["policy_loss"] / norms["q_loss"].clamp_min(1e-30)
    return probe


class KlentTrainer:
    """Owns the fp32 master weights, the compiled forward+loss and the optimizer.

    Mirrors supervised training (scripts/train.py): bf16 autocast with TF32
    matmuls, the loss inside the compiled graph, and the triton optimizer. The
    next batch is built and pinned on a background thread, and no step reads a
    value back from the GPU: losses accumulate on the device and are checked
    for finiteness once per epoch, which always precedes the checkpoint write.
    """

    def __init__(self, model, *, device, start_id, batch_tokens, grad_clip, compile_model,
                 train_dtype="float32", freeze_value_head=False, probe_every=64,
                 advantage=False):
        self.model, self.device, self.start_id = model, device, start_id
        self.freeze_value_head = freeze_value_head
        self.advantage = advantage
        # Every probe_every-th step also measures per-loss trunk gradients.
        self.probe_every = probe_every
        self.batch_tokens, self.grad_clip = batch_tokens, grad_clip
        cuda = device.type == "cuda"
        if compile_model and cuda and probe_every:
            # The gradient probe differentiates the compiled graph once per loss
            # (retain_graph=True), which donated backward buffers forbid.
            torch._functorch.config.donated_buffer = False
        self.forward_loss = (
            torch.compile(_forward_loss, dynamic=True, fullgraph=True)
            if compile_model and cuda else _forward_loss
        )
        self.autocast_dtype = {"float32": None, "bfloat16": torch.bfloat16}[train_dtype]
        if cuda:
            torch.set_float32_matmul_precision("high")
        self.optimizer = None
        self.warmup = None

    def set_phase(self, *, warmup, lr, weight_decay):
        """Warm-up trains only the action-value and value heads (trunk and policy
        frozen); the main phase trains everything.

        A checkpoint's value head may be badly calibrated for self-play (the
        supervised 53250 head never predicts draws), so it is fitted alongside
        the fresh Q head before any gradient reaches the shared trunk. With
        freeze_value_head the value head never trains in either phase: fitting
        it to draw-heavy self-play outcomes erases its ability to tell
        positions apart, which the bootstrap and search rely on.
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
        self.model.value_head.requires_grad_(not self.freeze_value_head)
        self.optimizer = StableAdamW(
            build_decay_param_groups(self.model, weight_decay=weight_decay),
            lr=lr,
            triton=self.device.type == "cuda",
            kahan_sum=True,
        )

    def _mask(self, batch):
        factory = create_batch_block_mask if self.device.type == "cuda" else create_batch_dense_mask
        with torch.no_grad():
            return factory(
                batch["seq_offsets"], total_tokens=batch["total_tokens"], device=self.device
            )

    def _autocast(self):
        if self.autocast_dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def train_epoch(self, games, rng, *, policy_weight, value_weight):
        lengths = [len(g["prev_move_id"]) for g in games]
        plan = pack(lengths, self.batch_tokens, rng)
        sums, tokens, start = {}, 0, time.perf_counter()
        probes, probe_count = {}, 0
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        trunk = trunk_parameters(self.model)
        loss_weights = dict(policy_loss=policy_weight, q_loss=1.0, value_loss=value_weight)
        pin = self.device.type == "cuda"
        self.model.train()
        with ThreadPoolExecutor(max_workers=1) as pool:
            submit = lambda i: pool.submit(_prepare, [games[j] for j in plan[i]], self.start_id, pin)
            pending = submit(0) if plan else None
            try:
                for step in range(len(plan)):
                    batch = pending.result()
                    pending = submit(step + 1) if step + 1 < len(plan) else None
                    tokens += batch["total_tokens"]
                    batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                    self.optimizer.zero_grad(set_to_none=True)
                    with self._autocast():
                        losses = self.forward_loss(
                            self.model, batch, self._mask(batch), policy_weight, value_weight,
                            self.advantage,
                        )
                    if trunk and self.probe_every and step % self.probe_every == 0:
                        for key, value in trunk_gradient_probe(losses, loss_weights, trunk).items():
                            probes[key] = probes[key] + value.detach() if key in probes else value.detach()
                        probe_count += 1
                    losses["loss"].backward()
                    losses["gradient_norm"] = torch.nn.utils.clip_grad_norm_(
                        parameters, self.grad_clip
                    )
                    self.optimizer.step()
                    for key, value in losses.items():
                        value = value.detach().float()
                        sums[key] = sums[key] + value if key in sums else value
            finally:
                if pending is not None:
                    pending.cancel()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.eval()
        steps = len(plan)
        # One device->host read per epoch. A non-finite loss or gradient norm
        # anywhere in the epoch makes its sum non-finite, and this raises before
        # the caller can write a checkpoint.
        values = torch.stack(list(sums.values())).tolist() if sums else []
        if not all(np.isfinite(values)):
            raise FloatingPointError("nonfinite KLENT loss or gradient norm in this epoch")
        elapsed = time.perf_counter() - start
        metrics = {f"train/{k}": v / max(steps, 1) for k, v in zip(sums, values)}
        if probes:
            probe_values = torch.stack([v.float() for v in probes.values()]).tolist()
            metrics.update({f"train/{k}": v / probe_count for k, v in zip(probes, probe_values)})
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
