#!/usr/bin/env python3
"""How much does the value head already know that Stockfish knows?

This is the zero-training feasibility check for distilling Lichess's
`[%eval]` annotations into the value head. Training a candidate arm costs a
run plus a 750-game SF2200 eval; this costs one inference pass, and it can
kill or confirm the idea first.

The value head is the head that stopped learning: over the last 75% of the
ckpt23 run, policy_loss fell 8.5% while value_loss fell 1.7%. The hypothesis
is that its target -- the final result of a HUMAN game -- is too noisy to
carry more signal (this corpus: even mate-scored positions convert only
85.1%), and that a Stockfish-derived target would carry more.

Reported on held-out positions that carry an eval:

  CE(model || sf)   -- cross-entropy of the model against the calibrated
                       Stockfish WDL.
  H(sf)             -- entropy of that target: the floor CE(model || sf) can
                       reach. The gap between them is KL(sf || model), which
                       IS the headroom, in nats, that distillation could close.
  CE(model || out)  -- the model against the actual game outcome; today's
                       training target, for reference.
  CE(sf || out)     -- Stockfish's OWN score against game outcomes. If this is
                       worse than CE(model || out), the model is already the
                       better predictor of human results and the case for
                       distillation rests entirely on ranking quality, not
                       calibration.

Two passes over the same stream (beta=1 then beta=0) recover the two targets
at identical tokens; the model is deterministic and num_workers is 0, so the
token order is identical between passes -- asserted, not assumed.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import LichessDataset, build_event_dataloader, load_or_create_static_move_vocab
from imba_chess.data.rollout_store import load_rollout_lookup
from imba_chess.eval.position_evaluator import load_hstu_checkpoint
from imba_chess.model import create_batch_block_mask


def _collect(*, model, loader, device, autocast_ctx, max_games):
    """(model_logprobs, target_probs) at tokens carrying a rollout value target."""
    preds, targs = [], []
    seen = 0
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        for batch in tqdm(loader, desc="value-vs-sf", unit="batch"):
            seq_offsets = batch["seq_offsets"].to(device=device, dtype=torch.long)
            block_mask = create_batch_block_mask(
                seq_offsets=seq_offsets, total_tokens=int(batch["total_tokens"]),
                device=device,
            )
            out = model(batch, block_mask=block_mask, return_loss=False)
            mask = batch["has_rollout_value_target"].to(device).bool()
            if mask.any():
                vl = out["value_logits"].float()[mask]
                preds.append(torch.log_softmax(vl, dim=-1).cpu().numpy())
                targs.append(batch["value_target_soft"].to(device)[mask].cpu().numpy())
            seen += int(batch["num_games"])
            if seen >= max_games:
                break
    return np.concatenate(preds), np.concatenate(targs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--local-corpus", type=Path, required=True)
    ap.add_argument("--eval-targets", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=4000)
    args = ap.parse_args()

    repo_config = load_repo_config(args.config)
    if int(repo_config.dataloader.num_workers) != 0:
        raise ValueError(
            "dataloader.num_workers must be 0: >0 shards the stream and silently "
            "zeroes every rollout target (the 2026-07-25 sharding bug)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path, include_unk=repo_config.vocab.include_unk
    )
    model, _ = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint, repo_config=repo_config,
        move_vocab=move_vocab, device=device, compile_model=False,
        require_value_head=True,
    )
    lookup = load_rollout_lookup(args.eval_targets)
    print(f"eval targets loaded: {len(lookup)} (game_id, ply) keys")

    autocast_ctx = contextlib.nullcontext()
    results = {}
    for beta in (1.0, 0.0):
        dataset = LichessDataset(
            min_avg_elo=repo_config.dataset.min_avg_elo,
            min_time_control_sec=repo_config.dataset.min_time_control_sec,
            split="val", dataset_name=repo_config.dataset.dataset_name,
            cache_dir=repo_config.dataset.cache_dir,
            parquet_batch_size=repo_config.dataset.parquet_batch_size,
            max_seq_len=repo_config.dataset.max_seq_len,
            board_state_config=repo_config.board_state,
        )
        dataset_cfg = replace(repo_config.dataset, split="val",
                              val_max_games=args.max_games)
        loader = build_event_dataloader(
            lichess_dataset=_LocalGames(dataset, str(args.local_corpus), args.max_games),
            config=replace(repo_config, dataset=dataset_cfg),
            move_vocab=move_vocab,
            rollout_lookup=lookup,
            rollout_beta=beta,
        )
        results[beta] = _collect(model=model, loader=loader, device=device,
                                 autocast_ctx=autocast_ctx, max_games=args.max_games)

    logp, sf = results[1.0]
    logp0, out = results[0.0]
    if logp.shape != logp0.shape or not np.allclose(logp, logp0, atol=1e-5):
        raise RuntimeError(
            "the two passes did not produce identical model outputs at identical "
            "tokens -- the streams diverged and the comparison is invalid"
        )

    n = len(logp)
    eps = 1e-9
    ce_model_sf = float(-(sf * logp).sum(1).mean())
    h_sf = float(-(sf * np.log(sf + eps)).sum(1).mean())
    ce_model_out = float(-(out * logp).sum(1).mean())
    ce_sf_out = float(-(out * np.log(sf + eps)).sum(1).mean())

    p = np.exp(logp)
    score_model = p[:, 2] - p[:, 0]
    score_sf = sf[:, 2] - sf[:, 0]
    pear = float(np.corrcoef(score_model, score_sf)[0, 1])
    rank = float(np.corrcoef(np.argsort(np.argsort(score_model)),
                             np.argsort(np.argsort(score_sf)))[0, 1])
    sign = float((np.sign(score_model) == np.sign(score_sf)).mean())

    print(f"\nheld-out positions with a Stockfish eval : {n}")
    print(f"  CE(model || sf)            {ce_model_sf:.4f}")
    print(f"  H(sf)          (the floor) {h_sf:.4f}")
    print(f"  -> KL(sf || model)         {ce_model_sf - h_sf:.4f}  <-- HEADROOM (nats)")
    print(f"  CE(model || outcome)       {ce_model_out:.4f}   (today's target)")
    print(f"  CE(sf    || outcome)       {ce_sf_out:.4f}")
    print(f"\n  E[score] Pearson r        {pear:.4f}")
    print(f"  E[score] Spearman r       {rank:.4f}")
    print(f"  sign agreement            {sign:.2%}")

    # WHERE the disagreement lives decides whether closing it would matter.
    # A gap concentrated in near-equal positions is mostly draw-rate
    # calibration; a gap in decided positions means the model misjudges who is
    # winning, which is exactly what a search ranking on value would get wrong.
    kl_row = (sf * (np.log(sf + eps) - logp)).sum(1)
    edges = [0.0, 0.05, 0.15, 0.30, 0.50, 1.01]
    print(f"\n  {'|sf E[score]-0.5|':<20}{'n':>8}{'KL(sf||model)':>16}"
          f"{'sign agree':>13}{'mean |d score|':>16}")
    absadv = np.abs(score_sf) / 2.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (absadv >= lo) & (absadv < hi)
        if m.sum() < 50:
            continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':<20}{int(m.sum()):>8}{kl_row[m].mean():>16.4f}"
              f"{(np.sign(score_model[m]) == np.sign(score_sf[m])).mean():>12.1%}"
              f"{np.abs(score_model[m] - score_sf[m]).mean():>16.4f}")
    np.savez_compressed("artifacts/corpus/value_vs_sf_arrays.npz",
                        logp=logp, sf=sf, out=out)
    print("\n  arrays saved -> artifacts/corpus/value_vs_sf_arrays.npz")


class _LocalGames:
    """Adapts a materialized corpus to the iterable the dataloader expects."""

    def __init__(self, dataset, path, max_games):
        self._dataset, self._path, self._max = dataset, path, max_games

    def as_torch_iterable(self, **_kwargs):
        outer = self

        class _It(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter(outer._dataset.stream_local(outer._path,
                                                        max_games=outer._max))

        return _It()


if __name__ == "__main__":
    main()
