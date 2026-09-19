"""Read-only per-game search-surprise distribution audit (no model required)."""
import argparse
import json
import torch
from imba_chess.data.self_play_store import SelfPlayStore, validate_game
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.dataset import policy_weights


def audit(store):
    rows = []
    ids = store.game_ids('train')
    for gid in ids:
        game = store.read_game(gid)
        validate_game(game)
        rows.append(policy_weights(game['targets'], learning=LearningConfig(policy_surprise_enabled=True)))
    if not rows:
        raise ValueError('no training games')
    data = {k: torch.tensor([v for row in rows for v in row[k]]) for k in rows[0]}
    eligible = data['policy_surprise_eligible']
    policy = data['policy_eligible']
    def distribution(t):
        t = t.double()
        return dict(count=t.numel(), mean=t.mean().item(), p95=torch.quantile(t, .95).item(), max=t.max().item()) if t.numel() else dict(count=0)
    effective = data['policy_training_weight'].double()
    return dict(games=len(ids), positions=len(policy),
                eligible_surprise=distribution(data['policy_surprise'][eligible]),
                final_weight=distribution(data['policy_surprise_weight'][policy]),
                missing_prior_fraction=((policy & ~eligible).sum() / policy.sum().clamp_min(1)).item(),
                clipping_fraction=(data['policy_surprise_clipped'].sum() / eligible.sum().clamp_min(1)).item(),
                effective_sample_size=(effective.sum().square() / effective.square().sum().clamp_min(1e-38)).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(SelfPlayStore(args.replay, read_only=True)), indent=2))


if __name__ == '__main__':
    main()
