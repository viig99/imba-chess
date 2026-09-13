"""Developer-only fixture generator; JAX/mctx are NOT project dependencies.

Run with PYTHONPATH pointing at a checkout of the pinned mctx revision and
an isolated JAX environment. The recurrent function is a solved one-step
bandit with zero discount, matching a chess search capped at depth one.
"""

import json
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as jnp
import mctx

REVISION = "88f92056a420c2673bed282f5a0c00211f126e78"
priors = [float((i % 7) - 3) for i in range(20)]
noise = [float((i * 3 % 11) - 5) / 2 for i in range(20)]
returns = [float(i % 5 - 2) / 2 for i in range(20)]


def recurrent(params, key, action, embedding):
    return mctx.RecurrentFnOutput(
        reward=jnp.array(returns)[action],
        discount=jnp.zeros_like(action, dtype=jnp.float32),
        prior_logits=jnp.zeros((1, 20)),
        value=jnp.zeros((1,)),
    ), embedding


cases = []
for budget, top_m in [
    (1, 16),
    (2, 16),
    (3, 16),
    (16, 3),
    (32, 7),
    (64, 16),
    (128, 16),
    (256, 16),
]:
    with patch("jax.random.gumbel", return_value=jnp.array([noise])):
        result = mctx.gumbel_muzero_policy(
            None,
            jax.random.PRNGKey(42),
            mctx.RootFnOutput(
                prior_logits=jnp.array([priors]),
                value=jnp.array([0.2]),
                embedding=jnp.zeros((1, 1)),
            ),
            recurrent,
            num_simulations=budget,
            max_num_considered_actions=min(budget, top_m),
            max_depth=1,
        )
    cases.append(
        dict(
            budget=budget,
            top_m=top_m,
            action=int(result.action[0]),
            policy=result.action_weights[0].tolist(),
            visits=result.search_tree.children_visits[0, 0].tolist(),
        )
    )
Path(__file__).with_name("search.json").write_text(
    json.dumps(
        dict(
            revision=REVISION,
            priors=priors,
            noise=noise,
            returns=returns,
            root_value=0.2,
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
