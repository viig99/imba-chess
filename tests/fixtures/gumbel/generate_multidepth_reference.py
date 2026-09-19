"""Generate multi-depth reference fixtures in an isolated CPU JAX/mctx environment.

Use mctx revision 88f92056a420c2673bed282f5a0c00211f126e78 on PYTHONPATH alongside
the repository root and src. The audit used JAX 0.11.2, Chex 0.1.92, x64 enabled.
Both backends receive identical fixed noise, deterministic values and dynamics.
An absorbing terminal state keeps its side-to-move value with discount +1;
nonterminal chess-like alternating-player transitions have discount -1.
No JAX dependency is added to the project's runtime or default tests.
"""
import json, math, random, functools, sys
from unittest.mock import patch
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import mctx
from mctx._src import qtransforms
from imba_chess.eval.gumbel_search import GumbelConfig, select_gumbel
from imba_chess.eval.search import PositionEval


def evaluate_values(node, k, seed, xp=np):
    logits = (((node[..., None] + 3) * (xp.arange(k) + 7) * 17 + seed * 11) % 113 - 56) / 16.
    value = ((node * 37 + seed * 13) % 101 - 50) / 64.
    return logits, value


from tests.test_gumbel_mctx_audit import run_synthetic


def upstream(k, seed, budget, top_m, depth, terminal, scale, noise):
    def recurrent(params, key, action, embedding):
        node, d, done = embedding[:,0], embedding[:,1], embedding[:,2]
        child = jnp.where(done, node, node * k + action + 1)
        new_d = jnp.where(done, d, d + 1)
        new_done = done | (terminal & (new_d >= 2) & (child % 11 == 0))
        logits, value = evaluate_values(child, k, seed, jnp)
        value = jnp.where(new_done, -1., value)
        return mctx.RecurrentFnOutput(reward=jnp.zeros_like(value),discount=jnp.where(done,1.,-1.),prior_logits=logits,value=value), jnp.stack((child,new_d,new_done),-1)
    logits, value = evaluate_values(jnp.asarray([1]),k,seed,jnp)
    root = mctx.RootFnOutput(prior_logits=logits,value=value,embedding=jnp.array([[1,0,0]], dtype=jnp.int32))
    with patch('jax.random.gumbel', return_value=jnp.asarray([noise])):
        result = jax.jit(lambda: mctx.gumbel_muzero_policy(None,jax.random.key(42),root,recurrent,num_simulations=budget,max_depth=depth,max_num_considered_actions=min(top_m,k,budget),qtransform=functools.partial(qtransforms.qtransform_completed_by_mix_value,value_scale=scale)))()
    return dict(action=int(result.action[0]),visits=result.search_tree.children_visits[0,0].tolist(),qvalues=result.search_tree.qvalues(jnp.array([0]))[0].tolist(),policy=result.action_weights[0].tolist())


def main():
    rng=random.Random(72)
    rows=[]
    cases=[(4, 17, 128, 4, 6, False, .1), (4, 17, 128, 4, 6, True, 1.)]
    cases += [(rng.choice([2,3,4,7]),s,rng.choice([3,16,32,64,128,200]),rng.choice([1,3,7,16]),rng.choice([1,2,3,5]),bool(s%2),rng.choice([.1,1.])) for s in range(20)]
    for k,seed,budget,top_m,depth,terminal,scale in cases:
        noise=[-math.log(-math.log(rng.random())) for _ in range(k)]
        actual=run_synthetic(k=k,seed=seed,budget=budget,top_m=top_m,depth=depth,terminal=terminal,scale=scale,noise=noise)
        expected=upstream(k,seed,budget,top_m,depth,terminal,scale,noise)
        passed=actual.move_id==expected['action'] and actual.visits==expected['visits'] and np.allclose(actual.policy,expected['policy'],atol=1e-10,rtol=1e-10) and np.allclose(actual.qvalues,expected['qvalues'],atol=1e-10,rtol=1e-10)
        row=dict(k=k,seed=seed,budget=budget,top_m=top_m,depth=depth,terminal=terminal,scale=scale,noise=noise,expected=expected,passed=passed,policy_error=float(np.max(np.abs(np.array(actual.policy)-expected['policy']))),q_error=float(np.max(np.abs(np.array(actual.qvalues)-expected['qvalues']))))
        rows.append(row)
        print({key:row[key] for key in ['k','seed','budget','depth','passed','policy_error','q_error']},flush=True)
    fixture = dict(revision="88f92056a420c2673bed282f5a0c00211f126e78",
                   reference_dtype="float64", generator="generate_multidepth_reference.py",
                   cases=[dict(inputs={k:r[k] for k in ['k','seed','budget','top_m','depth','terminal','scale','noise']},
                               expected=r['expected']) for r in rows])
    from pathlib import Path
    Path(__file__).with_name('multidepth_mctx.json').write_text(json.dumps(fixture, indent=2) + '\n')
    assert all(r['passed'] for r in rows)
if __name__=='__main__': main()
