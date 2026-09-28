"""Search driven by one network's policy and another network's value.

Two ordinary search coroutines run in lockstep over identical composed
predictions. Each coroutine owns its own complete network and its own KV tree;
only prediction outputs cross the boundary, never representations, handles or
caches. This is a full-network composition, not a head transplant.

Lockstep requires both coroutines to follow the same search path. The only
nondeterminism in Gumbel search is the root noise vector, so it is drawn once
here and passed explicitly to both sides. Drawing it here consumes the caller's
generator in exactly the pattern a single ordinary runtime would, which keeps
existing per-game RNG streams reproducible.
"""

import math
import random
from dataclasses import asdict

from .batch_scheduler import WorkRequest


def compose_nodes(policy, value):
    """Take legal actions and priors from `policy`, side-to-move value from `value`."""
    if len(policy) != len(value):
        raise ValueError("composed prediction count mismatch")
    output = []
    for p, v in zip(policy, value):
        if p.legal_ids != v.legal_ids or p.legal_ucis != v.legal_ucis:
            raise ValueError("composed legal alignment mismatch")
        output.append(p._replace(value_stm=v.value_stm, wdl=v.wdl))
    return output


def gumbel_noise(count, rng):
    """The root noise vector, drawn exactly as `gumbel_stepwise` would."""
    rng = rng or random.Random()
    return [-math.log(-math.log(max(rng.random(), 1e-12))) for _ in range(count)]


class ComposedRuntime:
    """Policy priors from one runtime, node values from another."""

    algorithm = "gumbel"
    allow_noise = True

    def __init__(self, policy, value):
        if policy is value:
            raise ValueError("composition requires two distinct runtimes")
        for runtime in (policy, value):
            if getattr(runtime, "algorithm", None) != "gumbel":
                raise ValueError("composition requires Gumbel runtimes")
        self.policy, self.value = policy, value
        self.move_vocab, self.encoder = policy.move_vocab, policy.encoder
        self.options = dict(
            algorithm="gumbel-composed",
            dtype="float32",
            tf32=False,
            policy=getattr(policy, "options", {}),
            value=getattr(value, "options", {}),
        )
        self.executors = {
            f"{role}_{kind}": executor
            for role, runtime in (("policy", policy), ("value", value))
            for kind, executor in runtime.executors.items()
        }

    def clear_caches(self):
        for runtime in (self.policy, self.value):
            runtime.clear_caches()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_caches()

    def _resolve_noise(self, kwargs):
        noise = kwargs.get("noise")
        if noise == 0.0:
            return 0.0
        if noise is not None:
            return noise
        if not self.allow_noise:
            raise ValueError("this composition requires zero noise")
        board = kwargs.get("board")
        if board is None:
            raise ValueError("composed search needs the board to size root noise")
        return gumbel_noise(board.legal_moves.count(), kwargs.get("rng"))

    def search(self, **kwargs):
        # Both sides receive the same explicit noise, so neither draws from the
        # caller's generator and the two trees cannot diverge on exploration.
        kwargs = dict(kwargs, noise=self._resolve_noise(kwargs))
        generators = [runtime.search(**kwargs) for runtime in (self.policy, self.value)]
        responses = [None, None]
        try:
            while True:
                requests, finished = [], []
                for gen, response in zip(generators, responses):
                    try:
                        requests.append(gen.send(response))
                        finished.append(None)
                    except StopIteration as stop:
                        finished.append(stop.value)
                if any(r is not None for r in finished):
                    if len(requests) or asdict(finished[0]) != asdict(finished[1]):
                        raise ValueError("composed search trees diverged")
                    return finished[0]
                p, v = requests
                if p.kind != v.kind or p.payload[0] != v.payload[0]:
                    raise ValueError("composed requests diverged")
                if p.kind == "decode_wave":
                    pb, vb = p.payload[1][1], v.payload[1][1]
                    if len(pb) != len(vb) or any(
                        a[1].fen() != b[1].fen() for a, b in zip(pb, vb)
                    ):
                        raise ValueError("composed tree paths diverged")
                po = yield WorkRequest("policy_" + p.kind, p.payload)
                vo = yield WorkRequest("value_" + v.kind, v.payload)
                if po[0] != p.payload[0] or vo[0] != v.payload[0]:
                    raise ValueError("composed result owner mismatch")
                if p.kind == "root_eval":
                    # Preserve each network's own kv_caches.
                    responses = [
                        (po[0], dict(po[1], value_logits=vo[1]["value_logits"])),
                        (vo[0], dict(vo[1], logits=po[1]["logits"])),
                    ]
                else:
                    composed = compose_nodes(po[1], vo[1])
                    responses = [(po[0], composed), (vo[0], composed)]
        finally:
            for gen in generators:
                gen.close()
