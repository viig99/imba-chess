Cool. Here’s the “minimal surgery” way to bolt **depth-2 lookahead** onto your existing **policy+value transformer** (with your global move vocab + legality mask), without turning your codebase into AlphaZero.

I’ll show a practical recipe that works even if you have **no Stockfish**, only your learned **value head**.

---

# Goal

Given a position (s), your model gives:

* policy logits (\pi(a\mid s)) over moves (masked to legal)
* value logits (p(\text{loss/draw/win}\mid s)) → scalar value
  [
  V(s) = p(\text{win}) - p(\text{loss}) \in [-1,1]
  ]

We want a move chooser that is stronger than sampling/greedy by doing a tiny amount of “calculation”.

Depth-2 means:

**We choose our move assuming the opponent responds best** (one-ply response).

That alone kills a ton of tactical blunders.

---

# Minimal depth-2 search (one-step minimax)

Let it be our turn at state (s).

For each candidate move (a):

1. Apply it → (s' = T(s,a))
2. Opponent chooses best response (b) from (s'):
   [
   b^* = \arg\max_b V_{\text{opp}}(T(s',b))
   ]
3. We score our move by assuming opponent plays that (b^*):
   [
   Q(a) = - \max_b V(T(s',b))
   ]
   The minus is because after opponent moves, value is from **side-to-move** perspective again (now it’s us to move if game isn’t over). Easiest mental model:

* **Always compute (V(s)) as “good for side-to-move”.**
* When you simulate opponent’s move, they are the side-to-move at that node, so they maximize (V).
* From our perspective, that’s bad, so we negate it.

Then pick:
[
a^* = \arg\max_a Q(a)
]

That’s depth-2 minimax.

---

# The “don’t rewrite everything” trick: restrict branching using your policy head

Chess branching factor is 30–40. Depth-2 full search is ~1200 evals per move, still doable but maybe heavy.

So we do this:

* Top-K candidate moves for **our move** using policy: (K_1) (e.g. 8–16)
* Top-K candidate moves for **opponent response** using policy: (K_2) (e.g. 8–16)

This is “policy-pruned minimax”.

It’s the smallest step toward AlphaZero-style search, but very simple.

---

# Required building blocks

You only need 3 functions:

### 1) `model_policy_value(state, history)` → (policy_probs, V_scalar)

* runs your transformer
* masks illegal moves
* outputs:

  * `probs: [move_vocab]` for legal moves (or logits)
  * `V: float` derived from value logits

### 2) `legal_moves(state)` → list[move_id]

* generate legal move ids in your move vocab (UCI→id mapping)

### 3) `apply_move(state, move_id)` → new_state

* pure chess logic (python-chess or your own)

That’s it.

---

# Pseudocode: depth-2 policy-pruned minimax

```python
def scalar_value_from_value_logits(value_logits_3):
    # value_logits_3: [3] for [loss, draw, win] from side-to-move POV
    p = softmax(value_logits_3)
    return float(p[2] - p[0])  # win - loss in [-1, 1]

def topk_legal_moves(policy_logits, legal_mask, k):
    # mask illegal
    masked = policy_logits.clone()
    masked[~legal_mask] = -1e9
    # take top-k by logits
    return masked.topk(k).indices.tolist()

def choose_move_depth2(state, history, K1=12, K2=12):
    # 1) get policy+value for root
    policy_logits, value_logits = model_forward(state, history)
    legal_mask = make_legal_mask(state)  # bool [move_vocab]
    cand_moves = topk_legal_moves(policy_logits, legal_mask, K1)

    best_move = None
    best_score = -1e9

    for a in cand_moves:
        s1 = apply_move(state, a)
        if is_terminal(s1):
            # if we just delivered mate, take it
            return a

        # opponent node: they are side-to-move in s1
        pol1_logits, _ = model_forward(s1, history + [(a, s1)])
        legal1_mask = make_legal_mask(s1)
        opp_moves = topk_legal_moves(pol1_logits, legal1_mask, K2)

        # opponent chooses response that maximizes V at resulting state
        worst_for_us = -1e9  # from opponent POV it’s best; we negate later
        for b in opp_moves:
            s2 = apply_move(s1, b)
            if is_terminal(s2):
                # if opponent can mate us, this move is terrible
                v_us = -1.0
            else:
                _, v2_logits = model_forward(s2, history + [(a, s1), (b, s2)])
                v = scalar_value_from_value_logits(v2_logits)
                # v is from side-to-move POV at s2 (which is us again, usually)
                v_us = v

            # opponent wants to MINIMIZE our value,
            # equivalently maximize their value.
            # Since v_us is our perspective (side-to-move at s2), opponent chooses b that makes it smallest.
            worst_for_us = min(worst_for_us, v_us)

        score = worst_for_us  # our move quality under best opponent response
        if score > best_score:
            best_score = score
            best_move = a

    return best_move
```

### Tiny but important clarification

At `s2`, it’s usually **our turn** again, so (V(s2)) is “good for us”. That’s why opponent chooses the response that makes (V(s2)) **smallest**.

So your minimax at depth-2 can simply be:

* our move: maximize
* opponent move: minimize

No sign-flipping required if you always evaluate at positions where it’s our turn (depth even). Depth-2 ends on our turn, so it’s nice.

---

# Make it fast without changing your model

### Batch the evaluations

Instead of calling the transformer per node, batch them:

1. Evaluate root once.
2. Generate (K_1) child states (s1_i), batch-evaluate their policies to get opponent top-K.
3. Generate (K_1 \times K_2) grandchild states (s2_{i,j}), batch-evaluate their values.
4. Reduce with `min` over j and `max` over i.

This can be 10–100× faster on GPU.

You don’t have to rewrite your model, just add an inference wrapper that accepts a list of states/histories.

---

# How to integrate with your existing `HSTUChessModel` cleanly

### Inference wrapper idea

You already build per-token embeddings using the current state + prev move, etc.

For search, you just need to create a tiny batch representing:

* current position token (or a short recent history window)
* correct `seq_offsets` for those tiny sequences

If you don’t want to pass the full history, you can do **context truncation**:

* use last N plies (say 32)
* search doesn’t need opening context if the state encoding is rich

So your wrapper can be:

`evaluate_positions(states, prev_moves, ...) -> policy_logits, value_logits`

Where each “position” is one token (or a short window), batched.

That’s minimal.

---

# What parameters to start with

* `K1 = 12` (your candidate moves)
* `K2 = 12` (opponent responses)
* choose best by minimax on value at depth-2

Then later:

* increase to `K1=16`, `K2=16` if compute allows
* add a small policy prior bonus:
  [
  \text{score}(a) = \min_b V(s2_{a,b}) + \lambda \log \pi(a|s)
  ]
  with tiny (\lambda) (0.05–0.2) to break ties.

---

# Why this helps more than PPO early

Your biggest losses right now are likely:

* hanging pieces
* missing simple tactics
* stepping into forks/pins
* not seeing mate threats

Depth-2 catches a huge fraction of those because it explicitly checks:

* “what is the opponent’s best immediate response?”

That’s tactical hygiene.

RL fine-tuning can learn hygiene too, but it’s slower and noisier.

---

# If you want one extra lever: “blunder filter”

Even simpler than minimax:

* sample N candidate moves from policy
* for each candidate, evaluate **opponent’s best response** value drop
* reject moves that lead to very low value after best response

It’s basically depth-2 but used as a filter.

---

If you tell me what you’re using for move generation (python-chess or custom) and whether your inference expects a full history vs just state+prev_move, I’ll write a concrete batching plan for `K1*K2` nodes that fits your `seq_offsets` jagged layout cleanly.
