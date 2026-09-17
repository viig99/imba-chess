# Window 16 plus eight KV heads: exact component arithmetic

These are theoretical component counts for v4 (8 layers, 16 query heads, K and V widths 64), not measured speedups. W=16 counts the current position and 15 preceding positions. Sixteen previous plies plus the current position would instead be W=17. Every temporal layer is windowed; the square encoder remains unchanged.

For a single sequence with T positions, compare full causal attention and 16 KV heads against W=16 and 8 KV heads:

| T | Raw path-KV reduction | One-query QK/AV arithmetic reduction | Causal training pair reduction |
|---:|---:|---:|---:|
| 64 | 8× | 4× | 2.30× |
| 128 | 16× | 8× | 4.28× |
| 256 | 32× | 16× | 8.27× |
| 512 | 64× | 32× | 16.27× |

Cache ratio = (T × 16)/(16 × 8) = T/8.

For one query with T visible entries including itself, full attention reads T entries per query head and windowed attention reads 16. Query-head count stays 16, so GQA does not add another factor of two to QK/AV arithmetic. It reduces unique KV storage, potentially its memory traffic, and K/V projection work.

For T >= 16, a full causal training sequence has T(T+1)/2 valid query/key pairs per head. Windowed attention has sum(min(t,16), t=1..T) = 16T-120. At T=256: 32,896 versus 3,976, ratio 8.27364. For packed games, sum these counts per game; total packed batch length is not a single game's context length. Executed GPU work includes tile padding/masking, so these are logical pairs, not exact kernel FLOPs.

In FP32, raw KV per retained position across the stack is 64 KiB with 16 KV heads and 32 KiB with 8. One 256-position path therefore changes from 16 MiB to 16 × 32 KiB = 0.5 MiB. For 24 such game prefixes: 384 MiB → 12 MiB. This excludes tree branch entries, padded workspaces, temporary buffers and allocator overhead. Different search branches may keep many additional unique KV entries; the entire search tree is not bounded to one 16-entry window per game. Memory savings require window-aware retention and compact GQA buffers, not just setting scores to negative infinity.

## Projection and training-memory effects

Current HSTU UVQK widths are U=1024, V=1024, Q=1024, K=1024. With eight KV heads, they become U=1024, V=512, Q=1024, K=512. The joint projection output width changes 4096 → 3072: 25% less work and parameters in that projection. Including the unchanged 1024→1024 output projection, the large temporal projection arithmetic changes from 5D² to 4D² per token: 20% less in those projections. Board encoding, policy/value heads and other operations remain.

Training does not use a single 16-token rolling cache for the whole batch: all training positions still require representations and backward state. Compiled FlexAttention already avoids saving the full quadratic score matrix. Windowing can therefore reduce attention work substantially without producing the inference-cache ratio as a total activation-memory reduction. GQA reduces K/V activations and some parameters/optimizer state, while other activations remain.

Consequently, a larger microbatch may fit, but its size must be measured. Increasing microbatch tokens also changes the number of optimizer updates per exposure budget; use gradient accumulation or an explicitly matched update schedule when comparing learning behavior.

For collection, smaller caches may permit more concurrent games. More games/hour depends on forward latency, CPU search, scheduling and saturation. If an illustrative 30% of total runtime becomes 16× faster, whole-run speedup is only 1/(0.70+0.30/16) = 1.39×. This fraction is hypothetical, not a profile of the bot.

Strength must be evaluated after adaptation to the new attention mask and head sharing. See [the history-use study plan](ATTENTION_HISTORY_STUDY_PLAN.md) before choosing a window.

Sources: [v4 configuration](../config/imba_chess_v4.toml), [HSTU implementation](../src/imba_chess/model/hstu_attention.py), [GQA paper](https://arxiv.org/abs/2305.13245), [FlexAttention and its block-sparse mask semantics](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html).
