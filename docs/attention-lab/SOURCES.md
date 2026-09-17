# Attention laboratory: implementation and source notes

Open [the interactive guide](index.html) in a browser, or read [the exported 25-page slide PDF](slides.pdf). It contains 25 slides, expandable explanations, four interactive laboratories and a printable slide layout. Choose **Read all** for continuous reading, **Present** for a focused deck, or **Print slides** for every slide with notes. The guide uses local CSS/JavaScript and the repository-root [tokens.css](../../tokens.css); it needs no server or network connection. To share it, keep these relative files together. The original PDF remains in the user's Downloads folder.

Reviewed 2026-09-13 against commit `1e9e64e` plus its working tree. Existing changes were present in the self-play runner/losses/tests and readiness report. This task adds educational documentation and interface assets, without changing model or training behavior. Installed PyTorch was `2.14.0+cu130`. Historical benchmarks below were not rerun.

## What the implementation actually does

| Question | Evidence |
|---|---|
| Where does square attention happen? | [hstu_model.py](../../src/imba_chess/model/hstu_model.py), `_SquareAttentionBlock` at line 208 and `BoardSquareEncoder` at 237. Two 64-wide blocks, four heads, bidirectional SDPA, residual attention and MLP, then final norm, mean pooling and projection. |
| What is a temporal token? | [hstu_model.py](../../src/imba_chess/model/hstu_model.py), `_build_content` at 376. Board vector plus sequence-token kind, turn, castling, EP, halfmove/fullmove buckets and previous-move embeddings. |
| What is the trunk equation? | [hstu_attention.py](../../src/imba_chess/model/hstu_attention.py), `forward` at 303: normalized input → joint linear UVQK → SiLU → split → softmax attention → normalize → U gate → dropout → output projection → residual. |
| What are positions? | [position_embedding.py](../../src/imba_chess/model/position_embedding.py): learned absolute embedding plus `sqrt(D)` times content. `_position_score_mod` at hstu_attention.py:254 adds a learned, per-head, per-layer relative-distance table. No RoPE. |
| What prevents game mixing and future leakage? | [hstu_model.py](../../src/imba_chess/model/hstu_model.py), `create_batch_block_mask` at 89 and `create_batch_dense_mask` at 110. Same game, key ≤ query; packed game position indices restart. |
| What selects SDPA versus Flex? | `forward` above branches on mask type. A Tensor goes to SDPA with an explicitly constructed additive bias. A BlockMask goes to FlexAttention with a score modifier. CPU compatibility has a dense path. This is not controlled just by `.training`. |
| Why is dense risky? | `_additive_mask` at hstu_attention.py:271 constructs `[1,H,S,S]`. Its per-mask guard is 256 MiB. Indexing and other temporaries are additional. |
| What happens in cached Stockfish search? | `forward_decode` at hstu_attention.py:402 and `forward_decode_grouped` at 514 explicitly form prefix/suffix/self scores, apply one softmax and mix values. [position_evaluator.py](../../src/imba_chess/eval/position_evaluator.py) and [merged_executors.py](../../src/imba_chess/eval/merged_executors.py) call these methods. |
| What happens in CUDA self-play collection? | [runtime.py](../../src/imba_chess/self_play/runtime.py):42 defaults to `compiled` when optimized CUDA is enabled. [tensor_decoder.py](../../src/imba_chess/model/tensor_decoder.py):13 defines `TensorDecoder`; `DecoderRunner` at 102 compiles the whole decoder in compiled modes. Default `sdpa=False` calls `_one_query_attention` at hstu_attention.py:153. Collection requests FP32. |
| Is SDPA decode available? | TensorDecoder's separate `sdpa` branch uses a reusable KV workspace. `sdpa` and `compiled-sdpa` are experimental benchmark/profiler options, not the adopted production default. |
| Is supervised training compiled? | [scripts/train.py](../../scripts/train.py):453 uses `torch.compile(model, dynamic=True, fullgraph=True)` when configured; training forward supplies a BlockMask and configured CUDA autocast. v4 requests BF16. |
| Is stage-2 learning compiled? | [trainer.py](../../src/imba_chess/self_play/trainer.py):105–116 creates the CUDA BlockMask, then directly calls the model. The runtime loads an eager FP32 model. The compiled collection runner is a separate wrapper. This training path does not itself compile the model or enter autocast. |
| Is this original HSTU attention? | No. The repository has softmax attention. SiLU on projected UVQK features is different from original HSTU's pointwise SiLU attention weighting. |

The local PyTorch implementation explicitly warns that uncompiled FlexAttention materializes the score matrix: `.venv/lib/python3.*/site-packages/torch/nn/attention/flex_attention.py`, around line 2599 in this environment. Compiling `create_block_mask` does not compile a later model's attention call. A future performance experiment should validate gradients and allocation behavior before adopting regional/whole-model compilation in stage-2 learning.

Configuration sources: [original lineage](../../config/imba_chess.toml), [v4](../../config/imba_chess_v4.toml), [default self-play](../../config/self_play.toml). Historical ckpt34 benchmark numbers must not be described as v4 measurements. The checkpoint loader builds the requested architecture from config and then loads weights strictly.

## Exact cache arithmetic

For L layers, H KV heads, key/value widths dₖ,dᵥ and b bytes/element:

`C = L × H × (dₖ + dᵥ) × b` bytes per temporal token across the stack.

| Architecture | FP32 | BF16 / FP16 |
|---|---:|---:|
| ckpt34 lineage: L=8, H=12, dₖ=dᵥ=64 | 49,152 bytes = 48 KiB | 24 KiB |
| v4: L=8, H=16, dₖ=dᵥ=64 | 65,536 bytes = 64 KiB | 32 KiB |

For 24 games × 256 prefix positions, raw FP32 prefix cache is 288 MiB / 384 MiB respectively. Retaining 128 unique branch entries per game adds 144 MiB / 192 MiB. These are illustrative assumptions, not an inventory of live allocations. Padded prefixes, suffix gathers, workspace, per-wave new K/V and allocator reservations add to these totals. The existing arena already removes repeated full-path storage.

Dense additive mask: `H × S² × b` **per layer**, where S is total packed tokens. With 16 heads and FP32, S=1024 costs 64 MiB, S=2048 costs 256 MiB, S=4096 costs 1 GiB. At v4's configured supervised token budget of 40,960, a hypothetical dense mask would be 100 GiB per layer; that configuration relies on the compiled sparse route. These numbers are not total forward/backward memory.

Full QK+AV prefill arithmetic before causal savings: approximately `2 L H Σ Tᵢ² (dₖ+dᵥ)` FLOPs. One-query cached attention: approximately `2 L H (T+suffix+1) (dₖ+dᵥ)` FLOPs per query. Neither formula includes UVQK/output projections, board encoding, softmax, readouts or data movement. The logical causal pair count is approximately half of full per-game squares, with practical tile overhead.

The calculator's GQA, latent, layer-sharing and FP4 rows are separate hypothetical storage estimates. They do not simulate accuracy, latency or the precise DeepSeek architecture. The latent example uses 256 channels and no extra positional cache because it is a proposed additive-bias chess design. DeepSeek's published architecture has different dimensions and positional components.

## Historical timing evidence

[Readiness review](../SELF_PLAY_READINESS_REVIEW_2026-09-11.md), “Measured laptop performance,” with supporting [analysis-summary.json](../../artifacts/self_play_validation/decoder_campaign/analysis-summary.json).

- RTX 3070 Ti Laptop, FP32, TF32 disabled, four Torch CPU threads, ckpt34, 24 concurrent slots, 128 simulations / 16 candidates / depth 32.
- Each final trial: 32 completed games, 2,775 searched positions, 2,767 eligible positions, 338,353 neural evaluations. Disk caches warmed; model loading excluded.
- Eager: 272.55 / 276.00 seconds; 36,319 eligible positions/hour median; 0.966 GB peak allocated.
- Compiled manual: 226.64 / 219.13 seconds; 44,705/hour; 0.966 GB.
- Compiled SDPA: 234.46 / 205.07 seconds; 45,530/hour; 1.113 GB. CUTLASS memory-efficient SDPA, not FlashAttention. ~1.85% incremental throughput gain did not meet the adoption gate.

The source records a 0.001105 maximum target-probability outlier for compiled SDPA. Matching root visits or game trajectories does not prove all internal search arithmetic was identical. Two trials are too few for a strong estimate of a small variable difference. These are collection measurements, not per-attention-call latency or strength tests.

[Maintenance review](../MAINTAINABILITY_REVIEW_2026-09-13.md) records the disposable stage-2 replay sweep: 1,024 tokens, 8,669 supervised exposures / 15 optimizer steps in 15.88 / 10.47 / 9.12 seconds, peak allocated 2.725 GB. At 2,048 tokens, backward ran out of memory. These values have a different workload and phase from the collection benchmark.

A new benchmark should use identical checkpoints/replay and shapes; isolate cold compilation from warmed trials; synchronize completed CUDA work or use `torch.utils.benchmark.Timer`; report median/dispersion and peak allocated/reserved memory; collect a profiler trace separately. Inspect actual backend names and score/bias allocations. Measure both model and end-to-end search time. Amdahl's law is a scenario calculation, not a device benchmark.

## DeepSeek report reading map

Primary source: the user-provided [DeepSeek-V4.1-Flash report](/home/vigi99/Downloads/DeepSeek_V41_Tech_Report.pdf), read directly using PDF text extraction and visual inspection of Figure 4. The report is titled **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression**. The [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) corroborates its identity and links the public report.

| Read | Learn |
|---|---|
| §1, p.4 | Distinguish global runtime KV (main KV + indexer K) from persistent offloaded prefix state; local SWA is bounded in length. |
| §2.2, p.9 | CED: upper global caches projected from final encoder output; local decoder computation still needs replay. |
| §2.3, pp.9–10 | Compress channels, time and layer ownership; index reuse alone does not save main KV storage. V4 CSA/HCA versus V4.1 CSA2. |
| §2.3.1, Fig.4, pp.10–11 | Full / Reindex / Reuse: separate cache ownership, index selection and fresh per-layer queries/local K/V. |
| §2.3.2, Fig.5, pp.11–12 | First decoder indexer scans visible global entries; later indexers select within its block-derived candidate pool. |
| §2.4.1–2.4.3, pp.12–14 | Single-Pass mHC, Engram and DSpark solve other problems. |
| §2.4.4, pp.14–15 | Main KV FP4 values with scale overhead, quantization-aware post-training; local SWA remains FP8. |
| §3.1.2, pp.17–18 | Cross-stage sharing during training needs ownership, gradient aggregation and correct shared-state lifetimes. |
| §3.2.2, p.20 | Bounded replay is explicitly approximate; it is not exact restoration of missing local states. |
| §4.2.1, pp.21–22 | Actual schedule: 40 layers, 20+20; local window 128; encoder m=2, decoder m=1; Top-512; later pool up to 16,384. |

The 890 bytes/token figure is reported **global KV across the network**, not total device memory or a promised chess cache size. Storage reductions, faster prefill, faster decode and model quality are distinct claims. Do not transfer DeepSeek's large-model benchmark ratios to a short-context chess bot.

Additional primary sources:

- [HSTU paper](https://arxiv.org/html/2402.17152v3): identify the original attention formulation.
- [DeepSeek-V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1): MLA and decoupled positional representation.
- [GQA paper](https://arxiv.org/abs/2305.13245): shared K/V groups with multiple query heads.
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/) and [SDPA reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention): API/kernel distinction.
- [FIDE Laws](https://handbook.fide.com/chapter/e012023), articles 9.2–9.6: repetition and move-count draw rules.

## Repetition counterexample, verified locally

From the starting board:

- A: `g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8`
- B: `g1f3 g8f6 f3g5 f6g4 g5f3 g4f6 f3g1 f6g8`

Both end at `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5` with the same previous move. python-chess verified every move and returned `True / False` respectively for both `is_repetition(3)` and `can_claim_threefold_repetition()`.

A current-position repetition count alone is insufficient to determine every future repetition possibility. Exact rule state includes occurrence information for relevant prior positions; search can keep this outside a neural sequence. Learned rule features and exact engine adjudication should have explicit, different contracts. The current model's bucketed halfmove feature is not an exact counter.

## Suggested experiments, not implementation changes

1. Confirm and fix the stage-2 unfused attention bottleneck in a disposable comparison: gradients, losses, time and memory.
2. Train current-only / short-history / full-history controls with explicit input contracts. Distinguish a per-layer window from a hard re-encoded last-k segment. Test truncation only as a diagnostic first; inference-only truncation is a distribution shift.
3. Isolate rule-feature changes, previous-move/absolute-position dependence and board encoder depth/readout changes. Do not bundle all of them and attribute results to history removal.
4. Measure policy CE/top-k, value Brier/calibration, tactical tasks, repetition/draw behavior, game-length robustness and path consistency on transpositions. Use held-out games and leakage-aware repeated-position splits.
5. Run paired strength evaluations at equal neural-evaluation budgets and equal wall-clock budgets, keeping opponents and search settings matched. Report uncertainty and compute/exposure differences.
6. Only if retained temporal attention still earns its cost, investigate GQA, latent caching or limited cross-layer sharing. All change model/cache contracts and need training or adaptation. Approximate FP4/replay needs its own accuracy campaign.

State-only softmax with one key is identically one: Q/K do not contribute to its mixture. A fair state-only model should spend its parameters usefully, such as on gated pointwise layers or deeper square attention. State-only inference still incurs encoder, projections and heads; zero temporal cache does not mean zero memory or latency.

## Artifact validation

The guide was exercised in local headless Chromium: slide navigation, read-all/presentation modes, mask counts, the one-key softmax result, cache arithmetic, invalid input recovery, v4's dense-mask guard and repetition case switching passed. No JavaScript runtime exceptions were reported. Layout checks found no horizontal overflow at 320, 375, 414, 768 and 1280 CSS pixels; screenshots were visually reviewed. The PDF export contains 25 landscape A4 pages with expanded notes. HTML IDs and local links were checked. No production training/evaluation tests or GPU performance runs were needed for this documentation-only task.
