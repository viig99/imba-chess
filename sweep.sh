for l in 0.0 0.15 0.35 0.7 1.5 3.0; do
  python scripts/eval_vs_stockfish.py \
    --checkpoint artifacts/checkpoints/best_hr10_checkpoint_2_hr10=0.8902.pt \
    --model-move-policy value_rerank \
    --value-rerank-top-k 8 \
    --value-rerank-lambda "$l" \
    --seed 42 \
    --no-compile \
    --debug-trace-games 0 --debug-topk 0 \
    --ladder-elos 1380 \
    --ladder-games-per-segment 40 \
    --no-include-full-strength-segment \
    --output-json "artifacts/eval/value_rerank_elo1380_g40_l${l}.json" \
    2>&1 | tee -a logs.txt
done

# Best params found per 0.35 and 0.7