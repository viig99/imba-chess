for l in 0.05 0.1 0.2; do
  python scripts/eval_vs_stockfish.py \
    --checkpoint "artifacts/checkpoints/best_hr10_checkpoint_5_hr10=0.9208.pt" \
    --model-move-policy value_search_d2 \
    --value-rerank-top-k 16 \
    --value-rerank-lambda "$l" \
    --seed 42 \
    --no-compile \
    --opening-random-plies=0 \
    --debug-trace-games 0 --debug-topk 0 \
    --ladder-elos 1400 \
    --ladder-games-per-segment 100 \
    --no-include-full-strength-segment \
    --output-json "artifacts/eval/value_search_d2_elo1400_g100_k16_l${l}.json" \
    2>&1 | tee -a logs.txt
done

# Value-dominant scoring: score = worst_reply_value + lambda * log_prob(move).
# lambda is now the weight of the policy log-prob prior (sensible range 0.05-0.2).
