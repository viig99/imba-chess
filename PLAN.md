## State-Conditioned STU Transformer for Chess + RL Self-Play

---

# 1. Project Goal

Build a compact (~20–30M parameter) state-conditioned STU transformer that:

1. Pretrains on Lichess PGNs to predict the next legal UCI move.
2. Fine-tunes via RL (PPO / GRPO) using:

   * Parallel self-play
   * Optional play vs Stockfish / Leela
3. Evaluates via Elo and win % against engines.
4. Runs efficiently on a single RTX 4090.

---

# 2. High-Level Architecture

Each timestep (ply) is treated as a structured event:

```
e_t = f(
    board_state_t,
    previous_move_{t-1},
    time_taken_t,
    metadata
)
```

The STU backbone runs causally over `e_1 … e_T` and predicts `move_t`.

---

# 3. Data Pipeline

## 3.1 Dataset

Source:

* [https://database.lichess.org/](https://database.lichess.org/)
* Standard rated games only

Filter:

* Exclude corrupted / aborted games
* Optional Elo filtering
* Extract clock comments if available (`[%clk]`)

---

## 3.2 Replay Engine

Use `python-chess`.

For each game:

* Initialize `board = chess.Board()`
* Maintain a `piece_ids` buffer (64 bytes)
* Incrementally update buffer per move
* Extract global state fields per ply

---

## 3.3 Board State Representation

### Per Position (ply t)

Structured state:

* `piece_ids`: 64 bytes
* `turn_id`: 0/1
* `castle_id`: 0..15 (bitmask KQkq)
* `ep_file_id`: 0..8
* `halfmove_bucket_id`
* `fullmove_bucket_id`

All derived directly from `python-chess` board.

---

## 3.4 Move Representation

* Convert SAN → UCI during ingestion
* Maintain vocabulary of all legal UCI moves
* Use legality masking during training and inference

---

## 3.5 Time Features

If clock available:

* `dt_bucket_id` (log2 buckets)
* Optional: remaining time bucket

If not available:

* Skip initially

---

## 3.6 Metadata Features

Optional but planned:

* Player ID → QR embeddings
* Elo bucket
* Time control bucket
* Opening/ECO bucket

Cold-start:

* Randomly replace some player IDs with UNK

---

# 4. Model Architecture

## 4.1 Embedding Layers

* `E_piece (13, d)`
* `E_square (64, d)`
* `E_turn (2, d)`
* `E_castle (16, d)`
* `E_ep (9, d)`
* `E_halfmove`
* `E_fullmove`
* `E_move`
* `E_time`
* `E_player` (QR hashing)
* `E_elo`

---

## 4.2 Board Embedding

Option A (baseline):

```
sq_emb = E_piece(piece_ids) + E_square(index)
board_emb = mean(sq_emb)
```

Option B (stronger):

* Add BOARD_CLS token
* 1–2 self-attention layers over 65 tokens
* Use CLS output

Start with Option A.

---

## 4.3 Event Construction

```
event_t = LN(
    W concat(
        board_emb,
        move_emb_{t-1},
        time_emb,
        meta_emb
    )
)
```

Feed `event_t` into STU backbone.

---

## 4.4 STU Backbone

* Causal sequence modeling
* Relative positional bias (ply index)
* Optional additional bias:

  * wall-clock importance bias
  * recency bias

Flex attention:

* Use 1D dynamic batching
* Support variable sequence lengths

Target size:

* 20–30M params

---

# 5. Training Phases

---

# 5.1 Stage 1: Supervised Pretraining

Objective:

* Cross-entropy on next legal UCI move

Loss:

```
CE(logits_masked, move_target)
```

Optional:

* Value head (predict outcome)
* Auxiliary engine eval head

Regularization:

* UNK-player masking
* Metadata dropout

Goal:

* Strong imitation policy

---

# 5.2 Stage 2: RL Fine-Tuning

Environment:

* gym-chess
* pufferlib parallel rollouts (~1000 workers)

Policy:

* Same STU model
* Add value head V(s)

Reward:

* +1 win
* 0 draw
* -1 loss
* Optional shaping (engine eval delta)

Algorithm:

* PPO or KL-regularized PPO
* Possibly GRPO-style grouped updates

League:

* Self-play vs current + past checkpoints
* Optional Stockfish/Leela matches

---

# 6. Evaluation

Metrics:

* Win % vs Stockfish (fixed time or nodes)
* Win % vs Leela
* Elo estimate (confidence intervals)
* Illegal move rate (~0)
* Blunder rate vs engine
* Top-k move accuracy on held-out PGNs

---

# 7. Performance Targets

* 20–30M parameters
* ~1000 parallel rollouts
* Single RTX 4090
* Efficient incremental board encoding
* Avoid rebuilding state per ply

---

# 8. Engineering Priorities

### Critical

* Incremental board update (avoid piece_map per ply)
* Legality masking from same board instance
* Packed state format (64 bytes + globals)
* Efficient dataloader (Parquet / binary shards)

### Secondary

* QR embeddings for players
* Relative time bias
* Importance bias on old moves
* VQ compression experiments

---

# 9. Milestones

1. Implement incremental state extractor
2. Build dataset shards from Lichess
3. Train small STU baseline (no metadata)
4. Validate legality masking
5. Scale to 20–30M params
6. Integrate pufferlib
7. Run PPO self-play
8. Benchmark vs Stockfish

---

# 10. Future Experiments

* Compare:

  * sequence-only vs state-conditioned
  * with/without QR player embeddings
  * with/without clock features
* Scaling laws (model size vs Elo)
* KL stabilization vs pure PPO
* Flex attention vs standard attention

---

If you'd like, I can also generate:

* A directory layout scaffold
* A `TODO.md` broken into executable tickets
* A minimal starter training loop template
* A recommended dataset schema for Parquet / binary shards