# Cozy-Native Search Tree (Stage 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove python-chess from the search hot path: cozy-native encoder, cozy evaluator movegen, cozy-only tree with native terminal/repetition — per `docs/superpowers/specs/2026-07-18-cozy-native-tree-design.md`.

**Architecture:** Step 0 canonicalizes legal-move order (sort by UCI) so both movegens agree, resetting the byte-identity baseline. Then, harness-first: `encode_cozy` on `BoardStateEncoder`, native terminal/repetition in `cozy_bridge` (Zobrist chains + exact python-chess claim semantics), each differentially gated before `search.py`/`position_evaluator.py` cut over to a cozy-only tree in one final wiring task with the G=1 byte gate as judge.

**Tech Stack:** Python 3.13, cozy-chess-py 0.1.1 (already pinned), python-chess (oracle + interface + data pipeline), pytest.

## Global Constraints

- Public search API unchanged: `select_value_search_halving/d2/rerank` take a python-chess board + `legal_moves: list[chess.Move]`, return an index into the CALLER's list. `eval_vs_stockfish.py` must need zero changes.
- `search.py` and `batch_scheduler.py` stay torch-free.
- python-chess remains: the oracle in all differential tests, the data pipeline's `encode()` path (untouched), and the interface currency at script boundaries.
- No leftover flags/dead code at the end: `_dual_push`'s py half, tree `_search_copy` usage, per-edge `py_move_to_cozy`, and `terminal_value_fast`'s py-board fallback are all DELETED in the final wiring task. `IMBA_DUAL_PUSH_VERIFY` is retargeted or deleted there too (no orphan).
- `tests/test_search.py`'s test DOUBLES (fake evaluators) may be updated for the internal `PositionEval` type change; its BEHAVIORAL assertions (which move wins, budgets spent, elimination behavior) must not change.
- Test command: `.venv/bin/pytest` (baseline 198 passing). GPU runs: only where a step says so; controller runs them (user has approved GPU use this session); never two rollout processes at once.
- Rollout gate command (~60-90s):
  `.venv/bin/python scripts/generate_search_rollouts.py --config config/imba_chess_exit_full.toml --checkpoint "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt" --output-path <out> --max-games 20 --profile [--concurrent-games 1]`
  (defaults are fp32 G=8; pass `--concurrent-games 1` where a step says G=1).
- Commit messages end with: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`

## Verified API facts (probed 2026-07-18, do not re-derive)

- cozy `Board` exposes `halfmove_clock`, `fullmove_number`, `hash()` (Zobrist int), `same_position`, `is_legal(move)`, `castle_rights(color)`, `colors(color)`/`pieces(piece)` BitBoards (`int()` converts to u64).
- **cozy `en_passant()` returns the FILE even when no capturer exists** (FEN-style; probed: after 1.e4 from startpos it returns `e`). Production config is `en_passant = "legal"` (`config/imba_chess_exit_full.toml:26`) — `encode_cozy` must probe capture legality natively (see Task 2).
- `cc.File` has no int conversion — map via a dict built from `[cc.File.A..cc.File.H]`.
- python-chess move order ≠ cozy move order; Step 0 fixes this by sorting.
- The py root board passed by both drivers carries its real move stack (bounded copies happen only inside the tree), so the root hash-chain seed can be derived internally — no public API change.

---

### Task 1 (Step 0): canonical legal-move order + new baselines

**Files:**
- Modify: `src/imba_chess/eval/position_evaluator.py:215-240` (`_project_legal_logits`)
- Test: `tests/test_prefix_decode.py` or wherever `_project_legal_logits` is covered (survey), plus one new test

**Interfaces:**
- Produces: `_project_legal_logits` returns legal moves sorted by UCI string ascending (with `legal_logits` index-aligned to the sorted order). All later tasks assume canonical order.

- [ ] **Step 1: Write the failing test** (add to the file that already tests `_project_legal_logits` — survey with `grep -rn "_project_legal_logits" tests/`):

```python
def test_project_legal_logits_returns_moves_sorted_by_uci():
    import torch
    import chess
    from imba_chess.eval.position_evaluator import _project_legal_logits

    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    logits = torch.arange(4096, dtype=torch.float32)  # adapt size to vocab size used by other tests
    legal_logits, legal_moves, total, mapped = _project_legal_logits(
        logits=logits, board=board, move_vocab=..._make_vocab_like_other_tests(),
    )
    ucis = [m.uci() for m in legal_moves]
    assert ucis == sorted(ucis)
    # alignment: each row of legal_logits is the vocab logit of the SAME-index move
    for row, move in zip(legal_logits.tolist(), legal_moves):
        assert row == ..._vocab_id_of(move.uci())  # arange logits make id==logit
```
Adapt vocab construction to the conventions of the surrounding test file (they already build a MoveVocab). The alignment assertion is the important one — sorting moves without re-gathering logits is the bug this test exists to catch.

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest <that file> -k sorted_by_uci -v` → FAIL (order mismatch).

- [ ] **Step 3: Implement.** In `_project_legal_logits`, after building `legal_move_ids`/`legal_moves_with_ids`, sort jointly before the `index_select`:

```python
    order = sorted(range(len(legal_moves_with_ids)), key=lambda i: legal_moves_with_ids[i].uci())
    legal_moves_with_ids = [legal_moves_with_ids[i] for i in order]
    legal_move_ids = [legal_move_ids[i] for i in order]
```
(One sort per evaluated node, ~30 elements — negligible. Comment WHY: canonical order shared by python-chess and cozy movegen; Gumbel draws and prior-tie-breaks are index-based, so order is behavior.)

- [ ] **Step 4:** `.venv/bin/pytest -q` → all pass (some search tests may need no change since fakes bypass `_project_legal_logits` — verify).

- [ ] **Step 5 (GPU, controller): statistical gate + NEW baselines.** Run the rollout gate command twice: `--concurrent-games 1` → `step0_g1.parquet`, and defaults (G=8) → `step0_g8.parquet`. Compare `step0_g8` against the pre-change fp32 G=8 parquet (scratchpad `clean_default.parquet`): expect best-arm move agreement in the ~88-96% band with shared-arm |Δ backed_value| p99 ≤ 1e-4 (fp32; pure resampling relabels Gumbel draws, values on shared arms barely move). Record numbers. These two parquets are THE byte-identity baselines for Tasks 4-5; keep them.

- [ ] **Step 6: Commit** — `git add` the two touched files; message `feat: canonical UCI-sorted legal-move order (Stage 3 Step 0)` + trailer.

---

### Task 2: encode_cozy on BoardStateEncoder

**Files:**
- Modify: `src/imba_chess/data/board_state.py`
- Test: `tests/test_cozy_differential.py` (permanent harness)

**Interfaces:**
- Consumes: `cozy_bridge.board_to_cozy`, `py_move_to_cozy` (existing).
- Produces: `BoardStateEncoder.encode_cozy(cozy_board) -> BoardState`, exactly equal to `encode(py_board)` for the same position under the production `en_passant="legal"` mode (and `fen`/`xfen` modes too).

- [ ] **Step 1: Failing differential test** (append to `tests/test_cozy_differential.py`):

```python
def test_encode_cozy_matches_encode_on_conversions_and_played_lines():
    import random
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.models import BoardTokenConfig

    for mode in ("legal", "fen", "xfen"):
        enc = BoardStateEncoder(BoardTokenConfig(en_passant=mode))
        # Conversion equivalence on edge FENs + random boards
        for board in [chess.Board(f) for f in EDGE_FENS] + _random_boards(30, seed=21):
            assert vars(enc.encode(board)) == vars(enc.encode_cozy(board_to_cozy(board))), (mode, board.fen())
        # Played-line equivalence: cozy board reached via play(), NOT conversion —
        # catches ep-semantics drift (cozy reports the ep file even with no capturer).
        rng = random.Random(31)
        for _ in range(40):
            pyb = chess.Board()
            cb = board_to_cozy(pyb)
            for _ in range(rng.randrange(10, 80)):
                moves = list(pyb.legal_moves)
                if not moves:
                    break
                mv = rng.choice(moves)
                cb2 = __import__("copy").copy(cb)
                cb2.play(py_move_to_cozy(pyb, mv))
                pyb.push(mv)
                cb = cb2
                assert vars(enc.encode(pyb)) == vars(enc.encode_cozy(cb)), (mode, pyb.fen())
                if pyb.is_game_over():
                    break
```

- [ ] **Step 2:** Run it → FAIL with `AttributeError: ... encode_cozy`.

- [ ] **Step 3: Implement** in `board_state.py` (import `cozy_chess as cc` lazily/module-level; this file already has no torch):

```python
_CC_FILES = None  # built lazily: {cc.File.X: 0-based index}


def _cc_file_index(file_obj) -> int:
    global _CC_FILES
    if _CC_FILES is None:
        import cozy_chess as cc
        _CC_FILES = {f: i for i, f in enumerate(
            (cc.File.A, cc.File.B, cc.File.C, cc.File.D, cc.File.E, cc.File.F, cc.File.G, cc.File.H)
        )}
    return _CC_FILES[file_obj]
```

Inside `BoardStateEncoder`:

```python
    def _ep_file_id_cozy(self, board) -> int:
        import cozy_chess as cc
        ep_file = board.en_passant()
        if ep_file is None:
            return 0
        file_idx = _cc_file_index(ep_file)
        if self._ep_ok is None:              # "fen" mode: report as-is
            return file_idx + 1
        # cozy reports the file after ANY double push (FEN-style). "legal" and
        # "xfen" modes require an actual capturer; probe the <=2 candidate
        # en-passant captures. cozy only generates fully LEGAL moves, and its
        # is_legal() is exact (ep pins included) — which matches "legal" mode.
        # For "xfen" (pseudo-legal capturer exists), a legal capture implies a
        # pseudo-legal one; the reverse gap (pinned capturer) is the ep-pin
        # case — handle by checking pawn adjacency for xfen.
        stm = board.side_to_move()
        to_rank = "6" if stm == cc.Color.White else "3"
        from_rank = "5" if stm == cc.Color.White else "4"
        file_char = "abcdefgh"[file_idx]
        legal_capture = False
        adjacent_pawn = False
        pawns = int(board.colors(stm) & board.pieces(cc.Piece.Pawn))
        for adj in (file_idx - 1, file_idx + 1):
            if not 0 <= adj <= 7:
                continue
            from_sq_index = (int(from_rank) - 1) * 8 + adj
            if not (pawns >> from_sq_index) & 1:
                continue
            adjacent_pawn = True
            mv = cc.Move.from_str(f"{'abcdefgh'[adj]}{from_rank}{file_char}{to_rank}")
            if board.is_legal(mv):
                legal_capture = True
                break
        if self._ep_ok is chess.Board.has_legal_en_passant:
            return file_idx + 1 if legal_capture else 0
        return file_idx + 1 if adjacent_pawn else 0   # xfen: pseudo-legal capturer

    def encode_cozy(self, board) -> BoardState:
        import cozy_chess as cc
        cfg = self.config
        ids = [0] * 64
        white = int(board.colors(cc.Color.White))
        for offset, piece in (
            (0, cc.Piece.Pawn), (1, cc.Piece.Knight), (2, cc.Piece.Bishop),
            (3, cc.Piece.Rook), (4, cc.Piece.Queen), (5, cc.Piece.King),
        ):
            bb = int(board.pieces(piece))
            for square in chess.scan_forward(bb & white):
                ids[square] = offset + 1
            for square in chess.scan_forward(bb & ~white & bb):
                ids[square] = offset + 7
        rights_white = board.castle_rights(cc.Color.White)
        rights_black = board.castle_rights(cc.Color.Black)
        castle_id = (
            (1 if rights_white.short is not None else 0)
            | (2 if rights_white.long is not None else 0)
            | (4 if rights_black.short is not None else 0)
            | (8 if rights_black.long is not None else 0)
        )
        return BoardState(
            piece_ids=ids,
            turn_id=int(board.side_to_move() == cc.Color.Black),
            castle_id=castle_id,
            ep_file_id=self._ep_file_id_cozy(board),
            halfmove_bucket_id=_bucket(board.halfmove_clock, cfg.halfmove_max, cfg.halfmove_bucket_size),
            fullmove_bucket_id=_bucket(board.fullmove_number, cfg.fullmove_max, cfg.fullmove_bucket_size),
        )
```
SURVEY REQUIRED in-step: `castle_rights(color)` return shape (`.short`/`.long` File-or-None per the cozy-chess-py docs) and `halfmove_clock`/`fullmove_number` being methods vs properties — probe in a REPL and adapt (`board.halfmove_clock()` vs `.halfmove_clock`); `chess.scan_forward` works on any int, reuse it. If `xfen` adjacency semantics fail the differential test, the oracle (python-chess `has_pseudo_legal_en_passant`) governs — fix the implementation, never the test.

- [ ] **Step 4:** `.venv/bin/pytest tests/test_cozy_differential.py -q` then full suite → green.

- [ ] **Step 5: Commit** — `feat: cozy-native BoardStateEncoder.encode_cozy, differentially gated incl played lines` + trailer.

---

### Task 3: native terminal + repetition in cozy_bridge (not yet wired)

**Files:**
- Modify: `src/imba_chess/eval/cozy_bridge.py`
- Test: `tests/test_cozy_differential.py`

**Interfaces:**
- Consumes: cozy Board API (`status`, `hash`, `halfmove_clock`, pieces/colors bitboards).
- Produces:
  - `insufficient_material(cozy_board) -> bool` — exact python-chess `Board.is_insufficient_material()` semantics.
  - `terminal_value_native(cozy_board, *, color_is_stm: bool, hash_history: Sequence[int]) -> float | None` — drop-in semantics for today's `terminal_value_for_color(board, color=...)` where `hash_history` holds Zobrist hashes of PRIOR positions since the last irreversible move (most-recent-last; excludes the current position), and `color_is_stm` says whether the caller's `color` equals the board's side to move. Values: mate → ±1.0 by color, stalemate/insufficient/claimable draw → 0.0, else None.

- [ ] **Step 1: Failing differential tests** (append to harness):

```python
def test_insufficient_material_matches_python_chess():
    fens = [
        "8/8/3k4/8/8/3KB3/8/8 w - - 0 1",     # KB vs K -> True
        "8/8/3k4/8/8/3KN3/8/8 w - - 0 1",     # KN vs K -> True
        "8/8/3k4/8/8/3K4/8/8 w - - 0 1",      # K vs K -> True
        "8/2b5/3k4/8/8/3KB3/8/8 w - - 0 1",   # KB vs KB (check same/opposite bishops semantics vs oracle)
        "8/2n5/3k4/8/8/3KN3/8/8 w - - 0 1",   # KN vs KN -> oracle decides
        "8/8/3k4/8/8/3KP3/8/8 w - - 0 1",     # pawn -> False
        "8/8/3k4/8/8/2NKN3/8/8 w - - 0 1",    # two knights same side -> oracle decides
    ]
    from imba_chess.eval.cozy_bridge import insufficient_material
    for fen in fens:
        b = chess.Board(fen)
        assert insufficient_material(board_to_cozy(b)) == b.is_insufficient_material(), fen
    for board in _random_boards(60, seed=41):
        assert insufficient_material(board_to_cozy(board)) == board.is_insufficient_material(), board.fen()


def test_terminal_value_native_matches_oracle_on_replayed_games():
    import copy as copymod
    import random
    from imba_chess.eval.cozy_bridge import terminal_value_native
    from imba_chess.eval.search import terminal_value_for_color

    rng = random.Random(77)
    terminal_seen = draw_claims_seen = 0
    for g in range(400):
        pyb = chess.Board()
        cb = board_to_cozy(pyb)
        hash_history = []          # hashes of prior positions since last irreversible move
        for _ in range(220):
            moves = list(pyb.legal_moves)
            if not moves:
                break
            quiet = [m for m in moves if not pyb.is_capture(m) and m.promotion is None]
            mv = rng.choice(quiet if (quiet and rng.random() < 0.85) else moves)
            prev_hash = cb.hash()
            prev_halfmove = pyb.halfmove_clock
            cb2 = copymod.copy(cb)
            cb2.play(py_move_to_cozy(pyb, mv))
            pyb.push(mv)
            hash_history = ([] if pyb.halfmove_clock <= prev_halfmove else hash_history + [prev_hash])
            cb = cb2
            expected = terminal_value_for_color(pyb, color=pyb.turn)
            got = terminal_value_native(cb, color_is_stm=True, hash_history=hash_history)
            assert got == expected, (pyb.fen(), len(hash_history), expected, got)
            if expected is not None:
                terminal_seen += 1
                if expected == 0.0 and not pyb.is_stalemate() and not pyb.is_insufficient_material():
                    draw_claims_seen += 1
                break
    assert terminal_seen >= 30
    assert draw_claims_seen >= 5   # repetition/50-move path must actually be exercised
```
(If `draw_claims_seen` comes up short, raise game count / quiet bias — never weaken.) Note the oracle here is the CURRENT `terminal_value_for_color` (cozy fast path + py `outcome(claim_draw=halfmove>=7)` fallback) — semantics equality with it is exactly what lets Task 5 delete the fallback.

- [ ] **Step 2:** Run → FAIL with ImportError.

- [ ] **Step 3: Implement** in `cozy_bridge.py`:

```python
def insufficient_material(cozy_board: cc.Board) -> bool:
    """Exact python-chess Board.is_insufficient_material() semantics:
    True iff NEITHER side has a winning-material possibility — implemented as
    python-chess does via has_insufficient_material(color) for both colors:
    a side has sufficient material iff it has a pawn/rook/queen, or more than
    one minor piece, or (exactly one knight? sufficient per python-chess —
    SURVEY the oracle's has_insufficient_material and mirror it precisely;
    the differential test decides)."""
    ...


def terminal_value_native(cozy_board, *, color_is_stm, hash_history):
    status = cozy_board.status()
    if status == cc.GameStatus.Won:      # side to move is checkmated
        return -1.0 if color_is_stm else 1.0
    if status == cc.GameStatus.Drawn:    # stalemate
        return 0.0
    if _no_heavy_pieces(cozy_board) and insufficient_material(cozy_board):
        return 0.0
    halfmove = cozy_board.halfmove_clock
    if halfmove >= 100:                  # fifty-move claim (mate already excluded above)
        return 0.0
    if halfmove >= 7:                    # repetition claims impossible below (existing proven guard)
        current = cozy_board.hash()
        window = hash_history[-halfmove:] if halfmove < len(hash_history) else hash_history
        if sum(1 for h in window if h == current) >= 2:
            return 0.0                   # third occurrence reached
        # python-chess also allows claiming one reversible ply early: any legal
        # move that REACHES the third occurrence. Probe children (~200ns each).
        for move in cozy_board.generate_moves():
            child = copy.copy(cozy_board)
            child.play(move)
            if child.halfmove_clock == 0:
                continue                 # irreversible move cannot repeat
            child_hash = child.hash()
            if sum(1 for h in window + [current] if h == child_hash) >= 2:
                return 0.0
    return None
```
The `insufficient_material` body: read python-chess's `has_insufficient_material` source (`chess/__init__.py`) and transcribe its exact rule onto cozy bitboards (per-color: any pawn/rook/queen → sufficient; knights/bishops counting incl. the same-colored-bishops case). The differential sweep + fixtures are the judge. Also mirror python-chess's fifty-move claim precondition exactly — SURVEY `can_claim_fifty_moves` (it also requires a legal move to exist / handles mate-at-100 edge); the replayed-game test will expose any mismatch. Careful with the repetition window: python-chess compares transposition keys within the reversible-move window on the move stack; the `hash_history` contract (reset on irreversible, prior positions only) mirrors it — but the oracle test on replayed games is the final word; adjust the window logic to match the oracle, never the test.

- [ ] **Step 4:** Harness green, full suite green.

- [ ] **Step 5: Commit** — `feat: native cozy terminal detection incl. repetition/fifty-move claims, oracle-gated` + trailer.

---

### Task 4: cozy-native evaluator

**Files:**
- Modify: `src/imba_chess/eval/position_evaluator.py` (`_project_legal_logits` → cozy variant; `CachedPositionEvaluator.build_decode_request/consume_decode_result/extend`; `_select_model_move`/root paths — survey callers)
- Modify: `src/imba_chess/eval/search.py` (only the `PositionEval`/`PositionEvaluator` type docs — moves become cozy internally)
- Test: existing suites + `tests/test_search.py` doubles updated (behavioral assertions unchanged)

**Interfaces:**
- Consumes: `encode_cozy` (Task 2), canonical order (Task 1), `cozy_move_to_uci` (existing).
- Produces: `_project_legal_logits_cozy(*, logits, cozy_board, move_vocab) -> (legal_logits, legal_moves: list[cc.Move], legal_ucis: list[str], total, mapped)` — UCI-sorted; `CachedPositionEvaluator.evaluate(batch: list[(handle, cozy_board)])` returning `PositionEval(value_stm, legal_moves: list[cc.Move], legal_ucis: list[str], legal_log_priors)`; `extend(handle, uci: str)` (or keeps `(handle, board_before, move)` shape with cozy args — pick ONE and survey all call sites; `extend` only needs the uci for vocab encoding, so the simplest honest signature is `(handle, move_uci: str)` — update the Protocol in `search.py`, all call sites, and test doubles together).

Key decisions locked here:
- `PositionEval` gains `legal_ucis` aligned with `legal_moves` (computed once via `cozy_move_to_uci` during projection) so search/rows never re-derive UCI strings (kills the 7.5M `uci()` calls).
- Root boundary: the sync wrappers translate the CALLER's py `legal_moves` list to `(cozy_move, uci)` pairs sorted the same canonical way, asserting the uci set equals the evaluator's root projection — index mapping back to the caller's list is by uci lookup built once per call.
- The py `_project_legal_logits` is DELETED in this task (its only consumers move to the cozy variant) unless a survey finds a caller that cannot switch yet — if so, STOP and report (plan bug), don't fork.

Steps: survey all `_project_legal_logits` / `.extend(` / `PositionEval(` call sites (`grep -rn` across src/tests/scripts); update `tests/test_search.py` doubles to produce cozy moves + ucis (assertions unchanged); implement; `.venv/bin/pytest -q` green; commit `refactor: cozy-native evaluator (movegen, vocab mapping, encode_cozy)` + trailer. This task intentionally does NOT change the tree — `search.py` still holds dual boards; `evaluate` receives the node's cozy board (already threaded since Stage 2), so this task is independently shippable and byte-gateable: **GPU gate (controller): G=1 rollout vs `step0_g1.parquet` must be byte-identical** (same movegen result post-Step-0, same batches).

---

### Task 5: cozy-only tree, native terminal wiring, deletions

**Files:**
- Modify: `src/imba_chess/eval/search.py` (tree structures, `_dual_push` → `_cozy_push`, terminal wiring, root hash seed)
- Modify: `src/imba_chess/eval/cozy_bridge.py` (delete `terminal_value_fast`'s py-board fallback path if superseded)
- Test: `tests/test_search.py` (doubles only), `tests/test_search_stepwise.py`, harness

**Interfaces:**
- Consumes: `terminal_value_native` (Task 3), Task 4's evaluator shapes.
- Produces: `_TreeNode`/`_RootCandidate` carry `cozy_board` + `hash_history: tuple[int, ...]` and NO py board; `_cozy_push(cozy_board, cozy_move, hash_history) -> (child_board, child_history)` (append parent hash if the child's `halfmove_clock` didn't reset, else empty); `_root_hash_seed(board: chess.Board) -> list[int]` (replay the last `min(halfmove_clock, len(move_stack))` stack moves on a cozy board reconstructed from the earlier position, collecting hashes; empty stack → empty seed, matching today's stackless behavior); `terminal_value_for_color(board, *, color, cozy_board=None)` public shim RETAINED for external callers (tests/harness) but internally search calls `terminal_value_native` directly with per-node histories.

Deletions in this task (grep-clean checks in-step): py half of `_dual_push` (+ rename to `_cozy_push`), `_search_copy` tree usage (keep only if `_root_hash_seed` needs a bounded copy — otherwise delete), per-edge `py_move_to_cozy` (root-only remains), `terminal_value_fast(cozy, py, color)`'s py fallback (`terminal_value_for_color` shim now wraps `terminal_value_native` + `_root_hash_seed`), `IMBA_DUAL_PUSH_VERIFY` → replaced by `IMBA_COZY_TREE_VERIFY` asserting `child.hash()` consistency against a py-chess replay oracle in the same opt-in style (keep the permanent sync test in the harness, retargeted).

Gate sequence (controller, GPU):
1. `.venv/bin/pytest -q` green.
2. **G=1 rollout vs `step0_g1.parquet`: byte-identical** (the ultimate judge — identical movegen order + identical terminal semantics + identical batches).
3. G=8 rollout vs `step0_g8.parquet`: byte-identical (same reasoning under merged batches).
4. `eval_vs_stockfish` smoke: ~4 games at small budget (survey its CLI; e.g. `--max-games 4` equivalent) — runs to completion, no crash; score not asserted (Stockfish nondeterminism).
Commit `perf: cozy-only search tree with native terminal/repetition (Stage 3 cutover)` + trailer.

---

### Task 6: profile re-measurement + docs

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-cozy-native-tree-design.md` (append Results)
- Modify: memory `imba-chess-elo-goal.md` (status line; not committed)

Steps: 20-game fp32 G=8 `--profile` run + cProfile 10-game decomposition (expect `generate_legal_moves`/`uci`/py-`push`/`py_move_to_cozy` gone from the top table); append Results with per-gate outcomes + before/after profile tables + s/game; note remaining top CPU items to inform any Stage-4 discussion; full suite; commit docs `docs: Stage 3 results` + trailer; update memory status line.
