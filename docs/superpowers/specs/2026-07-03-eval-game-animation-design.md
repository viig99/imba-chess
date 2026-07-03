# Eval Game Replay & Animation — Design

## Purpose

`scripts/eval_vs_stockfish.py` plays games against Stockfish but discards the
move history once a game ends — only aggregate W/D/L stats and (for the first
`debug_trace_games` games per segment) console debug traces survive. There is
no way to go back and actually look at what happened in a specific game.

This adds:

1. Saving the move history of traced games as standard PGN.
2. A small, dependency-free HTML animator to step/play through a saved game
   in a browser, with a header showing who played which color and against
   what Stockfish strength.

Scope is intentionally limited to the games already covered by
`debug_trace_games` (default 3 per segment) — the same games you already get
console traces for today. Showing which alternative moves the model
considered (from `_select_model_move`'s `debug_info`) is an explicit
follow-up, not part of this design.

## 1. Capturing game history

`_run_segment`'s game loop already accumulates the full move stack on `board`
via `board.push(move)`. No new move-tracking is needed during play.

When a traced game finishes (`game_idx < debug_trace_games`), build a PGN
from the finished board using python-chess's built-in helper:

```python
game = chess.pgn.Game.from_board(board)
game.headers["White"] = "imba-chess" if model_color == chess.WHITE else stockfish_label
game.headers["Black"] = stockfish_label if model_color == chess.WHITE else "imba-chess"
game.headers["Result"] = board.result(claim_draw=True) if completed else "*"
game.headers["ModelColor"] = "white" if model_color == chess.WHITE else "black"
game.headers["StockfishLimitStrength"] = str(spec.limit_strength)
game.headers["StockfishElo"] = str(spec.elo) if spec.elo is not None else "full_strength"
game.headers["Segment"] = segment_name
```

`stockfish_label` is e.g. `"Stockfish (elo=1400)"` or `"Stockfish (full strength)"`.

`chess.pgn.Game.from_board` walks `board.move_stack` from the initial
position and reconstructs the full mainline — this is a few lines, not a
manual move-tree build. PGN is kept as a durable artifact regardless of the
HTML animator: it opens in any standard chess GUI or lichess's PGN importer
as a fallback viewer.

## 2. Animator (`GameAnimator`)

New module: `src/imba_chess/eval/game_animation.py`.

```python
class GameAnimator:
    def render_html(self, game: chess.pgn.Game, *, metadata: dict[str, str]) -> str: ...
    def save(self, path: Path, game: chess.pgn.Game, *, metadata: dict[str, str]) -> None: ...
```

Implementation:

- Replay `game.mainline_moves()` on a fresh `chess.Board()`.
- For each ply, render `chess.svg.board(board, size=400, lastmove=move, coordinates=True)`
  — a built-in python-chess function returning a self-contained SVG string.
  No new dependency, no external image/CDN fetch.
- Collect per-ply `(svg, san, ply_number, side_to_move)` into a JS array
  embedded directly in one `.html` file.

The generated page is plain HTML/CSS/JS, no CDN, works fully offline (opens
directly in a browser, or over `rsync` from a remote box):

- **Header bar**: model's color, Stockfish strength (elo or "full strength"),
  segment name, result, ply count — sourced from the PGN headers passed in
  as `metadata`.
- **Board**: swaps the currently visible SVG frame.
- **Controls**: prev/next ply buttons, Play/Pause (auto-advance ~600ms/ply),
  and a scrubber (`<input type=range>`) over all plies.
- **Move list panel**: clickable two-column (White/Black) SAN move list;
  clicking a move jumps the board to that ply.

### Explicit extension point (not built now)

`_select_model_move` already computes `debug_info` (topk legal moves,
`value_rerank_candidates`, `value_search_d2_candidates`) for these same
traced games, currently only printed to console. A future iteration can pass
this through as an optional `candidates_by_ply: dict[int, list[dict]]` to
`GameAnimator`, to overlay "moves considered but not played" per ply. Not
implemented in this design.

## 3. Config, file naming, wiring

### Config (`config/imba_chess.toml`, `[eval_vs_stockfish]`)

```toml
save_games = true
save_games_dir = "artifacts/eval/games"
```

### CLI overrides (`scripts/eval_vs_stockfish.py`)

- `--save-games` / `--no-save-games` (`argparse.BooleanOptionalAction`,
  matching the existing `--compile` pattern).
- `--save-games-dir` (`Path`).

### File naming

One PGN + one HTML per traced game, written into `save_games_dir`:

```
{segment_name}_game{game_idx+1:03d}_{outcome}.pgn
{segment_name}_game{game_idx+1:03d}_{outcome}.html
```

`outcome` is one of `model_win` / `model_loss` / `draw` / `incomplete`, so a
plain directory listing already indicates which games are worth opening.

### Wiring point

Inside `_run_segment`, immediately after the existing
`result = board.result(...)` / `_update_summary(...)` calls: if
`game_idx < debug_trace_games and save_games`, build the PGN (Section 1) and
call `GameAnimator().save(...)` (Section 2). The move-selection loop itself
is untouched.

### Overwrite behavior

Files overwrite on repeated runs using the same `segment_name` (no
timestamp) — simplest behavior for a debugging tool iterating on one
checkpoint at a time. Pass a different `--save-games-dir` to keep multiple
runs side by side.

## Out of scope

- Showing considered-but-not-played moves (see extension point above).
- Saving all 100 games (only the `debug_trace_games` subset, matching
  existing console tracing scope).
- Any server process or external JS library — everything is a static,
  self-contained HTML file.
