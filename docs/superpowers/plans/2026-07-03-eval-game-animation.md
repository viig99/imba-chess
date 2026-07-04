# Eval Game Replay & Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the traced games from `scripts/eval_vs_stockfish.py` as PGN and render them as a self-contained, offline HTML replay viewer, so a specific eval game can be watched move-by-move instead of only reflected in aggregate stats.

**Architecture:** `_run_segment`'s existing game loop already accumulates the full move stack on `board` for the games already covered by `debug_trace_games`. When such a game finishes, build a `chess.pgn.Game` from that board (`chess.pgn.Game.from_board`), tag it with match metadata (colors, Stockfish strength, segment, result), write it as `.pgn`, and hand it to a new `GameAnimator` class that pre-renders every ply as an SVG frame (via `chess.svg.board`) and embeds them in one static `.html` file with play/pause/scrub controls and a clickable move list. No server, no CDN, no new dependency — `python-chess` (already a dependency) provides both PGN and SVG rendering.

**Tech Stack:** Python, `python-chess` (`chess.pgn`, `chess.svg`), plain HTML/CSS/vanilla JS embedded as a template string, `pytest`.

## Global Constraints

- No new third-party dependencies — use only `python-chess` (already in `pyproject.toml`) and the Python standard library.
- The generated HTML must be a single self-contained file: no external CDN scripts/styles/fonts, everything inline.
- Only games already covered by `debug_trace_games` are saved (matches existing console-tracing scope) — not all games in a run.
- Files are named `{segment_name}_game{N:03d}_{outcome}.{pgn,html}` and overwrite on repeated runs with the same segment name (no timestamping).
- Config keys live under `[eval_vs_stockfish]` in `config/imba_chess.toml`, following the existing config-with-CLI-override pattern in `scripts/eval_vs_stockfish.py`.

---

### Task 1: Add `save_games` / `save_games_dir` config fields

**Files:**
- Modify: `src/imba_chess/config.py:142-144` (end of `EvalVsStockfishConfig`)
- Modify: `config/imba_chess.toml:121-123` (end of `[eval_vs_stockfish]` section)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `EvalVsStockfishConfig.save_games: bool` (default `True`), `EvalVsStockfishConfig.save_games_dir: str` (default `"artifacts/eval/games"`) — consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, change the import line at the top from:

```python
from imba_chess.config import load_repo_config
```

to:

```python
from imba_chess.config import EvalVsStockfishConfig, load_repo_config
```

Then append these two tests to the end of the file:

```python
def test_eval_vs_stockfish_config_game_saving_defaults():
    config = EvalVsStockfishConfig()
    assert config.save_games is True
    assert config.save_games_dir == "artifacts/eval/games"


def test_load_repo_config_reads_eval_vs_stockfish_game_saving_fields(tmp_path):
    config_path = tmp_path / "imba_chess.toml"
    config_path.write_text(
        """
[eval_vs_stockfish]
debug_topk = 5
save_games = false
save_games_dir = "artifacts/eval/custom_games"
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_repo_config(config_path)
    assert config.eval_vs_stockfish.debug_topk == 5
    assert config.eval_vs_stockfish.save_games is False
    assert config.eval_vs_stockfish.save_games_dir == "artifacts/eval/custom_games"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: both new tests FAIL — `test_eval_vs_stockfish_config_game_saving_defaults` with `AttributeError` (or `TypeError` from the import), and `test_load_repo_config_reads_eval_vs_stockfish_game_saving_fields` with `ValueError: Unknown keys in [eval_vs_stockfish]` (since `save_games`/`save_games_dir` aren't recognized fields yet).

- [ ] **Step 3: Add the fields to `EvalVsStockfishConfig`**

In `src/imba_chess/config.py`, find:

```python
    debug_trace_games: int = 0
    debug_trace_max_plies: int = 80
    debug_topk: int = 5
```

Replace with:

```python
    debug_trace_games: int = 0
    debug_trace_max_plies: int = 80
    debug_topk: int = 5
    save_games: bool = True
    save_games_dir: str = "artifacts/eval/games"
```

- [ ] **Step 4: Document the new keys in the checked-in config**

In `config/imba_chess.toml`, find the end of the `[eval_vs_stockfish]` section:

```toml
debug_trace_games = 3
debug_trace_max_plies = 80
debug_topk = 5
```

Replace with:

```toml
debug_trace_games = 3
debug_trace_max_plies = 80
debug_topk = 5
save_games = true
save_games_dir = "artifacts/eval/games"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, including the two new ones).

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/config.py config/imba_chess.toml tests/test_config.py
git commit -m "feat: add save_games config fields for eval game replay"
```

---

### Task 2: `GameAnimator` — render a PGN game as a self-contained HTML replay

**Files:**
- Create: `src/imba_chess/eval/game_animation.py`
- Test: `tests/test_game_animation.py`

**Interfaces:**
- Consumes: standard `chess.pgn.Game` objects (from `chess.pgn.Game.from_board`, python-chess).
- Produces: `GameAnimator.render_html(game: chess.pgn.Game, *, metadata: dict[str, str]) -> str` and `GameAnimator.save(path: Path, game: chess.pgn.Game, *, metadata: dict[str, str]) -> None`. `metadata` must contain string keys `"white"`, `"black"`, `"result"`, `"segment"`, `"ply_count"`. Consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_game_animation.py`:

```python
from __future__ import annotations

import json

import chess
import chess.pgn

from imba_chess.eval.game_animation import GameAnimator


def _short_game() -> chess.pgn.Game:
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(move_uci)
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "imba-chess"
    game.headers["Black"] = "Stockfish (elo=1400)"
    game.headers["Result"] = "*"
    return game


def _metadata() -> dict[str, str]:
    return {
        "white": "imba-chess",
        "black": "Stockfish (elo=1400)",
        "result": "*",
        "segment": "sf_elo_1400",
        "ply_count": "3",
    }


def _extract_frames(html: str) -> list[dict]:
    start = html.index("const FRAMES = ") + len("const FRAMES = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def test_render_html_embeds_frames_moves_and_metadata():
    game = _short_game()
    html = GameAnimator().render_html(game, metadata=_metadata())

    assert html.startswith("<!doctype html>")
    assert "imba-chess vs Stockfish (elo=1400)" in html
    assert "Segment: sf_elo_1400" in html
    assert "Plies: 3" in html

    frames = _extract_frames(html)
    assert len(frames) == 4  # start position + 3 plies
    assert frames[0]["san"] is None
    assert [frame["san"] for frame in frames[1:]] == ["e4", "e5", "Nf3"]
    assert all("<svg" in frame["svg"] for frame in frames)


def test_render_html_handles_game_with_no_moves():
    board = chess.Board()
    game = chess.pgn.Game.from_board(board)
    metadata = {
        "white": "imba-chess",
        "black": "Stockfish (full strength)",
        "result": "*",
        "segment": "sf_full_strength",
        "ply_count": "0",
    }

    html = GameAnimator().render_html(game, metadata=metadata)

    frames = _extract_frames(html)
    assert len(frames) == 1
    assert frames[0]["san"] is None


def test_save_writes_html_file_and_creates_parent_dirs(tmp_path):
    game = _short_game()
    metadata = _metadata()
    output_path = tmp_path / "nested" / "game.html"

    GameAnimator().save(output_path, game, metadata=metadata)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == GameAnimator().render_html(
        game, metadata=metadata
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_game_animation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.eval.game_animation'`.

- [ ] **Step 3: Implement `GameAnimator`**

Create `src/imba_chess/eval/game_animation.py`:

```python
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import chess.svg

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1e1e1e; color: #eee; margin: 0; padding: 24px; }
  .header { margin-bottom: 16px; }
  .header .title { font-size: 1.25rem; font-weight: 600; }
  .header .meta { color: #aaa; margin-top: 4px; }
  .layout { display: flex; gap: 24px; flex-wrap: wrap; }
  .board-panel { flex: 0 0 auto; }
  #board svg { width: 400px; height: 400px; display: block; }
  .controls { margin-top: 12px; display: flex; gap: 8px; align-items: center; }
  .controls button { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 6px 10px; cursor: pointer; }
  .controls button:hover { background: #444; }
  #slider { flex: 1; min-width: 200px; }
  .moves-panel { flex: 1 1 240px; max-height: 460px; overflow-y: auto; background: #262626; border-radius: 6px; padding: 12px; }
  .move-row { display: grid; grid-template-columns: 2.5em 1fr 1fr; gap: 8px; padding: 2px 0; }
  .move-row .num { color: #888; }
  .move-cell { cursor: pointer; padding: 2px 6px; border-radius: 3px; }
  .move-cell:hover { background: #3a3a3a; }
  .move-cell.active { background: #4a6a8a; color: #fff; }
</style>
</head>
<body>
__HEADER_HTML__
<div class="layout">
  <div class="board-panel">
    <div id="board"></div>
    <div class="controls">
      <button id="btnFirst">|&lt;</button>
      <button id="btnPrev">&lt;</button>
      <button id="btnPlay">Play</button>
      <button id="btnNext">&gt;</button>
      <button id="btnLast">&gt;|</button>
      <input type="range" id="slider" min="0" value="0">
    </div>
    <div id="plyLabel" style="margin-top: 6px; color: #aaa;"></div>
  </div>
  <div class="moves-panel" id="movesPanel"></div>
</div>
<script>
const FRAMES = __FRAMES_JSON__;

const boardEl = document.getElementById("board");
const sliderEl = document.getElementById("slider");
const plyLabelEl = document.getElementById("plyLabel");
const movesPanelEl = document.getElementById("movesPanel");
const btnPlay = document.getElementById("btnPlay");

let idx = 0;
let playTimer = null;

function render() {
  boardEl.innerHTML = FRAMES[idx].svg;
  sliderEl.value = String(idx);
  const frame = FRAMES[idx];
  plyLabelEl.textContent = idx === 0
    ? "Start position"
    : `Ply ${idx}: ${frame.san} (${frame.side_to_move} to move)`;
  document.querySelectorAll(".move-cell").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.ply) === idx);
  });
}

function stopPlaying() {
  if (playTimer !== null) {
    clearInterval(playTimer);
    playTimer = null;
    btnPlay.textContent = "Play";
  }
}

function goTo(newIdx) {
  idx = Math.max(0, Math.min(FRAMES.length - 1, newIdx));
  render();
}

document.getElementById("btnFirst").addEventListener("click", () => { stopPlaying(); goTo(0); });
document.getElementById("btnPrev").addEventListener("click", () => { stopPlaying(); goTo(idx - 1); });
document.getElementById("btnNext").addEventListener("click", () => { stopPlaying(); goTo(idx + 1); });
document.getElementById("btnLast").addEventListener("click", () => { stopPlaying(); goTo(FRAMES.length - 1); });
sliderEl.addEventListener("input", (event) => { stopPlaying(); goTo(Number(event.target.value)); });

btnPlay.addEventListener("click", () => {
  if (playTimer !== null) {
    stopPlaying();
    return;
  }
  btnPlay.textContent = "Pause";
  playTimer = setInterval(() => {
    if (idx >= FRAMES.length - 1) {
      stopPlaying();
      return;
    }
    goTo(idx + 1);
  }, 600);
});

sliderEl.max = String(FRAMES.length - 1);

let moveRowHtml = "";
for (let ply = 1; ply < FRAMES.length; ply += 2) {
  const moveNumber = Math.ceil(ply / 2);
  const whiteFrame = FRAMES[ply];
  const blackFrame = FRAMES[ply + 1];
  const whiteCell = `<span class="move-cell" data-ply="${ply}">${whiteFrame.san}</span>`;
  const blackCell = blackFrame ? `<span class="move-cell" data-ply="${ply + 1}">${blackFrame.san}</span>` : "";
  moveRowHtml += `<div class="move-row"><span class="num">${moveNumber}.</span>${whiteCell}${blackCell}</div>`;
}
movesPanelEl.innerHTML = moveRowHtml;
movesPanelEl.querySelectorAll(".move-cell").forEach((el) => {
  el.addEventListener("click", () => { stopPlaying(); goTo(Number(el.dataset.ply)); });
});

render();
</script>
</body>
</html>"""


class GameAnimator:
    """Renders a chess.pgn.Game as a self-contained, offline HTML replay viewer."""

    def render_html(self, game: chess.pgn.Game, *, metadata: dict[str, str]) -> str:
        frames = self._build_frames(game)
        frames_json = json.dumps(frames).replace("</", "<\\/")
        title = escape(f"{metadata['white']} vs {metadata['black']}")
        header_html = _render_header(metadata)
        return (
            _PAGE_TEMPLATE.replace("__TITLE__", title)
            .replace("__HEADER_HTML__", header_html)
            .replace("__FRAMES_JSON__", frames_json)
        )

    def save(
        self, path: Path, game: chess.pgn.Game, *, metadata: dict[str, str]
    ) -> None:
        page_html = self.render_html(game, metadata=metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page_html, encoding="utf-8")

    def _build_frames(self, game: chess.pgn.Game) -> list[dict[str, Any]]:
        board = game.board()
        frames: list[dict[str, Any]] = [
            {
                "svg": chess.svg.board(board, size=400, coordinates=True),
                "san": None,
                "ply": 0,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
            }
        ]
        for move in game.mainline_moves():
            san = board.san(move)
            board.push(move)
            frames.append(
                {
                    "svg": chess.svg.board(
                        board, size=400, lastmove=move, coordinates=True
                    ),
                    "san": san,
                    "ply": len(frames),
                    "side_to_move": "white" if board.turn == chess.WHITE else "black",
                }
            )
        return frames


def _render_header(metadata: dict[str, str]) -> str:
    return (
        '<div class="header">'
        f'<div class="title">{escape(metadata["white"])} vs {escape(metadata["black"])}</div>'
        '<div class="meta">'
        f'Result: {escape(metadata["result"])} &middot; '
        f'Segment: {escape(metadata["segment"])} &middot; '
        f'Plies: {escape(metadata["ply_count"])}'
        "</div>"
        "</div>"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_game_animation.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/game_animation.py tests/test_game_animation.py
git commit -m "feat: add GameAnimator for self-contained HTML game replays"
```

---

### Task 3: Wire PGN capture + `GameAnimator` into `eval_vs_stockfish.py`

**Files:**
- Modify: `scripts/eval_vs_stockfish.py`
- Test: `tests/test_eval_vs_stockfish.py`

**Interfaces:**
- Consumes: `EvalVsStockfishConfig.save_games` / `.save_games_dir` (Task 1), `GameAnimator.save(path, game, *, metadata)` (Task 2).
- Produces: `_stockfish_label(*, limit_strength: bool, elo: int | None) -> str`, `_outcome_label(*, completed: bool, result: str, model_color: chess.Color) -> str`, `_build_game_pgn(*, board, model_color, result, segment_name, stockfish_limit_strength, stockfish_elo) -> chess.pgn.Game`, `_save_traced_game(*, board, model_color, result, completed, segment_name, stockfish_limit_strength, stockfish_elo, game_idx, save_games_dir) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_vs_stockfish.py`:

```python
def test_stockfish_label_formats_limited_and_full_strength():
    module = _load_eval_script_module()

    assert module._stockfish_label(limit_strength=True, elo=1400) == "Stockfish (elo=1400)"
    assert (
        module._stockfish_label(limit_strength=False, elo=None)
        == "Stockfish (full strength)"
    )


def test_outcome_label_covers_all_cases():
    module = _load_eval_script_module()

    assert (
        module._outcome_label(completed=False, result="*", model_color=chess.WHITE)
        == "incomplete"
    )
    assert (
        module._outcome_label(
            completed=True, result="1/2-1/2", model_color=chess.WHITE
        )
        == "draw"
    )
    assert (
        module._outcome_label(completed=True, result="1-0", model_color=chess.WHITE)
        == "model_win"
    )
    assert (
        module._outcome_label(completed=True, result="1-0", model_color=chess.BLACK)
        == "model_loss"
    )
    assert (
        module._outcome_label(completed=True, result="0-1", model_color=chess.BLACK)
        == "model_win"
    )


def test_build_game_pgn_sets_headers_for_model_as_black():
    module = _load_eval_script_module()
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5"]:
        board.push_uci(move_uci)

    game = module._build_game_pgn(
        board=board,
        model_color=chess.BLACK,
        result="*",
        segment_name="sf_elo_1400",
        stockfish_limit_strength=True,
        stockfish_elo=1400,
    )

    assert game.headers["White"] == "Stockfish (elo=1400)"
    assert game.headers["Black"] == "imba-chess"
    assert game.headers["Result"] == "*"
    assert game.headers["ModelColor"] == "black"
    assert game.headers["StockfishLimitStrength"] == "True"
    assert game.headers["StockfishElo"] == "1400"
    assert game.headers["Segment"] == "sf_elo_1400"
    assert [move.uci() for move in game.mainline_moves()] == ["e2e4", "e7e5"]


def test_save_traced_game_writes_pgn_and_html(tmp_path):
    module = _load_eval_script_module()
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5"]:
        board.push_uci(move_uci)
    save_games_dir = tmp_path / "games"

    module._save_traced_game(
        board=board,
        model_color=chess.WHITE,
        result="*",
        completed=False,
        segment_name="sf_elo_1400",
        stockfish_limit_strength=True,
        stockfish_elo=1400,
        game_idx=1,
        save_games_dir=save_games_dir,
    )

    pgn_path = save_games_dir / "sf_elo_1400_game002_incomplete.pgn"
    html_path = save_games_dir / "sf_elo_1400_game002_incomplete.html"
    assert pgn_path.exists()
    assert html_path.exists()
    assert "1. e4 e5" in pgn_path.read_text(encoding="utf-8")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py -v -k "stockfish_label or outcome_label or build_game_pgn or save_traced_game"`
Expected: FAIL with `AttributeError` (module has no attribute `_stockfish_label` / `_outcome_label` / `_build_game_pgn` / `_save_traced_game`).

- [ ] **Step 3: Add imports**

In `scripts/eval_vs_stockfish.py`, find:

```python
import chess
import chess.engine
import torch
```

Replace with:

```python
import chess
import chess.engine
import chess.pgn
import torch
```

Find:

```python
from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab
from imba_chess.model import (
    HSTUChessModel,
    build_hstu_chess_config,
    create_batch_block_mask,
)
```

Replace with:

```python
from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab
from imba_chess.eval.game_animation import GameAnimator
from imba_chess.model import (
    HSTUChessModel,
    build_hstu_chess_config,
    create_batch_block_mask,
)
```

- [ ] **Step 4: Add the new helper functions**

In `scripts/eval_vs_stockfish.py`, find the `_build_segment_options` function (defined just before `_build_segment_specs`) and insert these four new functions immediately above it:

```python
def _stockfish_label(*, limit_strength: bool, elo: int | None) -> str:
    if limit_strength:
        return f"Stockfish (elo={elo})"
    return "Stockfish (full strength)"


def _outcome_label(
    *, completed: bool, result: str, model_color: chess.Color
) -> str:
    if not completed:
        return "incomplete"
    if result == "1/2-1/2":
        return "draw"
    model_won = (model_color == chess.WHITE and result == "1-0") or (
        model_color == chess.BLACK and result == "0-1"
    )
    return "model_win" if model_won else "model_loss"


def _build_game_pgn(
    *,
    board: chess.Board,
    model_color: chess.Color,
    result: str,
    segment_name: str,
    stockfish_limit_strength: bool,
    stockfish_elo: int | None,
) -> chess.pgn.Game:
    game = chess.pgn.Game.from_board(board)
    stockfish_label = _stockfish_label(
        limit_strength=stockfish_limit_strength, elo=stockfish_elo
    )
    game.headers["White"] = (
        "imba-chess" if model_color == chess.WHITE else stockfish_label
    )
    game.headers["Black"] = (
        stockfish_label if model_color == chess.WHITE else "imba-chess"
    )
    game.headers["Result"] = result
    game.headers["ModelColor"] = "white" if model_color == chess.WHITE else "black"
    game.headers["StockfishLimitStrength"] = str(stockfish_limit_strength)
    game.headers["StockfishElo"] = (
        str(stockfish_elo) if stockfish_elo is not None else "full_strength"
    )
    game.headers["Segment"] = segment_name
    return game


def _save_traced_game(
    *,
    board: chess.Board,
    model_color: chess.Color,
    result: str,
    completed: bool,
    segment_name: str,
    stockfish_limit_strength: bool,
    stockfish_elo: int | None,
    game_idx: int,
    save_games_dir: Path,
) -> None:
    game = _build_game_pgn(
        board=board,
        model_color=model_color,
        result=result,
        segment_name=segment_name,
        stockfish_limit_strength=stockfish_limit_strength,
        stockfish_elo=stockfish_elo,
    )
    outcome = _outcome_label(completed=completed, result=result, model_color=model_color)
    base_name = f"{segment_name}_game{game_idx + 1:03d}_{outcome}"
    save_games_dir.mkdir(parents=True, exist_ok=True)

    pgn_path = save_games_dir / f"{base_name}.pgn"
    pgn_path.write_text(str(game), encoding="utf-8")

    metadata = {
        "white": game.headers["White"],
        "black": game.headers["Black"],
        "result": result,
        "segment": segment_name,
        "ply_count": str(len(board.move_stack)),
    }
    GameAnimator().save(save_games_dir / f"{base_name}.html", game, metadata=metadata)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py -v -k "stockfish_label or outcome_label or build_game_pgn or save_traced_game"`
Expected: PASS (all 4 new tests).

- [ ] **Step 6: Commit the tested helper functions**

```bash
git add scripts/eval_vs_stockfish.py tests/test_eval_vs_stockfish.py
git commit -m "feat: add PGN + HTML game-saving helpers to eval_vs_stockfish"
```

- [ ] **Step 7: Wire the helpers into `_run_segment`**

In `scripts/eval_vs_stockfish.py`, find the `_run_segment` signature:

```python
def _run_segment(
    *,
    engine: chess.engine.SimpleEngine,
    segment_name: str,
    model: torch.nn.Module,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    games: int,
    max_plies: int,
    engine_limit: chess.engine.Limit,
    device: torch.device,
    dtype: torch.dtype,
    model_move_policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    opening_random_plies: int,
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
) -> EvalSummary:
```

Replace with:

```python
def _run_segment(
    *,
    engine: chess.engine.SimpleEngine,
    segment_name: str,
    model: torch.nn.Module,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    games: int,
    max_plies: int,
    engine_limit: chess.engine.Limit,
    device: torch.device,
    dtype: torch.dtype,
    model_move_policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    opening_random_plies: int,
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
    stockfish_limit_strength: bool,
    stockfish_elo: int | None,
    save_games: bool,
    save_games_dir: Path,
) -> EvalSummary:
```

Then find, inside `_run_segment`'s per-game loop:

```python
            result = board.result(claim_draw=True) if completed else "*"
            _update_summary(
                summary,
                result=result,
                model_color=model_color,
                completed=completed,
                plies=plies,
            )
```

Replace with:

```python
            result = board.result(claim_draw=True) if completed else "*"
            if game_idx < debug_trace_games and save_games:
                _save_traced_game(
                    board=board,
                    model_color=model_color,
                    result=result,
                    completed=completed,
                    segment_name=segment_name,
                    stockfish_limit_strength=stockfish_limit_strength,
                    stockfish_elo=stockfish_elo,
                    game_idx=game_idx,
                    save_games_dir=save_games_dir,
                )
            _update_summary(
                summary,
                result=result,
                model_color=model_color,
                completed=completed,
                plies=plies,
            )
```

- [ ] **Step 8: Pass the new arguments from `main()`**

In `scripts/eval_vs_stockfish.py`, find the `_run_segment(...)` call site inside `main()`:

```python
            segment_summary = _run_segment(
                engine=engine,
                segment_name=spec.name,
                model=model,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                games=spec.games,
                max_plies=args.max_plies,
                engine_limit=engine_limit,
                device=device,
                dtype=dtype,
                model_move_policy=str(args.model_move_policy),
                value_rerank_top_k=int(args.value_rerank_top_k),
                value_rerank_lambda=float(args.value_rerank_lambda),
                opening_random_plies=int(args.opening_random_plies),
                debug_trace_games=max(0, int(args.debug_trace_games)),
                debug_trace_max_plies=max(0, int(args.debug_trace_max_plies)),
                debug_topk=max(0, int(args.debug_topk)),
            )
```

Replace with:

```python
            segment_summary = _run_segment(
                engine=engine,
                segment_name=spec.name,
                model=model,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                games=spec.games,
                max_plies=args.max_plies,
                engine_limit=engine_limit,
                device=device,
                dtype=dtype,
                model_move_policy=str(args.model_move_policy),
                value_rerank_top_k=int(args.value_rerank_top_k),
                value_rerank_lambda=float(args.value_rerank_lambda),
                opening_random_plies=int(args.opening_random_plies),
                debug_trace_games=max(0, int(args.debug_trace_games)),
                debug_trace_max_plies=max(0, int(args.debug_trace_max_plies)),
                debug_topk=max(0, int(args.debug_topk)),
                stockfish_limit_strength=bool(spec.limit_strength),
                stockfish_elo=(int(spec.elo) if spec.elo is not None else None),
                save_games=bool(args.save_games),
                save_games_dir=Path(args.save_games_dir),
            )
```

- [ ] **Step 9: Add CLI flags**

In `scripts/eval_vs_stockfish.py`, find:

```python
    parser.add_argument(
        "--debug-topk",
        type=int,
        default=None,
        help="Top-k legal model moves to print in debug traces.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
```

Replace with:

```python
    parser.add_argument(
        "--debug-topk",
        type=int,
        default=None,
        help="Top-k legal model moves to print in debug traces.",
    )
    parser.add_argument(
        "--save-games",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save PGN + HTML replay for each debug-traced game.",
    )
    parser.add_argument(
        "--save-games-dir",
        type=Path,
        default=None,
        help="Directory to write saved game PGN/HTML files into.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
```

- [ ] **Step 10: Resolve the new args against config in `main()`**

In `scripts/eval_vs_stockfish.py`, find:

```python
    args.debug_topk = int(
        eval_cfg.debug_topk if args.debug_topk is None else args.debug_topk
    )
```

Replace with:

```python
    args.debug_topk = int(
        eval_cfg.debug_topk if args.debug_topk is None else args.debug_topk
    )
    args.save_games = bool(
        eval_cfg.save_games if args.save_games is None else args.save_games
    )
    args.save_games_dir = Path(
        eval_cfg.save_games_dir
        if args.save_games_dir is None
        else args.save_games_dir
    )
```

- [ ] **Step 11: Run the full eval test suite**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py -v`
Expected: PASS (all tests, including the pre-existing ones — this step only added new parameters with call sites updated consistently, so no existing behavior changed).

- [ ] **Step 12: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (no regressions anywhere in the repo).

- [ ] **Step 13: Manual end-to-end smoke check**

This exercises the real CLI wiring end-to-end with a tiny checkpoint-free run scope isn't possible (the script requires a real checkpoint and Stockfish binary), so instead confirm wiring by running with `--games 1 --ladder-elos` unset and a real checkpoint you already have, e.g.:

```bash
.venv/bin/python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/<some_checkpoint>.pt \
  --games 2 --max-plies 40 --debug-trace-games 2 \
  --save-games-dir /tmp/eval_games_smoke
ls /tmp/eval_games_smoke
```

Expected: two `.pgn` and two `.html` files named like
`sf_full_strength_game001_<outcome>.html`. Open one `.html` file in a
browser and confirm the header shows the correct colors/result, the move
list is clickable, and Play/Pause/scrubber all move the board through the
game.

- [ ] **Step 14: Commit**

```bash
git add scripts/eval_vs_stockfish.py
git commit -m "feat: save and animate debug-traced eval games end to end"
```
