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
