from __future__ import annotations

import json
from html import escape

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
  .move-cell.model { color: #9fd0ff; }
  .move-cell .score { color: #888; font-size: 0.8em; margin-left: 4px; }
  .players { margin-top: 6px; display: flex; gap: 18px; }
  .chip { display: inline-block; width: 0.9em; height: 0.9em; border-radius: 2px; vertical-align: -0.1em; margin-right: 5px; border: 1px solid #888; }
  .chip.white { background: #f0f0f0; } .chip.black { background: #111; }
  .result { margin-top: 6px; font-weight: 600; color: #ffd27f; }
  .side { flex: 1 1 360px; display: flex; flex-direction: column; gap: 12px; min-width: 300px; }
  .info { background: #262626; border-radius: 6px; padding: 12px; }
  .info .move-title { font-weight: 600; margin-bottom: 6px; }
  .info .muted { color: #999; }
  .info table { border-collapse: collapse; width: 100%; margin-top: 8px; font-variant-numeric: tabular-nums; }
  .info th, .info td { text-align: right; padding: 3px 6px; border-bottom: 1px solid #333; }
  .info th:first-child, .info td:first-child { text-align: left; }
  .info tr.played td { background: #2f4a2f; font-weight: 600; }
  .bar { display: inline-block; height: 0.6em; background: #6aa0d8; vertical-align: middle; margin-right: 4px; }
  .wdl { display: flex; height: 14px; border-radius: 3px; overflow: hidden; margin: 4px 0; max-width: 360px; }
  .wdl span { display: block; } .wdl .w { background: #6fbf73; } .wdl .d { background: #9e9e9e; } .wdl .l { background: #d9534f; }
  .game-over { background: #4a3a1a; color: #ffd27f; border-radius: 6px; padding: 10px 12px; font-weight: 600; }
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
  <div class="side">
    <div class="info" id="info"></div>
    <div class="moves-panel" id="movesPanel"></div>
  </div>
</div>
<script>
const FRAMES = __FRAMES_JSON__;

const GAME = __GAME_JSON__;
const infoEl = document.getElementById("info");
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function wdlBar(wdl, label) {
  // wdl is [loss, draw, win] for the side that moved.
  const [l, d, w] = wdl;
  return `<div>${label}: win ${pct(w)} &middot; draw ${pct(d)} &middot; loss ${pct(l)} &middot; expected score ${pct(w + d / 2)}</div>` +
    `<div class="wdl"><span class="w" style="width:${100 * w}%"></span><span class="d" style="width:${100 * d}%"></span><span class="l" style="width:${100 * l}%"></span></div>`;
}

function renderInfo() {
  const f = FRAMES[idx];
  let html = "";
  if (idx === 0) {
    html = `<div class="move-title">Start position</div><div class="muted">White moves first. Use the arrows (or the left/right keys) to step through the game.</div>`;
  } else {
    const dots = f.color === "White" ? "." : "...";
    html = `<div class="move-title">${f.move_number}${dots} ${esc(f.san)} &mdash; ${f.color}: ${esc(f.player)}</div>`;
    const a = f.analysis;
    if (a) {
      html += `<div class="muted">${esc(f.player)}'s search before this move: ${a.simulations} simulations over ${a.legal_moves} legal moves. All numbers are from ${esc(f.player)}'s point of view.</div>`;
      if (a.wdl) html += wdlBar(a.wdl, "Network");
      if (a.search_wdl) html += wdlBar(a.search_wdl, "Search");
      html += `<table><tr><th>Move</th><th title="Raw network policy">Prior</th><th title="Simulations spent on this move">Visits</th><th title="Search value of this move, -1 to +1, and as an expected score">Q</th><th title="Improved policy the search produced">Search policy</th></tr>`;
      for (const c of a.candidates) {
        const q = c.visits ? `${c.q >= 0 ? "+" : ""}${c.q.toFixed(2)} (${pct((c.q + 1) / 2)})` : "&ndash;";
        html += `<tr class="${c.uci === f.uci ? "played" : ""}"><td>${c.uci === f.uci ? "&#9654; " : ""}${esc(c.san)}</td><td>${pct(c.prior)}</td><td>${c.visits}</td><td>${q}</td>` +
          `<td><span class="bar" style="width:${Math.round(60 * c.policy)}px"></span>${pct(c.policy)}</td></tr>`;
      }
      html += `</table>`;
    } else {
      html += `<div class="muted">${esc(f.player)} reports no analysis for its moves.</div>`;
    }
  }
  if (idx === FRAMES.length - 1) html += `<div class="game-over" style="margin-top:10px">${esc(GAME.summary)}</div>`;
  infoEl.innerHTML = html;
}

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
  plyLabelEl.textContent = idx === 0
    ? "Start position"
    : `Ply ${idx} of ${FRAMES.length - 1}: ${FRAMES[idx].color} played ${FRAMES[idx].san}`;
  document.querySelectorAll(".move-cell").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.ply) === idx);
  });
  renderInfo();
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
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") { stopPlaying(); goTo(idx - 1); }
  if (event.key === "ArrowRight") { stopPlaying(); goTo(idx + 1); }
});

function moveCell(ply) {
  const f = FRAMES[ply];
  if (!f) return "";
  const played = f.analysis && f.analysis.candidates.find((c) => c.uci === f.uci);
  const score = played && played.visits ? `<span class="score">${Math.round(50 * (played.q + 1))}%</span>` : "";
  return `<span class="move-cell${f.analysis ? " model" : ""}" data-ply="${ply}">${esc(f.san)}${score}</span>`;
}

let moveRowHtml = "";
// A game set up from a position where Black moves first starts its list with "1... move".
const first = FRAMES.length > 1 && FRAMES[1].color === "Black" ? 0 : 1;
for (let ply = first; ply < FRAMES.length; ply += 2) {
  const moveNumber = ply === 0 ? FRAMES[1].move_number : FRAMES[ply].move_number;
  moveRowHtml += `<div class="move-row"><span class="num">${moveNumber}.</span>${ply === 0 ? "<span></span>" : moveCell(ply)}${moveCell(ply + 1)}</div>`;
}
movesPanelEl.innerHTML = moveRowHtml;
movesPanelEl.querySelectorAll(".move-cell").forEach((el) => {
  el.addEventListener("click", () => { stopPlaying(); goTo(Number(el.dataset.ply)); });
});

render();
</script>
</body>
</html>"""


def render_game_html(
    game: chess.pgn.Game, analyses: dict[int, dict] | None = None
) -> str:
    """Render a game as a self-contained, offline HTML replay viewer.

    The page header is read from the game's White/Black/Result/Event (and
    optional Termination) headers. analyses maps a 1-based ply to the search
    analysis of the move played there (eval_vs_stockfish._gumbel_analysis);
    those moves show the searcher's WDL and candidate table.
    """
    frames = _build_frames(game, analyses or {})
    summary = _result_summary(game.headers)
    title = escape(f"{game.headers['White']} vs {game.headers['Black']}")
    header_html = _render_header(game, ply_count=len(frames) - 1, summary=summary)
    return (
        _PAGE_TEMPLATE.replace("__TITLE__", title)
        .replace("__HEADER_HTML__", header_html)
        .replace("__GAME_JSON__", _script_json(dict(summary=summary)))
        .replace("__FRAMES_JSON__", _script_json(frames))
    )


def _script_json(value) -> str:
    return json.dumps(value).replace("</", "<\\/")


def _build_frames(game: chess.pgn.Game, analyses: dict[int, dict]) -> list[dict]:
    board = game.board()
    frames: list[dict] = [
        {"svg": chess.svg.board(board, size=400, coordinates=True), "san": None}
    ]
    for ply, move in enumerate(game.mainline_moves(), start=1):
        color = "White" if board.turn == chess.WHITE else "Black"
        frame = {
            "san": board.san(move),
            "uci": move.uci(),
            "color": color,
            "player": game.headers[color],
            "move_number": board.fullmove_number,
            "analysis": None,
        }
        if ply in analyses:
            analysis = dict(analyses[ply])
            analysis["candidates"] = [
                dict(c, san=board.san(chess.Move.from_uci(c["uci"])))
                for c in analysis["candidates"]
            ]
            frame["analysis"] = analysis
        board.push(move)
        frame["svg"] = chess.svg.board(board, size=400, lastmove=move, coordinates=True)
        frames.append(frame)
    return frames


def _result_summary(headers) -> str:
    result = headers.get("Result", "*")
    termination = headers.get("Termination", "")
    by = f" by {termination}" if termination else ""
    if result == "1-0":
        return f"{headers['White']} (White) wins{by} · 1-0"
    if result == "0-1":
        return f"{headers['Black']} (Black) wins{by} · 0-1"
    if result == "1/2-1/2":
        return f"Draw{by} · ½-½"
    return f"Unfinished{f' ({termination})' if termination else ''}"


def _render_header(game: chess.pgn.Game, *, ply_count: int, summary: str) -> str:
    headers = game.headers
    return (
        '<div class="header">'
        f'<div class="title">{escape(headers["White"])} vs {escape(headers["Black"])}</div>'
        '<div class="players">'
        f'<span><span class="chip white"></span>White: {escape(headers["White"])}</span>'
        f'<span><span class="chip black"></span>Black: {escape(headers["Black"])}</span>'
        "</div>"
        f'<div class="result">{escape(summary)}</div>'
        '<div class="meta">'
        f'Event: {escape(headers["Event"])} &middot; '
        f"Plies: {ply_count}"
        "</div>"
        "</div>"
    )
