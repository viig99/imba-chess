from __future__ import annotations

import base64
import json
from html import escape

import chess
import chess.pgn
import chess.svg

# The look follows chessframe.com: its "Atelier" paper-and-walnut page, the Slate board
# colours, and the Classical tempo with its default settings (square 50 px, 24 fps, a
# 5-frame glide, a 1 s pause after each move, a 3 s final hold, title 28, caption 15,
# move list 11, 40 move-list rows per column).
_SQUARE_PX = 50
_GLIDE_MS = round(1000 * 5 / 24)
_PAUSE_MS = 1000
_FINAL_HOLD_MS = 3000
_SCORESHEET_ROWS = 40

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
  :root {
    --paper: #f5f0e6; --paper-2: #fbf8f1; --paper-3: #ede6d7; --paper-4: #e2d9c6;
    --ink: #1f1b17; --ink-2: #433c34; --ink-3: #7a7163;
    --rule: rgba(31, 27, 23, 0.14);
    --walnut: #5a3d2c; --brass: #a17f3c; --brass-2: #c9a45f;
    --ebony: #241f1b; --ebony-2: #2f2823; --ebony-3: #3d342d;
    --oxblood: #7c2f2f; --moss: #4c6b47;
    --board-light: #dee3e6; --board-dark: #8ca2ad;
    --font-display: "Cormorant Garamond", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    --font-ui: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, "Courier New", monospace;
    --square: __SQUARE_PX__px;
    --glide: __GLIDE_MS__ms;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 28px 16px 40px; font-family: var(--font-ui); font-size: 15px; color: var(--ink);
    background: radial-gradient(1100px 520px at 50% -8%, #fffaee 0%, rgba(255, 250, 238, 0) 65%), var(--paper);
    -webkit-font-smoothing: antialiased;
  }
  .page { max-width: 1180px; margin: 0 auto; }
  header { margin-bottom: 22px; }
  .eyebrow { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: var(--brass); }
  h1 { font-family: var(--font-display); font-size: 28px; font-weight: 600; margin: 4px 0 6px; line-height: 1.15; }
  .meta { color: var(--ink-3); font-size: 0.86rem; display: flex; flex-wrap: wrap; gap: 4px 16px; }
  .meta .result { color: var(--oxblood); font-weight: 600; }
  .studio { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }

  /* Board panel: the walnut case */
  .board-panel {
    flex: 0 0 auto; max-width: 100%; padding: 20px 20px 18px; border-radius: 6px; color: var(--paper);
    background: linear-gradient(170deg, var(--ebony-2), var(--ebony) 60%); border: 1px solid var(--ebony-3);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.05) inset, 0 24px 44px -24px rgba(31, 27, 23, 0.7);
  }
  .plate { display: flex; align-items: center; gap: 8px; font-size: 0.9rem; color: var(--paper-3); margin: 0 0 10px 30px; }
  .plate.bottom { margin: 10px 0 0 30px; }
  .chip { width: 12px; height: 12px; border-radius: 2px; border: 1px solid rgba(245, 240, 230, 0.5); flex: none; }
  .chip.white { background: #f7f1e3; } .chip.black { background: #111; }
  .plate .name { font-weight: 600; color: var(--paper); }
  .tag { font-size: 0.6rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: var(--brass-2);
         border: 1px solid rgba(201, 164, 95, 0.45); border-radius: 999px; padding: 1px 7px; }
  .board-row { display: flex; gap: 10px; align-items: stretch; }
  .evalbar { position: relative; width: 20px; border-radius: 3px; overflow: hidden; background: #2b2420;
             border: 1px solid #55453a; flex: none; }
  .evalbar .fill { position: absolute; left: 0; right: 0; bottom: 0; background: #f7f1e3; transition: height 0.35s ease; }
  .evalbar .num { position: absolute; left: 0; right: 0; font-family: var(--font-mono); font-size: 8.5px; font-weight: 600;
                  text-align: center; }
  .evalbar .num.top { top: 3px; color: #f7f1e3; } .evalbar .num.bottom { bottom: 3px; color: var(--ebony); }
  .board-frame {
    padding: 14px; border-radius: 4px; background: linear-gradient(160deg, #4a3a2f, #2b221c); border: 1px solid #55453a;
    box-shadow: inset 0 0 0 6px rgba(0, 0, 0, 0.2), inset 0 0 0 7px rgba(201, 164, 95, 0.4), 0 18px 40px -12px rgba(0, 0, 0, 0.65);
  }
  .board { position: relative; width: min(calc(8 * var(--square)), calc(100vw - 132px)); aspect-ratio: 1; border-radius: 2px;
           overflow: hidden; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45); display: grid;
           grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr); }
  .sq { position: relative; }
  .sq.light { background: var(--board-light); } .sq.dark { background: var(--board-dark); }
  .sq.hl::after { content: ""; position: absolute; inset: 0; background: rgba(201, 164, 95, 0.45);
                  box-shadow: inset 0 0 0 2px rgba(161, 127, 60, 0.55); }
  .sq.check::before { content: ""; position: absolute; inset: 0;
                      background: radial-gradient(circle, rgba(124, 47, 47, 0.9) 0%, rgba(124, 47, 47, 0) 72%); }
  .coord { position: absolute; font-size: 9px; font-weight: 600; opacity: 0.75; z-index: 1; pointer-events: none; }
  .coord.file { right: 3px; bottom: 1px; } .coord.rank { left: 3px; top: 1px; }
  .sq.light .coord { color: var(--board-dark); } .sq.dark .coord { color: var(--board-light); }
  .pieces { position: absolute; inset: 0; pointer-events: none; }
  .pc { position: absolute; width: 12.5%; height: 12.5%; background-size: 100% 100%; z-index: 2; will-change: transform; }
  .pc.sliding { transition: transform var(--glide) cubic-bezier(0.22, 1, 0.36, 1); z-index: 3; }
  .card-over { position: absolute; inset: 0; z-index: 5; display: grid; place-items: center; text-align: center; padding: 24px;
               background: rgba(36, 31, 27, 0.78); backdrop-filter: blur(2px); color: var(--paper-3);
               font-family: var(--font-display); cursor: pointer; }
  .card-over[hidden] { display: none; }
  .card-over .big { font-size: 28px; font-weight: 600; color: var(--paper); line-height: 1.2; }
  .card-over .sub { font-size: 1.05rem; font-style: italic; margin-top: 8px; color: var(--brass-2); }
  .caption { text-align: center; margin-top: 12px; min-height: 1.4em; font-family: var(--font-display); font-size: 15px;
             font-weight: 700; letter-spacing: 0.03em; color: var(--brass-2); }
  .controls { display: flex; gap: 6px; margin-top: 12px; }
  .btn { flex: 1; display: grid; place-items: center; padding: 7px 0; background: transparent; color: var(--paper-3);
         border: 1px solid rgba(245, 240, 230, 0.2); border-radius: 4px; cursor: pointer; }
  .btn:hover { color: var(--brass-2); border-color: var(--brass-2); }
  .btn.play { flex: 1.3; background: var(--brass-2); border-color: var(--brass-2); color: var(--ebony); }
  .btn.play:hover { background: var(--brass); border-color: var(--brass); color: var(--ebony); }
  .btn svg { width: 15px; height: 15px; fill: currentColor; }
  .scrub { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
  .scrub input { flex: 1; accent-color: var(--brass-2); }
  .ply { font-family: var(--font-mono); font-size: 0.75rem; color: var(--paper-3); min-width: 64px; text-align: right; }
  .hint { margin: 10px 0 0; text-align: center; font-size: 0.7rem; letter-spacing: 0.04em; color: rgba(245, 240, 230, 0.4); }

  /* Paper cards */
  .side { flex: 1 1 380px; min-width: 0; display: flex; flex-direction: column; gap: 18px; }
  .card { background: var(--paper-2); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 18px;
          box-shadow: 0 1px 0 rgba(255, 255, 255, 0.75) inset, 0 14px 34px -22px rgba(60, 40, 20, 0.45); }
  .card h2 { font-family: var(--font-display); font-size: 1.35rem; font-weight: 600; margin: 0 0 4px; }
  .muted { color: var(--ink-3); font-size: 0.84rem; line-height: 1.45; }
  .wdl-label { font-size: 0.84rem; margin-top: 10px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .wdl { display: flex; height: 10px; border-radius: 2px; overflow: hidden; margin-top: 4px; background: var(--paper-4); }
  .wdl .w { background: var(--moss); } .wdl .d { background: #b7ad9b; } .wdl .l { background: var(--oxblood); }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.82rem; font-variant-numeric: tabular-nums; }
  th { font-size: 0.64rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3);
       text-align: right; padding: 4px 6px; border-bottom: 1px solid var(--rule); cursor: help; }
  td { text-align: right; padding: 4px 6px; border-bottom: 1px solid var(--rule); }
  th:first-child, td:first-child { text-align: left; }
  td.mv { font-family: var(--font-mono); font-weight: 600; }
  tr.played td { background: rgba(201, 164, 95, 0.18); }
  .bar { display: inline-block; height: 7px; background: var(--brass-2); border-radius: 1px; vertical-align: middle; margin-right: 5px; }
  .over { margin-top: 12px; padding: 9px 12px; border-radius: 4px; background: rgba(124, 47, 47, 0.1); color: var(--oxblood);
          font-weight: 600; font-size: 0.88rem; }
  .sheet { overflow-x: auto; margin-top: 8px; display: grid; grid-auto-flow: column; grid-auto-columns: minmax(158px, 1fr);
           column-gap: 14px; font-family: var(--font-mono); font-size: 11px; line-height: 1.9; }
  .row { display: grid; grid-template-columns: 2.6em 1fr 1fr; }
  .row .no { color: var(--ink-3); }
  .cell { cursor: pointer; padding: 0 4px; border-radius: 3px; white-space: nowrap; }
  .cell:hover { background: var(--paper-3); }
  .cell.model { color: var(--walnut); font-weight: 600; }
  .cell.current { background: var(--brass-2); color: var(--ebony); }
  .cell .es { font-size: 0.8em; font-weight: 400; color: var(--ink-3); margin-left: 3px; }
  .cell.current .es { color: var(--ebony-3); }
</style>
</head>
<body>
<div class="page">
__HEADER_HTML__
<div class="studio">
  <section class="board-panel" aria-label="Board">
    <div class="plate" id="plateTop"></div>
    <div class="board-row">
      <div class="evalbar" id="evalbar" title="">
        <div class="fill" id="evalFill"></div><span class="num top" id="evalTop"></span><span class="num bottom" id="evalBottom"></span>
      </div>
      <div class="board-frame">
        <div class="board" id="board">
          <div class="pieces" id="pieces"></div>
          <div class="card-over" id="overlay"><div><div class="big" id="overlayBig"></div><div class="sub" id="overlaySub"></div></div></div>
        </div>
      </div>
    </div>
    <div class="plate bottom" id="plateBottom"></div>
    <div class="caption" id="caption"></div>
    <div class="controls">
      <button class="btn" id="btnFirst" title="First (Home)"><svg viewBox="0 0 16 16"><path d="M2 2h2v12H2zM14 2v12L5 8z"/></svg></button>
      <button class="btn" id="btnPrev" title="Previous (left arrow)"><svg viewBox="0 0 16 16"><path d="M12 2v12L3 8z"/></svg></button>
      <button class="btn play" id="btnPlay" title="Play / pause (space)"><svg viewBox="0 0 16 16" id="playIcon"><path d="M4 2v12l10-6z"/></svg></button>
      <button class="btn" id="btnNext" title="Next (right arrow)"><svg viewBox="0 0 16 16"><path d="M4 2v12l9-6z"/></svg></button>
      <button class="btn" id="btnLast" title="Last (End)"><svg viewBox="0 0 16 16"><path d="M12 2h2v12h-2zM2 2v12l9-6z"/></svg></button>
      <button class="btn" id="btnFlip" title="Flip board (f)"><svg viewBox="0 0 16 16"><path d="M4 1l3 4H5v6H3V5H1zm8 14l-3-4h2V5h2v6h2z"/></svg></button>
    </div>
    <div class="scrub"><input type="range" id="slider" min="0" value="0" aria-label="Ply"><span class="ply" id="plyLabel"></span></div>
    <p class="hint">Arrow keys step &middot; space plays &middot; f flips</p>
  </section>
  <div class="side">
    <section class="card" id="info"></section>
    <section class="card">
      <h2>Scoresheet</h2>
      <div class="muted">imba-chess's moves are in brown, with its search's expected score for the move it played.</div>
      <div class="sheet" id="sheet"></div>
    </section>
  </div>
</div>
</div>
<script>
const FRAMES = __FRAMES_JSON__;

const GAME = __GAME_JSON__;
const PIECES = __PIECES_JSON__;
const GLIDE_MS = __GLIDE_MS__, PAUSE_MS = __PAUSE_MS__, FINAL_HOLD_MS = __FINAL_HOLD_MS__, ROWS = __ROWS__;
const $ = (id) => document.getElementById(id);
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const FILES = "abcdefgh";

let idx = 0, flipped = false, timer = null;

// Square index (0 = a1) to display row/column.
const place = (sq) => flipped ? [sq >> 3, 7 - (sq & 7)] : [7 - (sq >> 3), sq & 7];

function buildSquares() {
  const board = $("board");
  board.querySelectorAll(".sq").forEach((el) => el.remove());
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const file = flipped ? 7 - c : c, rank = flipped ? r : 7 - r;
    const el = document.createElement("div");
    el.className = `sq ${(file + rank) % 2 ? "light" : "dark"}`;
    el.dataset.sq = String(rank * 8 + file);
    el.style.gridRow = String(r + 1); el.style.gridColumn = String(c + 1);
    if (r === 7) el.insertAdjacentHTML("beforeend", `<span class="coord file">${FILES[file]}</span>`);
    if (c === 0) el.insertAdjacentHTML("beforeend", `<span class="coord rank">${rank + 1}</span>`);
    board.insertBefore(el, $("pieces"));
  }
}

function placement(fen) {
  const out = {};
  fen.split("/").forEach((row, i) => {
    let file = 0;
    for (const ch of row) {
      if (/\\d/.test(ch)) file += Number(ch);
      else { out[(7 - i) * 8 + file] = ch; file++; }
    }
  });
  return out;
}

function drawBoard(glide) {
  const f = FRAMES[idx];
  const layer = $("pieces");
  layer.innerHTML = "";
  const pieces = placement(f.fen);
  const slides = glide ? Object.fromEntries(f.slides.map(([from, to]) => [to, from])) : {};
  for (const [sq, ch] of Object.entries(pieces)) {
    const [r, c] = place(Number(sq));
    const el = document.createElement("div");
    el.className = "pc";
    el.style.left = `${12.5 * c}%`; el.style.top = `${12.5 * r}%`;
    el.style.backgroundImage = `url("${PIECES[ch]}")`;
    layer.appendChild(el);
    if (sq in slides) {
      const [r0, c0] = place(slides[sq]);
      el.style.transform = `translate(${100 * (c0 - c)}%, ${100 * (r0 - r)}%)`;
      void el.offsetWidth;
      el.classList.add("sliding");
      el.style.transform = "";
    }
  }
  const lit = new Set(f.uci ? [f.slides[0][0], f.slides[0][1]] : []);
  document.querySelectorAll(".sq").forEach((el) => {
    const sq = Number(el.dataset.sq);
    el.classList.toggle("hl", lit.has(sq));
    el.classList.toggle("check", sq === f.check);
  });
}

function plate(color) {
  const name = GAME.players[color];
  const model = name === GAME.model ? `<span class="tag">model</span>` : "";
  return `<span class="chip ${color.toLowerCase()}"></span><span class="name">${esc(name)}</span>${model}`;
}

function drawPlates() {
  $("plateTop").innerHTML = plate(flipped ? "White" : "Black");
  $("plateBottom").innerHTML = plate(flipped ? "Black" : "White");
}

function drawEval() {
  const white = FRAMES[idx].white_score;
  const bottomWhite = !flipped;
  const fill = $("evalFill");
  fill.style.background = bottomWhite ? "#f7f1e3" : "#111";
  $("evalbar").style.background = bottomWhite ? "#2b2420" : "#f7f1e3";
  const known = white !== null;
  const share = known ? (bottomWhite ? white : 1 - white) : 0.5;
  fill.style.height = `${100 * share}%`;
  const bottom = $("evalBottom"), top = $("evalTop");
  bottom.style.color = bottomWhite ? "var(--ebony)" : "#f7f1e3";
  top.style.color = bottomWhite ? "#f7f1e3" : "var(--ebony)";
  // Like lichess, only the side ahead shows its number, at its own end of the bar.
  bottom.textContent = known && share >= 0.5 ? Math.round(100 * share) : "";
  top.textContent = known && share < 0.5 ? Math.round(100 * (1 - share)) : "";
  $("evalbar").title = known
    ? `${GAME.model}'s search: White's expected score ${pct(white)} (updated on ${GAME.model}'s moves)`
    : `No estimate yet: ${GAME.model} has not moved`;
}

function wdlBar(wdl, label) {
  const [l, d, w] = wdl;  // [loss, draw, win] for the side that moved
  return `<div class="wdl-label">${label}: win ${pct(w)} &middot; draw ${pct(d)} &middot; loss ${pct(l)} &middot; expected score <b>${pct(w + d / 2)}</b></div>` +
    `<div class="wdl"><span class="w" style="width:${100 * w}%"></span><span class="d" style="width:${100 * d}%"></span><span class="l" style="width:${100 * l}%"></span></div>`;
}

function drawInfo() {
  const f = FRAMES[idx];
  let html;
  if (idx === 0) {
    html = `<h2>Start position</h2><div class="muted">White to move. Press play, or step through with the arrow keys.</div>`;
  } else {
    html = `<h2>${f.move_number}${f.color === "White" ? "." : "&hellip;"} ${esc(f.san)}</h2>` +
      `<div class="muted">${f.color}: ${esc(f.player)}</div>`;
    const a = f.analysis;
    if (a) {
      html += `<div class="muted" style="margin-top:8px">The search before this move: ${a.simulations} simulations over ${a.legal_moves} legal moves. All numbers are from ${esc(f.player)}'s point of view.</div>`;
      if (a.wdl) html += wdlBar(a.wdl, "Network");
      if (a.search_wdl) html += wdlBar(a.search_wdl, "Search");
      html += `<table><tr><th>Move</th><th title="Raw network policy">Prior</th><th title="Simulations spent on this move">Visits</th>` +
        `<th title="Search value of this move, -1 to +1, and as an expected score">Q</th><th title="Improved policy the search produced">Search policy</th></tr>`;
      for (const c of a.candidates) {
        const q = c.visits ? `${c.q >= 0 ? "+" : ""}${c.q.toFixed(2)} (${pct((c.q + 1) / 2)})` : "&ndash;";
        html += `<tr class="${c.uci === f.uci ? "played" : ""}"><td class="mv">${c.uci === f.uci ? "&#9656; " : ""}${esc(c.san)}</td>` +
          `<td>${pct(c.prior)}</td><td>${c.visits}</td><td>${q}</td>` +
          `<td><span class="bar" style="width:${Math.round(56 * c.policy)}px"></span>${pct(c.policy)}</td></tr>`;
      }
      html += `</table>`;
    } else {
      html += `<div class="muted" style="margin-top:8px">${esc(f.player)} reports no analysis for its moves.</div>`;
    }
  }
  if (idx === FRAMES.length - 1) html += `<div class="over">${esc(GAME.summary)}</div>`;
  $("info").innerHTML = html;
}

function show(newIdx, glide = false) {
  glide = glide && newIdx === idx + 1 && newIdx < FRAMES.length;
  idx = Math.max(0, Math.min(FRAMES.length - 1, newIdx));
  const f = FRAMES[idx];
  drawBoard(glide);
  drawEval();
  drawInfo();
  $("slider").value = String(idx);
  $("plyLabel").textContent = `${idx} / ${FRAMES.length - 1}`;
  $("caption").textContent = idx === 0 ? "" : `${f.move_number}${f.color === "White" ? "." : "\\u2026"} ${f.san}`;
  document.querySelectorAll(".cell").forEach((el) => el.classList.toggle("current", Number(el.dataset.ply) === idx));
  const current = document.querySelector(".cell.current");
  if (current) current.scrollIntoView({block: "nearest", inline: "nearest"});
}

function overlay(big, sub) {
  $("overlayBig").textContent = big;
  $("overlaySub").textContent = sub;
  $("overlay").hidden = false;
}

const PLAY_ICON = `<path d="M4 2v12l10-6z"/>`, PAUSE_ICON = `<path d="M3 2h4v12H3zM9 2h4v12H9z"/>`;

function stop() {
  if (timer !== null) { clearTimeout(timer); timer = null; }
  $("playIcon").innerHTML = PLAY_ICON;
}

function tick() {
  if (idx >= FRAMES.length - 1) {
    timer = setTimeout(() => { stop(); overlay(GAME.summary, `${FRAMES.length - 1} plies`); }, FINAL_HOLD_MS);
    return;
  }
  show(idx + 1, true);
  timer = setTimeout(tick, GLIDE_MS + PAUSE_MS);
}

function play() {
  if (timer !== null) { stop(); return; }
  $("overlay").hidden = true;
  $("playIcon").innerHTML = PAUSE_ICON;
  if (idx >= FRAMES.length - 1) show(0);
  timer = setTimeout(tick, idx === 0 ? 400 : 0);
}

function step(delta) {
  stop();
  $("overlay").hidden = true;
  show(idx + delta, delta === 1);
}

function flip() {
  flipped = !flipped;
  buildSquares();
  drawPlates();
  show(idx);
}

function sheet() {
  const cell = (ply) => {
    const f = FRAMES[ply];
    if (!f) return "<span></span>";
    const played = f.analysis && f.analysis.candidates.find((c) => c.uci === f.uci);
    const es = played && played.visits ? `<span class="es">${Math.round(50 * (played.q + 1))}%</span>` : "";
    return `<span class="cell${f.analysis ? " model" : ""}" data-ply="${ply}">${esc(f.san)}${es}</span>`;
  };
  // A game set up with Black to move starts its list with "1... move".
  const first = FRAMES.length > 1 && FRAMES[1].color === "Black" ? 0 : 1;
  let rows = 0, html = "";
  for (let ply = first; ply < FRAMES.length; ply += 2, rows++) {
    const no = ply === 0 ? FRAMES[1].move_number : FRAMES[ply].move_number;
    html += `<div class="row"><span class="no">${no}.</span>${ply === 0 ? "<span>&hellip;</span>" : cell(ply)}${cell(ply + 1)}</div>`;
  }
  $("sheet").style.gridTemplateRows = `repeat(${Math.max(1, Math.min(rows, ROWS))}, auto)`;
  $("sheet").innerHTML = html;
  $("sheet").querySelectorAll(".cell").forEach((el) => el.addEventListener("click", () => {
    stop(); $("overlay").hidden = true; show(Number(el.dataset.ply));
  }));
}

$("btnFirst").addEventListener("click", () => { stop(); show(0); });
$("btnPrev").addEventListener("click", () => step(-1));
$("btnNext").addEventListener("click", () => step(1));
$("btnLast").addEventListener("click", () => { stop(); $("overlay").hidden = true; show(FRAMES.length - 1); });
$("btnPlay").addEventListener("click", play);
$("btnFlip").addEventListener("click", flip);
$("overlay").addEventListener("click", () => { $("overlay").hidden = true; if (idx === 0 && timer === null) play(); });
$("slider").max = String(FRAMES.length - 1);
$("slider").addEventListener("input", (event) => { stop(); $("overlay").hidden = true; show(Number(event.target.value)); });
document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" && event.key !== " ") return;
  if (event.key === "ArrowLeft") { event.preventDefault(); step(-1); }
  else if (event.key === "ArrowRight") { event.preventDefault(); step(1); }
  else if (event.key === "Home") { event.preventDefault(); stop(); show(0); }
  else if (event.key === "End") { event.preventDefault(); stop(); $("overlay").hidden = true; show(FRAMES.length - 1); }
  else if (event.key === " ") { event.preventDefault(); play(); }
  else if (event.key === "f") flip();
});

buildSquares();
drawPlates();
sheet();
show(0);
overlay(`${GAME.players.White} vs ${GAME.players.Black}`, GAME.event);
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
    those moves show the searcher's WDL and candidate table, and drive the
    eval bar.
    """
    frames = _build_frames(game, analyses or {})
    headers = game.headers
    summary = _result_summary(headers)
    analysed = {f["player"] for f in frames[1:] if f["analysis"] is not None}
    model = next(iter(analysed)) if len(analysed) == 1 else "imba-chess"
    game_json = dict(
        summary=summary,
        event=headers.get("Event", "?"),
        model=model,
        players=dict(White=headers["White"], Black=headers["Black"]),
    )
    title = escape(f"{headers['White']} vs {headers['Black']}")
    replacements = {
        "__TITLE__": title,
        "__HEADER_HTML__": _render_header(game, ply_count=len(frames) - 1, summary=summary),
        "__SQUARE_PX__": str(_SQUARE_PX),
        "__GLIDE_MS__": str(_GLIDE_MS),
        "__PAUSE_MS__": str(_PAUSE_MS),
        "__FINAL_HOLD_MS__": str(_FINAL_HOLD_MS),
        "__ROWS__": str(_SCORESHEET_ROWS),
        "__PIECES_JSON__": _script_json(_piece_images()),
        "__GAME_JSON__": _script_json(game_json),
        "__FRAMES_JSON__": _script_json(frames),
    }
    page = _PAGE_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


def _script_json(value) -> str:
    return json.dumps(value).replace("</", "<\\/")


def _piece_images() -> dict[str, str]:
    images = {}
    for symbol in "PNBRQKpnbrqk":
        svg = chess.svg.piece(chess.Piece.from_symbol(symbol))
        images[symbol] = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return images


def _slides(board: chess.Board, move: chess.Move) -> list[list[int]]:
    """The (from, to) squares of every piece the move slides, king first when castling."""
    if not board.is_castling(move):
        return [[move.from_square, move.to_square]]
    rank = chess.square_rank(move.from_square)
    kingside = board.is_kingside_castling(move)
    king_to = chess.square(6 if kingside else 2, rank)
    rook_from = move.to_square if board.chess960 else chess.square(7 if kingside else 0, rank)
    rook_to = chess.square(5 if kingside else 3, rank)
    return [[move.from_square, king_to], [rook_from, rook_to]]


def _check_square(board: chess.Board) -> int | None:
    return board.king(board.turn) if board.is_check() else None


def _build_frames(game: chess.pgn.Game, analyses: dict[int, dict]) -> list[dict]:
    board = game.board()
    frames: list[dict] = [
        {"fen": board.board_fen(), "san": None, "uci": None, "slides": [],
         "check": _check_square(board), "white_score": None}
    ]
    white_score = None
    for ply, move in enumerate(game.mainline_moves(), start=1):
        color = "White" if board.turn == chess.WHITE else "Black"
        frame = {
            "san": board.san(move),
            "uci": move.uci(),
            "color": color,
            "player": game.headers[color],
            "move_number": board.fullmove_number,
            "slides": _slides(board, move),
            "analysis": None,
        }
        if ply in analyses:
            analysis = dict(analyses[ply])
            analysis["candidates"] = [
                dict(c, san=board.san(chess.Move.from_uci(c["uci"])))
                for c in analysis["candidates"]
            ]
            frame["analysis"] = analysis
            wdl = analysis.get("search_wdl") or analysis.get("wdl")
            if wdl is not None:
                mover = wdl[2] + wdl[1] / 2
                white_score = mover if color == "White" else 1 - mover
        board.push(move)
        frame["fen"] = board.board_fen()
        frame["check"] = _check_square(board)
        frame["white_score"] = white_score
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
        "<header>"
        '<div class="eyebrow">imba-chess &middot; saved game</div>'
        f'<h1>{escape(headers["White"])} vs {escape(headers["Black"])}</h1>'
        '<div class="meta">'
        f'<span>White: {escape(headers["White"])}</span>'
        f'<span>Black: {escape(headers["Black"])}</span>'
        f'<span>Event: {escape(headers["Event"])}</span>'
        f"<span>Plies: {ply_count}</span>"
        f'<span class="result">{escape(summary)}</span>'
        "</div>"
        "</header>"
    )
