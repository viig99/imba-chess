"""Animated GIF of a saved game, in the look of the HTML viewer (game_animation.py).

It follows chessframe.com's Classical tempo with its default settings: 50 px squares, 24 fps,
a 5-frame glide, a 1 s pause after each move, a 3 s final hold, a 3 s title card and a 2 s
result card, title 28, caption 15, a move list at 11 with 40 rows per column, and the Slate
board. The eval bar is the model's search estimate of White's expected score, as on the page.

The piece images in assets/pieces are python-chess's SVG pieces rasterised to 100 px with
`magick -background none -density 1200 piece.svg -resize 100x100 PNG32:piece.png`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chess
import chess.pgn
from PIL import Image, ImageDraw, ImageFont

from imba_chess.eval.game_animation import _build_frames, _result_summary

SQUARE = 50
FPS = 24
GLIDE_FRAMES = 5
PAUSE_S = 1.0
FINAL_HOLD_S = 3.0
INTRO_S, OUTRO_S = 3.0, 2.0
TITLE_SIZE, CAPTION_SIZE, MOVES_SIZE, ROWS = 28, 15, 11, 40

LIGHT, DARK = (222, 227, 230), (140, 162, 173)
EBONY = (36, 31, 27)
WALNUT_CASE, BRASS, BRASS_2 = (58, 46, 37), (161, 127, 60), (201, 164, 95)
PAPER, PAPER_3, MUTED = (245, 240, 230), (237, 230, 215), (150, 140, 126)
IVORY, OXBLOOD = (247, 241, 227), (124, 47, 47)
HIGHLIGHT = (201, 164, 95)

_ASSETS = Path(__file__).with_name("assets") / "pieces"
_FONTS = {
    "serif": ["/usr/share/fonts/google-noto-vf/NotoSerif[wght].ttf",
              "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    "sans": ["/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono": ["/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}

# Layout, in pixels.
MARGIN, FRAME_PAD, BAR_W, GAP = 20, 14, 18, 10
COL_W, LINE_H = 132, 12
BOARD = 8 * SQUARE
CASE = BOARD + 2 * FRAME_PAD
PLATE_H, CAPTION_H = 28, 30


@lru_cache(maxsize=None)
def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONTS[kind]:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            if "[wght]" in path:
                font.set_variation_by_axes([600])
            return font
    raise FileNotFoundError(f"no {kind} font found; tried {_FONTS[kind]}")


@lru_cache(maxsize=None)
def _sprite(symbol: str) -> Image.Image:
    name = ("w" if symbol.isupper() else "b") + symbol.upper()
    return Image.open(_ASSETS / f"{name}.png").convert("RGBA").resize((SQUARE, SQUARE), Image.LANCZOS)


def _xy(square: int) -> tuple[int, int]:
    """Top-left of a square on the board, White at the bottom."""
    return chess.square_file(square) * SQUARE, (7 - chess.square_rank(square)) * SQUARE


class _Layout:
    def __init__(self, move_rows: int):
        self.columns = max(1, -(-move_rows // ROWS))
        self.list_x = MARGIN
        self.bar_x = self.list_x + self.columns * COL_W + GAP * 2
        self.case_x = self.bar_x + BAR_W + GAP
        self.case_y = MARGIN + PLATE_H
        self.board_x, self.board_y = self.case_x + FRAME_PAD, self.case_y + FRAME_PAD
        self.width = self.case_x + CASE + MARGIN
        self.height = max(self.case_y + CASE + PLATE_H + CAPTION_H, self.case_y + min(move_rows, ROWS) * LINE_H) + MARGIN


def _board_image(fen: str, lit: tuple[int, ...], check: int | None, hidden: set[int]) -> Image.Image:
    image = Image.new("RGB", (BOARD, BOARD))
    draw = ImageDraw.Draw(image)
    coord = _font("sans", 9)
    for square in chess.SQUARES:
        x, y = _xy(square)
        light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        colour = LIGHT if light else DARK
        if square in lit:
            colour = tuple(round(0.55 * c + 0.45 * h) for c, h in zip(colour, HIGHLIGHT))
        draw.rectangle([x, y, x + SQUARE - 1, y + SQUARE - 1], fill=colour)
        if square in lit:
            draw.rectangle([x, y, x + SQUARE - 1, y + SQUARE - 1], outline=BRASS, width=2)
        if check == square:
            glow = Image.new("RGBA", (SQUARE, SQUARE), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse([3, 3, SQUARE - 4, SQUARE - 4], fill=(*OXBLOOD, 190))
            image.paste(glow, (x, y), glow)
        ink = DARK if light else LIGHT
        if chess.square_rank(square) == 0:
            draw.text((x + SQUARE - 3, y + SQUARE - 1), chess.FILE_NAMES[chess.square_file(square)], font=coord,
                      fill=ink, anchor="rd")
        if chess.square_file(square) == 0:
            draw.text((x + 3, y + 1), str(chess.square_rank(square) + 1), font=coord, fill=ink, anchor="lt")
    board = chess.Board(fen + " w - - 0 1")
    for square, piece in board.piece_map().items():
        if square not in hidden:
            sprite = _sprite(piece.symbol())
            image.paste(sprite, _xy(square), sprite)
    return image


def _ease(t: float) -> float:
    # Gentler than the page's CSS ease-out, so the last glide frame still differs from the
    # landing frame (identical consecutive GIF frames are merged).
    return 1 - (1 - t) ** 2


class _Renderer:
    def __init__(self, game: chess.pgn.Game, frames: list[dict]):
        self.game, self.frames = game, frames
        self.white, self.black = game.headers["White"], game.headers["Black"]
        analysed = {f["player"] for f in frames[1:] if f["analysis"] is not None}
        self.model = next(iter(analysed)) if len(analysed) == 1 else "imba-chess"
        self.rows = self._rows()
        self.layout = _Layout(len(self.rows))

    def _rows(self) -> list[tuple[int, int | None, int | None]]:
        rows = []
        first = 0 if len(self.frames) > 1 and self.frames[1]["color"] == "Black" else 1
        for ply in range(first, len(self.frames), 2):
            number = self.frames[1]["move_number"] if ply == 0 else self.frames[ply]["move_number"]
            rows.append((number, ply or None, ply + 1 if ply + 1 < len(self.frames) else None))
        return rows

    def chrome(self, idx: int) -> Image.Image:
        """Everything but the board for frame idx: move list, eval bar, case, plates, caption."""
        lay = self.layout
        image = Image.new("RGB", (lay.width, lay.height), EBONY)
        draw = ImageDraw.Draw(image)
        mono = _font("mono", MOVES_SIZE)
        for row, (number, white, black) in enumerate(self.rows):
            if (white or black or 0) > idx:
                break
            x = lay.list_x + (row // ROWS) * COL_W
            y = lay.case_y + (row % ROWS) * LINE_H
            draw.text((x, y), f"{number}.", font=mono, fill=MUTED)
            for offset, ply in ((30, white), (82, black)):
                if ply is None or ply > idx:
                    continue
                frame = self.frames[ply]
                colour = BRASS_2 if frame["player"] == self.model else PAPER_3
                if ply == idx:
                    box = draw.textbbox((x + offset, y), frame["san"], font=mono)
                    draw.rectangle([box[0] - 3, y - 1, box[2] + 3, y + LINE_H - 1], fill=BRASS_2)
                    colour = EBONY
                draw.text((x + offset, y), frame["san"], font=mono, fill=colour)
        self._eval_bar(draw, self.frames[idx]["white_score"])
        draw.rectangle([lay.case_x, lay.case_y, lay.case_x + CASE - 1, lay.case_y + CASE - 1], fill=WALNUT_CASE,
                       outline=(85, 69, 58))
        draw.rectangle([lay.case_x + 6, lay.case_y + 6, lay.case_x + CASE - 7, lay.case_y + CASE - 7],
                       outline=(124, 102, 66))
        sans = _font("sans", 13)
        for name, colour, y in ((self.black, "black", MARGIN + 4), (self.white, "white", lay.case_y + CASE + 8)):
            chip = (17, 17, 17) if colour == "black" else IVORY
            draw.rectangle([lay.case_x, y + 2, lay.case_x + 11, y + 13], fill=chip, outline=MUTED)
            draw.text((lay.case_x + 18, y), name, font=sans, fill=PAPER)
            if name == self.model:
                box = draw.textbbox((lay.case_x + 18, y), name, font=sans)
                tag = _font("sans", 9)
                tx = box[2] + 10
                tb = draw.textbbox((tx + 7, y + 2), "MODEL", font=tag)
                draw.rounded_rectangle([tx, y, tb[2] + 7, y + 15], radius=7, outline=BRASS)
                draw.text((tx + 7, y + 2), "MODEL", font=tag, fill=BRASS_2)
        frame = self.frames[idx]
        if idx:
            dots = "." if frame["color"] == "White" else "…"
            caption = f"{frame['move_number']}{dots} {frame['san']}"
            draw.text((lay.case_x + CASE // 2, lay.case_y + CASE + PLATE_H + CAPTION_H // 2 + 2), caption,
                      font=_font("serif", CAPTION_SIZE), fill=BRASS_2, anchor="mm")
        return image

    def _eval_bar(self, draw: ImageDraw.ImageDraw, white: float | None):
        lay = self.layout
        top, bottom = lay.case_y, lay.case_y + CASE - 1
        draw.rectangle([lay.bar_x, top, lay.bar_x + BAR_W - 1, bottom], fill=(43, 36, 32), outline=(85, 69, 58))
        share = 0.5 if white is None else white
        split = max(top + 1, round(bottom - share * (bottom - top)))
        if split < bottom:  # nothing to fill when White's score rounds to 0
            draw.rectangle([lay.bar_x + 1, split, lay.bar_x + BAR_W - 2, bottom - 1], fill=IVORY)
        if white is not None:
            num = _font("mono", 9)
            cx = lay.bar_x + BAR_W // 2
            if white >= 0.5:
                draw.text((cx, bottom - 3), str(round(100 * white)), font=num, fill=EBONY, anchor="mb")
            else:
                draw.text((cx, top + 4), str(round(100 * (1 - white))), font=num, fill=IVORY, anchor="mt")

    def card(self, idx: int, big: str, sub: str) -> Image.Image:
        """A title or result card over frame idx's position."""
        image = self.position(idx)
        lay = self.layout
        veil = Image.new("RGBA", (BOARD, BOARD), (*EBONY, 205))
        image.paste(veil, (lay.board_x, lay.board_y), veil)
        draw = ImageDraw.Draw(image)
        title = _font("serif", TITLE_SIZE)
        lines = _wrap(draw, big, title, BOARD - 40)
        y = lay.board_y + BOARD // 2 - (len(lines) * (TITLE_SIZE + 8)) // 2 - 10
        for line in lines:
            draw.text((lay.board_x + BOARD // 2, y), line, font=title, fill=PAPER, anchor="mt")
            y += TITLE_SIZE + 8
        draw.text((lay.board_x + BOARD // 2, y + 6), sub, font=_font("serif", CAPTION_SIZE + 2), fill=BRASS_2,
                  anchor="mt")
        return image

    def position(self, idx: int, glide: float | None = None) -> Image.Image:
        """Frame idx's position; with glide in (0, 1) its moving pieces are part-way there."""
        frame = self.frames[idx]
        image = self.chrome(idx)
        lit = tuple(frame["slides"][0]) if frame["slides"] else ()
        moving = {to: frm for frm, to in frame["slides"]} if glide is not None else {}
        board = _board_image(frame["fen"], lit, frame["check"], set(moving))
        if moving:
            placed = chess.Board(frame["fen"] + " w - - 0 1")
            t = _ease(glide)
            for to, frm in moving.items():
                (x0, y0), (x1, y1) = _xy(frm), _xy(to)
                sprite = _sprite(placed.piece_at(to).symbol())
                board.paste(sprite, (round(x0 + t * (x1 - x0)), round(y0 + t * (y1 - y0))), sprite)
        image.paste(board, (self.layout.board_x, self.layout.board_y))
        return image


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def render_game_gif(game: chess.pgn.Game, path: Path, analyses: dict[int, dict] | None = None) -> None:
    """Write the game as an animated GIF to path; analyses as in render_game_html."""
    frames = _build_frames(game, analyses or {})
    renderer = _Renderer(game, frames)
    summary = _result_summary(game.headers)
    tick_ms = 1000 / FPS
    images = [renderer.card(0, f"{renderer.white} vs {renderer.black}", game.headers.get("Event", "?"))]
    durations = [INTRO_S * 1000]
    images.append(renderer.position(0))
    durations.append(PAUSE_S * 1000)
    for idx in range(1, len(frames)):
        for k in range(1, GLIDE_FRAMES + 1):
            images.append(renderer.position(idx, glide=k / (GLIDE_FRAMES + 1)))
            durations.append(tick_ms)
        images.append(renderer.position(idx))
        last = idx == len(frames) - 1
        durations.append(tick_ms + 1000 * (FINAL_HOLD_S if last else PAUSE_S))
    images.append(renderer.card(len(frames) - 1, summary, f"{len(frames) - 1} plies"))
    durations.append(OUTRO_S * 1000)
    palette = _palette(images)
    quantised = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=[round(d) for d in durations],
                      loop=0, optimize=False, disposal=1)


def _palette(images: list[Image.Image]) -> Image.Image:
    """One shared 256-colour palette, so colours don't flicker between frames."""
    picks = [images[0], images[1], images[len(images) // 2], images[-2], images[-1]]
    width, height = picks[0].size
    sheet = Image.new("RGB", (width, height * len(picks)))
    for i, image in enumerate(picks):
        sheet.paste(image, (0, i * height))
    return sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
