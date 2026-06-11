#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a DiffusionGemma generation trace JSON as an animated GIF."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


BACKGROUND = (7, 11, 18)
PANEL = (13, 20, 31)
GRID = (43, 61, 76)
TEXT = (226, 242, 235)
MUTED = (118, 141, 151)
GREEN = (82, 255, 186)
AMBER = (255, 194, 84)
RED = (255, 93, 93)
CYAN = (88, 213, 255)
PURPLE = (184, 125, 255)


def _load_font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
  candidates = [
      "/System/Library/Fonts/Menlo.ttc" if mono else "",
      "/System/Library/Fonts/SFNS.ttf",
      "/System/Library/Fonts/Helvetica.ttc",
      "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if mono else "",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
  ]
  for candidate in candidates:
    if candidate and pathlib.Path(candidate).exists():
      return ImageFont.truetype(candidate, size=size)
  return ImageFont.load_default(size=size)


def _clean_piece(piece: str, limit: int = 14) -> str:
  piece = piece.replace("\n", "\\n").replace("\t", "\\t")
  piece = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "?", piece)
  if len(piece) > limit:
    return piece[: limit - 1] + "~"
  return piece


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
  box = draw.textbbox((0, 0), text, font=font)
  return box[2] - box[0], box[3] - box[1]


def _draw_fit(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font,
    fill: tuple[int, int, int],
    max_width: int,
) -> None:
  if _text_size(draw, text, font)[0] <= max_width:
    draw.text(xy, text, font=font, fill=fill)
    return
  clipped = text
  while clipped and _text_size(draw, clipped + "~", font)[0] > max_width:
    clipped = clipped[:-1]
  draw.text(xy, clipped + "~", font=font, fill=fill)


def _draw_scanlines(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
  for y in range(0, height, 4):
    draw.line((0, y, width, y), fill=(255, 255, 255, 13), width=1)


def _glitch_overlay(
    image: Image.Image,
    *,
    rng: random.Random,
    strength: float,
) -> Image.Image:
  if strength <= 0:
    return image
  out = image.copy()
  width, height = out.size
  draw = ImageDraw.Draw(out, "RGBA")
  for _ in range(int(8 + 20 * strength)):
    y = rng.randrange(70, height - 20)
    h = rng.randrange(2, 8)
    dx = rng.randrange(-18, 19)
    band = out.crop((0, y, width, min(height, y + h)))
    out.paste(band, (dx, y))
    color = rng.choice([(GREEN, 35), (CYAN, 30), (RED, 30), (PURPLE, 28)])
    draw.rectangle(
        (0, y, width, min(height, y + h)), fill=(*color[0], color[1])
    )
  for _ in range(int(20 + 80 * strength)):
    x = rng.randrange(width)
    y = rng.randrange(height)
    draw.point((x, y), fill=(*rng.choice([GREEN, CYAN, RED, TEXT]), 120))
  return out


def _draw_noise_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    noise: float,
) -> None:
  draw.rounded_rectangle((x, y, x + width, y + 12), radius=4, fill=(24, 35, 47))
  fill_width = int(width * max(0.0, min(1.0, noise)))
  draw.rounded_rectangle((x, y, x + fill_width, y + 12), radius=4, fill=GREEN)


def _frame_image(
    payload: dict[str, Any],
    frame: dict[str, Any],
    *,
    index: int,
    total: int,
    width: int,
    height: int,
    rng: random.Random,
    glitch_phase: int,
) -> Image.Image:
  del payload
  image = Image.new("RGB", (width, height), BACKGROUND)
  draw = ImageDraw.Draw(image, "RGBA")
  title_font = _load_font(30)
  ui_font = _load_font(18)
  small_font = _load_font(14, mono=True)
  token_font = _load_font(15, mono=True)

  draw.rectangle((0, 0, width, height), fill=BACKGROUND)
  draw.rectangle((0, 0, width, 92), fill=(10, 17, 27))
  draw.text(
      (36, 24), "DiffusionGemma denoising trace", font=title_font, fill=TEXT
  )
  draw.text(
      (36, 60),
      f"frame {index + 1}/{total} | canvas {frame['canvas']} | step"
      f" {frame['step']}",
      font=ui_font,
      fill=MUTED,
  )
  _draw_noise_bar(draw, x=width - 360, y=36, width=270, noise=frame["noise"])
  draw.text(
      (width - 360, 56),
      f"noise {frame['noise']:.3f} -> {frame['target_noise']:.3f}",
      font=small_font,
      fill=MUTED,
  )

  metrics = [
      (
          "accepted",
          f"{frame['accepted_tokens']}/{len(frame['token_ids'])}",
          GREEN,
      ),
      ("changed", str(frame["changed_tokens"]), RED),
      (
          "entropy",
          "-"
          if frame["mean_entropy"] is None
          else f"{frame['mean_entropy']:.4f}",
          AMBER,
      ),
      ("phase", frame["phase"], CYAN),
  ]
  mx = 36
  for label, value, color in metrics:
    draw.rounded_rectangle((mx, 108, mx + 182, 162), radius=8, fill=PANEL)
    draw.text((mx + 14, 116), label, font=small_font, fill=MUTED)
    draw.text((mx + 14, 135), value, font=ui_font, fill=color)
    mx += 194

  cols = min(8, max(1, int(math.ceil(math.sqrt(len(frame["token_ids"]))))))
  rows = math.ceil(len(frame["token_ids"]) / cols)
  grid_x = 36
  grid_y = 190
  cell_gap = 8
  cell_w = (width - grid_x * 2 - cell_gap * (cols - 1)) // cols
  cell_h = min(74, (height - grid_y - 88 - cell_gap * (rows - 1)) // rows)

  for pos, token_id in enumerate(frame["token_ids"]):
    row, col = divmod(pos, cols)
    x = grid_x + col * (cell_w + cell_gap)
    y = grid_y + row * (cell_h + cell_gap)
    selected = frame["selected_mask"][pos]
    changed = frame["changed_mask"][pos]
    border = GREEN if selected else GRID
    if changed:
      border = RED if glitch_phase % 2 == 0 else AMBER
    fill = (18, 28, 40) if selected else (13, 22, 32)
    if changed and glitch_phase:
      fill = (36, 18, 29)
    draw.rounded_rectangle(
        (x, y, x + cell_w, y + cell_h),
        radius=7,
        fill=fill,
        outline=border,
        width=2,
    )
    piece = _clean_piece(frame["token_texts"][pos])
    if changed and glitch_phase == 1 and rng.random() < 0.45:
      piece = rng.choice(["////", "::::", "0x??", "zzzt", "####", "<<>>"])
    color = TEXT
    if selected:
      color = GREEN
    if changed:
      color = RED if glitch_phase != 2 else AMBER
    _draw_fit(
        draw,
        (x + 10, y + 12),
        piece,
        font=token_font,
        fill=color,
        max_width=cell_w - 20,
    )
    draw.text(
        (x + 10, y + cell_h - 23),
        f"#{pos} id:{token_id}",
        font=small_font,
        fill=MUTED,
    )

  output = frame["text"]
  if len(output) > 220:
    output = output[:219] + "~"
  draw.rounded_rectangle(
      (36, height - 72, width - 36, height - 26), radius=8, fill=PANEL
  )
  _draw_fit(
      draw,
      (52, height - 59),
      _clean_piece(output, limit=170),
      font=small_font,
      fill=TEXT,
      max_width=width - 104,
  )

  _draw_scanlines(draw, width, height)
  return _glitch_overlay(
      image,
      rng=rng,
      strength=(
          0.18 if frame["phase"] == "initial" else 0.10 + 0.03 * glitch_phase
      ),
  )


def render_gif(args: argparse.Namespace) -> None:
  payload = json.loads(
      pathlib.Path(args.trace_json).read_text(encoding="utf-8")
  )
  frames = payload["frames"]
  rng = random.Random(args.seed)
  images = []
  durations = []
  for index, frame in enumerate(frames):
    for phase in range(args.glitch_frames):
      images.append(
          _frame_image(
              payload,
              frame,
              index=index,
              total=len(frames),
              width=args.width,
              height=args.height,
              rng=rng,
              glitch_phase=phase,
          ).convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
      )
      durations.append(args.glitch_duration_ms)
    images.append(
        _frame_image(
            payload,
            frame,
            index=index,
            total=len(frames),
            width=args.width,
            height=args.height,
            rng=rng,
            glitch_phase=0,
        ).convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
    )
    durations.append(
        args.hold_ms if index < len(frames) - 1 else args.final_hold_ms
    )

  output = pathlib.Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  images[0].save(
      output,
      save_all=True,
      append_images=images[1:],
      duration=durations,
      loop=0,
      optimize=True,
  )
  print(
      json.dumps({
          "output": str(output),
          "input_frames": len(frames),
          "gif_frames": len(images),
          "bytes": output.stat().st_size,
      })
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("trace_json")
  parser.add_argument("--output", required=True)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--glitch_frames", type=int, default=2)
  parser.add_argument("--glitch_duration_ms", type=int, default=90)
  parser.add_argument("--hold_ms", type=int, default=620)
  parser.add_argument("--final_hold_ms", type=int, default=1600)
  return parser.parse_args()


def main() -> None:
  render_gif(parse_args())


if __name__ == "__main__":
  main()
