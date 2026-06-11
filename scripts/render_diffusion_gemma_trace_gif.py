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
NOISE_LABELS = ("...", "???", "...", "???")


def _load_font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
  if mono:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
  else:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font,
    max_width: int,
    max_lines: int,
) -> list[str]:
  """Wraps text for PIL drawing, splitting long spans when needed."""
  words = text.split(" ")
  lines: list[str] = []
  current = ""

  def flush_long_span(span: str) -> str:
    nonlocal lines
    remaining = span
    while remaining and len(lines) < max_lines:
      take = remaining
      while take and _text_size(draw, take, font)[0] > max_width:
        take = take[:-1]
      if not take:
        take = remaining[:1]
      lines.append(take)
      remaining = remaining[len(take) :]
    return remaining

  for word in words:
    candidate = word if not current else f"{current} {word}"
    if _text_size(draw, candidate, font)[0] <= max_width:
      current = candidate
      continue
    if current:
      lines.append(current)
      current = ""
      if len(lines) >= max_lines:
        break
    if _text_size(draw, word, font)[0] <= max_width:
      current = word
    else:
      flush_long_span(word)
      if len(lines) >= max_lines:
        break

  if current and len(lines) < max_lines:
    lines.append(current)
  if len(lines) == max_lines and words:
    line = lines[-1]
    while line and _text_size(draw, line + "~", font)[0] > max_width:
      line = line[:-1]
    lines[-1] = line + "~"
  return lines


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font,
    fill: tuple[int, int, int],
    max_width: int,
    max_lines: int,
    line_height: int,
) -> None:
  for line_index, line in enumerate(
      _wrap_text(
          draw, text, font=font, max_width=max_width, max_lines=max_lines
      )
  ):
    draw.text(
        (xy[0], xy[1] + line_index * line_height),
        line,
        font=font,
        fill=fill,
    )


def _mix(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
  amount = max(0.0, min(1.0, amount))
  return tuple(int(a + (b - a) * amount) for a, b in zip(left, right))


def _draw_denoise_pixels(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    rng: random.Random,
    strength: float,
    selected: bool,
    changed: bool,
) -> None:
  """Draws local pixel noise inside one token cell.

  The noise is intentionally clipped to token cells so the overall screen stays
  still while the canvas appears to denoise.
  """
  if strength <= 0:
    return
  x0, y0, x1, y1 = rect
  width = x1 - x0
  height = y1 - y0
  palette = [CYAN, PURPLE, GREEN, AMBER, RED, (230, 240, 255), (38, 51, 68)]
  block = max(2, int(12 * strength))
  count = int((10 + width * height / 260) * strength)
  alpha_base = int(35 + 150 * strength)
  if changed:
    count = int(count * 1.25)
  if selected:
    count = int(count * 0.55)
    alpha_base = int(alpha_base * 0.65)

  for _ in range(max(1, count)):
    bw = rng.randint(2, block)
    bh = rng.randint(2, block)
    x = rng.randint(x0 + 2, max(x0 + 2, x1 - bw - 2))
    y = rng.randint(y0 + 2, max(y0 + 2, y1 - bh - 2))
    color = rng.choice(palette)
    alpha = rng.randint(max(18, alpha_base // 2), min(220, alpha_base))
    draw.rectangle((x, y, x + bw, y + bh), fill=(*color, alpha))

  haze = int(75 * strength)
  if haze:
    draw.rounded_rectangle(
        rect,
        radius=7,
        fill=(226, 242, 235, haze if not selected else haze // 2),
    )


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


def _token_label(
    piece: str, *, noise: float, selected: bool, rng: random.Random
) -> str:
  text = _clean_piece(piece, limit=28)
  if not text:
    text = "<empty>"
  elif not text.strip():
    text = "<space>"

  if selected or noise < 0.24:
    return text
  if noise > 0.66:
    return rng.choice(NOISE_LABELS)

  keep = max(1, int(len(text) * (1.0 - noise * 0.75)))
  return text[:keep] + rng.choice(("...", "??"))


def _draw_token_canvas(
    draw: ImageDraw.ImageDraw,
    frame: dict[str, Any],
    *,
    rect: tuple[int, int, int, int],
    rng: random.Random,
    transition_phase: int,
    transition_frames: int,
    width: int,
    height: int,
) -> None:
  x0, y0, x1, y1 = rect
  panel_w = x1 - x0
  header_font = _load_font(19)
  token_size = 22
  if len(frame["token_ids"]) > 28 or width < 1050:
    token_size = 20
  if height < 620:
    token_size = 18
  token_font = _load_font(token_size)
  small_font = _load_font(13, mono=True)
  chip_h = max(36, token_size + 22)
  gap = 9

  draw.rounded_rectangle(rect, radius=8, fill=PANEL, outline=GRID, width=1)
  draw.text((x0 + 18, y0 + 15), "Denoising canvas", font=header_font, fill=TEXT)
  draw.text(
      (x0 + panel_w - 260, y0 + 18),
      "green = accepted | amber = changing",
      font=small_font,
      fill=MUTED,
  )

  progress = (
      1.0
      if transition_frames <= 1
      else transition_phase / max(1, transition_frames - 1)
  )
  chip_x = x0 + 18
  chip_y = y0 + 56
  max_chip_x = x1 - 18
  max_chip_y = y1 - 18

  for pos, _ in enumerate(frame["token_ids"]):
    selected = frame["selected_mask"][pos]
    changed = frame["changed_mask"][pos]
    denoise_strength = max(0.0, frame["noise"] * (1.0 - 0.68 * progress))
    if selected:
      denoise_strength *= 0.28
    elif changed:
      denoise_strength = max(denoise_strength, 0.20 * (1.0 - progress))

    label = _token_label(
        frame["token_texts"][pos],
        noise=denoise_strength,
        selected=selected,
        rng=rng,
    )
    label_w, _ = _text_size(draw, label, token_font)
    chip_w = min(max(label_w + 28, 58), 230)
    if chip_x + chip_w > max_chip_x:
      chip_x = x0 + 18
      chip_y += chip_h + gap
    if chip_y + chip_h > max_chip_y:
      remaining = len(frame["token_ids"]) - pos
      more = f"+{remaining} more"
      more_w = min(
          max(_text_size(draw, more, token_font)[0] + 28, 92),
          max_chip_x - chip_x,
      )
      draw.rounded_rectangle(
          (chip_x, max_chip_y - chip_h, chip_x + more_w, max_chip_y),
          radius=9,
          fill=(18, 28, 40),
          outline=GRID,
          width=1,
      )
      draw.text(
          (chip_x + 14, max_chip_y - chip_h + 10),
          more,
          font=token_font,
          fill=MUTED,
      )
      break

    fill = (17, 29, 40)
    border = GRID
    text_color = TEXT
    if selected:
      fill = (15, 42, 35)
      border = GREEN
      text_color = GREEN
    elif changed:
      fill = (39, 33, 35)
      border = AMBER
      text_color = AMBER
    elif denoise_strength > 0.35:
      fill = _mix(fill, (25, 34, 52), min(0.7, denoise_strength))
      text_color = _mix(MUTED, TEXT, max(0.0, 1.0 - denoise_strength))

    chip_rect = (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h)
    draw.rounded_rectangle(
        chip_rect,
        radius=9,
        fill=fill,
        outline=border,
        width=2 if selected or changed else 1,
    )
    _draw_denoise_pixels(
        draw,
        (chip_x + 3, chip_y + 3, chip_x + chip_w - 3, chip_y + chip_h - 3),
        rng=rng,
        strength=denoise_strength,
        selected=selected,
        changed=changed,
    )
    _draw_fit(
        draw,
        (chip_x + 14, chip_y + (chip_h - token_size) // 2 - 2),
        label,
        font=token_font,
        fill=text_color,
        max_width=chip_w - 28,
    )
    chip_x += chip_w + gap


def _frame_image(
    payload: dict[str, Any],
    frame: dict[str, Any],
    *,
    index: int,
    total: int,
    width: int,
    height: int,
    rng: random.Random,
    transition_phase: int,
    transition_frames: int,
) -> Image.Image:
  del payload
  image = Image.new("RGB", (width, height), BACKGROUND)
  draw = ImageDraw.Draw(image, "RGBA")
  title_font = _load_font(32)
  ui_font = _load_font(18)
  small_font = _load_font(14, mono=True)
  body_font = _load_font(19)

  draw.rectangle((0, 0, width, height), fill=BACKGROUND)
  draw.rectangle((0, 0, width, 96), fill=(10, 17, 27))
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
    draw.rounded_rectangle((mx, 112, mx + 182, 164), radius=8, fill=PANEL)
    draw.text((mx + 14, 119), label, font=small_font, fill=MUTED)
    draw.text((mx + 14, 138), value, font=ui_font, fill=color)
    mx += 194

  output_panel_h = max(132, min(170, height // 4))
  canvas_rect = (36, 186, width - 36, height - output_panel_h - 28)
  _draw_token_canvas(
      draw,
      frame,
      rect=canvas_rect,
      rng=rng,
      transition_phase=transition_phase,
      transition_frames=transition_frames,
      width=width,
      height=height,
  )

  output = _clean_piece(frame["text"], limit=520)
  output_y = height - output_panel_h + 8
  draw.rounded_rectangle(
      (36, output_y, width - 36, height - 24),
      radius=8,
      fill=PANEL,
      outline=GRID,
      width=1,
  )
  draw.text(
      (54, output_y + 14),
      "Current decoded text",
      font=small_font,
      fill=MUTED,
  )
  _draw_wrapped_text(
      draw,
      (54, output_y + 40),
      output,
      font=body_font,
      fill=TEXT,
      max_width=width - 108,
      max_lines=max(2, (output_panel_h - 66) // 25),
      line_height=26,
  )

  return image


def render_gif(args: argparse.Namespace) -> None:
  payload = json.loads(
      pathlib.Path(args.trace_json).read_text(encoding="utf-8")
  )
  frames = payload["frames"]
  rng = random.Random(args.seed)
  images = []
  durations = []
  for index, frame in enumerate(frames):
    for phase in range(args.transition_frames):
      images.append(
          _frame_image(
              payload,
              frame,
              index=index,
              total=len(frames),
              width=args.width,
              height=args.height,
              rng=rng,
              transition_phase=phase,
              transition_frames=args.transition_frames,
          ).convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
      )
      durations.append(args.transition_duration_ms)
    images.append(
        _frame_image(
            payload,
            frame,
            index=index,
            total=len(frames),
            width=args.width,
            height=args.height,
            rng=rng,
            transition_phase=args.transition_frames,
            transition_frames=args.transition_frames + 1,
        ).convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
    )
    durations.append(
        args.hold_ms if index < len(frames) - 1 else args.final_hold_ms
    )
    if index == len(frames) - 1:
      for _ in range(max(0, args.final_hold_frames - 1)):
        images.append(
            _frame_image(
                payload,
                frame,
                index=index,
                total=len(frames),
                width=args.width,
                height=args.height,
                rng=rng,
                transition_phase=args.transition_frames,
                transition_frames=args.transition_frames + 1,
            ).convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        )
        durations.append(args.final_hold_ms)

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
          "final_hold_frames": args.final_hold_frames,
          "total_duration_ms": sum(durations),
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
  parser.add_argument("--transition_frames", type=int, default=5)
  parser.add_argument("--transition_duration_ms", type=int, default=70)
  parser.add_argument("--hold_ms", type=int, default=80)
  parser.add_argument("--final_hold_ms", type=int, default=850)
  parser.add_argument("--final_hold_frames", type=int, default=3)
  return parser.parse_args()


def main() -> None:
  render_gif(parse_args())


if __name__ == "__main__":
  main()
