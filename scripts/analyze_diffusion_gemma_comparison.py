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

"""Analyze matched DiffusionGemma official/Tunix H100x2 training runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import statistics
from typing import Any

try:
  from PIL import Image
  from PIL import ImageDraw
  from PIL import ImageFont
except ImportError:  # pragma: no cover - report still works without plots.
  Image = None
  ImageDraw = None
  ImageFont = None


LOSS_METRICS = (
    "losses/total",
    "losses/diffusion_loss",
    "losses/encoder_loss",
)
WARNING_PATTERNS = (
    "CUDA_ERROR_NOT_PERMITTED",
    "TF-TRT Warning",
    "Unable to register",
    "Could not find cuda drivers",
    "NCCL",
    "Traceback",
    "RuntimeError",
    "OutOfMemory",
    "RESOURCE_EXHAUSTED",
    "nan",
)
COLORS = {
    "official": (64, 158, 255),
    "tunix": (65, 196, 139),
    "grid": (55, 70, 86),
    "text": (226, 236, 244),
    "muted": (143, 157, 171),
    "bg": (11, 16, 24),
    "panel": (18, 27, 39),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--official_dir", required=True)
  parser.add_argument("--tunix_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--title", default="DiffusionGemma H100x2 comparison")
  parser.add_argument("--animation", default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output_dir = pathlib.Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  official = _load_run("official", pathlib.Path(args.official_dir))
  tunix = _load_run("tunix", pathlib.Path(args.tunix_dir))
  comparison = _compare(official, tunix, args.title, args.animation)
  summary_path = output_dir / "comparison_summary.json"
  summary_path.write_text(
      json.dumps(comparison, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )

  plot_paths = {}
  if Image is not None:
    plot_paths["loss_curve"] = _draw_loss_plot(
        official, tunix, output_dir / "loss_curve.png"
    )
    plot_paths["gpu_memory"] = _draw_gpu_plot(
        official, tunix, output_dir / "gpu_memory.png"
    )
  comparison["plots"] = {
      key: str(path.relative_to(output_dir.parent))
      for key, path in plot_paths.items()
      if path is not None
  }

  report_path = output_dir / "README.md"
  report_path.write_text(
      _markdown_report(comparison, official, tunix, plot_paths),
      encoding="utf-8",
  )
  summary_path.write_text(
      json.dumps(comparison, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({"report": str(report_path), "summary": str(summary_path)}))


def _load_run(label: str, path: pathlib.Path) -> dict[str, Any]:
  train_log = path / "train.log"
  gpu_csv = path / "gpu_memory.csv"
  job_result = _read_json(path / "job_result.json")
  run_summary = _read_json(path / "run_summary.json")
  jl_status = _read_json(path / "jl_run_status.json")
  losses, events, warnings = _read_train_log(train_log)
  gpu = _read_gpu_csv(gpu_csv)
  return {
      "label": label,
      "path": str(path),
      "train_log": str(train_log),
      "gpu_csv": str(gpu_csv),
      "job_result": job_result,
      "run_summary": run_summary,
      "jl_status": jl_status,
      "losses": losses,
      "events": events,
      "warnings": warnings,
      "gpu": gpu,
      "artifacts": _artifact_inventory(path),
      "selected": _selected_stats(losses, gpu, job_result, jl_status),
  }


def _read_json(path: pathlib.Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    return {"error": f"{exc!r}", "path": str(path)}
  return value if isinstance(value, dict) else {"value": value}


def _read_train_log(
    path: pathlib.Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, int]]:
  losses: dict[str, list[dict[str, Any]]] = {metric: [] for metric in LOSS_METRICS}
  events: dict[str, dict[str, Any]] = {}
  warnings = {pattern: 0 for pattern in WARNING_PATTERNS}
  if not path.exists():
    return losses, events, warnings

  for line_no, line in enumerate(
      path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
  ):
    lowered = line.lower()
    for pattern in WARNING_PATTERNS:
      if pattern == "nan":
        if "nan" in lowered:
          warnings[pattern] += 1
      elif pattern in line:
        warnings[pattern] += 1
    stripped = line.strip()
    if not stripped.startswith("{"):
      continue
    try:
      payload = json.loads(stripped)
    except json.JSONDecodeError:
      continue
    if not isinstance(payload, dict):
      continue
    event_name = payload.get("event")
    if isinstance(event_name, str):
      entry = events.setdefault(
          event_name, {"count": 0, "first": None, "last": None}
      )
      record = {"line": line_no, **payload}
      entry["count"] += 1
      if entry["first"] is None:
        entry["first"] = record
      entry["last"] = record
    metric = payload.get("metric")
    value = payload.get("value")
    if metric in losses and isinstance(value, (int, float)):
      losses[metric].append({
          "line": line_no,
          "step": int(payload.get("step", payload.get("state_step", -1))),
          "loop_step": int(payload.get("loop_step", -1)),
          "value": float(value),
      })
  return losses, events, warnings


def _read_gpu_csv(path: pathlib.Path) -> dict[str, Any]:
  if not path.exists():
    return {"path": str(path), "samples": 0, "per_gpu": {}, "timeline": []}

  by_gpu: dict[str, list[dict[str, Any]]] = {}
  timeline: list[dict[str, Any]] = []
  with path.open(newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
      index = str(row.get("index", "unknown")).strip()
      try:
        used = float(str(row.get("memory_used_mib", "nan")).strip())
        free = float(str(row.get("memory_free_mib", "nan")).strip())
        util = float(str(row.get("utilization_gpu_percent", "nan")).strip())
        power = float(str(row.get("power_draw_w", "nan")).strip())
      except ValueError:
        continue
      record = {
          "timestamp": str(row.get("timestamp", "")).strip(),
          "index": index,
          "used_mib": used,
          "free_mib": free,
          "util_percent": util,
          "power_w": power,
      }
      by_gpu.setdefault(index, []).append(record)
      timeline.append(record)

  per_gpu = {}
  for index, rows in sorted(by_gpu.items()):
    used = [row["used_mib"] for row in rows if math.isfinite(row["used_mib"])]
    free = [row["free_mib"] for row in rows if math.isfinite(row["free_mib"])]
    util = [
        row["util_percent"] for row in rows if math.isfinite(row["util_percent"])
    ]
    power = [row["power_w"] for row in rows if math.isfinite(row["power_w"])]
    per_gpu[index] = {
        "samples": len(rows),
        "peak_used_mib": max(used) if used else None,
        "median_used_mib": statistics.median(used) if used else None,
        "min_free_mib": min(free) if free else None,
        "mean_util_percent": statistics.fmean(util) if util else None,
        "mean_power_w": statistics.fmean(power) if power else None,
    }

  timestamps = [_parse_timestamp(row["timestamp"]) for row in timeline]
  timestamps = [stamp for stamp in timestamps if stamp is not None]
  duration_seconds = None
  if timestamps:
    duration_seconds = (max(timestamps) - min(timestamps)).total_seconds()
  return {
      "path": str(path),
      "samples": len(timeline),
      "per_gpu": per_gpu,
      "duration_seconds": duration_seconds,
      "timeline": timeline,
  }


def _artifact_inventory(path: pathlib.Path) -> dict[str, Any]:
  entries = {}
  for name in (
      "train.log",
      "gpu_memory.csv",
      "job_result.json",
      "run_summary.json",
      "hybrid_loop_start.json",
      "hybrid_loop_progress.json",
      "hybrid_loop_state.json",
      "checkpoint_inventory.txt",
  ):
    file_path = path / name
    entries[name] = {
        "present": file_path.exists(),
        "bytes": file_path.stat().st_size if file_path.exists() else 0,
    }
  checkpoint_inventory = path / "checkpoint_inventory.txt"
  if checkpoint_inventory.exists():
    lines = [
        line
        for line in checkpoint_inventory.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    entries["checkpoint_inventory.txt"]["line_count"] = len(lines)
  return entries


def _parse_timestamp(value: str) -> dt.datetime | None:
  value = value.strip()
  for fmt in (
      "%Y/%m/%d %H:%M:%S.%f",
      "%Y/%m/%d %H:%M:%S",
      "%a %b %d %H:%M:%S %Y",
  ):
    try:
      return dt.datetime.strptime(value, fmt)
    except ValueError:
      continue
  return None


def _selected_stats(
    losses: dict[str, list[dict[str, Any]]],
    gpu: dict[str, Any],
    job_result: dict[str, Any],
    jl_status: dict[str, Any],
) -> dict[str, Any]:
  selected = {
      "timed_out": job_result.get("timed_out"),
      "train_status": job_result.get("train_status"),
      "exit_code": jl_status.get("exit_code"),
      "state": jl_status.get("state"),
      "gemma_revision": job_result.get("gemma_revision"),
      "hackable_diffusion_revision": job_result.get(
          "hackable_diffusion_revision"
      ),
      "jax_package_spec": job_result.get("jax_package_spec"),
      "train_loop": job_result.get("train_loop"),
      "sync_after_step": job_result.get("sync_after_step"),
  }
  for metric, records in losses.items():
    values = [record["value"] for record in records]
    finite = [value for value in values if math.isfinite(value)]
    selected[metric] = _series_stats(records, values, finite)
  peak_used = []
  min_free = []
  util = []
  power = []
  for stats in gpu.get("per_gpu", {}).values():
    _append_number(peak_used, stats.get("peak_used_mib"))
    _append_number(min_free, stats.get("min_free_mib"))
    _append_number(util, stats.get("mean_util_percent"))
    _append_number(power, stats.get("mean_power_w"))
  selected["gpu"] = {
      "samples": gpu.get("samples", 0),
      "duration_seconds": gpu.get("duration_seconds"),
      "max_peak_used_mib": max(peak_used) if peak_used else None,
      "min_free_mib": min(min_free) if min_free else None,
      "mean_util_percent": statistics.fmean(util) if util else None,
      "mean_power_w": statistics.fmean(power) if power else None,
      "per_gpu": gpu.get("per_gpu", {}),
  }
  steps = selected["losses/total"].get("last_step")
  duration = gpu.get("duration_seconds")
  if isinstance(steps, int) and duration:
    selected["steps_per_hour_by_gpu_window"] = steps / (duration / 3600)
  return selected


def _series_stats(
    records: list[dict[str, Any]], values: list[float], finite: list[float]
) -> dict[str, Any]:
  if not records:
    return {
        "count": 0,
        "finite": True,
        "nan_count": 0,
        "inf_count": 0,
    }
  last_values = finite[-50:] if finite else []
  first_values = finite[:50] if finite else []
  return {
      "count": len(records),
      "first_step": records[0].get("step"),
      "last_step": records[-1].get("step"),
      "first": values[0],
      "last": values[-1],
      "min": min(finite) if finite else None,
      "max": max(finite) if finite else None,
      "mean": statistics.fmean(finite) if finite else None,
      "first_50_mean": statistics.fmean(first_values) if first_values else None,
      "last_50_mean": statistics.fmean(last_values) if last_values else None,
      "finite": len(finite) == len(values),
      "nan_count": sum(math.isnan(value) for value in values),
      "inf_count": sum(math.isinf(value) for value in values),
  }


def _append_number(target: list[float], value: Any) -> None:
  if isinstance(value, (int, float)) and math.isfinite(value):
    target.append(float(value))


def _compare(
    official: dict[str, Any],
    tunix: dict[str, Any],
    title: str,
    animation: str | None,
) -> dict[str, Any]:
  rows = []
  for run in (official, tunix):
    selected = run["selected"]
    rows.append({
        "label": run["label"],
        "state": selected.get("state"),
        "exit_code": selected.get("exit_code"),
        "timed_out": selected.get("timed_out"),
        "steps": selected["losses/total"].get("last_step"),
        "total_first": selected["losses/total"].get("first"),
        "total_last": selected["losses/total"].get("last"),
        "total_last_50_mean": selected["losses/total"].get("last_50_mean"),
        "diffusion_last_50_mean": selected["losses/diffusion_loss"].get(
            "last_50_mean"
        ),
        "encoder_last_50_mean": selected["losses/encoder_loss"].get(
            "last_50_mean"
        ),
        "gpu_peak_used_mib": selected["gpu"].get("max_peak_used_mib"),
        "gpu_min_free_mib": selected["gpu"].get("min_free_mib"),
        "mean_gpu_util_percent": selected["gpu"].get("mean_util_percent"),
        "steps_per_hour_by_gpu_window": selected.get(
            "steps_per_hour_by_gpu_window"
        ),
    })

  return {
      "title": title,
      "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
      "animation": animation,
      "rows": rows,
      "revision_match": {
          "gemma": official["selected"].get("gemma_revision")
          == tunix["selected"].get("gemma_revision"),
          "hackable_diffusion": official["selected"].get(
              "hackable_diffusion_revision"
          )
          == tunix["selected"].get("hackable_diffusion_revision"),
      },
      "runs": {
          "official": official["selected"],
          "tunix": tunix["selected"],
      },
      "warnings": {
          "official": official["warnings"],
          "tunix": tunix["warnings"],
      },
  }


def _draw_loss_plot(
    official: dict[str, Any], tunix: dict[str, Any], path: pathlib.Path
) -> pathlib.Path | None:
  metric_labels = [
      ("losses/total", "total"),
      ("losses/diffusion_loss", "diffusion"),
      ("losses/encoder_loss", "encoder"),
  ]
  series = []
  for run in (official, tunix):
    for metric, name in metric_labels:
      values = [
          (record["step"], record["value"])
          for record in run["losses"][metric]
          if math.isfinite(record["value"])
      ]
      if values:
        series.append((run["label"], name, metric, values))
  if not series:
    return None

  image = Image.new("RGB", (1280, 760), COLORS["bg"])
  draw = ImageDraw.Draw(image)
  font = _load_font(18)
  small = _load_font(14)
  draw.text((36, 26), "Loss curves", font=_load_font(30), fill=COLORS["text"])
  for panel_index, (metric, name) in enumerate(metric_labels):
    x0, y0 = 60, 96 + panel_index * 205
    x1, y1 = 1220, y0 + 160
    _draw_panel(draw, x0, y0, x1, y1)
    draw.text((x0 + 12, y0 + 10), name, font=font, fill=COLORS["text"])
    metric_series = [
        item for item in series if item[2] == metric
    ]
    _plot_series(draw, metric_series, (x0 + 64, y0 + 28, x1 - 22, y1 - 30))
    draw.text((x0 + 64, y1 - 24), "step", font=small, fill=COLORS["muted"])
  _draw_legend(draw, 1040, 30)
  path.parent.mkdir(parents=True, exist_ok=True)
  image.save(path)
  return path


def _draw_gpu_plot(
    official: dict[str, Any], tunix: dict[str, Any], path: pathlib.Path
) -> pathlib.Path | None:
  series = []
  for run in (official, tunix):
    averaged = _average_gpu_timeline(run["gpu"].get("timeline", []))
    if averaged:
      series.append((run["label"], "memory used", "gpu", averaged))
  if not series:
    return None
  image = Image.new("RGB", (1280, 420), COLORS["bg"])
  draw = ImageDraw.Draw(image)
  draw.text(
      (36, 26), "Average GPU memory used", font=_load_font(30), fill=COLORS["text"]
  )
  _draw_panel(draw, 60, 96, 1220, 340)
  _plot_series(draw, series, (122, 128, 1198, 300), y_suffix=" MiB")
  _draw_legend(draw, 1040, 30)
  path.parent.mkdir(parents=True, exist_ok=True)
  image.save(path)
  return path


def _load_font(size: int):
  for candidate in (
      "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
      "/System/Library/Fonts/Helvetica.ttc",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ):
    if pathlib.Path(candidate).exists():
      return ImageFont.truetype(candidate, size=size)
  return ImageFont.load_default()


def _draw_panel(draw, x0: int, y0: int, x1: int, y1: int) -> None:
  draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=COLORS["panel"])
  for i in range(5):
    y = y0 + 28 + i * ((y1 - y0 - 58) // 4)
    draw.line((x0 + 64, y, x1 - 22, y), fill=COLORS["grid"], width=1)


def _plot_series(draw, series, rect, y_suffix: str = "") -> None:
  x0, y0, x1, y1 = rect
  xs = [point[0] for _, _, _, values in series for point in values]
  ys = [point[1] for _, _, _, values in series for point in values]
  if not xs or not ys:
    return
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  if min_y == max_y:
    min_y -= 1.0
    max_y += 1.0
  small = _load_font(13)
  draw.text((x0 - 54, y0 - 4), f"{max_y:.2f}{y_suffix}", font=small, fill=COLORS["muted"])
  draw.text((x0 - 54, y1 - 14), f"{min_y:.2f}{y_suffix}", font=small, fill=COLORS["muted"])
  for label, _, _, values in series:
    color = COLORS[label]
    points = []
    for step, value in values:
      px = x0 if max_x == min_x else x0 + (step - min_x) / (max_x - min_x) * (x1 - x0)
      py = y1 - (value - min_y) / (max_y - min_y) * (y1 - y0)
      points.append((px, py))
    if len(points) == 1:
      x, y = points[0]
      draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    else:
      draw.line(points, fill=color, width=3)


def _draw_legend(draw, x: int, y: int) -> None:
  font = _load_font(15)
  for index, label in enumerate(("official", "tunix")):
    yy = y + index * 24
    draw.rounded_rectangle((x, yy + 3, x + 22, yy + 15), radius=3, fill=COLORS[label])
    draw.text((x + 32, yy), label, font=font, fill=COLORS["text"])


def _average_gpu_timeline(timeline: list[dict[str, Any]]) -> list[tuple[int, float]]:
  by_timestamp: dict[str, list[float]] = {}
  for row in timeline:
    timestamp = row.get("timestamp")
    used = row.get("used_mib")
    if isinstance(timestamp, str) and isinstance(used, (int, float)):
      by_timestamp.setdefault(timestamp, []).append(float(used))
  points = []
  for index, timestamp in enumerate(sorted(by_timestamp)):
    values = by_timestamp[timestamp]
    if values:
      points.append((index, statistics.fmean(values)))
  return points


def _markdown_report(
    comparison: dict[str, Any],
    official: dict[str, Any],
    tunix: dict[str, Any],
    plot_paths: dict[str, pathlib.Path],
) -> str:
  lines = [
      "# DiffusionGemma H100x2 Three-Hour Comparison",
      "",
      "This report compares the official DeepMind DiffusionGemma Hackable Diffusion training path with the Tunix wrapper path under matched H100x2 conditions.",
      "",
      "## Verdict",
      "",
      _verdict_text(comparison),
      "",
      "## Run Matrix",
      "",
      "| Run | State | Exit | Timed out | Steps | Total first | Total last | Total last-50 mean | Peak used MiB | Min free MiB | GPU util mean |",
      "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for row in comparison["rows"]:
    lines.append(
        "| {label} | {state} | {exit_code} | {timed_out} | {steps} | {total_first} | {total_last} | {total_last_50_mean} | {gpu_peak_used_mib} | {gpu_min_free_mib} | {mean_gpu_util_percent} |".format(
            **{key: _fmt(value) for key, value in row.items()}
        )
    )
  lines.extend([
      "",
      "## Loss Detail",
      "",
      "| Run | Metric | Count | First | Last | First-50 mean | Last-50 mean | Min | Max | Finite |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ])
  for run in (official, tunix):
    for metric in LOSS_METRICS:
      stats = run["selected"][metric]
      lines.append(
          "| {run} | {metric} | {count} | {first} | {last} | {first_50_mean} | {last_50_mean} | {min} | {max} | {finite} |".format(
              run=run["label"],
              metric=metric,
              count=_fmt(stats.get("count")),
              first=_fmt(stats.get("first")),
              last=_fmt(stats.get("last")),
              first_50_mean=_fmt(stats.get("first_50_mean")),
              last_50_mean=_fmt(stats.get("last_50_mean")),
              min=_fmt(stats.get("min")),
              max=_fmt(stats.get("max")),
              finite=_fmt(stats.get("finite")),
          )
      )
  lines.extend([
      "",
      "## Environment",
      "",
      "| Field | Official | Tunix wrapper | Match |",
      "| --- | --- | --- | --- |",
  ])
  for field in (
      "gemma_revision",
      "hackable_diffusion_revision",
      "jax_package_spec",
      "train_loop",
      "sync_after_step",
  ):
    official_value = official["selected"].get(field)
    tunix_value = tunix["selected"].get(field)
    lines.append(
        f"| `{field}` | `{_fmt(official_value)}` | `{_fmt(tunix_value)}` | {_fmt(official_value == tunix_value)} |"
    )
  lines.extend([
      "",
      "## Artifact Inventory",
      "",
      "| Artifact | Official | Tunix wrapper |",
      "| --- | ---: | ---: |",
  ])
  artifact_names = sorted(
      set(official["artifacts"]).union(set(tunix["artifacts"]))
  )
  for name in artifact_names:
    lines.append(
        f"| `{name}` | {_artifact_cell(official['artifacts'].get(name))} | {_artifact_cell(tunix['artifacts'].get(name))} |"
    )
  lines.extend([
      "",
      "## Plots",
      "",
  ])
  for title, path in (
      ("Loss curve", plot_paths.get("loss_curve")),
      ("GPU memory", plot_paths.get("gpu_memory")),
  ):
    if path is not None:
      lines.append(f"![{title}]({path.name})")
      lines.append("")
  animation = comparison.get("animation")
  if animation:
    lines.extend([
        "## Generation Animation",
        "",
        f"![DiffusionGemma denoising trace]({pathlib.Path(animation).name})",
        "",
  ])
  lines.extend([
      "## Notes",
      "",
      "- The two long runs are independent stochastic training runs. Exact step-by-step loss identity is not expected; parity of deterministic helper paths and logits is covered by the dedicated parity scripts in `scripts/verify_diffusion_gemma_official_parity.py` and `scripts/verify_diffusion_gemma_official_logits.py`.",
      "- The Tunix path intentionally wraps the official Hackable Diffusion backend for the GPU-heavy model/loss path, while exposing a Tunix-facing integration layer. This avoids reimplementing the fragile multi-GPU diffusion internals before a full NNX/Qwix port.",
      "- CUDA VMM/TensorFlow GPU visibility warnings were counted when present. They are treated as non-fatal only when the training process reaches finite losses and exits with code 0.",
      "",
      "## Warning Counts",
      "",
      "| Pattern | Official | Tunix wrapper |",
      "| --- | ---: | ---: |",
  ])
  for pattern in WARNING_PATTERNS:
    lines.append(
        f"| `{pattern}` | {official['warnings'].get(pattern, 0)} | {tunix['warnings'].get(pattern, 0)} |"
    )
  lines.append("")
  return "\n".join(lines)


def _verdict_text(comparison: dict[str, Any]) -> str:
  rows = comparison["rows"]
  finite = all(
      run["losses/total"].get("finite")
      for run in comparison["runs"].values()
      if isinstance(run.get("losses/total"), dict)
  )
  ok_exit = all(row.get("exit_code") in (0, "0") for row in rows)
  steps = [row.get("steps") for row in rows if isinstance(row.get("steps"), int)]
  if finite and ok_exit and steps:
    return (
        "Both matched H100x2 runs completed the requested timed training window "
        "with finite losses. The wrapper is therefore operational for this "
        "official-backend LoRA training path; it is not yet a native NNX/Qwix "
        "DiffusionGemma implementation."
    )
  return (
      "The comparison did not fully satisfy the pass criteria. Inspect the run "
      "matrix, warning counts, and raw logs before treating this integration as "
      "ready for longer training."
  )


def _fmt(value: Any) -> str:
  if value is None:
    return "-"
  if isinstance(value, bool):
    return "yes" if value else "no"
  if isinstance(value, float):
    if not math.isfinite(value):
      return str(value)
    if abs(value) >= 100:
      return f"{value:.1f}"
    return f"{value:.4f}".rstrip("0").rstrip(".")
  return str(value)


def _artifact_cell(entry: dict[str, Any] | None) -> str:
  if not entry or not entry.get("present"):
    return "missing"
  size = int(entry.get("bytes", 0))
  line_count = entry.get("line_count")
  if isinstance(line_count, int):
    return f"{size} B, {line_count} lines"
  return f"{size} B"


if __name__ == "__main__":
  main()
