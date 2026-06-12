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

"""Summarizes DiffusionGemma training metrics from event files and logs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
from typing import Any


_LOG_METRIC_RE = re.compile(
    r"(?P<name>(?:losses|perf_stats|timers|stats)/[A-Za-z0-9_./-]+)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_STEP_RE = re.compile(r"(?:step[ =]|global_step=)(?P<step>\d+)")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--workdir", required=True)
  parser.add_argument("--log", default=None)
  parser.add_argument("--gpu_csv", default=None)
  parser.add_argument("--output", default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  workdir = pathlib.Path(args.workdir)
  summary = {
      "workdir": str(workdir),
      "event_scalars": _read_event_scalars(workdir),
      "log_scalars": (
          _read_log_scalars(pathlib.Path(args.log)) if args.log else {}
      ),
      "gpu_memory": (
          _read_gpu_csv(pathlib.Path(args.gpu_csv)) if args.gpu_csv else {}
      ),
  }
  summary["selected"] = _select_metrics(summary)
  text = json.dumps(summary, indent=2, sort_keys=True)
  if args.output:
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
  print(text)


def _read_event_scalars(workdir: pathlib.Path) -> dict[str, Any]:
  try:
    from tensorboard.backend.event_processing import event_accumulator
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return {"error": f"tensorboard unavailable: {exc!r}", "metrics": {}}

  metrics: dict[str, list[dict[str, float | int]]] = {}
  event_dirs = {
      path.parent
      for path in workdir.rglob("events.out.tfevents*")
      if path.is_file()
  }
  for event_dir in sorted(event_dirs):
    try:
      accumulator = event_accumulator.EventAccumulator(str(event_dir))
      accumulator.Reload()
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    for tag in accumulator.Tags().get("scalars", []):
      if not _is_interesting_metric(tag):
        continue
      for event in accumulator.Scalars(tag):
        metrics.setdefault(tag, []).append({
            "step": int(event.step),
            "wall_time": float(event.wall_time),
            "value": float(event.value),
        })
  return {
      "event_dirs": [str(path) for path in sorted(event_dirs)],
      "metrics": _summarize_series(metrics),
  }


def _read_log_scalars(log_path: pathlib.Path) -> dict[str, Any]:
  if not log_path.exists():
    return {"error": f"log not found: {log_path}", "metrics": {}}

  metrics: dict[str, list[dict[str, float | int | None]]] = {}
  for line_no, line in enumerate(
      log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
  ):
    _read_json_line_scalars(line, line_no, metrics)
    step_match = _STEP_RE.search(line)
    step = int(step_match.group("step")) if step_match else None
    for match in _LOG_METRIC_RE.finditer(line):
      metrics.setdefault(match.group("name"), []).append({
          "step": step,
          "line": line_no,
          "value": float(match.group("value")),
      })
  return {
      "path": str(log_path),
      "metrics": _summarize_series(metrics),
  }


def _read_json_line_scalars(
    line: str,
    line_no: int,
    metrics: dict[str, list[dict[str, float | int | None]]],
) -> None:
  line = line.strip()
  if not line.startswith("{"):
    return
  try:
    payload = json.loads(line)
  except json.JSONDecodeError:
    return
  if not isinstance(payload, dict):
    return

  step = payload.get("state_step", payload.get("step"))
  if step is not None:
    try:
      step = int(step)
    except (TypeError, ValueError):
      step = None

  for container_name, prefix in (("losses", "losses"), ("metrics", "metrics")):
    values = payload.get(container_name)
    if not isinstance(values, dict):
      continue
    for name, value in values.items():
      if not isinstance(value, (int, float)):
        continue
      metric_name = name if "/" in name else f"{prefix}/{name}"
      metrics.setdefault(metric_name, []).append({
          "step": step,
          "line": line_no,
          "value": float(value),
      })


def _read_gpu_csv(path: pathlib.Path) -> dict[str, Any]:
  if not path.exists():
    return {"error": f"gpu csv not found: {path}"}

  used_by_gpu: dict[str, list[float]] = {}
  free_by_gpu: dict[str, list[float]] = {}
  with path.open(newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
      index = str(row.get("index", "unknown"))
      try:
        used = float(row.get("memory_used_mib", "nan"))
        free = float(row.get("memory_free_mib", "nan"))
      except ValueError:
        continue
      used_by_gpu.setdefault(index, []).append(used)
      free_by_gpu.setdefault(index, []).append(free)
  return {
      "path": str(path),
      "samples": sum(len(values) for values in used_by_gpu.values()),
      "per_gpu": {
          index: {
              "peak_used_mib": max(values),
              "median_used_mib": statistics.median(values),
              "min_free_mib": min(free_by_gpu.get(index, [0.0])),
          }
          for index, values in sorted(used_by_gpu.items())
          if values
      },
  }


def _summarize_series(
    metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
  summary = {}
  for name, records in sorted(metrics.items()):
    if not records:
      continue
    values = [float(record["value"]) for record in records]
    first = records[0]
    last = records[-1]
    summary[name] = {
        "count": len(records),
        "first": first,
        "last": last,
        "min": min(values),
        "max": max(values),
    }
  return summary


def _select_metrics(summary: dict[str, Any]) -> dict[str, Any]:
  selected = {}
  event_metrics = summary.get("event_scalars", {}).get("metrics", {})
  log_metrics = summary.get("log_scalars", {}).get("metrics", {})
  for name in (
      "losses/total",
      "losses/diffusion_loss",
      "losses/encoder_loss",
      "perf_stats/train/avg_time_sec",
  ):
    selected[name] = event_metrics.get(name) or log_metrics.get(name)
  return {key: value for key, value in selected.items() if value is not None}


def _is_interesting_metric(tag: str) -> bool:
  return tag.startswith(("losses/", "perf_stats/", "timers/", "stats/"))


if __name__ == "__main__":
  main()
