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

"""Runs DiffusionGemma through the official Hackable Diffusion backend.

This is a Tunix compatibility entrypoint. It does not reimplement the official
SFT math in NNX/Qwix; instead it loads the official Gemma/Hackable Diffusion
recipe and applies only run-environment overrides such as checkpoint path,
workdir, and training step count.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _load_hackable_adapter():
  module_path = (
      REPO_ROOT / "tunix" / "models" / "diffusion_gemma" / "hackable_adapter.py"
  )
  spec = importlib.util.spec_from_file_location(
      "_tunix_diffusion_gemma_hackable_adapter", module_path
  )
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load hackable_adapter from {module_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


hackable_adapter = _load_hackable_adapter()


def _parse_key_value(items: list[str]) -> dict[str, Any]:
  values: dict[str, Any] = {}
  for item in items:
    key, sep, raw_value = item.partition("=")
    if not sep or not key:
      raise ValueError(f"Override {item!r} must use dotted.path=value syntax.")
    try:
      values[key] = json.loads(raw_value)
    except json.JSONDecodeError:
      values[key] = raw_value
  return values


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--recipe",
      choices=["pubmedqa", "sudoku", "sudoku_full"],
      default="pubmedqa",
  )
  parser.add_argument("--gemma_ref", default=None)
  parser.add_argument("--hackable_diffusion_ref", default=None)
  parser.add_argument("--workdir", default=None)
  parser.add_argument("--checkpoint_path", default=None)
  parser.add_argument("--num_train_steps", type=int, default=1)
  parser.add_argument(
      "--run_steps",
      type=int,
      default=None,
      help=(
          "Limit the hybrid loop to this many local steps without changing "
          "the official recipe's num_train_steps schedule."
      ),
  )
  parser.add_argument("--checkpoint_every_n_steps", type=int, default=None)
  parser.add_argument("--lora_rank", type=int, default=None)
  parser.add_argument(
      "--dataset_batch_size",
      type=int,
      default=None,
      help=(
          "Override official dataset builder batch_size kwargs. Unset keeps "
          "the official recipe default."
      ),
  )
  parser.add_argument(
      "--use_early_stopping",
      action=argparse.BooleanOptionalAction,
      default=None,
  )
  parser.add_argument(
      "--disable_evals",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--build_config_only",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Only import the official recipe and build the overridden Kauldron "
          "config. This validates the compatibility wrapper without loading "
          "the public checkpoint or launching training."
      ),
  )
  parser.add_argument(
      "--skip_step_metrics",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Skip official Kauldron per-step metric materialization after the "
          "train step. This is intended for environments with multi-GPU NCCL "
          "metric-gather failures."
      ),
  )
  parser.add_argument(
      "--log_losses",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "In the hybrid loop, request and log loss states from the official "
          "train step. Disable this to isolate train-step execution from "
          "host-side loss materialization."
      ),
  )
  parser.add_argument(
      "--sync_after_step",
      choices=["state", "losses", "none"],
      default="state",
      help=(
          "Hybrid-loop synchronization point. 'state' preserves the strict "
          "device-state block; 'losses' synchronizes by reading addressable "
          "loss shards only; 'none' only dispatches the step."
      ),
  )
  parser.add_argument(
      "--train_loop",
      choices=["kauldron", "hybrid"],
      default="kauldron",
      help=(
          "Use the official Kauldron Trainer loop, or a Tunix-owned hybrid "
          "loop that reuses official model/data/loss/trainstep objects but "
          "avoids Kauldron post-step metric and final-sync paths."
      ),
  )
  parser.add_argument(
      "--config_override",
      action="append",
      default=[],
      help=(
          "Post-build Kauldron config override as dotted.path=JSON. Example: "
          "--config_override aux.eval_num_batches=1"
      ),
  )
  parser.add_argument(
      "--module_override",
      action="append",
      default=[],
      help=(
          "Pre-build official recipe module override as NAME=JSON. Use for "
          "official path constants that must be changed before get_config()."
      ),
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = hackable_adapter.OfficialSFTConfig(
      recipe=args.recipe,
      gemma_ref=args.gemma_ref,
      hackable_diffusion_ref=args.hackable_diffusion_ref,
      workdir=args.workdir,
      checkpoint_path=args.checkpoint_path,
      num_train_steps=args.num_train_steps,
      run_steps=args.run_steps,
      checkpoint_every_n_steps=args.checkpoint_every_n_steps,
      lora_rank=args.lora_rank,
      dataset_batch_size=args.dataset_batch_size,
      skip_step_metrics=args.skip_step_metrics,
      log_losses=args.log_losses,
      sync_after_step=args.sync_after_step,
      train_loop=args.train_loop,
      use_early_stopping=args.use_early_stopping,
      disable_evals=args.disable_evals,
      module_overrides=_parse_key_value(args.module_override),
      config_overrides=_parse_key_value(args.config_override),
  )
  dependency_report = hackable_adapter.check_dependencies(
      [path for path in (args.hackable_diffusion_ref, args.gemma_ref) if path]
  )
  print(
      json.dumps(
          {
              "event": "official_backend_dependencies",
              **dependency_report,
          },
          default=str,
      ),
      flush=True,
  )
  if not dependency_report["available"]:
    raise SystemExit("Official DiffusionGemma backend dependencies missing.")
  trainer = hackable_adapter.OfficialDiffusionGemmaTrainer(config)
  if args.build_config_only:
    cfg = trainer.build_config()
    print(
        json.dumps(
            {
                "event": "official_backend_config_built",
                "recipe": args.recipe,
                "workdir": getattr(cfg, "workdir", None),
                "num_train_steps": getattr(cfg, "num_train_steps", None),
                "checkpoint_path": getattr(
                    getattr(cfg, "init_transform", None), "path", None
                ),
                "use_lora": getattr(
                    getattr(cfg, "aux", None), "use_lora", None
                ),
                "lora_rank": getattr(
                    getattr(cfg, "aux", None), "lora_rank", None
                ),
                "dataset_batch_size": args.dataset_batch_size,
                "prompt_len": getattr(
                    getattr(cfg, "aux", None), "prompt_len", None
                ),
                "num_canvases": getattr(
                    getattr(cfg, "aux", None), "num_canvases", None
                ),
                "canvas_size": getattr(
                    getattr(cfg, "aux", None), "canvas_size", None
                ),
                "evals": sorted(getattr(cfg, "evals", {}).keys()),
            },
            default=str,
        ),
        flush=True,
    )
    return
  trainer.train()
  print(
      json.dumps({
          "event": "official_backend_train_complete",
          "recipe": args.recipe,
          "workdir": args.workdir,
          "num_train_steps": args.num_train_steps,
          "run_steps": args.run_steps,
          "skip_step_metrics": args.skip_step_metrics,
          "log_losses": args.log_losses,
          "sync_after_step": args.sync_after_step,
          "train_loop": args.train_loop,
      }),
      flush=True,
  )


if __name__ == "__main__":
  main()
