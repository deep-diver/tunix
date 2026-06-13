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

"""Compare DiffusionGemma native Qwix LoRA coverage to official all-linear."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

from flax import nnx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tunix.models.diffusion_gemma import lora_inventory
from tunix.models.diffusion_gemma import model as diffusion_model
from tunix.models.diffusion_gemma import sft as diffusion_sft


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rank", type=int, default=2)
  parser.add_argument("--alpha", type=float, default=4.0)
  parser.add_argument(
      "--moe_lora_targets",
      default="router_logits",
      help=(
          "Comma-separated MoE raw-param targets for the native path. "
          "Use router_logits for the official-compatible default; add "
          "gating_einsum,linear to expose experimental extras."
      ),
  )
  parser.add_argument("--output_json", default=None)
  parser.add_argument("--fail_on_mismatch", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  comparison = build_comparison(
      rank=args.rank,
      alpha=args.alpha,
      moe_lora_targets=_parse_targets(args.moe_lora_targets),
  )
  payload = comparison.as_dict()
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if args.output_json:
    pathlib.Path(args.output_json).write_text(text, encoding="utf-8")
  print(text, end="")
  if args.fail_on_mismatch and not comparison.matches_official:
    raise SystemExit(1)


def build_comparison(
    *,
    rank: int = 2,
    alpha: float = 4.0,
    moe_lora_targets: tuple[str, ...] = ("router_logits",),
) -> lora_inventory.LoRATargetComparison:
  cfg = dataclasses.replace(
      diffusion_model.ModelConfig.tiny(
          vocab_size=32,
          num_layers=1,
          embed_dim=16,
          hidden_dim=32,
          num_heads=2,
          head_dim=8,
          num_kv_heads=1,
      ),
      enable_moe=True,
      num_experts=4,
      num_experts_per_tok=2,
      expert_dim=8,
      moe_dense_hidden_dim=16,
  )
  model = diffusion_model.DiffusionGemma_A26B_A4B(cfg, rngs=nnx.Rngs(0))
  model = diffusion_sft.apply_lora(
      model,
      rank=rank,
      alpha=alpha,
      moe_target_names=moe_lora_targets,
  )
  return lora_inventory.compare_model_to_official_all_linear(model)


def _parse_targets(value: str) -> tuple[str, ...]:
  targets = tuple(part.strip() for part in value.split(",") if part.strip())
  valid = {"router_logits", "gating_einsum", "linear"}
  invalid = sorted(set(targets) - valid)
  if invalid:
    raise ValueError(f"Invalid MoE LoRA targets: {invalid}")
  return targets


if __name__ == "__main__":
  main()
