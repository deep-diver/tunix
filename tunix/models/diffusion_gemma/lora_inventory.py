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

"""LoRA target inventory helpers for DiffusionGemma.

The official Hackable Diffusion backend applies LoRA to Linen modules selected
by ``target_modules="all-linear"``. Native Tunix uses Qwix on an NNX graph, so
the module paths are not textually identical. This module normalizes both sides
to canonical target families before comparing coverage.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

from flax import nnx
import jax


# Mirrors the effective official ``hd.lora.LoRA(..., target_modules='all-linear')``
# policy for DiffusionGemma/Gemma4 modules. The official supported-module tuple
# includes Dense/Einsum variants, so MoERagged's router Einsum is covered while
# its raw expert _Weight leaves are not.
OFFICIAL_ALL_LINEAR_FAMILIES: frozenset[str] = frozenset({
    "attention.attn_vec_einsum",
    "attention.k_einsum",
    "attention.kv_einsum",
    "attention.q_einsum",
    "ffw.gating_einsum",
    "ffw.linear",
    "moe.router_logits",
    "self_conditioner.ffw.gating_einsum",
    "self_conditioner.ffw.linear",
})


OFFICIAL_UNSUPPORTED_RAW_WEIGHT_FAMILIES: frozenset[str] = frozenset({
    "moe.gating_einsum",
    "moe.linear",
})

OFFICIAL_ATTENTION_KV_VARIANT_FAMILIES: frozenset[str] = frozenset({
    "attention.k_einsum",
    "attention.kv_einsum",
})


@dataclasses.dataclass(frozen=True)
class LoRATargetInventory:
  """Canonical LoRA target inventory."""

  leaf_paths: tuple[str, ...]
  family_to_leaf_paths: Mapping[str, tuple[str, ...]]
  unknown_leaf_paths: tuple[str, ...] = ()

  @property
  def families(self) -> frozenset[str]:
    return frozenset(self.family_to_leaf_paths)

  def as_dict(self) -> dict[str, Any]:
    return {
        "families": sorted(self.families),
        "family_to_leaf_paths": {
            family: list(paths)
            for family, paths in sorted(self.family_to_leaf_paths.items())
        },
        "leaf_count": len(self.leaf_paths),
        "leaf_paths": list(self.leaf_paths),
        "unknown_leaf_paths": list(self.unknown_leaf_paths),
    }


@dataclasses.dataclass(frozen=True)
class LoRATargetComparison:
  """Comparison against the official all-linear target policy."""

  inventory: LoRATargetInventory
  official_families: frozenset[str] = OFFICIAL_ALL_LINEAR_FAMILIES

  @property
  def missing_from_inventory(self) -> frozenset[str]:
    missing = (
        self.official_families
        - OFFICIAL_ATTENTION_KV_VARIANT_FAMILIES
        - self.inventory.families
    )
    if not (
        self.inventory.families & OFFICIAL_ATTENTION_KV_VARIANT_FAMILIES
    ):
      missing = missing | frozenset({"attention.k_or_kv_einsum"})
    return missing

  @property
  def extra_in_inventory(self) -> frozenset[str]:
    return self.inventory.families - self.official_families

  @property
  def matches_official(self) -> bool:
    return (
        not self.missing_from_inventory
        and not self.extra_in_inventory
        and not self.inventory.unknown_leaf_paths
    )

  def as_dict(self) -> dict[str, Any]:
    return {
        "matches_official": self.matches_official,
        "missing_from_inventory": sorted(self.missing_from_inventory),
        "extra_in_inventory": sorted(self.extra_in_inventory),
        "official_families": sorted(self.official_families),
        "official_attention_kv_variants": sorted(
            OFFICIAL_ATTENTION_KV_VARIANT_FAMILIES
        ),
        "official_unsupported_raw_weight_families": sorted(
            OFFICIAL_UNSUPPORTED_RAW_WEIGHT_FAMILIES
        ),
        "inventory": self.inventory.as_dict(),
    }


def inventory_from_lora_state(lora_state: Mapping[str, Any]) -> LoRATargetInventory:
  """Builds an inventory from an ``nnx.state(model, nnx.LoRAParam)`` dict."""
  leaf_paths = _flatten_lora_paths(lora_state)
  family_paths: dict[str, list[str]] = {}
  unknown = []
  for path in leaf_paths:
    family = canonical_family_for_lora_path(path)
    if family is None:
      unknown.append(path)
    else:
      family_paths.setdefault(family, []).append(path)
  return LoRATargetInventory(
      leaf_paths=tuple(leaf_paths),
      family_to_leaf_paths={
          family: tuple(paths) for family, paths in family_paths.items()
      },
      unknown_leaf_paths=tuple(unknown),
  )


def inventory_from_model(model: nnx.Module) -> LoRATargetInventory:
  """Builds an inventory from the LoRA leaves attached to an NNX model."""
  return inventory_from_lora_state(nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam)))


def compare_model_to_official_all_linear(
    model: nnx.Module,
) -> LoRATargetComparison:
  """Compares an NNX/Qwix LoRA model against official all-linear coverage."""
  return LoRATargetComparison(inventory=inventory_from_model(model))


def canonical_family_for_lora_path(path: str) -> str | None:
  """Normalizes an NNX/Qwix LoRA leaf path to an official target family."""
  parts = path.split("/")
  if len(parts) < 2:
    return None

  if parts[:2] == ["self_conditioner", "ffw"]:
    if len(parts) >= 3 and parts[2] in (
        "gate_proj",
        "gating_einsum",
        "up_proj",
    ):
      return "self_conditioner.ffw.gating_einsum"
    if len(parts) >= 3 and parts[2] in ("down_proj", "linear"):
      return "self_conditioner.ffw.linear"
    return None

  if "attn" in parts:
    idx = parts.index("attn")
    if len(parts) > idx + 1 and parts[idx + 1] in (
        "attn_vec_einsum",
        "k_einsum",
        "kv_einsum",
        "q_einsum",
    ):
      return f"attention.{parts[idx + 1]}"

  if "mlp" in parts:
    idx = parts.index("mlp")
    if len(parts) > idx + 1 and parts[idx + 1] in (
        "gate_proj",
        "gating_einsum",
        "up_proj",
    ):
      return "ffw.gating_einsum"
    if len(parts) > idx + 1 and parts[idx + 1] in ("down_proj", "linear"):
      return "ffw.linear"
    if len(parts) > idx + 1 and parts[idx + 1] == "router_logits":
      return "moe.router_logits"

  if "mlp2" in parts:
    idx = parts.index("mlp2")
    if len(parts) > idx + 1 and parts[idx + 1] == "gating_einsum":
      return "ffw.gating_einsum"
    if len(parts) > idx + 1 and parts[idx + 1] == "linear":
      return "ffw.linear"
    if len(parts) > idx + 1 and parts[idx + 1] == "router_logits":
      return "moe.router_logits"

  if "moe" in parts:
    idx = parts.index("moe")
    if len(parts) > idx + 1:
      name = parts[idx + 1].removesuffix("_lora_a").removesuffix("_lora_b")
    else:
      name = ""
    if name in ("gating_einsum", "linear", "router_logits"):
      return f"moe.{name}"

  return None


def _flatten_lora_paths(lora_state: Mapping[str, Any]) -> tuple[str, ...]:
  leaves = jax.tree_util.tree_flatten_with_path(lora_state)[0]
  return tuple(
      "/".join(str(part.key) for part in path)
      for path, _ in leaves
  )
