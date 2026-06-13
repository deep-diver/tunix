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

"""Experimental Qwix LoRA bridge for Linen DiffusionGemma models.

The validated H100x2 path still uses the official Hackable Diffusion LoRA
wrapper. This module is the first small step toward replacing that LoRA layer
with a Tunix/Qwix policy while keeping the official Linen model and method
surface intact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import flax.traverse_util
import jax

from tunix.models.diffusion_gemma import lora_inventory


DIFFUSION_GEMMA_LINEN_LORA_METHODS: tuple[str, ...] = (
    "__call__",
    "encoder_call",
    "init_cache",
)


OFFICIAL_COMPATIBLE_LINEN_TARGET_PATTERNS: tuple[str, ...] = (
    r"(.*/)?attn/(q_einsum|kv_einsum|k_einsum|attn_vec_einsum)$",
    r"(.*/)?(mlp|mlp2)/(gating_einsum|linear|router_logits)$",
    r"(.*/)?self_conditioner/ffw/(gating_einsum|linear)$",
)


def official_compatible_module_path(
    patterns: Sequence[str] = OFFICIAL_COMPATIBLE_LINEN_TARGET_PATTERNS,
) -> str:
  """Returns a single Qwix ``module_path`` regex for official-like coverage."""
  if not patterns:
    raise ValueError("At least one Linen LoRA target pattern is required.")
  return "|".join(f"(?:{pattern})" for pattern in patterns)


def create_linen_lora_provider(
    *,
    rank: int,
    alpha: float,
    module_path: str | None = None,
    dropout: float = 0.0,
):
  """Creates a Qwix LoRA provider for Linen DiffusionGemma modules."""
  import qwix  # pylint: disable=g-import-not-at-top

  return qwix.LoraProvider(
      module_path=module_path or official_compatible_module_path(),
      rank=rank,
      alpha=alpha,
      dropout=dropout,
  )


def apply_lora_to_linen_model(
    model: Any,
    *,
    rank: int,
    alpha: float,
    module_path: str | None = None,
    methods: Sequence[str] = DIFFUSION_GEMMA_LINEN_LORA_METHODS,
    dropout: float = 0.0,
) -> Any:
  """Applies Qwix LoRA to a Linen model with DiffusionGemma call coverage.

  DiffusionGemma uses more than ``__call__`` during SFT and generation. The
  official wrapper keeps LoRA active for ``encoder_call`` and ``init_cache`` as
  well; this bridge mirrors that method set for Qwix.
  """
  import qwix  # pylint: disable=g-import-not-at-top

  provider = create_linen_lora_provider(
      rank=rank,
      alpha=alpha,
      module_path=module_path,
      dropout=dropout,
  )
  return qwix.apply_lora_to_model(model, provider, methods=tuple(methods))


def inventory_from_linen_params(
    params: Mapping[str, Any],
) -> lora_inventory.LoRATargetInventory:
  """Builds a canonical inventory from a Linen Qwix-LoRA param tree."""
  flat = flax.traverse_util.flatten_dict(params)
  leaf_paths = tuple(
      "/".join(str(part) for part in path)
      for path, _ in flat.items()
      if _is_lora_leaf_path(path)
  )
  family_paths: dict[str, list[str]] = {}
  unknown = []
  for path in leaf_paths:
    family = lora_inventory.canonical_family_for_lora_path(path)
    if family is None:
      unknown.append(path)
    else:
      family_paths.setdefault(family, []).append(path)
  return lora_inventory.LoRATargetInventory(
      leaf_paths=leaf_paths,
      family_to_leaf_paths={
          family: tuple(paths) for family, paths in family_paths.items()
      },
      unknown_leaf_paths=tuple(unknown),
  )


def _is_lora_leaf_path(path: tuple[Any, ...]) -> bool:
  if not path:
    return False
  leaf_name = str(path[-1])
  if leaf_name.endswith(("_lora_a", "_lora_b")):
    return True
  tail = tuple(str(part) for part in path[-2:])
  return len(path) >= 2 and tail in (("lora", "a"), ("lora", "b"))
