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
import dataclasses
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp

try:
  from . import lora_inventory
except ImportError:  # Allows direct importlib loading without importing tunix.
  import importlib.util
  import pathlib
  import sys

  _LORA_INVENTORY_PATH = pathlib.Path(__file__).with_name("lora_inventory.py")
  _SPEC = importlib.util.spec_from_file_location(
      "_tunix_diffusion_gemma_lora_inventory", _LORA_INVENTORY_PATH
  )
  if _SPEC is None or _SPEC.loader is None:
    raise
  lora_inventory = importlib.util.module_from_spec(_SPEC)
  sys.modules[_SPEC.name] = lora_inventory
  _SPEC.loader.exec_module(lora_inventory)


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinenQwixLoRAConfig:
  """Qwix LoRA/QLoRA configuration for Linen DiffusionGemma models."""

  rank: int
  alpha: float
  module_path: str | None = None
  dropout: float = 0.0
  weight_qtype: str | type[Any] | jnp.dtype | None = None
  act_qtype: str | type[Any] | jnp.dtype | None = None
  tile_size: int | float | None = None
  weight_calibration_method: str = "absmax"
  act_calibration_method: str | None = None

  @property
  def is_qlora(self) -> bool:
    return self.weight_qtype is not None or self.act_qtype is not None


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
    weight_qtype: str | type[Any] | jnp.dtype | None = None,
    act_qtype: str | type[Any] | jnp.dtype | None = None,
    tile_size: int | float | None = None,
    weight_calibration_method: str = "absmax",
    act_calibration_method: str | None = None,
):
  """Creates a Qwix LoRA/QLoRA provider for Linen DiffusionGemma modules."""
  import qwix  # pylint: disable=g-import-not-at-top

  kwargs = {
      "module_path": module_path or official_compatible_module_path(),
      "rank": rank,
      "alpha": alpha,
      "dropout": dropout,
  }
  if weight_qtype is not None:
    kwargs["weight_qtype"] = weight_qtype
  if act_qtype is not None:
    kwargs["act_qtype"] = act_qtype
  if tile_size is not None:
    kwargs["tile_size"] = tile_size
  if weight_calibration_method is not None:
    kwargs["weight_calibration_method"] = weight_calibration_method
  if act_calibration_method is not None:
    kwargs["act_calibration_method"] = act_calibration_method
  return _NoDebugAttrLoraProvider(**kwargs)


def create_linen_lora_provider_from_config(config: LinenQwixLoRAConfig):
  """Creates a Qwix provider from a structured Linen bridge config."""
  return create_linen_lora_provider(
      rank=config.rank,
      alpha=config.alpha,
      module_path=config.module_path,
      dropout=config.dropout,
      weight_qtype=config.weight_qtype,
      act_qtype=config.act_qtype,
      tile_size=config.tile_size,
      weight_calibration_method=config.weight_calibration_method,
      act_calibration_method=config.act_calibration_method,
  )


def apply_lora_to_linen_model_from_config(
    model: Any,
    config: LinenQwixLoRAConfig,
    *,
    methods: Sequence[str] = DIFFUSION_GEMMA_LINEN_LORA_METHODS,
) -> Any:
  """Applies Qwix LoRA/QLoRA to a Linen model from a structured config."""
  return _apply_provider_to_linen_model(
      model,
      create_linen_lora_provider_from_config(config),
      methods=methods,
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
  """Applies Qwix LoRA to a Linen model with DiffusionGemma call coverage."""
  return apply_lora_to_linen_model_from_config(
      model,
      LinenQwixLoRAConfig(
          rank=rank,
          alpha=alpha,
          module_path=module_path,
          dropout=dropout,
      ),
      methods=methods,
  )


def apply_qlora_to_linen_model(
    model: Any,
    *,
    rank: int,
    alpha: float,
    weight_qtype: str | type[Any] | jnp.dtype = "int4",
    module_path: str | None = None,
    methods: Sequence[str] = DIFFUSION_GEMMA_LINEN_LORA_METHODS,
    dropout: float = 0.0,
    act_qtype: str | type[Any] | jnp.dtype | None = None,
    tile_size: int | float | None = None,
    weight_calibration_method: str = "absmax",
    act_calibration_method: str | None = None,
) -> Any:
  """Applies Qwix QLoRA to a Linen model with DiffusionGemma call coverage."""
  return apply_lora_to_linen_model_from_config(
      model,
      LinenQwixLoRAConfig(
          rank=rank,
          alpha=alpha,
          module_path=module_path,
          dropout=dropout,
          weight_qtype=weight_qtype,
          act_qtype=act_qtype,
          tile_size=tile_size,
          weight_calibration_method=weight_calibration_method,
          act_calibration_method=act_calibration_method,
      ),
      methods=methods,
  )


def _apply_provider_to_linen_model(
    model: Any,
    provider: Any,
    *,
    methods: Sequence[str],
) -> Any:
  """Applies a Qwix provider to a Linen model with DiffusionGemma methods."""
  import qwix  # pylint: disable=g-import-not-at-top

  return qwix.apply_lora_to_model(model, provider, methods=tuple(methods))


def _qwix_lora_module():
  import qwix._src.providers.lora as qwix_lora  # pylint: disable=g-import-not-at-top

  return qwix_lora


class _NoDebugAttrLoraProvider:
  """Qwix LoRA provider that does not mutate frozen Linen modules for debug.

  Qwix's standard Linen einsum path stores ``*_lora_einsum_str`` on the current
  module for debugging. Official Gemma custom Linen modules can be frozen at
  that interception point, so this bridge keeps Qwix's math and parameter
  creation but skips that debug-only attribute write.
  """

  def __new__(cls, *args, **kwargs):
    qwix_lora = _qwix_lora_module()

    class Provider(qwix_lora.LoraProvider):

      def einsum(self, einsum_str: str, *operands, **kwargs):  # pylint: disable=missing-function-docstring
        res = super().einsum(einsum_str, *operands, **kwargs)

        rule, _ = self._get_current_rule_and_op_id(
            "einsum", repeated_call=True
        )
        if not isinstance(rule, qwix_lora.LoraRule):
          return res
        if len(operands) != 2:
          raise ValueError(
              f"Unsupported einsum format: {einsum_str=} {operands=}"
          )
        lhs, rhs = operands
        weight_name = qwix_lora.flax_util.find_param(
            rhs, qwix_lora.ptq.WithAux
        )
        if weight_name is None:
          return res

        (
            a_shape,
            b_shape,
            lora_einsum_str,
            a_sharding_transpose,
            b_sharding_transpose,
        ) = qwix_lora._parse_einsum_str_for_lora(
            lhs.shape, rhs.shape, einsum_str, rule.rank
        )
        lora_a, lora_b = qwix_lora._get_or_create_lora_params(
            name=weight_name,
            rule=rule,
            a_shape=a_shape,
            b_shape=b_shape,
            a_sharding_transpose=a_sharding_transpose,
            b_sharding_transpose=b_sharding_transpose,
        )

        if rule.dropout > 0:
          lhs = qwix_lora.nnx.Dropout(
              rule.dropout, deterministic=False
          )(lhs, rngs=qwix_lora.flax_util.make_rng("dropout"))

        return res + (
            jnp.einsum(lora_einsum_str, lhs, lora_a, lora_b, **kwargs)
            * (rule.alpha / rule.rank)
        )

    return Provider(*args, **kwargs)


def quantized_base_leaf_paths(params: Mapping[str, Any]) -> tuple[str, ...]:
  """Returns Linen param paths whose base weights carry Qwix quantization aux."""
  flat = flax.traverse_util.flatten_dict(params)
  return tuple(
      "/".join(str(part) for part in path)
      for path, value in flat.items()
      if hasattr(value, "how") and hasattr(value, "array")
  )


def has_quantized_base_leaves(params: Mapping[str, Any]) -> bool:
  return bool(quantized_base_leaf_paths(params))


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
