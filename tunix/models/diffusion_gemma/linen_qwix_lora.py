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

"""Qwix LoRA bridge for Linen DiffusionGemma models.

This module replaces the official Hackable Diffusion LoRA layer with a
Tunix/Qwix LoRA policy while keeping the official Linen model and method
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
    r"(.*/)?(mlp|mlp2)$",
    r"(.*/)?(mlp|mlp2)/(gating_einsum|linear|router_logits)$",
    r"(.*/)?self_conditioner/ffw$",
    r"(.*/)?self_conditioner/ffw/(gating_einsum|linear)$",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinenQwixLoRAConfig:
  """Qwix LoRA configuration for Linen DiffusionGemma models."""

  rank: int
  alpha: float
  module_path: str | None = None
  dropout: float = 0.0


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

  kwargs = {
      "module_path": module_path or official_compatible_module_path(),
      "rank": rank,
      "alpha": alpha,
      "dropout": dropout,
  }
  return _NoDebugAttrLoraProvider(**kwargs)


def create_linen_lora_provider_from_config(config: LinenQwixLoRAConfig):
  """Creates a Qwix provider from a structured Linen bridge config."""
  return create_linen_lora_provider(
      rank=config.rank,
      alpha=config.alpha,
      module_path=config.module_path,
      dropout=config.dropout,
  )


def apply_lora_to_linen_model_from_config(
    model: Any,
    config: LinenQwixLoRAConfig,
    *,
    methods: Sequence[str] = DIFFUSION_GEMMA_LINEN_LORA_METHODS,
) -> Any:
  """Applies Qwix LoRA to a Linen model from a structured config."""
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
    from qwix._src.core import dot_general as qwix_dot_general  # pylint: disable=g-import-not-at-top
    from qwix._src.core import einsum as qwix_einsum  # pylint: disable=g-import-not-at-top
    from qwix._src.core import ragged_dot as qwix_ragged_dot  # pylint: disable=g-import-not-at-top

    def _force_fast_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
        **dot_general_kwargs,
    ):
      if not _contains_qarray(lhs, rhs, qwix_lora):
        return qwix_dot_general.dot_general(
            lhs,
            rhs,
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **dot_general_kwargs,
        )
      return qwix_dot_general._fast_dot_general(  # pylint: disable=protected-access
          _unwrap_with_aux(lhs, qwix_lora),
          _unwrap_with_aux(rhs, qwix_lora),
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          **dot_general_kwargs,
      )

    def _force_fast_einsum(
        einsum_str,
        *einsum_operands,
        preferred_element_type=None,
        **einsum_kwargs,
    ):
      return qwix_einsum.einsum(
          einsum_str,
          *einsum_operands,
          _qwix_dot_general=_force_fast_dot_general,
          preferred_element_type=preferred_element_type,
          **einsum_kwargs,
      )

    class Provider(qwix_lora.LoraProvider):

      def __init__(self, *provider_args, **provider_kwargs):
        super().__init__(*provider_args, **provider_kwargs)
        self._dot_general_fn = _force_fast_dot_general
        self._einsum_fn = _force_fast_einsum

      def nn_param(self, module, name: str, *args, **kwargs):  # pylint: disable=missing-function-docstring
        if not _is_linen_ragged_moe_weight(module, name):
          return super().nn_param(module, name, *args, **kwargs)

        rule, _ = self._get_current_rule_and_op_id("ragged_weight_param")
        if (
            not isinstance(rule, qwix_lora.LoraRule)
            or rule.weight_qtype is None
        ):
          return super().nn_param(module, name, *args, **kwargs)

        existing_param = module.get_variable("params", name)
        if existing_param is not None:
          unboxed = qwix_lora.nn.unbox(existing_param)
          if isinstance(unboxed, qwix_lora.ptq.WithAux):
            return _stop_gradient_qarray(unboxed.array, qwix_lora)
          if not module.is_initializing():
            raise ValueError(
                "It seems you're feeding an unquantized ragged MoE expert "
                "weight to a quantized Qwix LoRA model."
            )

        value = module.param(name, *args, **kwargs)
        how = _ragged_moe_weight_how(module, rule, qwix_lora.qarray)
        quantized = qwix_lora.ptq.create_quantized_param(
            name,
            value,
            how,
            _qarray_module=self._qarray_module,
        )
        return _stop_gradient_qarray(quantized.array, qwix_lora)

      def ragged_dot(  # pylint: disable=missing-function-docstring
          self,
          lhs,
          rhs,
          group_sizes,
          precision=None,
          preferred_element_type=None,
          group_offset=None,
          out_sharding=None,
      ):
        if not _contains_qarray(lhs, rhs, qwix_lora):
          return jax.lax.ragged_dot(
              lhs,
              rhs,
              group_sizes,
              precision=precision,
              preferred_element_type=preferred_element_type,
              group_offset=group_offset,
              out_sharding=out_sharding,
          )
        if out_sharding is not None:
          raise NotImplementedError(
              "Qwix ragged_dot QArray bridge does not support out_sharding."
          )
        lhs = _quantize_dense_lhs_for_qarray_ragged_dot(
            lhs,
            rhs,
            qwix_lora,
            qwix_ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS,  # pylint: disable=protected-access
        )
        # Qwix's public ragged_dot currently takes a dense/dequantize path for
        # bf16 activations against QArray weights. DiffusionGemma MoE weights
        # are too large for that temporary, so force the quantized fast path.
        return qwix_ragged_dot._fast_ragged_dot_general(  # pylint: disable=protected-access
            _unwrap_with_aux(lhs, qwix_lora),
            _unwrap_with_aux(rhs, qwix_lora),
            group_sizes,
            qwix_ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS,  # pylint: disable=protected-access
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
        )

      def ragged_dot_general(  # pylint: disable=missing-function-docstring
          self,
          lhs,
          rhs,
          group_sizes,
          ragged_dot_dimension_numbers,
          precision=None,
          preferred_element_type=None,
          group_offset=None,
          out_sharding=None,
      ):
        if not _contains_qarray(lhs, rhs, qwix_lora):
          return jax.lax.ragged_dot_general(
              lhs,
              rhs,
              group_sizes,
              ragged_dot_dimension_numbers,
              precision=precision,
              preferred_element_type=preferred_element_type,
              group_offset=group_offset,
              out_sharding=out_sharding,
          )
        if out_sharding is not None:
          raise NotImplementedError(
              "Qwix ragged_dot_general QArray bridge does not support "
              "out_sharding."
          )
        lhs = _quantize_dense_lhs_for_qarray_ragged_dot(
            lhs,
            rhs,
            qwix_lora,
            ragged_dot_dimension_numbers,
        )
        # See ragged_dot above: the public wrapper may dequantize full expert
        # weights before dispatching. Keep QArray operands on the fast path.
        return qwix_ragged_dot._fast_ragged_dot_general(  # pylint: disable=protected-access
            _unwrap_with_aux(lhs, qwix_lora),
            _unwrap_with_aux(rhs, qwix_lora),
            group_sizes,
            ragged_dot_dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
        )

      def transpose(self, a, axes=None):  # pylint: disable=missing-function-docstring
        a = _unwrap_with_aux(a, qwix_lora)
        if isinstance(a, qwix_lora.qarray.QArray):
          if axes is None:
            return a.transpose()
          return a.transpose(*tuple(axes))
        return jnp.transpose(a, axes=axes)

      def einsum(self, einsum_str: str, *operands, **kwargs):  # pylint: disable=missing-function-docstring
        pre_ptq_weight_name = None
        if len(operands) == 2:
          pre_ptq_weight_name = qwix_lora.flax_util.find_param(operands[1])
        base_operands = (
            (
                operands[0],
                _stop_gradient_base_operand(operands[1], qwix_lora),
            )
            if len(operands) == 2
            else operands
        )
        res = qwix_lora.ptq.PtqProvider.einsum(
            self, einsum_str, *base_operands, **kwargs
        )

        rule, _ = self._get_current_rule_and_op_id("einsum", repeated_call=True)
        if not isinstance(rule, qwix_lora.LoraRule):
          return res
        if len(operands) != 2:
          raise ValueError(
              f"Unsupported einsum format: {einsum_str=} {operands=}"
          )
        lhs, rhs = operands
        weight_name = qwix_lora.flax_util.find_param(rhs, qwix_lora.ptq.WithAux)
        if weight_name is None:
          weight_name = pre_ptq_weight_name
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
          lhs = qwix_lora.nnx.Dropout(rule.dropout, deterministic=False)(
              lhs, rngs=qwix_lora.flax_util.make_rng("dropout")
          )

        return res + (
            jnp.einsum(lora_einsum_str, lhs, lora_a, lora_b, **kwargs)
            * (rule.alpha / rule.rank)
        )

      def get_intercept_map(self):  # pylint: disable=missing-function-docstring
        return super().get_intercept_map() | {
            "jax.lax.ragged_dot": self.ragged_dot,
            "jax.lax.ragged_dot_general": self.ragged_dot_general,
            "jax.numpy.transpose": self.transpose,
        }

    return Provider(*args, **kwargs)


def _unwrap_with_aux(value: Any, qwix_lora_module: Any) -> Any:
  if isinstance(value, qwix_lora_module.ptq.WithAux):
    return value.array
  return value


def _contains_qarray(lhs: Any, rhs: Any, qwix_lora_module: Any) -> bool:
  return isinstance(
      _unwrap_with_aux(lhs, qwix_lora_module), qwix_lora_module.qarray.QArray
  ) or isinstance(
      _unwrap_with_aux(rhs, qwix_lora_module), qwix_lora_module.qarray.QArray
  )


def _stop_gradient_base_operand(value: Any, qwix_lora_module: Any) -> Any:
  if isinstance(value, qwix_lora_module.ptq.WithAux):
    return qwix_lora_module.ptq.WithAux(
        _stop_gradient_qarray(value.array, qwix_lora_module),
        value.how,
    )
  return _stop_gradient_qarray(value, qwix_lora_module)


def _stop_gradient_qarray(value: Any, qwix_lora_module: Any) -> Any:
  if isinstance(value, qwix_lora_module.qarray.QArray):
    return jax.tree.map(jax.lax.stop_gradient, value)
  return value


def _quantize_dense_lhs_for_qarray_ragged_dot(
    lhs: Any,
    rhs: Any,
    qwix_lora_module: Any,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
) -> Any:
  """Quantizes small activation lhs to keep QArray ragged dot memory-safe."""
  lhs = _unwrap_with_aux(lhs, qwix_lora_module)
  rhs = _unwrap_with_aux(rhs, qwix_lora_module)
  if isinstance(lhs, qwix_lora_module.qarray.QArray):
    return lhs
  if not isinstance(rhs, qwix_lora_module.qarray.QArray):
    return lhs
  if not isinstance(lhs, jax.Array):
    return lhs
  if not jnp.issubdtype(lhs.dtype, jnp.floating):
    return lhs

  (lhs_contracting_axes, _), _ = dimension_numbers.dot_dimension_numbers
  contracting = {int(axis) for axis in lhs_contracting_axes}
  channelwise_axes = tuple(
      axis for axis in range(len(lhs.shape)) if axis not in contracting
  )
  return qwix_lora_module.qarray.quantize(
      lhs,
      qwix_lora_module.qarray.HowToQuantize(
          qtype="int8",
          channelwise_axes=channelwise_axes,
      ),
  )


def _is_linen_ragged_moe_weight(module: Any, name: str) -> bool:
  if module.__class__.__name__ != "_Weight":
    return False
  if name != getattr(module, "weight_name", "w"):
    return False
  try:
    path_parts = tuple(str(part) for part in module.path)
  except AttributeError:
    path_parts = (getattr(module, "name", "") or "",)
  return len(path_parts) >= 2 and path_parts[-2:] in (
      ("mlp", "gating_einsum"),
      ("mlp", "linear"),
      ("mlp2", "gating_einsum"),
      ("mlp2", "linear"),
  )


def _ragged_moe_weight_how(module: Any, rule: Any, qarray_module: Any) -> Any:
  path = "/".join(str(part) for part in module.path)
  shape = tuple(int(dim) for dim in module.shape)
  if path.endswith("/linear"):
    contract_axis = 1
  else:
    contract_axis = len(shape) - 1
  tiled_axes = {contract_axis: rule.tile_size} if rule.tile_size else {}
  return qarray_module.HowToQuantize(
      qtype=rule.weight_qtype,
      channelwise_axes=tuple(
          axis for axis in range(len(shape)) if axis != contract_axis
      ),
      tiled_axes=tiled_axes,
      calibration_method=rule.weight_calibration_method,
  )


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
