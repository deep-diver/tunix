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

"""Official Linen/Hackable Diffusion QLoRA bridge for DiffusionGemma.

This module deliberately does not use Qwix.  It keeps the official
Hackable-Diffusion LoRA interception surface and swaps the frozen base weight
storage for symmetric packed int4 qvalue/scale leaves.  The LoRA adapters keep
the official ``lora/a`` and ``lora/b`` parameter layout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import functools
import json
import re
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


ALL_LINEAR = "all-linear"
_QVALUE_SUFFIX = "_qvalue"
_SCALE_SUFFIX = "_scale"
_QLORA_EINSUM_PREFIX = "_QLoRAEinsum_"
_QLORA_WEIGHT_PREFIX = "_QLoRAWeight_"
_LOGGED_DENSE_FALLBACKS: set[tuple[Any, ...]] = set()


@dataclasses.dataclass(frozen=True, kw_only=True)
class OfficialQLoRAConfig:
  """Quantized-base configuration for the official LoRA bridge."""

  qtype: str = "int4"
  scale_dtype: Any = jnp.bfloat16
  pack_int4: bool = True
  quantize_moe_weights: bool = True
  einsum_output_chunk_size: int = 256
  ragged_output_chunk_size: int = 256

  def validate(self) -> None:
    if self.qtype != "int4":
      raise ValueError(
          "Official DiffusionGemma QLoRA currently supports qtype='int4' "
          f"only, got {self.qtype!r}."
      )
    if not self.pack_int4:
      raise ValueError("Official DiffusionGemma QLoRA requires pack_int4=True.")
    if self.einsum_output_chunk_size <= 0:
      raise ValueError("einsum_output_chunk_size must be positive.")
    if self.ragged_output_chunk_size <= 0:
      raise ValueError("ragged_output_chunk_size must be positive.")


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class PackedInt4Weight:
  """Packed symmetric int4 weight plus broadcast scale."""

  qvalue: jax.Array
  scale: jax.Array
  shape: tuple[int, ...]
  qtype: str = "int4"
  source_shape: tuple[int, ...] | None = None
  axis_order: tuple[int, ...] | None = None
  dtype: Any = jnp.bfloat16

  def tree_flatten(self):
    return (self.qvalue, self.scale), (
        self.shape,
        self.qtype,
        self.source_shape,
        self.axis_order,
        jnp.dtype(self.dtype),
    )

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    shape, qtype, source_shape, axis_order, dtype = aux_data
    qvalue, scale = children
    return cls(
        qvalue=qvalue,
        scale=scale,
        shape=shape,
        qtype=qtype,
        source_shape=source_shape,
        axis_order=axis_order,
        dtype=dtype,
    )

  @property
  def logical_shape(self) -> tuple[int, ...]:
    return self.shape

  @property
  def storage_shape(self) -> tuple[int, ...]:
    return self.shape if self.source_shape is None else self.source_shape

  @property
  def ndim(self) -> int:
    return len(self.logical_shape)

  @property
  def size(self) -> int:
    return int(np.prod(self.logical_shape, dtype=np.int64))

  def astype(self, dtype: Any) -> "PackedInt4Weight":
    return dataclasses.replace(self, dtype=jnp.dtype(dtype))

  def transpose(self, axes: Sequence[int] | None = None) -> "PackedInt4Weight":
    if self.source_shape is not None:
      raise NotImplementedError(
          "PackedInt4Weight only supports one transpose before reshape."
      )
    current_order = (
        tuple(range(len(self.storage_shape)))
        if self.axis_order is None
        else self.axis_order
    )
    if axes is None:
      axes = tuple(reversed(range(len(current_order))))
    axes = tuple(int(axis) for axis in axes)
    if len(axes) != len(current_order):
      raise ValueError(
          f"PackedInt4Weight transpose axes {axes} do not match rank "
          f"{len(current_order)}."
      )
    return dataclasses.replace(
        self,
        shape=tuple(self.shape[axis] for axis in axes),
        source_shape=self.storage_shape,
        axis_order=tuple(current_order[axis] for axis in axes),
    )

  @property
  def T(self) -> "PackedInt4Weight":
    return self.transpose()

  def reshape(self, *shape: int | Sequence[int]) -> "PackedInt4Weight":
    if len(shape) == 1 and isinstance(shape[0], Sequence):
      shape = tuple(shape[0])
    new_shape = tuple(int(dim) for dim in shape)
    if -1 in new_shape:
      if new_shape.count(-1) != 1:
        raise ValueError(f"Only one inferred reshape dimension is allowed: {new_shape}.")
      known = int(np.prod([dim for dim in new_shape if dim != -1], dtype=np.int64))
      if known == 0 or self.size % known:
        raise ValueError(
            f"Cannot infer reshape dimension for {self.logical_shape} -> "
            f"{new_shape}."
        )
      inferred = self.size // known
      new_shape = tuple(inferred if dim == -1 else dim for dim in new_shape)
    if int(np.prod(new_shape, dtype=np.int64)) != self.size:
      raise ValueError(
          f"Cannot reshape packed int4 weight of shape {self.logical_shape} "
          f"to {new_shape}."
      )
    return dataclasses.replace(self, shape=new_shape)


def packed_int4_shape(shape: Sequence[int]) -> tuple[int, ...]:
  """Returns the rank-preserving packed uint8 shape for an int4 tensor."""
  shape = tuple(int(dim) for dim in shape)
  if not shape:
    return (1,)
  return shape[:-1] + ((shape[-1] + 1) // 2,)


def quantized_param_kind(path: str) -> str | None:
  """Returns ``qvalue``/``scale`` when ``path`` is an official QLoRA leaf."""
  leaf = path.rsplit("/", 1)[-1]
  if leaf.endswith(_QVALUE_SUFFIX):
    return "qvalue"
  if leaf.endswith(_SCALE_SUFFIX):
    return "scale"
  return None


def is_quantized_param_path(path: str) -> bool:
  return quantized_param_kind(path) is not None


def paired_quantized_path(path: str) -> str:
  """Returns the sibling scale/qvalue path for a quantized leaf."""
  if path.endswith(_QVALUE_SUFFIX):
    return path[: -len(_QVALUE_SUFFIX)] + _SCALE_SUFFIX
  if path.endswith(_SCALE_SUFFIX):
    return path[: -len(_SCALE_SUFFIX)] + _QVALUE_SUFFIX
  raise ValueError(f"Not an official QLoRA quantized path: {path!r}.")


def checkpoint_candidates_for_quantized_path(path: str) -> tuple[str, ...]:
  """Returns checkpoint dense-weight candidates for a qvalue/scale path."""
  kind = quantized_param_kind(path)
  if kind is None:
    return ()
  parts = path.split("/")
  leaf = parts[-1]
  suffix = _QVALUE_SUFFIX if kind == "qvalue" else _SCALE_SUFFIX
  weight_name = leaf[: -len(suffix)]

  if len(parts) >= 2 and (
      parts[-2].startswith(_QLORA_EINSUM_PREFIX)
      or parts[-2].startswith(_QLORA_WEIGHT_PREFIX)
  ):
    wrapper_name = parts[-2]
    prefix = (
        _QLORA_EINSUM_PREFIX
        if wrapper_name.startswith(_QLORA_EINSUM_PREFIX)
        else _QLORA_WEIGHT_PREFIX
    )
    encoded = wrapper_name[len(prefix) :]
    base_weight = "w" if encoded == "0" else encoded
    if weight_name in ("kernel", "w"):
      base_weight = weight_name
    base_scope = parts[:-2]
    candidates = [
        "/".join(base_scope),
        "/".join(base_scope + [base_weight]),
        "/".join(base_scope + [base_weight, "w"]),
    ]
  else:
    base_scope = parts[:-1]
    candidates = [
        "/".join(base_scope),
        "/".join(base_scope + [weight_name]),
        "/".join(base_scope + [weight_name, "w"]),
    ]

  return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def quantize_checkpoint_leaf(
    *,
    quantized_path: str,
    checkpoint_value: Any,
    model_flat: Mapping[str, Any],
    config: OfficialQLoRAConfig | None = None,
) -> Any:
  """Converts a dense checkpoint weight to a qvalue or scale model leaf."""
  config = config or OfficialQLoRAConfig()
  config.validate()
  kind = quantized_param_kind(quantized_path)
  if kind is None:
    raise ValueError(f"Not an official QLoRA quantized path: {quantized_path}.")

  if kind == "qvalue":
    qvalue_path = quantized_path
    scale_path = paired_quantized_path(quantized_path)
  else:
    scale_path = quantized_path
    qvalue_path = paired_quantized_path(quantized_path)

  if scale_path not in model_flat:
    raise KeyError(f"Missing paired QLoRA scale path {scale_path!r}.")
  scale_shape = tuple(int(dim) for dim in getattr(model_flat[scale_path], "shape"))
  quantized = quantize_symmetric_int4(
      checkpoint_value,
      scale_shape=scale_shape,
      scale_dtype=getattr(model_flat[scale_path], "dtype", config.scale_dtype),
  )

  if kind == "qvalue":
    expected_shape = tuple(int(dim) for dim in getattr(model_flat[qvalue_path], "shape"))
    if tuple(quantized.qvalue.shape) != expected_shape:
      raise ValueError(
          "Packed QLoRA qvalue shape mismatch for "
          f"{quantized_path!r}: got {quantized.qvalue.shape}, "
          f"expected {expected_shape}."
      )
    return quantized.qvalue.astype(getattr(model_flat[qvalue_path], "dtype"))
  return quantized.scale.astype(getattr(model_flat[scale_path], "dtype"))


def quantize_symmetric_int4(
    weight: Any,
    *,
    scale_shape: Sequence[int],
    scale_dtype: Any = jnp.bfloat16,
) -> PackedInt4Weight:
  """Quantizes ``weight`` with symmetric per-broadcast-scale int4."""
  weight = jnp.asarray(weight, dtype=jnp.float32)
  scale_shape = tuple(int(dim) for dim in scale_shape)
  if len(scale_shape) != weight.ndim:
    raise ValueError(
        f"scale_shape rank {len(scale_shape)} must match weight rank "
        f"{weight.ndim}: {scale_shape} vs {weight.shape}."
    )
  reduce_axes = tuple(
      axis
      for axis, (weight_dim, scale_dim) in enumerate(
          zip(weight.shape, scale_shape, strict=True)
      )
      if int(scale_dim) == 1 and int(weight_dim) != 1
  )
  scale = jnp.max(jnp.abs(weight), axis=reduce_axes, keepdims=True)
  scale = jnp.where(scale == 0, 1.0, scale / 7.0)
  qvalue = jnp.rint(weight / scale)
  qvalue = jnp.clip(qvalue, -8, 7).astype(jnp.int8)
  return PackedInt4Weight(
      qvalue=pack_int4(qvalue),
      scale=scale.astype(scale_dtype),
      shape=tuple(int(dim) for dim in weight.shape),
  )


def pack_int4(qvalue: Any) -> jax.Array:
  """Packs signed int4 values into uint8 nibbles along the last axis."""
  qvalue = jnp.asarray(qvalue, dtype=jnp.int8)
  if qvalue.ndim == 0:
    qvalue = qvalue.reshape((1,))
  qvalue = jnp.clip(qvalue, -8, 7)
  unsigned = jnp.bitwise_and(qvalue, jnp.asarray(0x0F, dtype=jnp.int8)).astype(
      jnp.uint8
  )
  if unsigned.shape[-1] % 2:
    unsigned = jnp.pad(unsigned, [(0, 0)] * (unsigned.ndim - 1) + [(0, 1)])
  low = unsigned[..., 0::2]
  high = unsigned[..., 1::2]
  return jnp.bitwise_or(low, jnp.left_shift(high, 4)).astype(jnp.uint8)


def unpack_int4(packed: Any, shape: Sequence[int]) -> jax.Array:
  """Unpacks uint8 nibbles into signed int4 values with ``shape``."""
  shape = tuple(int(dim) for dim in shape)
  if not shape:
    shape = (1,)
  packed = jnp.asarray(packed, dtype=jnp.uint8)
  low = jnp.bitwise_and(packed, jnp.asarray(0x0F, dtype=jnp.uint8))
  high = jnp.bitwise_and(jnp.right_shift(packed, 4), jnp.asarray(0x0F, dtype=jnp.uint8))
  values = jnp.stack((low, high), axis=-1).reshape(packed.shape[:-1] + (-1,))
  values = values[..., : shape[-1]].astype(jnp.int8)
  values = jnp.where(values >= 8, values - 16, values)
  return values.reshape(shape)


def dequantize_symmetric_int4(
    qvalue: Any,
    scale: Any,
    shape: Sequence[int],
    *,
    dtype: Any,
) -> jax.Array:
  compute_dtype = jnp.dtype(dtype)
  q = jax.lax.stop_gradient(unpack_int4(qvalue, shape).astype(compute_dtype))
  scale = jax.lax.stop_gradient(jnp.asarray(scale, dtype=compute_dtype))
  return q * scale


def deduce_einsum_scale_shape(eqn: str, weight_shape: Sequence[int]) -> tuple[int, ...]:
  """Deduces broadcast scale shape for a two-operand explicit einsum."""
  lhs_spec, weight_spec, output_spec = _parse_two_operand_einsum(eqn)
  if len(weight_spec) != len(weight_shape):
    raise ValueError(
        f"Einsum weight spec {weight_spec!r} does not match weight shape "
        f"{tuple(weight_shape)}."
    )
  scale_shape = list(int(dim) for dim in weight_shape)
  for axis, label in enumerate(weight_spec):
    if label in lhs_spec and label not in output_spec:
      scale_shape[axis] = 1
  if tuple(scale_shape) == tuple(int(dim) for dim in weight_shape):
    return tuple([1] * (len(scale_shape) - 1) + [scale_shape[-1]])
  return tuple(scale_shape)


def deduce_moe_weight_scale_shape(
    module_path: str | Sequence[str] | None,
    weight_shape: Sequence[int],
) -> tuple[int, ...]:
  """Deduces a broadcast scale shape for Gemma4 MoE ragged weights."""
  shape = tuple(int(dim) for dim in weight_shape)
  if not shape:
    raise ValueError("MoE QLoRA weight shape must not be scalar.")
  path = _module_path_string(module_path)
  scale_shape = list(shape)
  if path.endswith("/linear") and len(shape) == 3:
    # MoERagged linear weights are [expert, hidden, features]; ragged_dot
    # contracts hidden, so keep per-expert/per-output scales.
    scale_shape[1] = 1
  else:
    # MoERagged gate weights are [expert, 2, hidden, features] and are
    # transposed before ragged_dot; raw last axis is the contracted feature dim.
    scale_shape[-1] = 1
  return tuple(scale_shape)


def patch_official_lora_module(
    official_lora_module: Any,
    *,
    quantization_config: OfficialQLoRAConfig | None = None,
) -> None:
  """Patches official ``hd.lora.LoRA`` to construct quantized-base LoRA."""
  quantization_config = quantization_config or OfficialQLoRAConfig()
  quantization_config.validate()
  if getattr(official_lora_module, "_tunix_official_qlora_patch_applied", False):
    return
  if not hasattr(official_lora_module, "_tunix_original_lora_cls"):
    official_lora_module._tunix_original_lora_cls = official_lora_module.LoRA
  official_lora_module.LoRA = make_official_qlora_class(
      official_lora_module,
      quantization_config=quantization_config,
  )
  official_lora_module._tunix_official_qlora_patch_applied = True


def restore_official_lora_module(official_lora_module: Any) -> None:
  """Restores ``hd.lora.LoRA`` if this module patched it earlier."""
  original = getattr(official_lora_module, "_tunix_original_lora_cls", None)
  if original is None:
    return
  official_lora_module.LoRA = original
  official_lora_module._tunix_official_qlora_patch_applied = False


def make_official_qlora_class(
    official_lora_module: Any,
    *,
    quantization_config: OfficialQLoRAConfig,
):
  """Builds a LoRA-compatible wrapper class backed by int4 base weights."""
  peft_module = official_lora_module.peft
  supported_modules = official_lora_module._SUPPORTED_MODULES  # pylint: disable=protected-access
  matches_target_modules = official_lora_module._matches_target_modules  # pylint: disable=protected-access
  lora_einsum_adapter_cls = peft_module.LoRAEinsumAdapter
  lora_dense_adapter_cls = getattr(
      peft_module, "LoRADenseAdapter", _FallbackLoRADenseAdapter
  )
  lora_dense_general_adapter_cls = getattr(
      peft_module, "LoRADenseGeneralAdapter", None
  )

  class OfficialQLoRA(nn.Module):
    """Official Hackable-Diffusion LoRA API with quantized frozen base."""

    _: dataclasses.KW_ONLY

    rank: int
    model: nn.Module
    dtype: jnp.dtype = jnp.bfloat16
    verbose: bool = False
    target_modules: str | Sequence[str] | None = None

    def __post_init__(self):
      super().__post_init__()
      if self.scope is not None:
        nn.share_scope(self, self.model)

    def _lora_interceptor(self):
      replace_module_fn = functools.partial(
          _replace_by_official_qlora,
          rank=self.rank,
          dtype=self.dtype,
          target_modules=self.target_modules,
          supported_modules=supported_modules,
          matches_target_modules=matches_target_modules,
          lora_einsum_adapter_cls=lora_einsum_adapter_cls,
          lora_dense_adapter_cls=lora_dense_adapter_cls,
          lora_dense_general_adapter_cls=lora_dense_general_adapter_cls,
          scale_dtype=quantization_config.scale_dtype,
          quantize_moe_weights=quantization_config.quantize_moe_weights,
          einsum_output_chunk_size=(
              quantization_config.einsum_output_chunk_size
          ),
          ragged_output_chunk_size=(
              quantization_config.ragged_output_chunk_size
          ),
      )
      return peft_module.ModuleInterceptor(replace_module_fn)

    def _interceptors(self):
      return contextlib.ExitStack()

    @nn.compact
    def __call__(self, *args, **kwargs):
      with self._lora_interceptor(), _packed_int4_runtime_context(
          ragged_output_chunk_size=quantization_config.ragged_output_chunk_size
      ):
        return self.model(*args, **kwargs)

    @nn.compact
    def encoder_call(self, *args, **kwargs):
      with self._lora_interceptor(), _packed_int4_runtime_context(
          ragged_output_chunk_size=quantization_config.ragged_output_chunk_size
      ):
        return self.model.encoder_call(*args, **kwargs)

    @nn.compact
    def encoder_hidden_call(self, *args, **kwargs):
      with self._lora_interceptor(), _packed_int4_runtime_context(
          ragged_output_chunk_size=quantization_config.ragged_output_chunk_size
      ):
        if hasattr(self.model, "encoder_hidden_call"):
          return self.model.encoder_hidden_call(*args, **kwargs)
        return _encoder_hidden_call_from_gemma_network(self, *args, **kwargs)

    @nn.compact
    def decode_hidden(self, *args, **kwargs):
      with self._lora_interceptor(), _packed_int4_runtime_context(
          ragged_output_chunk_size=quantization_config.ragged_output_chunk_size
      ):
        if hasattr(self.model, "decode_hidden"):
          return self.model.decode_hidden(*args, **kwargs)
        return _decode_hidden_from_gemma_network(self, *args, **kwargs)

    @nn.compact
    def init_cache(self, *args, **kwargs):
      with self._lora_interceptor(), _packed_int4_runtime_context(
          ragged_output_chunk_size=quantization_config.ragged_output_chunk_size
      ):
        return self.model.init_cache(*args, **kwargs)

    def __kontext_keys__(self) -> dict[str, str]:
      return official_lora_module.kontext.get_keypaths(self.model)

    def __getattr__(self, name: str) -> Any:
      return getattr(self.model, name)

  OfficialQLoRA.__name__ = "OfficialQLoRA"
  return OfficialQLoRA


def _module_path_string(module_path: str | Sequence[str] | None) -> str:
  if module_path is None:
    return ""
  if isinstance(module_path, str):
    return module_path
  return "/".join(str(part) for part in module_path)


def _module_path(module: nn.Module) -> str:
  try:
    return _module_path_string(getattr(module, "path", None))
  except ValueError:
    return ""


def _matches_target_path(
    module: nn.Module,
    target_modules: str | Sequence[str] | None,
) -> bool:
  if target_modules is None or target_modules == ALL_LINEAR:
    return True
  if isinstance(target_modules, str):
    raise ValueError(
        f"Unsupported target_modules string: {target_modules!r}. "
        f"Use {ALL_LINEAR!r}, None, or a list of regex patterns."
    )
  path_str = _module_path(module)
  if not path_str:
    path_str = getattr(module, "name", None) or ""
  return any(re.search(pattern, path_str) for pattern in target_modules)


def _is_gemma_moe_weight_module(module: nn.Module) -> bool:
  """Returns whether ``module`` is Gemma4 MoERagged's private ``_Weight``."""
  if module.__class__.__name__ != "_Weight":
    return False
  if not hasattr(module, "shape"):
    return False
  shape = tuple(int(dim) for dim in getattr(module, "shape"))
  if len(shape) not in (3, 4):
    return False
  path_str = _module_path(module)
  module_name = getattr(module, "name", None) or ""
  return (
      path_str.endswith("/gating_einsum")
      or path_str.endswith("/linear")
      or module_name in ("gating_einsum", "linear")
  )


def _encoder_hidden_call_from_gemma_network(
    network: Any,
    *,
    x: Any,
    conditioning_embeddings: Mapping[str, Any],
) -> Any:
  tokens = x[..., 0] if len(x.shape) == 3 else x
  return network.gemma_model(
      tokens=tokens,
      cache=conditioning_embeddings.get("kv_cache", None),
      positions=conditioning_embeddings.get("positions", None),
      attention_mask=conditioning_embeddings.get("attention_mask", None),
      return_hidden_states=True,
  )


def _decode_hidden_from_gemma_network(network: Any, hidden: Any) -> Any:
  gemma_model = network.gemma_model
  logits = gemma_model.embedder.decode(hidden)
  softcap = gemma_model.config.final_logit_softcap
  if softcap is not None:
    logits /= softcap
    logits = jnp.tanh(logits) * softcap
  return logits


def _replace_by_official_qlora(
    module: nn.Module,
    *,
    rank: int,
    dtype: Any,
    target_modules: str | Sequence[str] | None,
    supported_modules: tuple[type[Any], ...],
    matches_target_modules: Any,
    lora_einsum_adapter_cls: Any,
    lora_dense_adapter_cls: Any,
    lora_dense_general_adapter_cls: Any,
    scale_dtype: Any,
    quantize_moe_weights: bool,
    einsum_output_chunk_size: int,
    ragged_output_chunk_size: int,
) -> nn.Module:
  """Replaces compatible modules by official-layout QLoRA versions."""
  if _is_gemma_moe_weight_module(module):
    if not quantize_moe_weights:
      return module
    if not _matches_target_path(module, target_modules):
      return module
    weight_name = getattr(module, "weight_name", "w")
    name = f"{_QLORA_WEIGHT_PREFIX}{weight_name if weight_name != 'w' else '0'}"
    return QuantizedWeightProvider(
        name=name,
        shape=tuple(int(dim) for dim in module.shape),
        weight_name=weight_name,
        initializer=getattr(module, "initializer", nn.initializers.normal()),
        dtype=getattr(module, "dtype", None) or dtype,
        scale_dtype=scale_dtype,
        module_path=_module_path(module),
        ragged_output_chunk_size=ragged_output_chunk_size,
    )
  if not isinstance(module, supported_modules):
    return module
  if not matches_target_modules(module, target_modules):
    return module

  if isinstance(module, nn.Dense):
    return QuantizedLoRADense(
        rank=rank,
        dtype=dtype,
        scale_dtype=scale_dtype,
        wrapped=module,
        lora_adapter_cls=lora_dense_adapter_cls,
        output_chunk_size=einsum_output_chunk_size,
    )
  if isinstance(module, nn.Einsum):
    return QuantizedLoRAFlaxEinsum(
        rank=rank,
        dtype=dtype,
        scale_dtype=scale_dtype,
        wrapped=module,
        lora_adapter_cls=lora_einsum_adapter_cls,
        output_chunk_size=einsum_output_chunk_size,
    )
  if isinstance(module, nn.DenseGeneral):
    return QuantizedLoRADenseGeneral(
        rank=rank,
        dtype=dtype,
        scale_dtype=scale_dtype,
        wrapped=module,
        lora_adapter_cls=lora_dense_general_adapter_cls,
        output_chunk_size=einsum_output_chunk_size,
    )

  weight_name = getattr(module, "weight_name", "w")
  if weight_name != "w":
    name = f"{_QLORA_EINSUM_PREFIX}{weight_name}"
  else:
    name = f"{_QLORA_EINSUM_PREFIX}0"
  return QuantizedLoRAEinsum(
      name=name,
      rank=rank,
      dtype=dtype,
      scale_dtype=scale_dtype,
      shape=tuple(int(dim) for dim in module.shape),
      weight_name=weight_name,
      initializer=getattr(module, "initializer", nn.initializers.normal()),
      w_scale=getattr(module, "w_scale", None),
      clipped=module.__class__.__name__ == "ClippedEinsum",
      lora_adapter_cls=lora_einsum_adapter_cls,
      output_chunk_size=einsum_output_chunk_size,
  )


class QuantizedWeightProvider(nn.Module):
  """Quantized frozen provider for Gemma4 MoERagged ``_Weight`` modules."""

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  weight_name: str = "w"
  dtype: Any = jnp.bfloat16
  scale_dtype: Any = jnp.bfloat16
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  module_path: str = ""
  ragged_output_chunk_size: int = 256

  @nn.compact
  def __call__(self) -> PackedInt4Weight:
    return _packed_weight_param(
        self,
        self.weight_name,
        self.shape,
        deduce_moe_weight_scale_shape(self.module_path, self.shape),
        scale_dtype=self.scale_dtype,
        output_dtype=self.dtype,
    )


class QuantizedLoRAEinsum(nn.Module):
  """LoRA wrapper around a Gemma custom Einsum with int4 base storage."""

  _: dataclasses.KW_ONLY
  rank: int
  shape: tuple[int, ...]
  weight_name: str = "w"
  dtype: Any = jnp.bfloat16
  scale_dtype: Any = jnp.bfloat16
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  w_scale: float | None = None
  clipped: bool = False
  lora_adapter_cls: Any | None = None
  output_chunk_size: int = 256

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    eqn = _normalize_einsum(eqn)
    compute_dtype = jnp.dtype(self.dtype)
    x = _cast_floating(x, compute_dtype)
    kernel = _packed_weight_param(
        self,
        self.weight_name,
        self.shape,
        deduce_einsum_scale_shape(eqn, self.shape),
        scale_dtype=self.scale_dtype,
        output_dtype=compute_dtype,
    )
    if self.w_scale is not None:
      kernel = kernel * self.w_scale
    if self.clipped:
      x = _apply_input_clip(self, x)
    y = _packed_int4_einsum(
        eqn,
        x,
        kernel,
        output_chunk_size=self.output_chunk_size,
    )
    if self.clipped:
      y = _apply_output_clip(self, y)
    adapter_cls = self.lora_adapter_cls or _FallbackLoRAEinsumAdapter
    adapter = adapter_cls(
        name="lora",
        rank=self.rank,
        dtype=self.dtype,
        einsum_str=eqn,
        shape=self.shape,
    )
    return y + adapter(x)


class QuantizedLoRAFlaxEinsum(nn.Module):
  """QLoRA wrapper for ``flax.linen.Einsum``."""

  _: dataclasses.KW_ONLY
  rank: int
  wrapped: nn.Einsum
  dtype: Any = jnp.bfloat16
  scale_dtype: Any = jnp.bfloat16
  lora_adapter_cls: Any | None = None
  output_chunk_size: int = 256

  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: jax.Array, einsum_str: str | None = None) -> jax.Array:
    eqn = nn.merge_param("einsum_str", self.wrapped.einsum_str, einsum_str)
    eqn = _normalize_einsum(eqn)
    compute_dtype = jnp.dtype(self.dtype)
    inputs = _cast_floating(inputs, compute_dtype)
    kernel = _packed_weight_param(
        self,
        "kernel",
        tuple(int(dim) for dim in self.wrapped.shape),
        deduce_einsum_scale_shape(eqn, self.wrapped.shape),
        scale_dtype=self.scale_dtype,
        output_dtype=compute_dtype,
    )
    adapter_cls = self.lora_adapter_cls or _FallbackLoRAEinsumAdapter
    adapter = adapter_cls(
        name="lora",
        rank=self.rank,
        dtype=self.dtype,
        einsum_str=eqn,
        shape=self.wrapped.shape,
    )
    return (
        _packed_int4_einsum(
            eqn,
            inputs,
            kernel,
            output_chunk_size=self.output_chunk_size,
        )
        + adapter(inputs)
    )


class QuantizedLoRADense(nn.Module):
  """QLoRA wrapper for ``flax.linen.Dense``."""

  _: dataclasses.KW_ONLY
  rank: int
  wrapped: nn.Dense
  dtype: Any = jnp.bfloat16
  scale_dtype: Any = jnp.bfloat16
  lora_adapter_cls: Any | None = None
  output_chunk_size: int = 256

  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    compute_dtype = jnp.dtype(self.dtype)
    inputs = _cast_floating(inputs, compute_dtype)
    kernel_shape = (int(inputs.shape[-1]), int(self.wrapped.features))
    kernel = _packed_weight_param(
        self,
        "kernel",
        kernel_shape,
        (1, int(self.wrapped.features)),
        scale_dtype=self.scale_dtype,
        output_dtype=compute_dtype,
    )
    y = _packed_int4_matmul(
        inputs,
        kernel,
        output_chunk_size=self.output_chunk_size,
    )
    if getattr(self.wrapped, "use_bias", False):
      bias = self.param(
          "bias",
          self.wrapped.bias_init,
          (int(self.wrapped.features),),
          self.wrapped.param_dtype,
      )
      y = y + bias.astype(y.dtype)
    adapter_cls = self.lora_adapter_cls or _FallbackLoRADenseAdapter
    adapter = adapter_cls(
        name="lora",
        rank=self.rank,
        features=self.wrapped.features,
        dtype=self.dtype,
    )
    return y + adapter(inputs)


class QuantizedLoRADenseGeneral(nn.Module):
  """QLoRA wrapper for the common non-batched ``DenseGeneral`` case."""

  _: dataclasses.KW_ONLY
  rank: int
  wrapped: nn.DenseGeneral
  dtype: Any = jnp.bfloat16
  scale_dtype: Any = jnp.bfloat16
  lora_adapter_cls: Any | None = None
  output_chunk_size: int = 256

  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    compute_dtype = jnp.dtype(self.dtype)
    inputs = _cast_floating(inputs, compute_dtype)
    batch_dims = nn.linear._canonicalize_tuple(self.wrapped.batch_dims)  # pylint: disable=protected-access
    if batch_dims:
      # DiffusionGemma does not use this path; keep exact behavior if it appears.
      return self.wrapped(inputs)
    axis = nn.linear._normalize_axes(  # pylint: disable=protected-access
        nn.linear._canonicalize_tuple(self.wrapped.axis), inputs.ndim  # pylint: disable=protected-access
    )
    features = nn.linear._canonicalize_tuple(self.wrapped.features)  # pylint: disable=protected-access
    kernel_shape = tuple(int(inputs.shape[axis_i]) for axis_i in axis) + tuple(
        int(feature) for feature in features
    )
    scale_shape = tuple(1 for _ in axis) + tuple(int(feature) for feature in features)
    kernel = _packed_weight_param(
        self,
        "kernel",
        kernel_shape,
        scale_shape,
        scale_dtype=self.scale_dtype,
        output_dtype=compute_dtype,
    )
    contract_ind = tuple(range(len(axis)))
    y = _packed_int4_dot_general(
        inputs,
        kernel,
        dimension_numbers=((axis, contract_ind), ((), ())),
        output_chunk_size=self.output_chunk_size,
    )
    if getattr(self.wrapped, "use_bias", False):
      bias = self.param(
          "bias",
          self.wrapped.bias_init,
          features,
          self.wrapped.param_dtype,
      )
      y = y + bias.astype(y.dtype)
    if self.lora_adapter_cls is None:
      return y
    adapter = self.lora_adapter_cls(
        name="lora",
        rank=self.rank,
        features=self.wrapped.features,
        axis=self.wrapped.axis,
        batch_dims=self.wrapped.batch_dims,
        dtype=self.dtype,
    )
    return y + adapter(inputs)


class _FallbackLoRADenseAdapter(nn.Module):
  """Small local LoRA adapter used by unit tests without gemma.peft."""

  _: dataclasses.KW_ONLY
  rank: int
  features: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    a = self.param(
        "a",
        nn.initializers.kaiming_uniform(),
        (inputs.shape[-1], self.rank),
        self.dtype,
    )
    b = self.param("b", nn.initializers.zeros, (self.rank, self.features), self.dtype)
    return inputs @ a @ b


class _FallbackLoRAEinsumAdapter(nn.Module):
  """Small local LoRA einsum adapter used by tests without gemma.peft."""

  _: dataclasses.KW_ONLY
  rank: int
  einsum_str: str
  shape: Sequence[int]
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    lora_einsum_str, a_shape, b_shape = _lora_einsum_str_and_shapes(
        self.einsum_str, self.shape, self.rank
    )
    a = self.param("a", nn.initializers.kaiming_uniform(), a_shape, self.dtype)
    b = self.param("b", nn.initializers.zeros, b_shape, self.dtype)
    return jnp.einsum(lora_einsum_str, inputs, a, b)


def _quantized_kernel_param(
    module: nn.Module,
    weight_name: str,
    shape: Sequence[int],
    scale_shape: Sequence[int],
    *,
    scale_dtype: Any,
    output_dtype: Any,
) -> jax.Array:
  packed = _packed_weight_param(
      module,
      weight_name,
      shape,
      scale_shape,
      scale_dtype=scale_dtype,
      output_dtype=output_dtype,
  )
  return dequantize_symmetric_int4(
      packed.qvalue, packed.scale, packed.shape, dtype=packed.dtype
  )


def _packed_weight_param(
    module: nn.Module,
    weight_name: str,
    shape: Sequence[int],
    scale_shape: Sequence[int],
    *,
    scale_dtype: Any,
    output_dtype: Any,
) -> PackedInt4Weight:
  shape = tuple(int(dim) for dim in shape)
  qvalue = module.param(
      f"{weight_name}{_QVALUE_SUFFIX}",
      lambda key, shape, dtype=jnp.uint8: jnp.zeros(shape, dtype=dtype),
      packed_int4_shape(shape),
      jnp.uint8,
  )
  scale = module.param(
      f"{weight_name}{_SCALE_SUFFIX}",
      lambda key, shape, dtype=scale_dtype: jnp.ones(shape, dtype=dtype),
      tuple(int(dim) for dim in scale_shape),
      scale_dtype,
  )
  return PackedInt4Weight(
      qvalue=qvalue,
      scale=scale,
      shape=shape,
      dtype=jnp.dtype(output_dtype),
  )


def _dequantize_last_axis_slice(
    weight: PackedInt4Weight,
    start: int,
    size: int,
) -> jax.Array:
  """Dequantizes a source-weight slice on the packed last axis."""
  shape = tuple(int(dim) for dim in weight.storage_shape)
  if not shape:
    raise ValueError("Cannot slice a scalar packed int4 weight.")
  start = int(start)
  size = int(size)
  if start < 0 or size < 0 or start + size > shape[-1]:
    raise ValueError(
        f"Invalid packed int4 slice [{start}, {start + size}) for shape {shape}."
    )
  byte_start = start // 2
  byte_end = (start + size + 1) // 2
  packed = weight.qvalue[..., byte_start:byte_end]
  unpack_shape = shape[:-1] + ((byte_end - byte_start) * 2,)
  q = unpack_int4(packed, unpack_shape)
  nibble_offset = start % 2
  q = q[..., nibble_offset : nibble_offset + size]
  compute_dtype = jnp.dtype(weight.dtype)
  q = jax.lax.stop_gradient(q.astype(compute_dtype))
  scale = jax.lax.stop_gradient(jnp.asarray(weight.scale, dtype=compute_dtype))
  if scale.shape and int(scale.shape[-1]) != 1:
    scale = scale[..., start : start + size]
  return q * scale


def _dequantize_packed_view(weight: PackedInt4Weight) -> jax.Array:
  dense = dequantize_symmetric_int4(
      weight.qvalue,
      weight.scale,
      weight.storage_shape,
      dtype=weight.dtype,
  )
  if weight.axis_order is not None:
    dense = jnp.transpose(dense, weight.axis_order)
  if dense.shape != weight.logical_shape:
    dense = dense.reshape(weight.logical_shape)
  return dense


def _dequantize_last_axis_slice_dynamic(
    weight: PackedInt4Weight,
    start: jax.Array,
    size: int,
) -> jax.Array:
  """Dynamically dequantizes an even slice on the packed last axis."""
  shape = tuple(int(dim) for dim in weight.storage_shape)
  size = int(size)
  if size % 2:
    raise NotImplementedError(
        "Dynamic packed int4 last-axis slicing requires an even chunk size."
    )
  byte_size = (size + 1) // 2
  byte_start = start // 2
  packed = jax.lax.dynamic_slice_in_dim(
      weight.qvalue, byte_start, byte_size, axis=-1
  )
  q = unpack_int4(packed, shape[:-1] + (byte_size * 2,))
  compute_dtype = jnp.dtype(weight.dtype)
  q = jax.lax.stop_gradient(q.astype(compute_dtype))
  scale = jax.lax.stop_gradient(jnp.asarray(weight.scale, dtype=compute_dtype))
  if scale.shape and int(scale.shape[-1]) != 1:
    scale = jax.lax.dynamic_slice_in_dim(scale, start, size, axis=-1)
  return q * scale


def _dequantize_view_output_slice(
    weight: PackedInt4Weight,
    start: int,
    size: int,
) -> jax.Array:
  """Dequantizes an output-axis slice of a supported packed-weight view."""
  axis_order = weight.axis_order
  logical_shape = weight.logical_shape
  if (
      weight.source_shape is None
      and (axis_order is None or axis_order == tuple(range(len(weight.shape))))
  ):
    return _dequantize_last_axis_slice(weight, start, size)

  # Gemma4 MoERagged gating path:
  #   source [expert, 2, hidden, features]
  #   transpose(0, 3, 1, 2).reshape(expert, features, 2 * hidden)
  if (
      len(weight.storage_shape) == 4
      and axis_order == (0, 3, 1, 2)
      and len(logical_shape) == 3
      and logical_shape[0] == weight.storage_shape[0]
      and logical_shape[1] == weight.storage_shape[3]
      and logical_shape[2] == weight.storage_shape[1] * weight.storage_shape[2]
  ):
    expert_dim, gate_dim, hidden_dim, feature_dim = weight.storage_shape
    del expert_dim
    pieces = []
    pos = int(start)
    end = int(start + size)
    while pos < end:
      gate_index = pos // hidden_dim
      hidden_start = pos % hidden_dim
      next_pos = min(end, (gate_index + 1) * hidden_dim)
      hidden_size = next_pos - pos
      packed = weight.qvalue[
          :,
          gate_index : gate_index + 1,
          hidden_start : hidden_start + hidden_size,
          :,
      ]
      q = unpack_int4(
          packed,
          (
              weight.storage_shape[0],
              1,
              hidden_size,
              feature_dim,
          ),
      )
      compute_dtype = jnp.dtype(weight.dtype)
      q = jax.lax.stop_gradient(q.astype(compute_dtype))
      scale = jax.lax.stop_gradient(
          jnp.asarray(
              weight.scale[
                  :,
                  gate_index : gate_index + 1,
                  hidden_start : hidden_start + hidden_size,
                  :,
              ],
              dtype=compute_dtype,
          )
      )
      dense = (q * scale)[:, 0, :, :]
      pieces.append(jnp.transpose(dense, (0, 2, 1)))
      pos = next_pos
    return jnp.concatenate(pieces, axis=-1)

  raise NotImplementedError(
      "Packed int4 view slicing does not support logical shape "
      f"{logical_shape} from source shape {weight.storage_shape} with axis_order "
      f"{axis_order}."
  )


def _dequantize_view_axis_slice(
    weight: PackedInt4Weight,
    axis: int,
    start: int,
    size: int,
) -> jax.Array:
  """Dequantizes a slice along a supported logical weight axis."""
  axis = int(axis)
  if axis < 0:
    axis += weight.ndim
  if axis < 0 or axis >= weight.ndim:
    raise ValueError(f"Invalid packed int4 axis {axis} for shape {weight.shape}.")
  if axis == weight.ndim - 1:
    return _dequantize_view_output_slice(weight, start, size)

  axis_order = weight.axis_order
  if (
      weight.source_shape is None
      and (axis_order is None or axis_order == tuple(range(len(weight.shape))))
  ):
    shape = tuple(int(dim) for dim in weight.storage_shape)
    start = int(start)
    size = int(size)
    if start < 0 or size < 0 or start + size > shape[axis]:
      raise ValueError(
          f"Invalid packed int4 slice [{start}, {start + size}) on axis "
          f"{axis} for shape {shape}."
      )
    slices = [slice(None)] * len(shape)
    slices[axis] = slice(start, start + size)
    packed = weight.qvalue[tuple(slices)]
    unpack_shape = list(shape)
    unpack_shape[axis] = size
    q = unpack_int4(packed, tuple(unpack_shape))
    compute_dtype = jnp.dtype(weight.dtype)
    q = jax.lax.stop_gradient(q.astype(compute_dtype))
    scale = jax.lax.stop_gradient(jnp.asarray(weight.scale, dtype=compute_dtype))
    if scale.shape and int(scale.shape[axis]) != 1:
      scale_slices = [slice(None)] * len(scale.shape)
      scale_slices[axis] = slice(start, start + size)
      scale = scale[tuple(scale_slices)]
    return q * scale

  raise NotImplementedError(
      "Packed int4 logical-axis slicing does not support logical shape "
      f"{weight.logical_shape} from source shape {weight.storage_shape} "
      f"with axis_order {axis_order}."
  )


def _dequantize_view_axis_slice_dynamic(
    weight: PackedInt4Weight,
    axis: int,
    start: jax.Array,
    size: int,
) -> jax.Array:
  """Dynamically dequantizes a slice along a supported logical weight axis."""
  axis = int(axis)
  if axis < 0:
    axis += weight.ndim
  if axis < 0 or axis >= weight.ndim:
    raise ValueError(f"Invalid packed int4 axis {axis} for shape {weight.shape}.")
  if axis == weight.ndim - 1:
    return _dequantize_view_output_slice_dynamic(weight, start, size)

  axis_order = weight.axis_order
  if (
      weight.source_shape is None
      and (axis_order is None or axis_order == tuple(range(len(weight.shape))))
  ):
    shape = tuple(int(dim) for dim in weight.storage_shape)
    size = int(size)
    qvalue_starts = [0] * len(weight.qvalue.shape)
    qvalue_starts[axis] = start
    qvalue_sizes = list(int(dim) for dim in weight.qvalue.shape)
    qvalue_sizes[axis] = size
    packed = jax.lax.dynamic_slice(
        weight.qvalue,
        tuple(qvalue_starts),
        tuple(qvalue_sizes),
    )
    unpack_shape = list(shape)
    unpack_shape[axis] = size
    q = unpack_int4(packed, tuple(unpack_shape))
    compute_dtype = jnp.dtype(weight.dtype)
    q = jax.lax.stop_gradient(q.astype(compute_dtype))
    scale = jax.lax.stop_gradient(jnp.asarray(weight.scale, dtype=compute_dtype))
    if scale.shape and int(scale.shape[axis]) != 1:
      scale = jax.lax.dynamic_slice_in_dim(scale, start, size, axis=axis)
    return q * scale

  raise NotImplementedError(
      "Dynamic packed int4 logical-axis slicing does not support logical "
      f"shape {weight.logical_shape} from source shape {weight.storage_shape} "
      f"with axis_order {axis_order}."
  )


def _dequantize_view_output_slice_dynamic(
    weight: PackedInt4Weight,
    start: jax.Array,
    size: int,
) -> jax.Array:
  """Dynamically dequantizes an output-axis slice for supported views."""
  axis_order = weight.axis_order
  logical_shape = weight.logical_shape
  if (
      weight.source_shape is None
      and (axis_order is None or axis_order == tuple(range(len(weight.shape))))
  ):
    return _dequantize_last_axis_slice_dynamic(weight, start, size)

  if (
      len(weight.storage_shape) == 4
      and axis_order == (0, 3, 1, 2)
      and len(logical_shape) == 3
      and logical_shape[0] == weight.storage_shape[0]
      and logical_shape[1] == weight.storage_shape[3]
      and logical_shape[2] == weight.storage_shape[1] * weight.storage_shape[2]
  ):
    expert_dim, gate_dim, hidden_dim, feature_dim = weight.storage_shape
    del gate_dim
    hidden_start = start % hidden_dim
    gate_index = start // hidden_dim
    packed = jax.lax.dynamic_slice(
        weight.qvalue,
        (0, gate_index, hidden_start, 0),
        (
            expert_dim,
            1,
            int(size),
            int(weight.qvalue.shape[-1]),
        ),
    )
    q = unpack_int4(packed, (expert_dim, 1, int(size), feature_dim))
    compute_dtype = jnp.dtype(weight.dtype)
    q = jax.lax.stop_gradient(q.astype(compute_dtype))
    scale = jax.lax.stop_gradient(jnp.asarray(weight.scale, dtype=compute_dtype))
    scale = jax.lax.dynamic_slice(
        scale,
        (0, gate_index, hidden_start, 0),
        (expert_dim, 1, int(size), int(scale.shape[-1])),
    )
    dense = (q * scale)[:, 0, :, :]
    return jnp.transpose(dense, (0, 2, 1))

  raise NotImplementedError(
      "Dynamic packed int4 view slicing does not support logical shape "
      f"{logical_shape} from source shape {weight.storage_shape} with axis_order "
      f"{axis_order}."
  )


def _shape_summary(shape: Any) -> tuple[str, ...] | None:
  if shape is None:
    return None
  try:
    return tuple(str(dim) for dim in shape)
  except TypeError:
    return (str(shape),)


def _log_dense_fallback(
    kind: str,
    reason: str,
    *,
    rhs: PackedInt4Weight,
    lhs_shape: Any = None,
    **extra: Any,
) -> None:
  """Logs one JSON event per packed-int4 dense fallback shape."""
  key = (
      kind,
      reason,
      _shape_summary(lhs_shape),
      tuple(str(dim) for dim in rhs.logical_shape),
      tuple(str(dim) for dim in rhs.storage_shape),
      tuple(str(dim) for dim in rhs.axis_order or ()),
      tuple(sorted((name, str(value)) for name, value in extra.items())),
  )
  if key in _LOGGED_DENSE_FALLBACKS:
    return
  _LOGGED_DENSE_FALLBACKS.add(key)
  payload = {
      "event": "official_qlora_dense_fallback",
      "kind": kind,
      "reason": reason,
      "lhs_shape": _shape_summary(lhs_shape),
      "rhs_logical_shape": tuple(int(dim) for dim in rhs.logical_shape),
      "rhs_storage_shape": tuple(int(dim) for dim in rhs.storage_shape),
      "rhs_axis_order": rhs.axis_order,
      "rhs_dtype": str(jnp.dtype(rhs.dtype)),
  }
  payload.update({name: str(value) for name, value in extra.items()})
  print(json.dumps(payload, sort_keys=True), flush=True)


def _einsum_spec_tokens(spec: str) -> tuple[str, ...]:
  tokens = []
  index = 0
  while index < len(spec):
    if spec.startswith("...", index):
      tokens.append("...")
      index += 3
    else:
      tokens.append(spec[index])
      index += 1
  return tuple(tokens)


def _einsum_output_axis_for_label(
    *,
    lhs_spec: str,
    output_spec: str,
    lhs_ndim: int,
    label: str,
) -> int | None:
  lhs_tokens = _einsum_spec_tokens(lhs_spec)
  output_tokens = _einsum_spec_tokens(output_spec)
  explicit_lhs_rank = sum(1 for token in lhs_tokens if token != "...")
  ellipsis_rank = int(lhs_ndim) - explicit_lhs_rank
  if ellipsis_rank < 0:
    return None
  output_axis = 0
  for token in output_tokens:
    if token == "...":
      output_axis += ellipsis_rank
      continue
    if token == label:
      return output_axis
    output_axis += 1
  return None


def _einsum_chunk_plan(
    *,
    lhs_spec: str,
    weight_spec: str,
    output_spec: str,
    lhs_ndim: int,
    rhs_shape: Sequence[int],
) -> tuple[int, str, int] | None:
  """Returns ``(weight_axis, label, output_axis)`` for packed chunking."""
  candidates = []
  for weight_axis, label in enumerate(weight_spec):
    if label in lhs_spec:
      continue
    if label not in output_spec:
      continue
    output_axis = _einsum_output_axis_for_label(
        lhs_spec=lhs_spec,
        output_spec=output_spec,
        lhs_ndim=lhs_ndim,
        label=label,
    )
    if output_axis is None:
      continue
    candidates.append((int(rhs_shape[weight_axis]), weight_axis, label, output_axis))
  if not candidates:
    return None
  _, weight_axis, label, output_axis = max(candidates)
  return weight_axis, label, output_axis


def _einsum_output_shape(
    *,
    lhs_spec: str,
    weight_spec: str,
    output_spec: str,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> tuple[int, ...] | None:
  """Infers the runtime output shape for the supported two-operand einsums."""
  lhs_tokens = _einsum_spec_tokens(lhs_spec)
  weight_tokens = _einsum_spec_tokens(weight_spec)
  output_tokens = _einsum_spec_tokens(output_spec)
  if "..." in weight_tokens:
    return None
  explicit_lhs_rank = sum(1 for token in lhs_tokens if token != "...")
  ellipsis_rank = len(lhs_shape) - explicit_lhs_rank
  if ellipsis_rank < 0:
    return None
  ellipsis_shape = tuple(int(dim) for dim in lhs_shape[:ellipsis_rank])
  dims: dict[str, int] = {}
  lhs_explicit_dims = tuple(int(dim) for dim in lhs_shape[ellipsis_rank:])
  for token, dim in zip(
      (token for token in lhs_tokens if token != "..."),
      lhs_explicit_dims,
      strict=True,
  ):
    dims[token] = dim
  for token, dim in zip(weight_tokens, rhs_shape, strict=True):
    dims[token] = int(dim)

  output_shape = []
  for token in output_tokens:
    if token == "...":
      output_shape.extend(ellipsis_shape)
    elif token in dims:
      output_shape.append(dims[token])
    else:
      return None
  return tuple(output_shape)


def _can_use_dynamic_packed_einsum(
    rhs: PackedInt4Weight,
    weight_axis: int,
    chunk_size: int,
) -> bool:
  if rhs.logical_shape[weight_axis] % chunk_size:
    return False
  axis_order = rhs.axis_order
  if (
      rhs.source_shape is None
      and (axis_order is None or axis_order == tuple(range(len(rhs.shape))))
  ):
    return weight_axis != rhs.ndim - 1 or chunk_size % 2 == 0
  return False


def _packed_int4_einsum_dynamic(
    eqn: str,
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    weight_axis: int,
    out_axis: int,
    output_chunk_size: int,
) -> jax.Array:
  lhs_spec, weight_spec, output_spec = _parse_two_operand_einsum(eqn)
  output_shape = _einsum_output_shape(
      lhs_spec=lhs_spec,
      weight_spec=weight_spec,
      output_spec=output_spec,
      lhs_shape=tuple(int(dim) for dim in lhs.shape),
      rhs_shape=rhs.logical_shape,
  )
  if output_shape is None:
    raise NotImplementedError(f"Cannot infer output shape for einsum {eqn!r}.")
  output_dtype = jnp.result_type(lhs.dtype, rhs.dtype)
  chunk_size = int(output_chunk_size)
  num_chunks = int(rhs.logical_shape[weight_axis]) // chunk_size
  output = jnp.zeros(output_shape, dtype=output_dtype)

  def body(chunk_index, acc):
    start = chunk_index * chunk_size
    rhs_chunk = _dequantize_view_axis_slice_dynamic(
        rhs, weight_axis, start, chunk_size
    )
    out_chunk = jnp.einsum(eqn, lhs, rhs_chunk).astype(output_dtype)
    return jax.lax.dynamic_update_slice_in_dim(
        acc, out_chunk, start, axis=out_axis
    )

  return jax.lax.fori_loop(0, num_chunks, body, output)


def _packed_int4_matmul_impl(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    output_chunk_size: int,
) -> jax.Array:
  if rhs.ndim != 2:
    _log_dense_fallback(
        "matmul",
        "rhs_rank_not_2",
        lhs_shape=getattr(lhs, "shape", None),
        rhs=rhs,
    )
    return jnp.matmul(lhs, _dequantize_packed_view(rhs))
  out_dim = rhs.logical_shape[-1]
  chunks = []
  for start in range(0, int(out_dim), int(output_chunk_size)):
    size = min(int(output_chunk_size), int(out_dim) - start)
    rhs_chunk = _dequantize_view_output_slice(rhs, start, size)
    chunks.append(jnp.matmul(lhs, rhs_chunk))
  return jnp.concatenate(chunks, axis=-1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _packed_int4_matmul_custom_vjp(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_matmul_impl(
      lhs,
      rhs,
      output_chunk_size=output_chunk_size,
  )


def _packed_int4_matmul_fwd(lhs, rhs, output_chunk_size):
  return (
      _packed_int4_matmul_impl(
          lhs,
          rhs,
          output_chunk_size=output_chunk_size,
      ),
      (rhs,),
  )


def _packed_int4_matmul_bwd(output_chunk_size, residual, cotangent):
  (rhs,) = residual
  if rhs.ndim != 2:
    return (
        jnp.matmul(cotangent, jnp.swapaxes(_dequantize_packed_view(rhs), -1, -2)),
        None,
    )
  return (
      _packed_int4_matmul_lhs_grad(
          cotangent,
          rhs,
          output_chunk_size=output_chunk_size,
      ),
      None,
  )


_packed_int4_matmul_custom_vjp.defvjp(
    _packed_int4_matmul_fwd,
    _packed_int4_matmul_bwd,
)


def _packed_int4_matmul(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_matmul_custom_vjp(lhs, rhs, int(output_chunk_size))


def _packed_int4_matmul_lhs_grad(
    cotangent: jax.Array,
    rhs: PackedInt4Weight,
    *,
    output_chunk_size: int,
) -> jax.Array:
  in_dim = int(rhs.logical_shape[0])
  out_dim = int(rhs.logical_shape[1])
  chunk_size = min(int(output_chunk_size), out_dim)
  output_dtype = jnp.result_type(cotangent.dtype, rhs.dtype)
  grad = jnp.zeros(cotangent.shape[:-1] + (in_dim,), dtype=output_dtype)

  if out_dim % chunk_size == 0 and chunk_size % 2 == 0:

    def body(chunk_index, acc):
      start = chunk_index * chunk_size
      grad_chunk = jax.lax.dynamic_slice_in_dim(
          cotangent, start, chunk_size, axis=-1
      )
      rhs_chunk = _dequantize_view_output_slice_dynamic(
          rhs, start, chunk_size
      )
      return acc + jnp.matmul(grad_chunk, jnp.swapaxes(rhs_chunk, -1, -2))

    return jax.lax.fori_loop(0, out_dim // chunk_size, body, grad)

  chunks = []
  for start in range(0, out_dim, chunk_size):
    size = min(chunk_size, out_dim - start)
    grad_chunk = jax.lax.dynamic_slice_in_dim(cotangent, start, size, axis=-1)
    rhs_chunk = _dequantize_view_output_slice(rhs, start, size)
    chunks.append(jnp.matmul(grad_chunk, jnp.swapaxes(rhs_chunk, -1, -2)))
  return functools.reduce(lambda acc, value: acc + value, chunks, grad)


def _packed_int4_einsum_impl(
    eqn: str,
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    output_chunk_size: int,
) -> jax.Array:
  lhs_spec, weight_spec, output_spec = _parse_two_operand_einsum(eqn)
  chunk_plan = _einsum_chunk_plan(
      lhs_spec=lhs_spec,
      weight_spec=weight_spec,
      output_spec=output_spec,
      lhs_ndim=len(getattr(lhs, "shape", ())),
      rhs_shape=rhs.logical_shape,
  )
  if chunk_plan is None:
    _log_dense_fallback(
        "einsum",
        "no_pure_output_weight_axis",
        eqn=eqn,
        lhs_shape=getattr(lhs, "shape", None),
        rhs=rhs,
    )
    return jnp.einsum(eqn, lhs, _dequantize_packed_view(rhs))
  weight_axis, _, out_axis = chunk_plan
  out_dim = rhs.logical_shape[weight_axis]
  chunk_size = min(int(output_chunk_size), int(out_dim))
  if _can_use_dynamic_packed_einsum(rhs, weight_axis, chunk_size):
    return _packed_int4_einsum_dynamic(
        eqn,
        lhs,
        rhs,
        weight_axis=weight_axis,
        out_axis=out_axis,
        output_chunk_size=chunk_size,
    )
  chunks = []
  for start in range(0, int(out_dim), chunk_size):
    size = min(chunk_size, int(out_dim) - start)
    rhs_chunk = _dequantize_view_axis_slice(rhs, weight_axis, start, size)
    chunks.append(jnp.einsum(eqn, lhs, rhs_chunk))
  return jnp.concatenate(chunks, axis=out_axis)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def _packed_int4_einsum_custom_vjp(
    eqn: str,
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_einsum_impl(
      eqn,
      lhs,
      rhs,
      output_chunk_size=output_chunk_size,
  )


def _packed_int4_einsum_fwd(eqn, lhs, rhs, output_chunk_size):
  return (
      _packed_int4_einsum_impl(
          eqn,
          lhs,
          rhs,
          output_chunk_size=output_chunk_size,
      ),
      (rhs,),
  )


def _packed_int4_einsum_bwd(eqn, output_chunk_size, residual, cotangent):
  (rhs,) = residual
  return (
      _packed_int4_einsum_lhs_grad(
          eqn,
          cotangent,
          rhs,
          output_chunk_size=output_chunk_size,
      ),
      None,
  )


_packed_int4_einsum_custom_vjp.defvjp(
    _packed_int4_einsum_fwd,
    _packed_int4_einsum_bwd,
)


def _packed_int4_einsum(
    eqn: str,
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_einsum_custom_vjp(
      eqn,
      lhs,
      rhs,
      int(output_chunk_size),
  )


def _einsum_output_axis_from_output_ndim(
    *,
    output_spec: str,
    output_ndim: int,
    label: str,
) -> int | None:
  output_tokens = _einsum_spec_tokens(output_spec)
  explicit_output_rank = sum(1 for token in output_tokens if token != "...")
  ellipsis_rank = int(output_ndim) - explicit_output_rank
  if ellipsis_rank < 0:
    return None
  output_axis = 0
  for token in output_tokens:
    if token == "...":
      output_axis += ellipsis_rank
      continue
    if token == label:
      return output_axis
    output_axis += 1
  return None


def _einsum_chunk_plan_from_output(
    *,
    lhs_spec: str,
    weight_spec: str,
    output_spec: str,
    output_ndim: int,
    rhs_shape: Sequence[int],
) -> tuple[int, str, int] | None:
  candidates = []
  for weight_axis, label in enumerate(weight_spec):
    if label in lhs_spec or label not in output_spec:
      continue
    output_axis = _einsum_output_axis_from_output_ndim(
        output_spec=output_spec,
        output_ndim=output_ndim,
        label=label,
    )
    if output_axis is None:
      continue
    candidates.append((int(rhs_shape[weight_axis]), weight_axis, label, output_axis))
  if not candidates:
    return None
  _, weight_axis, label, output_axis = max(candidates)
  return weight_axis, label, output_axis


def _einsum_lhs_shape_from_output(
    *,
    lhs_spec: str,
    weight_spec: str,
    output_spec: str,
    output_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> tuple[int, ...] | None:
  lhs_tokens = _einsum_spec_tokens(lhs_spec)
  output_tokens = _einsum_spec_tokens(output_spec)
  explicit_output_rank = sum(1 for token in output_tokens if token != "...")
  ellipsis_rank = len(output_shape) - explicit_output_rank
  if ellipsis_rank < 0:
    return None
  ellipsis_shape = tuple(int(dim) for dim in output_shape[:ellipsis_rank])
  output_explicit_dims = tuple(int(dim) for dim in output_shape[ellipsis_rank:])
  dims: dict[str, int] = {}
  for token, dim in zip(
      (token for token in output_tokens if token != "..."),
      output_explicit_dims,
      strict=True,
  ):
    dims[token] = int(dim)
  for token, dim in zip(weight_spec, rhs_shape, strict=True):
    dims.setdefault(token, int(dim))

  lhs_shape = []
  for token in lhs_tokens:
    if token == "...":
      lhs_shape.extend(ellipsis_shape)
    elif token in dims:
      lhs_shape.append(dims[token])
    else:
      return None
  return tuple(lhs_shape)


def _packed_int4_einsum_lhs_grad(
    eqn: str,
    cotangent: jax.Array,
    rhs: PackedInt4Weight,
    *,
    output_chunk_size: int,
) -> jax.Array:
  lhs_spec, weight_spec, output_spec = _parse_two_operand_einsum(eqn)
  grad_eqn = f"{output_spec},{weight_spec}->{lhs_spec}"
  chunk_plan = _einsum_chunk_plan_from_output(
      lhs_spec=lhs_spec,
      weight_spec=weight_spec,
      output_spec=output_spec,
      output_ndim=cotangent.ndim,
      rhs_shape=rhs.logical_shape,
  )
  lhs_shape = _einsum_lhs_shape_from_output(
      lhs_spec=lhs_spec,
      weight_spec=weight_spec,
      output_spec=output_spec,
      output_shape=tuple(int(dim) for dim in cotangent.shape),
      rhs_shape=rhs.logical_shape,
  )
  if lhs_shape is None:
    return jnp.einsum(grad_eqn, cotangent, _dequantize_packed_view(rhs))
  output_dtype = jnp.result_type(cotangent.dtype, rhs.dtype)
  lhs_grad = jnp.zeros(lhs_shape, dtype=output_dtype)
  if chunk_plan is None:
    _log_dense_fallback(
        "einsum_bwd",
        "no_pure_output_weight_axis",
        eqn=eqn,
        lhs_shape=lhs_shape,
        rhs=rhs,
    )
    return jnp.einsum(grad_eqn, cotangent, _dequantize_packed_view(rhs))

  weight_axis, _, out_axis = chunk_plan
  out_dim = int(rhs.logical_shape[weight_axis])
  chunk_size = min(int(output_chunk_size), out_dim)
  if (
      out_dim % chunk_size == 0
      and _can_use_dynamic_packed_einsum(rhs, weight_axis, chunk_size)
  ):

    def body(chunk_index, acc):
      start = chunk_index * chunk_size
      grad_chunk = jax.lax.dynamic_slice_in_dim(
          cotangent, start, chunk_size, axis=out_axis
      )
      rhs_chunk = _dequantize_view_axis_slice_dynamic(
          rhs, weight_axis, start, chunk_size
      )
      return acc + jnp.einsum(grad_eqn, grad_chunk, rhs_chunk).astype(
          output_dtype
      )

    return jax.lax.fori_loop(0, out_dim // chunk_size, body, lhs_grad)

  chunks = []
  for start in range(0, out_dim, chunk_size):
    size = min(chunk_size, out_dim - start)
    grad_chunk = jax.lax.dynamic_slice_in_dim(
        cotangent, start, size, axis=out_axis
    )
    rhs_chunk = _dequantize_view_axis_slice(rhs, weight_axis, start, size)
    chunks.append(jnp.einsum(grad_eqn, grad_chunk, rhs_chunk).astype(output_dtype))
  return functools.reduce(lambda acc, value: acc + value, chunks, lhs_grad)


def _packed_int4_dot_general_impl(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    dimension_numbers: Any,
    output_chunk_size: int,
) -> jax.Array:
  ((_, rhs_contract), (_, rhs_batch)) = dimension_numbers
  rhs_contract = tuple(int(axis) for axis in rhs_contract)
  rhs_batch = tuple(int(axis) for axis in rhs_batch)
  if rhs_contract != tuple(range(len(rhs_contract))) or rhs_batch:
    _log_dense_fallback(
        "dot_general",
        "unsupported_dimension_numbers",
        lhs_shape=getattr(lhs, "shape", None),
        rhs=rhs,
        dimension_numbers=dimension_numbers,
    )
    return jax.lax.dot_general(lhs, _dequantize_packed_view(rhs), dimension_numbers)
  out_dim = rhs.logical_shape[-1]
  chunks = []
  for start in range(0, int(out_dim), int(output_chunk_size)):
    size = min(int(output_chunk_size), int(out_dim) - start)
    rhs_chunk = _dequantize_view_output_slice(rhs, start, size)
    chunks.append(jax.lax.dot_general(lhs, rhs_chunk, dimension_numbers))
  return jnp.concatenate(chunks, axis=-1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _packed_int4_dot_general_custom_vjp(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    dimension_numbers: Any,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_dot_general_impl(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      output_chunk_size=output_chunk_size,
  )


def _packed_int4_dot_general_fwd(
    lhs,
    rhs,
    dimension_numbers,
    output_chunk_size,
):
  return (
      _packed_int4_dot_general_impl(
          lhs,
          rhs,
          dimension_numbers=dimension_numbers,
          output_chunk_size=output_chunk_size,
      ),
      (rhs,),
  )


def _packed_int4_dot_general_bwd(
    dimension_numbers,
    output_chunk_size,
    residual,
    cotangent,
):
  (rhs,) = residual
  return (
      _packed_int4_dot_general_lhs_grad(
          cotangent,
          rhs,
          dimension_numbers=dimension_numbers,
          output_chunk_size=output_chunk_size,
      ),
      None,
  )


_packed_int4_dot_general_custom_vjp.defvjp(
    _packed_int4_dot_general_fwd,
    _packed_int4_dot_general_bwd,
)


def _packed_int4_dot_general(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    *,
    dimension_numbers: Any,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_dot_general_custom_vjp(
      lhs,
      rhs,
      dimension_numbers,
      int(output_chunk_size),
  )


def _packed_int4_dot_general_lhs_grad(
    cotangent: jax.Array,
    rhs: PackedInt4Weight,
    *,
    dimension_numbers: Any,
    output_chunk_size: int,
) -> jax.Array:
  ((lhs_contract, rhs_contract), (_, rhs_batch)) = dimension_numbers
  lhs_contract = tuple(int(axis) for axis in lhs_contract)
  rhs_contract = tuple(int(axis) for axis in rhs_contract)
  rhs_batch = tuple(int(axis) for axis in rhs_batch)
  if rhs_contract != tuple(range(len(rhs_contract))) or rhs_batch:
    _log_dense_fallback(
        "dot_general_bwd",
        "unsupported_dimension_numbers",
        cotangent_shape=getattr(cotangent, "shape", None),
        rhs=rhs,
        dimension_numbers=dimension_numbers,
    )
    dense_rhs = _dequantize_packed_view(rhs)
    rhs_output_axes = tuple(
        axis for axis in range(rhs.ndim) if axis not in rhs_contract
    )
    cotangent_output_axes = tuple(
        range(cotangent.ndim - len(rhs_output_axes), cotangent.ndim)
    )
    return jax.lax.dot_general(
        cotangent,
        dense_rhs,
        ((cotangent_output_axes, rhs_output_axes), ((), ())),
    )

  lhs_rank = cotangent.ndim - (rhs.ndim - len(rhs_contract)) + len(lhs_contract)
  lhs_free_axes = tuple(axis for axis in range(lhs_rank) if axis not in lhs_contract)
  rhs_output_axes = tuple(axis for axis in range(rhs.ndim) if axis not in rhs_contract)
  cotangent_output_axes = tuple(
      range(len(lhs_free_axes), len(lhs_free_axes) + len(rhs_output_axes))
  )
  output_dtype = jnp.result_type(cotangent.dtype, rhs.dtype)
  lhs_shape_in_result_order = tuple(int(cotangent.shape[axis]) for axis in range(len(lhs_free_axes))) + tuple(
      int(rhs.logical_shape[axis]) for axis in rhs_contract
  )
  lhs_grad = jnp.zeros(lhs_shape_in_result_order, dtype=output_dtype)
  out_dim = int(rhs.logical_shape[-1])
  chunk_size = min(int(output_chunk_size), out_dim)

  def project(grad_chunk, rhs_chunk):
    return jax.lax.dot_general(
        grad_chunk,
        rhs_chunk,
        ((cotangent_output_axes, rhs_output_axes), ((), ())),
    ).astype(output_dtype)

  if (
      out_dim % chunk_size == 0
      and rhs.source_shape is None
      and (rhs.axis_order is None or rhs.axis_order == tuple(range(rhs.ndim)))
      and chunk_size % 2 == 0
  ):

    def body(chunk_index, acc):
      start = chunk_index * chunk_size
      grad_chunk = jax.lax.dynamic_slice_in_dim(
          cotangent, start, chunk_size, axis=-1
      )
      rhs_chunk = _dequantize_view_output_slice_dynamic(
          rhs, start, chunk_size
      )
      return acc + project(grad_chunk, rhs_chunk)

    lhs_grad = jax.lax.fori_loop(0, out_dim // chunk_size, body, lhs_grad)
  else:
    chunks = []
    for start in range(0, out_dim, chunk_size):
      size = min(chunk_size, out_dim - start)
      grad_chunk = jax.lax.dynamic_slice_in_dim(cotangent, start, size, axis=-1)
      rhs_chunk = _dequantize_view_output_slice(rhs, start, size)
      chunks.append(project(grad_chunk, rhs_chunk))
    lhs_grad = functools.reduce(lambda acc, value: acc + value, chunks, lhs_grad)

  result_axes = lhs_free_axes + lhs_contract
  transpose_axes = tuple(result_axes.index(axis) for axis in range(lhs_rank))
  return jnp.transpose(lhs_grad, transpose_axes)


def _is_basic_ragged_dot_dimension_numbers(
    ragged_dot_dimension_numbers: Any,
) -> bool:
  """Returns whether ``ragged_dot_general`` matches ``lax.ragged_dot``."""
  try:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = (
        ragged_dot_dimension_numbers.dot_dimension_numbers
    )
    lhs_ragged = ragged_dot_dimension_numbers.lhs_ragged_dimensions
    rhs_group = ragged_dot_dimension_numbers.rhs_group_dimensions
  except (AttributeError, TypeError, ValueError):
    return False
  return (
      tuple(int(axis) for axis in lhs_contract) == (1,)
      and tuple(int(axis) for axis in rhs_contract) == (1,)
      and tuple(int(axis) for axis in lhs_batch) == ()
      and tuple(int(axis) for axis in rhs_batch) == ()
      and tuple(int(axis) for axis in lhs_ragged) == (0,)
      and tuple(int(axis) for axis in rhs_group) == (0,)
  )


@contextlib.contextmanager
def _packed_int4_runtime_context(*, ragged_output_chunk_size: int):
  """Intercepts packed int4 views that flow through MoE transpose/ragged_dot."""
  original_transpose = jnp.transpose
  original_ragged_dot = jax.lax.ragged_dot
  original_ragged_dot_general = jax.lax.ragged_dot_general

  def _transpose(a, axes=None):
    if isinstance(a, PackedInt4Weight):
      return a.transpose(axes)
    return original_transpose(a, axes=axes)

  def _ragged_dot(
      lhs,
      rhs,
      group_sizes,
      precision=None,
      preferred_element_type=None,
      group_offset=None,
      out_sharding=None,
  ):
    if isinstance(rhs, PackedInt4Weight):
      if out_sharding is not None:
        raise NotImplementedError(
            "Official QLoRA packed ragged_dot does not support out_sharding."
        )
      return _packed_int4_ragged_dot(
          lhs,
          rhs,
          group_sizes,
          original_ragged_dot=original_ragged_dot,
          precision=precision,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
          output_chunk_size=ragged_output_chunk_size,
      )
    return original_ragged_dot(
        lhs,
        rhs,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        out_sharding=out_sharding,
    )

  def _ragged_dot_general(
      lhs,
      rhs,
      group_sizes,
      ragged_dot_dimension_numbers,
      precision=None,
      preferred_element_type=None,
      group_offset=None,
      out_sharding=None,
  ):
    if isinstance(rhs, PackedInt4Weight):
      if (
          out_sharding is None
          and _is_basic_ragged_dot_dimension_numbers(
              ragged_dot_dimension_numbers
          )
      ):
        return _packed_int4_ragged_dot(
            lhs,
            rhs,
            group_sizes,
            original_ragged_dot=original_ragged_dot,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
            output_chunk_size=ragged_output_chunk_size,
        )
      dense = _dequantize_packed_view(rhs)
      _log_dense_fallback(
          "ragged_dot_general",
          "unsupported_general_api",
          lhs_shape=getattr(lhs, "shape", None),
          rhs=rhs,
          ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
      )
      return original_ragged_dot_general(
          lhs,
          dense,
          group_sizes,
          ragged_dot_dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
          out_sharding=out_sharding,
      )
    return original_ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        out_sharding=out_sharding,
    )

  jnp.transpose = _transpose
  jax.lax.ragged_dot = _ragged_dot
  jax.lax.ragged_dot_general = _ragged_dot_general
  try:
    yield
  finally:
    jnp.transpose = original_transpose
    jax.lax.ragged_dot = original_ragged_dot
    jax.lax.ragged_dot_general = original_ragged_dot_general


def _packed_int4_ragged_dot_impl(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    group_sizes: jax.Array,
    *,
    original_ragged_dot: Any,
    precision: Any,
    preferred_element_type: Any,
    group_offset: Any,
    output_chunk_size: int,
) -> jax.Array:
  logical_shape = rhs.logical_shape
  if len(logical_shape) != 3:
    _log_dense_fallback(
        "ragged_dot",
        "rhs_rank_not_3",
        lhs_shape=getattr(lhs, "shape", None),
        rhs=rhs,
    )
    return original_ragged_dot(
        lhs,
        _dequantize_packed_view(rhs),
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
  out_dim = int(logical_shape[-1])
  chunk_size = min(int(output_chunk_size), out_dim)
  if _can_use_dynamic_packed_ragged_dot(rhs, chunk_size):
    return _packed_int4_ragged_dot_dynamic(
        lhs,
        rhs,
        group_sizes,
        original_ragged_dot=original_ragged_dot,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        output_chunk_size=chunk_size,
    )
  chunks = []
  for start in range(0, out_dim, chunk_size):
    size = min(chunk_size, out_dim - start)
    rhs_chunk = _dequantize_view_output_slice(rhs, start, size)
    chunks.append(
        original_ragged_dot(
            lhs,
            rhs_chunk,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
        )
    )
  return jnp.concatenate(chunks, axis=-1)


def _can_use_dynamic_packed_ragged_dot(
    rhs: PackedInt4Weight,
    chunk_size: int,
) -> bool:
  logical_shape = rhs.logical_shape
  if len(logical_shape) != 3 or logical_shape[-1] % chunk_size:
    return False
  axis_order = rhs.axis_order
  if (
      rhs.source_shape is None
      and (axis_order is None or axis_order == tuple(range(len(rhs.shape))))
  ):
    return chunk_size % 2 == 0
  if (
      len(rhs.storage_shape) == 4
      and axis_order == (0, 3, 1, 2)
      and logical_shape[0] == rhs.storage_shape[0]
      and logical_shape[1] == rhs.storage_shape[3]
      and logical_shape[2] == rhs.storage_shape[1] * rhs.storage_shape[2]
  ):
    return rhs.storage_shape[2] % chunk_size == 0
  return False


def _packed_int4_ragged_dot_dynamic(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    group_sizes: jax.Array,
    *,
    original_ragged_dot: Any,
    precision: Any,
    preferred_element_type: Any,
    group_offset: Any,
    output_chunk_size: int,
) -> jax.Array:
  out_dim = int(rhs.logical_shape[-1])
  chunk_size = int(output_chunk_size)
  num_chunks = out_dim // chunk_size
  output_dtype = (
      jnp.dtype(preferred_element_type)
      if preferred_element_type is not None
      else jnp.result_type(lhs.dtype, rhs.dtype)
  )
  output = jnp.zeros((int(lhs.shape[0]), out_dim), dtype=output_dtype)

  def body(chunk_index, acc):
    start = chunk_index * chunk_size
    rhs_chunk = _dequantize_view_output_slice_dynamic(rhs, start, chunk_size)
    out_chunk = original_ragged_dot(
        lhs,
        rhs_chunk,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    ).astype(output_dtype)
    return jax.lax.dynamic_update_slice_in_dim(
        acc, out_chunk, start, axis=-1
    )

  return jax.lax.fori_loop(0, num_chunks, body, output)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _packed_int4_ragged_dot_custom_vjp(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    group_sizes: jax.Array,
    original_ragged_dot: Any,
    precision: Any,
    preferred_element_type: Any,
    group_offset: Any,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_ragged_dot_impl(
      lhs,
      rhs,
      group_sizes,
      original_ragged_dot=original_ragged_dot,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
      output_chunk_size=output_chunk_size,
  )


def _packed_int4_ragged_dot_fwd(
    lhs,
    rhs,
    group_sizes,
    original_ragged_dot,
    precision,
    preferred_element_type,
    group_offset,
    output_chunk_size,
):
  return (
      _packed_int4_ragged_dot_impl(
          lhs,
          rhs,
          group_sizes,
          original_ragged_dot=original_ragged_dot,
          precision=precision,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
          output_chunk_size=output_chunk_size,
      ),
      (rhs, group_sizes),
  )


def _packed_int4_ragged_dot_bwd(
    original_ragged_dot,
    precision,
    preferred_element_type,
    group_offset,
    output_chunk_size,
    residual,
    cotangent,
):
  rhs, group_sizes = residual
  return (
      _packed_int4_ragged_dot_lhs_grad(
          cotangent,
          rhs,
          group_sizes,
          original_ragged_dot=original_ragged_dot,
          precision=precision,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
          output_chunk_size=output_chunk_size,
      ),
      None,
      None,
  )


_packed_int4_ragged_dot_custom_vjp.defvjp(
    _packed_int4_ragged_dot_fwd,
    _packed_int4_ragged_dot_bwd,
)


def _packed_int4_ragged_dot(
    lhs: jax.Array,
    rhs: PackedInt4Weight,
    group_sizes: jax.Array,
    *,
    original_ragged_dot: Any,
    precision: Any,
    preferred_element_type: Any,
    group_offset: Any,
    output_chunk_size: int,
) -> jax.Array:
  return _packed_int4_ragged_dot_custom_vjp(
      lhs,
      rhs,
      group_sizes,
      original_ragged_dot,
      precision,
      preferred_element_type,
      group_offset,
      int(output_chunk_size),
  )


def _packed_int4_ragged_dot_lhs_grad(
    cotangent: jax.Array,
    rhs: PackedInt4Weight,
    group_sizes: jax.Array,
    *,
    original_ragged_dot: Any,
    precision: Any,
    preferred_element_type: Any,
    group_offset: Any,
    output_chunk_size: int,
) -> jax.Array:
  logical_shape = rhs.logical_shape
  if len(logical_shape) != 3:
    dense_rhs = _dequantize_packed_view(rhs)
    return original_ragged_dot(
        cotangent,
        jnp.swapaxes(dense_rhs, -1, -2),
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

  in_dim = int(logical_shape[1])
  out_dim = int(logical_shape[2])
  output_dtype = (
      jnp.dtype(preferred_element_type)
      if preferred_element_type is not None
      else jnp.result_type(cotangent.dtype, rhs.dtype)
  )
  lhs_grad = jnp.zeros((int(cotangent.shape[0]), in_dim), dtype=output_dtype)
  chunk_size = min(int(output_chunk_size), out_dim)

  if _can_use_dynamic_packed_ragged_dot(rhs, chunk_size):

    def body(chunk_index, acc):
      start = chunk_index * chunk_size
      grad_chunk = jax.lax.dynamic_slice_in_dim(
          cotangent, start, chunk_size, axis=-1
      )
      rhs_chunk = _dequantize_view_output_slice_dynamic(
          rhs, start, chunk_size
      )
      return acc + original_ragged_dot(
          grad_chunk,
          jnp.swapaxes(rhs_chunk, -1, -2),
          group_sizes,
          precision=precision,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
      ).astype(output_dtype)

    return jax.lax.fori_loop(0, out_dim // chunk_size, body, lhs_grad)

  chunks = []
  for start in range(0, out_dim, chunk_size):
    size = min(chunk_size, out_dim - start)
    grad_chunk = jax.lax.dynamic_slice_in_dim(cotangent, start, size, axis=-1)
    rhs_chunk = _dequantize_view_output_slice(rhs, start, size)
    chunks.append(
        original_ragged_dot(
            grad_chunk,
            jnp.swapaxes(rhs_chunk, -1, -2),
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
        ).astype(output_dtype)
    )
  return functools.reduce(lambda acc, value: acc + value, chunks, lhs_grad)


def _cast_floating(value: Any, dtype: Any) -> Any:
  if not hasattr(value, "dtype") or not jnp.issubdtype(value.dtype, jnp.floating):
    return value
  return value.astype(dtype)


def _apply_input_clip(module: nn.Module, x: jax.Array) -> jax.Array:
  inf = float("inf")
  clip_input_min = module.param(
      "clip_input_min", lambda key, shape, dtype=None: jnp.array(-inf), ()
  )
  clip_input_max = module.param(
      "clip_input_max", lambda key, shape, dtype=None: jnp.array(inf), ()
  )
  return jnp.clip(x, clip_input_min, clip_input_max)


def _apply_output_clip(module: nn.Module, y: jax.Array) -> jax.Array:
  inf = float("inf")
  clip_output_min = module.param(
      "clip_output_min", lambda key, shape, dtype=None: jnp.array(-inf), ()
  )
  clip_output_max = module.param(
      "clip_output_max", lambda key, shape, dtype=None: jnp.array(inf), ()
  )
  return jnp.clip(y, clip_output_min, clip_output_max)


def _normalize_einsum(eqn: str) -> str:
  eqn = eqn.replace(" ", "")
  if "->" not in eqn or eqn.count(",") != 1:
    raise ValueError(
        "QLoRA einsum equations must be explicit two-operand equations, got "
        f"{eqn!r}."
    )
  return eqn


def _parse_two_operand_einsum(eqn: str) -> tuple[str, str, str]:
  eqn = _normalize_einsum(eqn)
  inputs, output_spec = eqn.split("->")
  lhs_spec, weight_spec = inputs.split(",")
  return lhs_spec, weight_spec, output_spec


def _lora_einsum_str_and_shapes(
    eqn: str,
    weight_shape: Sequence[int],
    rank: int,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
  lhs_spec, weight_spec, output_spec = _parse_two_operand_einsum(eqn)
  if len(weight_spec) != len(weight_shape):
    raise ValueError(
        f"Einsum weight spec {weight_spec!r} does not match weight shape "
        f"{tuple(weight_shape)}."
    )
  weight_dims = dict(zip(weight_spec, tuple(int(dim) for dim in weight_shape)))
  contracted = tuple(
      label for label in weight_spec if label in lhs_spec and label not in output_spec
  )
  produced = tuple(
      label for label in weight_spec if label in output_spec and label not in lhs_spec
  )
  rank_label = _unused_einsum_label(eqn)
  a_spec = "".join(contracted) + rank_label
  b_spec = rank_label + "".join(produced)
  a_shape = tuple(weight_dims[label] for label in contracted) + (rank,)
  b_shape = (rank,) + tuple(weight_dims[label] for label in produced)
  return f"{lhs_spec},{a_spec},{b_spec}->{output_spec}", a_shape, b_shape


def _unused_einsum_label(eqn: str) -> str:
  used = set(re.sub(r"[^A-Za-z]", "", eqn))
  for label in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if label not in used:
      return label
  raise ValueError(f"Cannot find unused einsum label for {eqn!r}.")
