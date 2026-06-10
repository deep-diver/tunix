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

"""Experimental DiffusionGemma checkpoint loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
import itertools
from typing import Any

from absl import logging
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
from tunix.models.diffusion_gemma import model as model_lib
from tunix.models.gemma4 import params as gemma4_params


DIFFUSIONGEMMA_A26B_A4B_IT = (
    "gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it"
)


def _normalise_path(key_path: tuple[Any, ...]) -> list[str]:
  return list(
      itertools.chain.from_iterable(
          str(segment).split("/") for segment in key_path
      )
  )


def _map_self_conditioner(
    params: Mapping[str, Any],
) -> dict[tuple[Any, ...], Any]:
  mapped = {}
  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    parts = _normalise_path(key_path)
    if parts and parts[0] == "transformer":
      parts = parts[1:]
    if not parts or parts[0] != "self_conditioner":
      continue

    module_path = parts[:-1]
    param_name = parts[-1]
    if module_path == ["self_conditioner", "pre_norm"]:
      mapped[("self_conditioner", "pre_norm", param_name)] = value
    elif module_path == ["self_conditioner", "post_norm"]:
      mapped[("self_conditioner", "post_norm", param_name)] = value
    elif module_path == ["self_conditioner", "ffw", "gating_einsum"]:
      if value.shape[0] != 2:
        raise ValueError(
            "Expected self_conditioner ffw gating_einsum shape[0] == 2, "
            f"got {value.shape[0]}."
        )
      mapped[("self_conditioner", "ffw", "gate_proj", "kernel")] = value[0].T
      mapped[("self_conditioner", "ffw", "up_proj", "kernel")] = value[1].T
    elif module_path == ["self_conditioner", "ffw", "linear"]:
      mapped[("self_conditioner", "ffw", "down_proj", "kernel")] = value
  return mapped


def map_from_upstream_checkpoint(params: Mapping[str, Any]) -> dict[str, Any]:
  mapped = flax.traverse_util.flatten_dict(
      gemma4_params.map_from_upstream_checkpoint(params)
  )
  mapped.update(_map_self_conditioner(params))
  return flax.traverse_util.unflatten_dict(mapped)


def _merge_with_initialized_state(
    initialized_state: Any,
    mapped_params: Mapping[str, Any],
) -> dict[str, Any]:
  state_dict = nnx.to_pure_dict(initialized_state)
  flat_state = flax.traverse_util.flatten_dict(state_dict)
  flat_mapped = flax.traverse_util.flatten_dict(mapped_params)
  usable = {}
  skipped = []
  for key, value in flat_mapped.items():
    if key not in flat_state:
      skipped.append(key)
      continue
    if getattr(value, "shape", None) != getattr(flat_state[key], "shape", None):
      raise ValueError(
          f"Shape mismatch for {key}: checkpoint={value.shape}, "
          f"model={flat_state[key].shape}"
      )
    usable[key] = value
  if skipped:
    logging.info(
        "Skipping %d DiffusionGemma checkpoint keys not present in the NNX "
        "model: %s",
        len(skipped),
        sorted(str(k) for k in skipped)[:20],
    )
  missing = set(flat_state) - set(usable)
  if missing:
    logging.warning(
        "DiffusionGemma checkpoint did not provide %d NNX parameters. They "
        "remain randomly initialized. First missing keys: %s",
        len(missing),
        sorted(str(k) for k in missing)[:20],
    )
  flat_state.update(usable)
  return flax.traverse_util.unflatten_dict(flat_state)


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> model_lib.DiffusionGemma_A26B_A4B:
  """Loads a DiffusionGemma model from an upstream Orbax checkpoint.

  This loader maps the Gemma4-compatible backbone and the known
  ``self_conditioner`` leaves. Missing diffusion-only leaves are kept at their
  deterministic initialization and reported as warnings so MVP smoke tests can
  validate the training path before full checkpoint parity is completed.
  """
  abs_model = nnx.eval_shape(
      lambda: model_lib.DiffusionGemma_A26B_A4B(model_config, rngs=nnx.Rngs(0))
  )
  raw_params = ocp.StandardCheckpointer().restore(checkpoint_path)
  mapped_params = map_from_upstream_checkpoint(raw_params)
  model_state = nnx.state(abs_model)
  merged_params = _merge_with_initialized_state(model_state, mapped_params)
  if mesh is not None:
    typed_params = jax.tree.map(
        lambda x, s: jnp.asarray(x, device=s, dtype=dtype),
        merged_params,
        nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh)),
    )
  else:
    typed_params = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=dtype), merged_params
    )
  nnx.update(abs_model, typed_params)
  return abs_model
