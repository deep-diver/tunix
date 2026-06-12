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
import dataclasses
import itertools
from typing import Any

from absl import logging
from etils import epath
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
from tunix.models import low_peak_params
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
    *,
    preserve_predicate: low_peak_params.LeafPredicate | None = None,
) -> dict[str, Any]:
  if hasattr(initialized_state, "_mapping"):
    state_dict = nnx.to_pure_dict(initialized_state)
  else:
    state_dict = initialized_state
  preserve_predicate = preserve_predicate or (lambda _path, _value: False)
  result = low_peak_params.merge_restored_state(
      state_dict,
      mapped_params,
      preserve_predicate=preserve_predicate,
      strict=False,
  )
  if result.report.extra_paths:
    logging.info(
        "Skipping %d DiffusionGemma checkpoint keys not present in the NNX "
        "model: %s",
        len(result.report.extra_paths),
        sorted(str(k) for k in result.report.extra_paths)[:20],
    )
  if result.report.skipped_preserved_restore_paths:
    logging.info(
        "Preserved %d initialized adapter leaves instead of overwriting them "
        "from the DiffusionGemma checkpoint. First paths: %s",
        len(result.report.skipped_preserved_restore_paths),
        sorted(str(k) for k in result.report.skipped_preserved_restore_paths)[
            :20
        ],
    )
  if result.report.missing_paths:
    logging.warning(
        "DiffusionGemma checkpoint did not provide %d NNX parameters. They "
        "remain randomly initialized. First missing keys: %s",
        len(result.report.missing_paths),
        sorted(str(k) for k in result.report.missing_paths)[:20],
    )
  return result.tree


def _load_raw_params(
    checkpoint_path: str,
    *,
    restore_concurrent_gb: int | None,
) -> Mapping[str, Any]:
  """Restores upstream DiffusionGemma params using the public metadata tree."""
  checkpointer_kwargs = {}
  handler_kwargs = {}
  if restore_concurrent_gb is not None:
    checkpointer_kwargs["restore_concurrent_gb"] = restore_concurrent_gb
    handler_kwargs["restore_concurrent_gb"] = restore_concurrent_gb
  ckpt = ocp.StandardCheckpointer(**checkpointer_kwargs)
  path = epath.Path(checkpoint_path)
  metadata = ckpt.metadata(path)
  if (
      metadata.item_metadata is None
      and path.joinpath("_CHECKPOINT_METADATA").exists()
  ):
    path = path / "default"
    metadata = ckpt.metadata(path)
  if metadata.item_metadata is None:
    raise ValueError(f"No item metadata found in {path}")

  target = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
      metadata.item_metadata.tree,
  )
  handler = ocp.StandardCheckpointHandler(**handler_kwargs)
  if str(path).startswith("gs://"):
    logging.warning(
        "Using Orbax handler-level restore for %s because this public GCS "
        "checkpoint may not have the finalization marker required by the "
        "Checkpointer wrapper.",
        path,
    )
    return handler.restore(path, args=ocp.args.StandardRestore(target))

  try:
    return ckpt.restore(path, target)
  except ValueError as err:
    if "Found incomplete checkpoint" not in str(err):
      raise
    logging.warning(
        "Falling back to Orbax handler-level restore for %s because the "
        "checkpoint does not have the finalization marker required by the "
        "Checkpointer wrapper.",
        path,
    )
    return handler.restore(path, args=ocp.args.StandardRestore(target))


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    restore_concurrent_gb: int | None = 16,
) -> model_lib.DiffusionGemma_A26B_A4B:
  """Loads a DiffusionGemma model from an upstream Orbax checkpoint.

  This loader maps the Gemma4-compatible backbone and the known
  ``self_conditioner`` leaves. Missing diffusion-only leaves are kept at their
  deterministic initialization and reported as warnings so MVP validation tests can
  validate the training path before full checkpoint parity is completed.
  """
  if dataclasses.is_dataclass(model_config):
    model_config = dataclasses.replace(
        model_config, dtype=dtype, param_dtype=dtype
    )
  abs_model = nnx.eval_shape(
      lambda: model_lib.DiffusionGemma_A26B_A4B(model_config, rngs=nnx.Rngs(0))
  )
  raw_params = _load_raw_params(
      checkpoint_path, restore_concurrent_gb=restore_concurrent_gb
  )
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
