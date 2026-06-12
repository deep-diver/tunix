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

"""Compatibility wrapper for official DiffusionGemma SFT recipes.

This module intentionally keeps the official Flax/Linen + Hackable Diffusion
training path intact. Tunix uses it as an optional backend for DiffusionGemma
when the official `gemma`, `hackable_diffusion`, and `kauldron` packages are
available, while the native NNX/Qwix implementation remains in `sft.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import contextlib
import dataclasses
import importlib
import json
import pathlib
import sys
from typing import Any


_RECIPE_MODULES = {
    "pubmedqa": (
        "gemma.diffusion.hackable_diffusion_adapter.configs.sft_pubmedqa"
    ),
    "sudoku": "gemma.diffusion.hackable_diffusion_adapter.configs.sft_sudoku",
    "sudoku_full": (
        "gemma.diffusion.hackable_diffusion_adapter.configs.sft_sudoku_full"
    ),
}


class OfficialBackendDependencyError(ImportError):
  """Raised when the optional official DiffusionGemma backend is unavailable."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class OfficialSFTConfig:
  """Configuration for the official Hackable Diffusion SFT backend.

  Attributes:
    recipe: Official recipe to load. Supported values are `pubmedqa`, `sudoku`,
      and `sudoku_full`.
    gemma_ref: Optional checkout/root path containing the official
      `gemma.diffusion` package. Added to `sys.path` only while importing and
      resolving the official config.
    hackable_diffusion_ref: Optional checkout/root path containing the
      `hackable_diffusion` package.
    workdir: Optional Kauldron workdir override.
    checkpoint_path: Optional public or local DiffusionGemma checkpoint path.
    num_train_steps: Optional train step override for controlled runs.
    checkpoint_every_n_steps: Optional checkpointer interval override.
    lora_rank: Optional LoRA rank override. This is applied before the official
      config factory is called so the official LoRA wrapper is constructed with
      the requested rank.
    dataset_batch_size: Optional batch-size override for official dataset
      builder calls. The official recipes keep their default batch size when
      this is unset.
    skip_step_metrics: If true, patches the resolved Kauldron writer to skip
      per-step metric materialization. This is intended only for controlled GPU
      runs on environments where the official multi-GPU metric all-gather fails
      after the train step has run.
    train_loop: Training loop implementation. `kauldron` delegates to the
      official `Trainer.train()`. `hybrid` uses the official resolved model,
      data, checkpoint loader, sharding, train step, optimizer, and LoRA mask,
      but drives the loop directly to avoid Kauldron's post-step metric
      materialization and final host sync.
    use_early_stopping: Optional PubMedQA early-stopping eval toggle.
    disable_evals: If true, drops official evals after the config is built.
      Useful for GPU runs that only need train-step evidence.
    module_overrides: Additional module-level overrides applied before
      `get_config()` is called. Use sparingly; this is intended for path-like
      constants in the official config modules.
    config_overrides: Additional dotted-path overrides applied after the
      official config is built, for example `{"aux.eval_num_batches": 1}`.
  """

  recipe: str = "pubmedqa"
  gemma_ref: str | pathlib.Path | None = None
  hackable_diffusion_ref: str | pathlib.Path | None = None
  workdir: str | pathlib.Path | None = None
  checkpoint_path: str | pathlib.Path | None = None
  num_train_steps: int | None = None
  checkpoint_every_n_steps: int | None = None
  lora_rank: int | None = None
  dataset_batch_size: int | None = None
  skip_step_metrics: bool = False
  train_loop: str = "kauldron"
  use_early_stopping: bool | None = None
  disable_evals: bool = False
  module_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  config_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def recipe_module_name(recipe: str) -> str:
  """Returns the import path for an official DiffusionGemma recipe."""
  try:
    return _RECIPE_MODULES[recipe]
  except KeyError as exc:
    choices = ", ".join(sorted(_RECIPE_MODULES))
    raise ValueError(
        f"Unknown DiffusionGemma recipe {recipe!r}; use {choices}."
    ) from exc


def check_dependencies(
    extra_paths: Sequence[str | pathlib.Path] = (),
) -> dict[str, Any]:
  """Checks whether the optional official backend can be imported."""
  missing = []
  versions = {}
  with _temporary_sys_path(extra_paths):
    for module_name in ("gemma", "hackable_diffusion", "kauldron"):
      try:
        module = importlib.import_module(module_name)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        missing.append({"module": module_name, "error": repr(exc)})
      else:
        versions[module_name] = getattr(module, "__version__", None)
  return {
      "available": not missing,
      "missing": missing,
      "versions": versions,
  }


def build_official_sft_config(config: OfficialSFTConfig):
  """Builds an official Kauldron config with Tunix-side overrides applied."""
  module = _import_recipe_module(config)
  module_overrides = dict(config.module_overrides)
  if config.checkpoint_path is not None:
    module_overrides["CHECKPOINT_PATH"] = str(config.checkpoint_path)
  if config.lora_rank is not None:
    module_overrides["_LORA_RANK"] = config.lora_rank

  with _patched_module_attrs(
      module, module_overrides
  ), _patched_dataset_batch_size(module, config.dataset_batch_size):
    cfg = _call_get_config(module, config)

  _apply_common_config_overrides(cfg, config)
  for dotted_path, value in config.config_overrides.items():
    _set_dotted_attr(cfg, dotted_path, value)
  return cfg


def resolve_official_trainer(config: OfficialSFTConfig):
  """Resolves the official Kauldron trainer for this backend."""
  extra_paths = _extra_paths(config)
  with _temporary_sys_path(extra_paths):
    try:
      from kauldron import konfig  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise OfficialBackendDependencyError(
          "The official DiffusionGemma backend requires kauldron. Install the "
          "official gemma and hackable_diffusion checkouts, then retry."
      ) from exc
    return konfig.resolve(build_official_sft_config(config))


@dataclasses.dataclass(frozen=True)
class OfficialDiffusionGemmaTrainer:
  """Thin Tunix-facing handle around the official Kauldron trainer."""

  config: OfficialSFTConfig = dataclasses.field(
      default_factory=OfficialSFTConfig
  )

  def build_config(self):
    return build_official_sft_config(self.config)

  def resolve(self):
    return resolve_official_trainer(self.config)

  def train(self):
    trainer = self.resolve()
    if self.config.train_loop == "hybrid":
      return run_hybrid_official_loop(
          trainer,
          num_steps=self.config.num_train_steps or 1,
      )
    if self.config.train_loop != "kauldron":
      raise ValueError(
          "Official DiffusionGemma train_loop must be 'kauldron' or "
          f"'hybrid', got {self.config.train_loop!r}."
      )
    if self.config.skip_step_metrics:
      _patch_trainer_to_skip_step_metrics(trainer)
    return trainer.train()


def run_hybrid_official_loop(trainer: Any, *, num_steps: int) -> dict[str, Any]:
  """Runs official DiffusionGemma train steps without Kauldron loop syncs."""
  if num_steps < 1:
    raise ValueError(f"num_steps must be positive, got {num_steps}.")

  try:
    import jax  # pylint: disable=g-import-not-at-top
    from kauldron.train import train_loop  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "The hybrid official DiffusionGemma loop requires kauldron and jax."
    ) from exc

  setup = trainer.setup
  setup.log_status("Configuring hybrid official DiffusionGemma loop ...")
  setup.run(trainer)

  trainstep = trainer.trainstep
  checkpointer = trainer.checkpointer
  latest_step = checkpointer.latest_step
  state = trainstep.init(
      elem_spec=trainer.train_ds.element_spec,
      skip_transforms=latest_step is not None,
  )
  chrono = trainer._chrono  # pylint: disable=protected-access
  ds_iter = iter(trainer.train_ds)
  state, chrono, ds_iter = checkpointer.restore(
      train_loop.checkpoint_state.CheckpointState(state, chrono, ds_iter),
      noop_if_missing=True,
  )

  workdir = pathlib.Path(str(trainer.workdir))
  workdir.mkdir(parents=True, exist_ok=True)
  _write_json(
      workdir / "hybrid_loop_start.json",
      {
          "event": "official_backend_hybrid_loop_start",
          "num_steps": num_steps,
          "workdir": str(workdir),
      },
  )

  for loop_step in range(num_steps):
    batch = next(ds_iter)
    batch = train_loop.sharding_lib.device_put(batch, trainer.sharding.batch)
    state, aux = trainstep.step(
        state,
        batch,
        return_losses=True,
        return_metrics=False,
        return_summaries=False,
        checkify_error_categories=trainer.checkify_error_categories,
    )
    if trainer.checkify_error_categories:
      jax.device_get(aux.error).throw()

    losses = _safe_average_loss_values(getattr(aux, "loss_states", None))
    print(
        _json_dumps({
            "event": "official_backend_hybrid_step_complete",
            "loop_step": loop_step,
            "state_step": loop_step + 1,
            "losses": losses,
        }),
        flush=True,
    )

  result = {
      "event": "official_backend_hybrid_train_complete",
      "num_steps": num_steps,
      "workdir": str(workdir),
      "state_step": num_steps,
  }
  _write_json(workdir / "hybrid_loop_state.json", result)
  print(_json_dumps(result), flush=True)
  return result


def _extra_paths(config: OfficialSFTConfig) -> list[str]:
  paths = []
  for path in (config.hackable_diffusion_ref, config.gemma_ref):
    if path is not None:
      paths.append(str(pathlib.Path(path).expanduser()))
  return paths


def _import_recipe_module(config: OfficialSFTConfig):
  extra_paths = _extra_paths(config)
  module_name = recipe_module_name(config.recipe)
  with _temporary_sys_path(extra_paths):
    try:
      return importlib.import_module(module_name)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      report = check_dependencies(extra_paths)
      raise OfficialBackendDependencyError(
          "Could not import the official DiffusionGemma recipe "
          f"{module_name!r}. Dependency report: {report}"
      ) from exc


def _call_get_config(module: Any, config: OfficialSFTConfig):
  if not hasattr(module, "get_config"):
    raise OfficialBackendDependencyError(
        f"Official recipe module {module.__name__!r} has no get_config()."
    )
  config_args_cls = getattr(module, "ConfigArgs", None)
  if config_args_cls is not None and config.use_early_stopping is not None:
    return module.get_config(
        config_args_cls(use_early_stopping=config.use_early_stopping)
    )
  return module.get_config()


def _apply_common_config_overrides(cfg: Any, config: OfficialSFTConfig) -> None:
  if config.workdir is not None:
    cfg.workdir = str(config.workdir)
  if config.num_train_steps is not None:
    cfg.num_train_steps = config.num_train_steps
  if config.checkpoint_every_n_steps is not None:
    _set_dotted_attr(
        cfg, "aux.checkpoint_every_n_steps", config.checkpoint_every_n_steps
    )
    checkpointer = getattr(cfg, "checkpointer", None)
    if checkpointer is not None and hasattr(
        checkpointer, "save_interval_steps"
    ):
      checkpointer.save_interval_steps = config.checkpoint_every_n_steps
  if config.disable_evals and hasattr(cfg, "evals"):
    cfg.evals = {}


def _set_dotted_attr(obj: Any, dotted_path: str, value: Any) -> None:
  if not dotted_path:
    raise ValueError("Override path must not be empty.")
  parts = dotted_path.split(".")
  target = obj
  for part in parts[:-1]:
    target = _get_child(target, part)
  _set_child(target, parts[-1], value)


def _get_child(target: Any, key: str) -> Any:
  if isinstance(target, Mapping):
    return target[key]
  return getattr(target, key)


def _set_child(target: Any, key: str, value: Any) -> None:
  if isinstance(target, MutableMapping):
    target[key] = value
    return
  setattr(target, key, value)


def _patch_trainer_to_skip_step_metrics(trainer: Any) -> None:
  writer = getattr(trainer, "writer", None)
  if writer is None:
    return

  def _skip_write_step_metrics(*args, **kwargs):
    step = kwargs.get("step")
    if step is None and args:
      step = args[0]
    print(
        f"Skipping official DiffusionGemma step metrics at step {step}.",
        flush=True,
    )

  object.__setattr__(writer, "write_step_metrics", _skip_write_step_metrics)


def _safe_average_loss_values(loss_states: Any) -> dict[str, float]:
  """Extracts Kauldron AverageState losses without global host all-gathers."""
  if loss_states is None:
    return {}

  try:
    import jax  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Loss extraction requires jax."
    ) from exc

  values: dict[str, float] = {}
  leaves = jax.tree_util.tree_flatten_with_path(
      loss_states,
      is_leaf=_is_loss_average_state,
  )[0]
  for path, state in leaves:
    if not _is_loss_average_state(state):
      continue
    total = _safe_array_scalar(
        getattr(state, "total", getattr(state, "value", 0.0))
    )
    count = _safe_array_scalar(state.count)
    value = 0.0 if count == 0.0 else total / count
    values[f"losses/{_jax_path_to_string(path)}"] = value

  if values:
    values["losses/total"] = sum(values.values())
  return values


def _addressable_param_checksum(
    tree: Any,
    predicate,
    *,
    max_leaves: int | None = None,
) -> dict[str, Any]:
  try:
    import jax  # pylint: disable=g-import-not-at-top
    import numpy as np  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Checksum computation requires jax and numpy."
    ) from exc

  checksum = 0.0
  num_leaves = 0
  num_elements = 0
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    path_str = _jax_path_to_string(path)
    if not predicate(path_str):
      continue
    if not hasattr(leaf, "addressable_shards") and not hasattr(leaf, "shape"):
      continue
    if max_leaves is not None and num_leaves >= max_leaves:
      break

    if isinstance(leaf, jax.Array):
      if hasattr(leaf, "addressable_data"):
        shard_arrays = [
            leaf.addressable_data(i)
            for i in range(len(leaf.addressable_shards))
        ]
      else:
        shard_arrays = [shard.data for shard in leaf.addressable_shards]
    else:
      shard_arrays = [leaf]
    for shard_array in shard_arrays:
      shard_host = np.asarray(jax.device_get(shard_array))
      checksum += float(shard_host.astype(np.float64).sum())
      num_elements += int(shard_host.size)
    num_leaves += 1

  return {
      "checksum": checksum,
      "num_leaves": num_leaves,
      "num_elements": num_elements,
  }


def _is_loss_average_state(value: Any) -> bool:
  return (
      hasattr(value, "count")
      and (hasattr(value, "total") or hasattr(value, "value"))
  )


def _safe_array_scalar(value: Any) -> float:
  try:
    import jax  # pylint: disable=g-import-not-at-top
    import numpy as np  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Scalar extraction requires jax and numpy."
    ) from exc

  if isinstance(value, jax.Array):
    if hasattr(value, "addressable_data"):
      value = value.addressable_data(0)
    elif getattr(value, "addressable_shards", None):
      value = value.addressable_shards[0].data
  host_value = np.asarray(jax.device_get(value))
  if host_value.size == 0:
    return 0.0
  return float(host_value.reshape(-1)[0])


def _jax_path_to_string(path: Any) -> str:
  parts = []
  for part in path:
    key = getattr(part, "key", None)
    if key is None:
      key = getattr(part, "name", None)
    if key is None:
      key = str(part)
    parts.append(str(key))
  return "/".join(parts)


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _json_dumps(payload: Mapping[str, Any]) -> str:
  return json.dumps(payload, default=str, sort_keys=True)


@contextlib.contextmanager
def _patched_dataset_batch_size(module: Any, batch_size: int | None):
  if batch_size is None:
    yield
    return

  patched_fns = []
  try:
    for name in dir(module):
      if not name.endswith("_data"):
        continue
      data_module = getattr(module, name)
      for fn_name in dir(data_module):
        if not (fn_name.startswith("make_") and fn_name.endswith("_ds")):
          continue
        original = getattr(data_module, fn_name)
        if not callable(original):
          continue

        def wrapped(*args, __original=original, **kwargs):
          if "batch_size" in kwargs:
            kwargs["batch_size"] = batch_size
          return __original(*args, **kwargs)

        patched_fns.append((data_module, fn_name, original))
        setattr(data_module, fn_name, wrapped)
    yield
  finally:
    for data_module, fn_name, original in patched_fns:
      setattr(data_module, fn_name, original)


@contextlib.contextmanager
def _temporary_sys_path(paths: Sequence[str | pathlib.Path]):
  old_path = list(sys.path)
  try:
    for path in reversed([str(pathlib.Path(p).expanduser()) for p in paths]):
      if path and path not in sys.path:
        sys.path.insert(0, path)
    yield
  finally:
    sys.path[:] = old_path


@contextlib.contextmanager
def _patched_module_attrs(module: Any, overrides: Mapping[str, Any]):
  sentinel = object()
  old_values = {}
  try:
    for name, value in overrides.items():
      old_values[name] = getattr(module, name, sentinel)
      setattr(module, name, value)
    yield
  finally:
    for name, old_value in old_values.items():
      if old_value is sentinel:
        delattr(module, name)
      else:
        setattr(module, name, old_value)
