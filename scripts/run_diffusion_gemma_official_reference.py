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

"""Runs the upstream DiffusionGemma SFT recipe without importing Tunix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextlib
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
      choices=sorted(_RECIPE_MODULES),
      default="pubmedqa",
  )
  parser.add_argument("--gemma_ref", default=None)
  parser.add_argument("--hackable_diffusion_ref", default=None)
  parser.add_argument("--workdir", default=None)
  parser.add_argument("--checkpoint_path", default=None)
  parser.add_argument("--num_train_steps", type=int, default=None)
  parser.add_argument("--checkpoint_every_n_steps", type=int, default=None)
  parser.add_argument("--lora_rank", type=int, default=None)
  parser.add_argument("--dataset_batch_size", type=int, default=None)
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
  )
  parser.add_argument(
      "--train_loop",
      choices=["kauldron", "hybrid"],
      default="kauldron",
      help=(
          "Use the upstream Kauldron Trainer loop, or a minimal loop that "
          "reuses upstream model/data/loss/trainstep objects and logs losses "
          "from addressable shards."
      ),
  )
  parser.add_argument("--config_override", action="append", default=[])
  parser.add_argument("--module_override", action="append", default=[])
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  extra_paths = [
      path for path in (args.hackable_diffusion_ref, args.gemma_ref) if path
  ]
  dependency_report = _check_dependencies(extra_paths)
  _json_event(event="official_reference_dependencies", **dependency_report)
  if not dependency_report["available"]:
    raise SystemExit("Official DiffusionGemma dependencies missing.")

  with _temporary_sys_path(extra_paths):
    from kauldron import konfig  # pylint: disable=g-import-not-at-top

    cfg = _build_config(
        recipe=args.recipe,
        workdir=args.workdir,
        checkpoint_path=args.checkpoint_path,
        num_train_steps=args.num_train_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        lora_rank=args.lora_rank,
        dataset_batch_size=args.dataset_batch_size,
        use_early_stopping=args.use_early_stopping,
        disable_evals=args.disable_evals,
        module_overrides=_parse_key_value(args.module_override),
        config_overrides=_parse_key_value(args.config_override),
    )
    _json_event(
        event="official_reference_config_built",
        recipe=args.recipe,
        workdir=getattr(cfg, "workdir", None),
        num_train_steps=getattr(cfg, "num_train_steps", None),
        checkpoint_path=getattr(
            getattr(cfg, "init_transform", None), "path", None
        ),
        use_lora=getattr(getattr(cfg, "aux", None), "use_lora", None),
        lora_rank=getattr(getattr(cfg, "aux", None), "lora_rank", None),
        dataset_batch_size=args.dataset_batch_size,
        prompt_len=getattr(getattr(cfg, "aux", None), "prompt_len", None),
        num_canvases=getattr(getattr(cfg, "aux", None), "num_canvases", None),
        canvas_size=getattr(getattr(cfg, "aux", None), "canvas_size", None),
        evals=sorted(getattr(cfg, "evals", {}).keys()),
    )
    if args.build_config_only:
      return
    trainer = konfig.resolve(cfg)
    if args.train_loop == "hybrid":
      _run_hybrid_official_loop(trainer, num_steps=args.num_train_steps or 1)
    else:
      trainer.train()
  _json_event(
      event="official_reference_train_complete",
      recipe=args.recipe,
      workdir=args.workdir,
      num_train_steps=args.num_train_steps,
      train_loop=args.train_loop,
  )


def _run_hybrid_official_loop(trainer: Any, *, num_steps: int) -> None:
  """Runs upstream train steps while extracting losses without all-gathers."""
  import jax  # pylint: disable=g-import-not-at-top
  from kauldron.train import train_loop  # pylint: disable=g-import-not-at-top

  setup = trainer.setup
  setup.log_status("Configuring upstream hybrid DiffusionGemma loop ...")
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
          "event": "official_reference_hybrid_loop_start",
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

    step_value = _safe_scalar_to_int(getattr(state, "step", loop_step + 1))
    losses = _safe_average_loss_values(getattr(aux, "loss_states", None))
    _json_event(
        event="official_reference_hybrid_step_complete",
        loop_step=loop_step,
        state_step=step_value,
        losses=losses,
    )

  result = {
      "event": "official_reference_hybrid_train_complete",
      "num_steps": num_steps,
      "workdir": str(workdir),
      "state_step": _safe_scalar_to_int(getattr(state, "step", num_steps)),
  }
  _write_json(workdir / "hybrid_loop_state.json", result)
  _json_event(**result)


def _build_config(
    *,
    recipe: str,
    workdir: str | None,
    checkpoint_path: str | None,
    num_train_steps: int | None,
    checkpoint_every_n_steps: int | None,
    lora_rank: int | None,
    dataset_batch_size: int | None,
    use_early_stopping: bool | None,
    disable_evals: bool,
    module_overrides: Mapping[str, Any],
    config_overrides: Mapping[str, Any],
):
  module = importlib.import_module(_RECIPE_MODULES[recipe])
  patched_attrs = dict(module_overrides)
  if checkpoint_path is not None:
    patched_attrs["CHECKPOINT_PATH"] = checkpoint_path
  if lora_rank is not None:
    patched_attrs["_LORA_RANK"] = lora_rank

  with _patched_module_attrs(
      module, patched_attrs
  ), _patched_dataset_batch_size(module, dataset_batch_size):
    config_args_cls = getattr(module, "ConfigArgs", None)
    if config_args_cls is not None and use_early_stopping is not None:
      cfg = module.get_config(
          config_args_cls(use_early_stopping=use_early_stopping)
      )
    else:
      cfg = module.get_config()

  if workdir is not None:
    cfg.workdir = workdir
  if num_train_steps is not None:
    cfg.num_train_steps = num_train_steps
  if checkpoint_every_n_steps is not None:
    _set_dotted_attr(
        cfg, "aux.checkpoint_every_n_steps", checkpoint_every_n_steps
    )
    checkpointer = getattr(cfg, "checkpointer", None)
    if checkpointer is not None and hasattr(
        checkpointer, "save_interval_steps"
    ):
      checkpointer.save_interval_steps = checkpoint_every_n_steps
  if disable_evals and hasattr(cfg, "evals"):
    cfg.evals = {}
  for dotted_path, value in config_overrides.items():
    _set_dotted_attr(cfg, dotted_path, value)
  return cfg


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
  if isinstance(target, dict):
    target[key] = value
    return
  setattr(target, key, value)


def _safe_average_loss_values(loss_states: Any) -> dict[str, float]:
  """Extracts Kauldron AverageState losses without global host all-gathers."""
  if loss_states is None:
    return {}

  import jax  # pylint: disable=g-import-not-at-top

  values: dict[str, float] = {}
  leaves = jax.tree_util.tree_flatten_with_path(
      loss_states,
      is_leaf=lambda x: hasattr(x, "total") and hasattr(x, "count"),
  )[0]
  for path, state in leaves:
    if not (hasattr(state, "total") and hasattr(state, "count")):
      continue
    total = _safe_array_scalar(state.total)
    count = _safe_array_scalar(state.count)
    value = 0.0 if count == 0.0 else total / count
    values[f"losses/{_jax_path_to_string(path)}"] = value

  if values:
    values["losses/total"] = sum(values.values())
  return values


def _safe_array_scalar(value: Any) -> float:
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  if isinstance(value, jax.Array):
    if hasattr(value, "addressable_data"):
      value = value.addressable_data(0)
    elif getattr(value, "addressable_shards", None):
      value = value.addressable_shards[0].data
  host_value = np.asarray(jax.device_get(value))
  if host_value.size == 0:
    return 0.0
  return float(host_value.reshape(-1)[0])


def _safe_scalar_to_int(value: Any) -> int:
  return int(round(_safe_array_scalar(value)))


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
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _check_dependencies(
    extra_paths: Sequence[str | pathlib.Path] = (),
) -> dict[str, Any]:
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


def _json_event(**payload: Any) -> None:
  print(json.dumps(payload, default=str, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
