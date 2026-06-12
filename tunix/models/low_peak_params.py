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

"""Low-peak-memory helpers for loading frozen base params plus adapters.

The utilities in this module intentionally operate on plain PyTrees. Model
families provide the checkpoint-specific mapping/filter callbacks, while the
generic code enforces the important PEFT invariant: restored base parameters
must not overwrite already-created adapter parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

from flax import traverse_util
import jax
import jax.numpy as jnp


Path = tuple[Any, ...]
LeafPredicate = Callable[[Path, Any], bool]
PathMapper = Callable[[Path, Any], Path | None]


@dataclasses.dataclass(frozen=True)
class RestoreTargetInfo:
  """Restore target tree plus accounting for excluded leaves."""

  target: Any
  restored_paths: tuple[Path, ...]
  preserved_paths: tuple[Path, ...]
  skipped_paths: tuple[Path, ...]


@dataclasses.dataclass(frozen=True)
class MergeReport:
  """Accounting produced while merging checkpoint leaves into model state."""

  restored_paths: tuple[Path, ...]
  preserved_paths: tuple[Path, ...]
  missing_paths: tuple[Path, ...]
  extra_paths: tuple[Path, ...]
  skipped_preserved_restore_paths: tuple[Path, ...]


@dataclasses.dataclass(frozen=True)
class MergeResult:
  """Merged tree and restore accounting."""

  tree: Any
  report: MergeReport


def _flatten(tree: Any) -> dict[Path, Any]:
  return dict(traverse_util.flatten_dict(tree))


def _unflatten(flat: Mapping[Path, Any]) -> Any:
  return traverse_util.unflatten_dict(dict(flat))


def _path_sort_key(path: Path) -> tuple[str, ...]:
  return tuple(str(part) for part in path)


def _shape_dtype_struct(
    value: Any,
    *,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> jax.ShapeDtypeStruct:
  if not hasattr(value, "shape") or not hasattr(value, "dtype"):
    raise TypeError(
        "Restore targets require array-like leaves with shape and dtype; "
        f"got {type(value)!r}."
    )
  target_dtype = dtype if dtype is not None else value.dtype
  if sharding is None:
    return jax.ShapeDtypeStruct(value.shape, target_dtype)
  return jax.ShapeDtypeStruct(value.shape, target_dtype, sharding=sharding)


def build_restore_target(
    initialized_state: Any,
    *,
    preserve_predicate: LeafPredicate,
    restore_predicate: LeafPredicate | None = None,
    dtype: jnp.dtype | None = None,
    sharding_tree: Any | None = None,
) -> RestoreTargetInfo:
  """Builds an Orbax/JAX restore target excluding adapter leaves.

  Args:
    initialized_state: Abstract or concrete model state tree.
    preserve_predicate: Returns True for leaves that must remain initialized,
      such as LoRA adapter parameters.
    restore_predicate: Optional additional filter for restorable base leaves.
    dtype: Optional dtype override for restored leaves.
    sharding_tree: Optional tree matching ``initialized_state`` containing
      sharding objects for the restored leaves.

  Returns:
    ``RestoreTargetInfo`` containing a target tree made only from restorable
    leaves and sorted path accounting.
  """
  restore_predicate = restore_predicate or (lambda _path, _leaf: True)
  flat_state = _flatten(initialized_state)
  flat_sharding = _flatten(sharding_tree) if sharding_tree is not None else {}
  flat_target = {}
  restored_paths = []
  preserved_paths = []
  skipped_paths = []
  for path, value in flat_state.items():
    if preserve_predicate(path, value):
      preserved_paths.append(path)
      continue
    if not restore_predicate(path, value):
      skipped_paths.append(path)
      continue
    flat_target[path] = _shape_dtype_struct(
        value, dtype=dtype, sharding=flat_sharding.get(path)
    )
    restored_paths.append(path)

  return RestoreTargetInfo(
      target=_unflatten(flat_target),
      restored_paths=tuple(sorted(restored_paths, key=_path_sort_key)),
      preserved_paths=tuple(sorted(preserved_paths, key=_path_sort_key)),
      skipped_paths=tuple(sorted(skipped_paths, key=_path_sort_key)),
  )


def map_checkpoint_tree(
    checkpoint_tree: Any,
    mapper: PathMapper,
) -> Any:
  """Maps a restored checkpoint tree through a caller-provided path mapper.

  ``mapper`` may return ``None`` to drop a checkpoint leaf. Collisions are
  rejected because silently keeping one copy is almost always a checkpoint
  conversion bug.
  """
  mapped = {}
  for path, value in _flatten(checkpoint_tree).items():
    mapped_path = mapper(path, value)
    if mapped_path is None:
      continue
    if mapped_path in mapped:
      raise ValueError(f"Multiple checkpoint leaves map to {mapped_path!r}.")
    mapped[mapped_path] = value
  return _unflatten(mapped)


def merge_restored_state(
    initialized_state: Any,
    restored_state: Any,
    *,
    preserve_predicate: LeafPredicate,
    strict: bool = True,
) -> MergeResult:
  """Merges restored base leaves while preserving adapter leaves.

  Args:
    initialized_state: Model state tree containing base and adapter leaves.
    restored_state: Restored checkpoint tree, already mapped into model layout.
    preserve_predicate: Returns True for leaves that restored checkpoints must
      not overwrite.
    strict: When True, all non-preserved initialized leaves must be restored and
      restored leaves must exist in ``initialized_state``.

  Returns:
    A ``MergeResult`` with the merged tree and path accounting.
  """
  flat_initialized = _flatten(initialized_state)
  flat_restored = _flatten(restored_state)
  flat_merged = dict(flat_initialized)
  restored_paths = []
  skipped_preserved = []
  extra_paths = []

  for path, value in flat_restored.items():
    if path not in flat_initialized:
      extra_paths.append(path)
      continue
    if preserve_predicate(path, flat_initialized[path]):
      skipped_preserved.append(path)
      continue
    expected = flat_initialized[path]
    if getattr(value, "shape", None) != getattr(expected, "shape", None):
      raise ValueError(
          f"Shape mismatch for {path}: checkpoint={getattr(value, 'shape', None)}, "
          f"model={getattr(expected, 'shape', None)}"
      )
    flat_merged[path] = value
    restored_paths.append(path)

  preserved_paths = tuple(
      sorted(
          (
              path
              for path, value in flat_initialized.items()
              if preserve_predicate(path, value)
          ),
          key=_path_sort_key,
      )
  )
  missing_paths = tuple(
      sorted(
          (
              path
              for path, value in flat_initialized.items()
              if (
                  not preserve_predicate(path, value)
                  and path not in flat_restored
              )
          ),
          key=_path_sort_key,
      )
  )
  extra_paths = tuple(sorted(extra_paths, key=_path_sort_key))
  skipped_preserved = tuple(sorted(skipped_preserved, key=_path_sort_key))

  if strict and (missing_paths or extra_paths):
    raise ValueError(
        "Restored checkpoint does not match model state: "
        f"{len(missing_paths)} missing, {len(extra_paths)} extra."
    )

  return MergeResult(
      tree=_unflatten(flat_merged),
      report=MergeReport(
          restored_paths=tuple(sorted(restored_paths, key=_path_sort_key)),
          preserved_paths=preserved_paths,
          missing_paths=missing_paths,
          extra_paths=extra_paths,
          skipped_preserved_restore_paths=skipped_preserved,
      ),
  )


def path_contains_lora(path: Path, _value: Any) -> bool:
  """Path predicate for common LoRA/adapter checkpoint leaves."""
  return any("lora" in str(part).lower() for part in path)
