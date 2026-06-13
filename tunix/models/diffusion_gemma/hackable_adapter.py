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
import functools
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

_LOGGED_DEVICE_LOSS_KEYS: set[tuple[str, int]] = set()
_QWIX_CHECKPOINTER_PATCHED = False
_QWIX_KAULDRON_SELECT_PATCHED = False
_QWIX_OPTAX_PATCHED = False


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
    run_steps: Optional hybrid-loop step limit. This leaves the official
      recipe schedule length unchanged while allowing short validation runs.
    checkpoint_every_n_steps: Optional checkpointer interval override.
    lora_rank: Optional LoRA rank override. This is applied before the official
      config factory is called so the official LoRA wrapper is constructed with
      the requested rank.
    lora_backend: LoRA implementation used by the official recipe. `official`
      preserves the DeepMind Hackable Diffusion wrapper. `qwix_lora` patches
      only the recipe's LoRA constructor so the Linen model is wrapped by
      Tunix/Qwix while the rest of the official backend remains intact.
    lora_alpha: Optional Qwix LoRA alpha. When unset, Qwix Linen LoRA uses
      `rank` so the adapter scale is 1.0, matching the official unscaled LoRA
      adapter more closely than the common `alpha=2*rank` recipe.
    qwix_lora_module_path: Optional Qwix `module_path` regex override.
    dataset_batch_size: Optional batch-size override for official dataset
      builder calls. The official recipes keep their default batch size when
      this is unset.
    skip_step_metrics: If true, patches the resolved Kauldron writer to skip
      per-step metric materialization. This is intended only for controlled GPU
      runs on environments where the official multi-GPU metric all-gather fails
      after the train step has run.
    log_losses: If true, the hybrid loop requests and logs official loss states
      from each train step. Disable this to isolate train-step execution from
      host-side loss materialization.
    sync_after_step: Hybrid-loop synchronization point. `state` preserves the
      strict device-state block. `losses` synchronizes by reading addressable
      loss shards only, avoiding full host all-gathers. `none` only dispatches
      the official step.
    save_final_checkpoint: If true, the hybrid loop writes a forced final
      Kauldron checkpoint after the last train step. This is disabled by
      default to preserve the lowest-memory validation path.
    train_loop: Training loop implementation. `kauldron` delegates to the
      official `Trainer.train()`. `hybrid` uses the official resolved model,
      data, checkpoint loader, sharding, train step, optimizer, and LoRA mask,
      but drives the loop directly to avoid Kauldron's post-step metric
      materialization and final host sync.
    use_early_stopping: Optional PubMedQA early-stopping eval toggle.
    disable_evals: If true, drops official evals after the config is built.
      Useful for GPU runs that only need train-step evidence.
    log_param_summary: If true, the hybrid loop writes metadata-only parameter
      summaries that separate LoRA leaves from dense non-LoRA leaves without
      materializing parameter values on host.
    encoder_loss_token_chunk_size: Optional token chunk size for a
      memory-safer exact encoder AR cross-entropy loss. When unset, the
      official EncoderARLoss is preserved. This is useful for LoRA runs where
      full-sequence full-vocab softmax intermediates dominate peak memory.
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
  run_steps: int | None = None
  checkpoint_every_n_steps: int | None = None
  lora_rank: int | None = None
  lora_backend: str = "official"
  lora_alpha: float | None = None
  qwix_lora_module_path: str | None = None
  dataset_batch_size: int | None = None
  skip_step_metrics: bool = False
  log_losses: bool = True
  sync_after_step: str = "state"
  save_final_checkpoint: bool = False
  train_loop: str = "kauldron"
  use_early_stopping: bool | None = None
  disable_evals: bool = False
  log_param_summary: bool = False
  encoder_loss_token_chunk_size: int | None = None
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
  initialize_jax_before_tensorflow(extra_paths)
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


def chunked_softmax_cross_entropy_with_integer_labels(
    logits: Any,
    labels: Any,
    *,
    token_chunk_size: int | None,
):
  """Computes exact token CE while chunking the sequence dimension."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  logits = jnp.asarray(logits, dtype=jnp.float32)
  labels = jnp.asarray(labels)
  if logits.ndim < 2:
    raise ValueError(
        f"logits must have at least 2 dimensions, got {logits.ndim}."
    )
  if labels.shape != logits.shape[:-1]:
    raise ValueError(
        "labels shape must match logits without the vocab dimension: "
        f"{labels.shape} vs {logits.shape[:-1]}."
    )

  seq_len = logits.shape[1] if logits.ndim >= 3 else 1
  if (
      token_chunk_size is None
      or token_chunk_size <= 0
      or token_chunk_size >= seq_len
  ):
    target_logits = jnp.take_along_axis(
        logits, labels[..., None], axis=-1
    ).squeeze(axis=-1)
    return jax.nn.logsumexp(logits, axis=-1) - target_logits

  if logits.ndim != 3:
    raise ValueError(
        "token_chunk_size currently expects rank-3 logits [batch, seq, vocab], "
        f"got shape {logits.shape}."
    )

  pad_len = (-seq_len) % token_chunk_size
  if pad_len:
    logits = jnp.pad(logits, ((0, 0), (0, pad_len), (0, 0)))
    labels = jnp.pad(labels, ((0, 0), (0, pad_len)), constant_values=0)

  batch_size, padded_seq_len, vocab_size = logits.shape
  num_chunks = padded_seq_len // token_chunk_size
  logits_chunks = logits.reshape(
      batch_size, num_chunks, token_chunk_size, vocab_size
  )
  label_chunks = labels.reshape(batch_size, num_chunks, token_chunk_size)

  def scan_body(_, chunk_inputs):
    logits_chunk, labels_chunk = chunk_inputs
    target_logits = jnp.take_along_axis(
        logits_chunk, labels_chunk[..., None], axis=-1
    ).squeeze(axis=-1)
    ce = jax.nn.logsumexp(logits_chunk, axis=-1) - target_logits
    return None, ce

  _, ce_chunks = jax.lax.scan(
      scan_body,
      None,
      (jnp.swapaxes(logits_chunks, 0, 1), jnp.swapaxes(label_chunks, 0, 1)),
  )
  ce = jnp.swapaxes(ce_chunks, 0, 1).reshape(batch_size, padded_seq_len)
  return ce[:, :seq_len]


def build_official_sft_config(config: OfficialSFTConfig):
  """Builds an official Kauldron config with Tunix-side overrides applied."""
  _validate_lora_backend(config)
  initialize_jax_before_tensorflow(_extra_paths(config))
  module = _import_recipe_module(config)
  if config.lora_backend != "official":
    _patch_kauldron_lora_select_for_qwix()
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
  initialize_jax_before_tensorflow(extra_paths)
  with _temporary_sys_path(extra_paths):
    try:
      from kauldron import konfig  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise OfficialBackendDependencyError(
          "The official DiffusionGemma backend requires kauldron. Install the "
          "official gemma and hackable_diffusion checkouts, then retry."
      ) from exc
    trainer = konfig.resolve(build_official_sft_config(config))
    _replace_resolved_lora_backend(trainer, config)
    _replace_resolved_encoder_ar_loss_with_memory_safe(trainer, config)
    return trainer


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
          num_steps=self.config.run_steps or self.config.num_train_steps or 1,
          log_losses=self.config.log_losses,
          sync_after_step=self.config.sync_after_step,
          save_final_checkpoint=self.config.save_final_checkpoint,
          log_param_summary=self.config.log_param_summary,
      )
    if self.config.train_loop != "kauldron":
      raise ValueError(
          "Official DiffusionGemma train_loop must be 'kauldron' or "
          f"'hybrid', got {self.config.train_loop!r}."
      )
    if self.config.skip_step_metrics:
      _patch_trainer_to_skip_step_metrics(trainer)
    return trainer.train()


def run_hybrid_official_loop(
    trainer: Any,
    *,
    num_steps: int,
    log_losses: bool = True,
    sync_after_step: str = "state",
    save_final_checkpoint: bool = False,
    log_param_summary: bool = False,
) -> dict[str, Any]:
  """Runs official DiffusionGemma train steps without Kauldron loop syncs."""
  if num_steps < 1:
    raise ValueError(f"num_steps must be positive, got {num_steps}.")
  if sync_after_step not in ("state", "losses", "none"):
    raise ValueError(
        "sync_after_step must be 'state', 'losses', or 'none', got "
        f"{sync_after_step!r}."
    )

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
  if log_param_summary:
    _emit_param_tree_summary(
        state,
        label="post_restore",
        workdir=workdir,
    )

  for loop_step in range(num_steps):
    batch = next(ds_iter)
    batch = train_loop.sharding_lib.device_put(batch, trainer.sharding.batch)
    state, aux = trainstep.step(
        state,
        batch,
        return_losses=log_losses,
        return_metrics=False,
        return_summaries=False,
        checkify_error_categories=trainer.checkify_error_categories,
    )
    if sync_after_step == "state":
      _block_first_array(state)
    if trainer.checkify_error_categories:
      jax.device_get(aux.error).throw()

    logged_losses = []
    if log_losses:
      logged_losses = _emit_device_loss_values(
          getattr(aux, "loss_states", None),
          loop_step=loop_step,
          state_step=loop_step + 1,
      )
    print(
        _json_dumps({
            "event": "official_backend_hybrid_step_complete",
            "loop_step": loop_step,
            "state_step": loop_step + 1,
            "logged_losses": logged_losses,
            "sync_after_step": sync_after_step,
        }),
        flush=True,
    )
    if loop_step == 0 or (loop_step + 1) % 10 == 0:
      _write_json(
          workdir / "hybrid_loop_progress.json",
          {
              "event": "official_backend_hybrid_loop_progress",
              "loop_step": loop_step,
              "state_step": loop_step + 1,
              "num_steps": num_steps,
              "logged_losses": logged_losses,
              "sync_after_step": sync_after_step,
              "workdir": str(workdir),
          },
      )
    if log_param_summary and (loop_step == 0 or loop_step + 1 == num_steps):
      _emit_param_tree_summary(
          state,
          label=f"after_step_{loop_step + 1}",
          workdir=workdir,
      )

  result = {
      "event": "official_backend_hybrid_train_complete",
      "num_steps": num_steps,
      "workdir": str(workdir),
      "state_step": num_steps,
  }
  if save_final_checkpoint:
    final_step = _safe_train_step(getattr(state, "step", num_steps), num_steps)
    checkpoint_state = train_loop.checkpoint_state.CheckpointState(
        state, chrono, ds_iter
    )
    save_result = checkpointer.save(
        checkpoint_state,
        step=final_step,
        force=True,
    )
    checkpointer.wait_until_finished()
    result.update({
        "checkpoint_save_returned": bool(save_result),
        "checkpoint_saved": True,
        "checkpoint_step": final_step,
    })
  _write_json(workdir / "hybrid_loop_state.json", result)
  print(_json_dumps(result), flush=True)
  return result


def _extra_paths(config: OfficialSFTConfig) -> list[str]:
  paths = []
  for path in (config.hackable_diffusion_ref, config.gemma_ref):
    if path is not None:
      paths.append(str(pathlib.Path(path).expanduser()))
  return paths


_JAX_PREINIT_DONE = False


def initialize_jax_before_tensorflow(
    extra_paths: Sequence[str | pathlib.Path] = (),
) -> dict[str, Any]:
  """Initializes JAX before TensorFlow/Kauldron can claim CUDA libraries."""
  global _JAX_PREINIT_DONE
  if _JAX_PREINIT_DONE:
    return {"event": "official_backend_jax_preinit", "already_done": True}

  report: dict[str, Any] = {"event": "official_backend_jax_preinit"}
  with _temporary_sys_path(extra_paths):
    try:
      import jax  # pylint: disable=g-import-not-at-top

      report["devices"] = [str(device) for device in jax.devices()]
      report["device_count"] = jax.device_count()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      report["jax_error"] = repr(exc)
      print(_json_dumps(report), flush=True)
      return report

    try:
      import tensorflow as tf  # pylint: disable=g-import-not-at-top

      tf.config.set_visible_devices([], "GPU")
      report["tensorflow_gpu_hidden"] = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
      report["tensorflow_error"] = repr(exc)

  _JAX_PREINIT_DONE = True
  print(_json_dumps(report), flush=True)
  return report


def _import_recipe_module(config: OfficialSFTConfig):
  extra_paths = _extra_paths(config)
  initialize_jax_before_tensorflow(extra_paths)
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


def _validate_lora_backend(config: OfficialSFTConfig) -> None:
  valid = {"official", "qwix_lora"}
  if config.lora_backend not in valid:
    raise ValueError(
        "Official DiffusionGemma lora_backend must be one of "
        f"{sorted(valid)}, got {config.lora_backend!r}."
    )


def _replace_resolved_lora_backend(
    trainer: Any,
    config: OfficialSFTConfig,
) -> None:
  """Replaces the resolved official LoRA module with a Qwix Linen bridge."""
  if config.lora_backend == "official":
    return
  _patch_qwix_checkpoint_loader(config)
  _patch_qwix_optax_apply_updates()

  model_candidates = _resolved_sft_model_candidates(trainer)
  if not model_candidates:
    raise OfficialBackendDependencyError(
        "Resolved official DiffusionGemma trainer does not expose "
        "an SFT model with gemma_network for Qwix LoRA replacement."
    )
  gemma_network = getattr(model_candidates[0], "gemma_network")

  bridge = _load_linen_qwix_lora_module()
  rank = config.lora_rank or getattr(gemma_network, "rank", None)
  if rank is None:
    aux = getattr(getattr(trainer, "cfg", None), "aux", None)
    rank = getattr(aux, "lora_rank", None)
  if rank is None:
    raise OfficialBackendDependencyError(
        "Could not infer LoRA rank for resolved Qwix Linen backend."
    )
  alpha = config.lora_alpha if config.lora_alpha is not None else float(rank)
  base_network = getattr(gemma_network, "model", gemma_network)
  qwix_methods = None
  if config.encoder_loss_token_chunk_size is not None:
    base_network = _with_encoder_hidden_methods(base_network)
    qwix_methods = _qwix_methods_with_encoder_hidden(bridge)
  target_modules = getattr(gemma_network, "target_modules", "all-linear")
  module_path = _qwix_module_path_from_official_targets(
      bridge,
      target_modules,
      override=config.qwix_lora_module_path,
  )
  replacement = bridge.apply_lora_to_linen_model(
      base_network,
      rank=int(rank),
      alpha=float(alpha),
      module_path=module_path,
      methods=qwix_methods or bridge.DIFFUSION_GEMMA_LINEN_LORA_METHODS,
  )

  for model in model_candidates:
    object.__setattr__(model, "gemma_network", replacement)
  print(
      _json_dumps({
          "event": "official_backend_qwix_lora_replaced",
          "lora_backend": config.lora_backend,
          "model_candidate_count": len(model_candidates),
          "rank": int(rank),
          "alpha": float(alpha),
          "module_path": module_path,
          "encoder_hidden_methods": bool(qwix_methods),
      }),
      flush=True,
  )


def _resolved_sft_model_candidates(trainer: Any) -> list[Any]:
  candidates = []
  seen = set()
  for owner in (trainer, getattr(trainer, "trainstep", None)):
    if owner is None:
      continue
    model = getattr(owner, "model", None)
    if model is None or not hasattr(model, "gemma_network"):
      continue
    marker = id(model)
    if marker not in seen:
      seen.add(marker)
      candidates.append(model)
  return candidates


def _qwix_module_path_from_official_targets(
    bridge: Any,
    target_modules: Any,
    *,
    override: str | None,
) -> str:
  if override is not None:
    return override
  if target_modules is None or target_modules == "all-linear":
    return bridge.official_compatible_module_path()
  if isinstance(target_modules, str):
    return target_modules
  try:
    patterns = tuple(str(pattern) for pattern in target_modules)
  except TypeError as exc:
    raise ValueError(
        f"Unsupported official target_modules value: {target_modules!r}"
    ) from exc
  return bridge.official_compatible_module_path(
      tuple(f".*{pattern}.*" for pattern in patterns)
  )


def _load_linen_qwix_lora_module():
  module_path = pathlib.Path(__file__).with_name("linen_qwix_lora.py")
  spec = importlib.util.spec_from_file_location(
      "_tunix_diffusion_gemma_linen_qwix_lora", module_path
  )
  if spec is None or spec.loader is None:
    raise OfficialBackendDependencyError(
        f"Could not load Qwix Linen LoRA bridge from {module_path}."
    )
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _qwix_methods_with_encoder_hidden(bridge: Any) -> tuple[str, ...]:
  return tuple(
      dict.fromkeys(
          tuple(bridge.DIFFUSION_GEMMA_LINEN_LORA_METHODS)
          + ("encoder_hidden_call", "decode_hidden")
      )
  )


@functools.cache
def _encoder_hidden_network_cls(base_cls: type[Any]):
  try:
    from flax import linen as nn  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Encoder-hidden DiffusionGemma network wrapping requires flax and jax."
    ) from exc

  class EncoderHiddenNetwork(base_cls):
    """Adds hidden-state encoder/decode methods for memory-safe CE."""

    @nn.compact
    def encoder_hidden_call(
        self,
        *,
        x: Any,
        conditioning_embeddings: dict[str, Any],
    ) -> Any:
      if len(x.shape) == 3:
        tokens = x[..., 0]
      else:
        tokens = x
      cache = conditioning_embeddings.get("kv_cache", None)
      positions = conditioning_embeddings.get("positions", None)
      attention_mask = conditioning_embeddings.get("attention_mask", None)
      return self.gemma_model(
          tokens=tokens,
          cache=cache,
          positions=positions,
          attention_mask=attention_mask,
          return_hidden_states=True,
      )

    @nn.compact
    def decode_hidden(self, hidden: Any) -> Any:
      logits = self.gemma_model.embedder.decode(hidden)
      softcap = self.gemma_model.config.final_logit_softcap
      if softcap is not None:
        logits /= softcap
        logits = jnp.tanh(logits) * softcap
      return logits

  EncoderHiddenNetwork.__name__ = f"EncoderHidden{base_cls.__name__}"
  return EncoderHiddenNetwork


def _with_encoder_hidden_methods(network: Any) -> Any:
  if hasattr(network, "encoder_hidden_call") and hasattr(
      network, "decode_hidden"
  ):
    return network
  cls = _encoder_hidden_network_cls(network.__class__)
  kwargs = _linen_module_init_values(network)
  return cls(**kwargs)


def _linen_module_init_values(module: Any) -> dict[str, Any]:
  values = {}
  for field in dataclasses.fields(module):
    if not field.init or field.name in ("parent", "name"):
      continue
    if hasattr(module, field.name):
      values[field.name] = getattr(module, field.name)
  return values


def _qwix_lora_select_patterns(pattern: str | Sequence[str]) -> tuple[str, ...]:
  """Expands official LoRA optimizer masks to include Qwix Linen names."""
  if isinstance(pattern, str):
    patterns = (pattern,)
  else:
    patterns = tuple(str(item) for item in pattern)
  expanded: list[str] = []
  for item in patterns:
    expanded.append(item)
    if item == "lora":
      expanded.extend((r".*_lora_a", r".*_lora_b"))
  return tuple(dict.fromkeys(expanded))


def _patch_kauldron_lora_select_for_qwix() -> None:
  """Makes the official LoRA optimizer mask include Qwix LoRA leaves."""
  global _QWIX_KAULDRON_SELECT_PATCHED
  if _QWIX_KAULDRON_SELECT_PATCHED:
    return
  try:
    from kauldron import kd  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Qwix LoRA optimizer mask patching requires kauldron."
    ) from exc

  if getattr(kd.optim, "_tunix_qwix_lora_select_patched", False):
    _QWIX_KAULDRON_SELECT_PATCHED = True
    return

  original_select = kd.optim.select

  def select(pattern):
    return original_select(_qwix_lora_select_patterns(pattern))

  kd.optim._tunix_original_select = original_select
  kd.optim.select = select
  kd.optim._tunix_qwix_lora_select_patched = True
  _QWIX_KAULDRON_SELECT_PATCHED = True
  print(
      _json_dumps({
          "event": "official_backend_qwix_kauldron_select_patched",
          "lora_patterns": _qwix_lora_select_patterns("lora"),
      }),
      flush=True,
  )


def _replace_resolved_encoder_ar_loss_with_memory_safe(
    trainer: Any,
    config: OfficialSFTConfig,
) -> None:
  """Moves resolved encoder AR loss behind a memory-safe model wrapper."""
  chunk_size = config.encoder_loss_token_chunk_size
  if chunk_size is None:
    return
  if chunk_size <= 0:
    raise ValueError(
        "encoder_loss_token_chunk_size must be positive when set, got "
        f"{chunk_size}."
    )
  _replace_resolved_sft_model_with_memory_safe_encoder_loss(
      trainer, int(chunk_size)
  )
  train_losses = getattr(trainer, "train_losses", None)
  if train_losses is None or "encoder_loss" not in train_losses:
    raise OfficialBackendDependencyError(
        "The resolved official trainer has no train_losses['encoder_loss'] "
        "to replace."
    )
  original = train_losses.get("encoder_loss")
  replacement_cls = _encoder_loss_value_cls()
  kwargs = {
      key: value
      for key, value in _dataclass_init_values(original).items()
      if key in ("step", "mask", "weight", "normalize_by")
  }
  replacement = replacement_cls(**kwargs)
  updated_losses = dict(train_losses)
  updated_losses["encoder_loss"] = replacement
  object.__setattr__(trainer, "train_losses", updated_losses)
  _replace_trainstep_aux_losses(
      getattr(trainer, "trainstep", None), updated_losses
  )
  aux = getattr(trainer, "aux", None)
  if isinstance(aux, MutableMapping):
    aux["encoder_loss_token_chunk_size"] = int(chunk_size)
  elif aux is not None:
    aux.encoder_loss_token_chunk_size = int(chunk_size)
  print(
      _json_dumps({
          "event": "official_backend_memory_safe_encoder_loss_replaced",
          "token_chunk_size": int(chunk_size),
          "original_loss": type(original).__name__,
          "replacement_loss": replacement_cls.__name__,
          "model_output": "preds.encoder_loss",
      }),
      flush=True,
  )


def _replace_resolved_sft_model_with_memory_safe_encoder_loss(
    trainer: Any,
    chunk_size: int,
) -> None:
  model = getattr(trainer, "model", None)
  if model is None or not hasattr(model, "gemma_network"):
    raise OfficialBackendDependencyError(
        "The resolved official trainer model cannot be wrapped for "
        "memory-safe encoder loss because it has no gemma_network."
    )
  gemma_network = getattr(model, "gemma_network")
  if not hasattr(gemma_network, "encoder_hidden_call") or not hasattr(
      gemma_network, "decode_hidden"
  ):
    raise OfficialBackendDependencyError(
        "Memory-safe encoder loss requires a gemma_network with "
        "encoder_hidden_call() and decode_hidden(). Use lora_backend "
        "'qwix_lora' for this path."
    )
  wrapper_cls = _memory_safe_sft_diffusion_cls(model.__class__)
  kwargs = _linen_module_init_values(model)
  kwargs["encoder_loss_token_chunk_size"] = int(chunk_size)
  replacement = wrapper_cls(**kwargs)
  for owner in (trainer, getattr(trainer, "trainstep", None)):
    if owner is not None and getattr(owner, "model", None) is model:
      object.__setattr__(owner, "model", replacement)
  print(
      _json_dumps({
          "event": "official_backend_memory_safe_sft_model_replaced",
          "original_model": type(model).__name__,
          "replacement_model": wrapper_cls.__name__,
          "token_chunk_size": int(chunk_size),
      }),
      flush=True,
  )


def _replace_trainstep_aux_losses(
    trainstep: Any, losses: Mapping[str, Any]
) -> None:
  """Keeps the resolved Kauldron TrainStep loss collection in sync."""
  if trainstep is None:
    return
  aux = getattr(trainstep, "aux", None)
  if aux is None or not hasattr(aux, "losses"):
    return
  replacement_losses = _losses_like(getattr(aux, "losses"), losses)
  try:
    new_aux = dataclasses.replace(aux, losses=replacement_losses)
  except TypeError:
    if hasattr(aux, "replace"):
      new_aux = aux.replace(losses=replacement_losses)
    else:
      object.__setattr__(aux, "losses", replacement_losses)
      new_aux = aux
  object.__setattr__(trainstep, "aux", new_aux)


def _losses_like(original: Any, losses: Mapping[str, Any]) -> Any:
  if original is None:
    return dict(losses)
  try:
    return type(original)(losses)
  except Exception:  # pylint: disable=broad-exception-caught
    return dict(losses)


@functools.cache
def _memory_safe_sft_diffusion_cls(base_cls: type[Any]):
  try:
    from flax import linen as nn  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Memory-safe SFTDiffusion wrapping requires flax, jax, and the "
        "official gemma DiffusionGemma SFT module."
    ) from exc

  class MemorySafeSFTDiffusion(base_cls):
    """Official SFTDiffusion with model-integrated chunked encoder CE."""

    encoder_loss_token_chunk_size: int = 128

    @nn.compact
    def __call__(
        self,
        x0: Any,
        prompt: Any,
        canvas_id: Any,
        canvas_mask: Any,
        encoder_target: Any,
        encoder_target_mask: Any,
        is_training: bool = True,
    ):
      time = self.time_sampler(self.make_rng("sampling"), x0)
      xt, target_info = self.corruption_process.corrupt(
          self.make_rng("sampling"), x0, time
      )

      first_token_indices = jnp.arange(self.num_canvases) * self.canvas_size
      canvas_validity = canvas_mask[:, first_token_indices]
      num_valid_canvases = jnp.sum(canvas_validity, axis=-1)
      num_valid_canvases = jnp.maximum(num_valid_canvases, 1)
      selected_canvas_idx = jax.random.randint(
          self.make_rng("sampling"),
          shape=num_valid_canvases.shape,
          minval=0,
          maxval=num_valid_canvases,
      )

      x0_tokens = x0[..., 0] if x0.ndim == 3 else x0
      encoder_hidden, kv_cache, positions, prompt_mask = _sft_encode_hidden(
          gemma_network=self.gemma_network,
          prompt=prompt,
          x0_tokens=x0_tokens,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          prompt_len=self.prompt_len,
          total_canvas_len=self.total_canvas_len,
          canvas_size=self.canvas_size,
          pad_token=self.pad_token,
      )
      encoder_loss = _encoder_loss_from_hidden(
          self.gemma_network,
          encoder_hidden,
          encoder_target,
          encoder_target_mask,
          chunk_size=self.encoder_loss_token_chunk_size,
      )

      if self.stop_gradient_from_denoiser_to_encoder:
        kv_cache = jax.lax.stop_gradient(kv_cache)

      decoder_kwargs = dict(
          gemma_network=self.gemma_network,
          xt=xt,
          time=time,
          kv_cache=kv_cache,
          positions=positions,
          prompt_mask=prompt_mask,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          prompt_len=self.prompt_len,
          total_canvas_len=self.total_canvas_len,
          canvas_size=self.canvas_size,
          is_training=is_training,
      )
      denoiser_output_first_pass = sft_model.sft_decode(**decoder_kwargs)

      target_mask = canvas_mask & (canvas_id == selected_canvas_idx[:, None])
      target_info["is_corrupted"] = (
          target_info["is_corrupted"] & target_mask[..., None]
      )
      target_info["target_mask"] = target_mask[..., None]

      converted_first_pass = self.corruption_process.convert_predictions(
          denoiser_output_first_pass, xt, time
      )
      converted_first_pass = jax.lax.stop_gradient(converted_first_pass)
      sc_logits = converted_first_pass["logits"]
      zero_logits = jnp.zeros_like(sc_logits)

      batch_size = xt.shape[0]
      do_self_cond = (
          jax.random.uniform(self.make_rng("sampling"), shape=(batch_size,))
          < self.self_cond_prob
      )
      do_self_cond = do_self_cond.reshape(
          (batch_size,) + (1,) * (sc_logits.ndim - 1)
      )
      sc_logits = jnp.where(do_self_cond, sc_logits, zero_logits)

      denoiser_output = sft_model.sft_decode(
          **decoder_kwargs, sc_logits=sc_logits
      )
      converted = self.corruption_process.convert_predictions(
          denoiser_output, xt, time
      )
      noise_info = self.corruption_process.get_schedule_info(time)
      return {
          "output": converted,
          "target": target_info,
          "xt": xt,
          "noise_info": noise_info,
          "encoder_loss": encoder_loss,
          "encoder_target": encoder_target,
          "encoder_target_mask": encoder_target_mask,
      }

  MemorySafeSFTDiffusion.__name__ = f"MemorySafe{base_cls.__name__}"
  return MemorySafeSFTDiffusion


def _sft_encode_hidden(
    gemma_network: Any,
    *,
    prompt: Any,
    x0_tokens: Any,
    canvas_mask: Any,
    selected_canvas_idx: Any,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    pad_token: int,
) -> tuple[Any, Any, Any, Any]:
  del total_canvas_len
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers  # pylint: disable=g-import-not-at-top

  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)
  prompt_mask = prompt != pad_token
  full_seq_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)
  batch_size, full_seq_len = full_seq.shape
  kv_cache = gemma_network.init_cache(
      batch_size=batch_size,
      cache_length=full_seq_len,
  )
  positions = mask_helpers.build_positions_from_mask(full_seq_mask)
  attention_mask = mask_helpers.make_causal_prefill_mask(
      full_seq_mask, full_seq_len
  )
  encoder_out = gemma_network.encoder_hidden_call(
      x=full_seq,
      conditioning_embeddings={
          "kv_cache": kv_cache,
          "positions": positions,
          "attention_mask": attention_mask,
      },
  )
  kv_cache = encoder_out.cache
  encoder_hidden = encoder_out.hidden_states
  if kv_cache is None:
    raise ValueError("KV cache should not be None after encoder pass.")
  if encoder_hidden is None:
    raise ValueError("Encoder hidden states should not be None.")
  end_index = prompt_len + selected_canvas_idx * canvas_size
  kv_cache = mask_helpers.set_cache_end_index(kv_cache, end_index)
  return encoder_hidden, kv_cache, positions, prompt_mask


def _encoder_loss_from_hidden(
    gemma_network: Any,
    hidden: Any,
    targets: Any,
    mask: Any,
    *,
    chunk_size: int | None,
) -> Any:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  hidden = jnp.asarray(hidden)
  targets = jnp.asarray(targets)
  mask = jnp.asarray(mask)
  if chunk_size is None or chunk_size <= 0 or chunk_size >= hidden.shape[1]:
    logits = gemma_network.decode_hidden(hidden).astype(jnp.float32)
    token_loss = _token_ce_from_logits(logits, targets)
    token_mask = mask.astype(token_loss.dtype)
    return jnp_sum_over_nonbatch(token_loss * token_mask) / jnp_maximum(
        jnp_sum_over_nonbatch(token_mask), 1.0
    )

  seq_len = hidden.shape[1]
  pad_len = (-seq_len) % chunk_size
  if pad_len:
    hidden = jnp.pad(hidden, ((0, 0), (0, pad_len), (0, 0)))
    targets = jnp.pad(targets, ((0, 0), (0, pad_len)), constant_values=0)
    mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=0)
  num_chunks = hidden.shape[1] // chunk_size
  hidden_chunks = hidden.reshape(
      hidden.shape[0], num_chunks, chunk_size, hidden.shape[-1]
  )
  target_chunks = targets.reshape(targets.shape[0], num_chunks, chunk_size)
  mask_chunks = mask.reshape(mask.shape[0], num_chunks, chunk_size)

  def scan_body(carry, chunk_inputs):
    total_loss, total_weight = carry
    hidden_chunk, target_chunk, mask_chunk = chunk_inputs
    logits = gemma_network.decode_hidden(hidden_chunk).astype(jnp.float32)
    token_loss = _token_ce_from_logits(logits, target_chunk)
    token_mask = mask_chunk.astype(token_loss.dtype)
    return (
        total_loss + jnp.sum(token_loss * token_mask, axis=1),
        total_weight + jnp.sum(token_mask, axis=1),
    ), None

  init = (
      jnp.zeros((hidden.shape[0],), dtype=jnp.float32),
      jnp.zeros((hidden.shape[0],), dtype=jnp.float32),
  )
  (loss_sum, weight_sum), _ = jax.lax.scan(
      scan_body,
      init,
      (
          jnp.swapaxes(hidden_chunks, 0, 1),
          jnp.swapaxes(target_chunks, 0, 1),
          jnp.swapaxes(mask_chunks, 0, 1),
      ),
  )
  return loss_sum / jnp.maximum(weight_sum, 1.0)


def _token_ce_from_logits(logits: Any, targets: Any) -> Any:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  target_logits = jnp.take_along_axis(
      logits, targets[..., None], axis=-1
  ).squeeze(axis=-1)
  return jax.nn.logsumexp(logits, axis=-1) - target_logits


def _dataclass_init_values(instance: Any) -> dict[str, Any]:
  try:
    fields = dataclasses.fields(instance)
  except TypeError:
    fields = ()
  values = {}
  for field in fields:
    if field.init and hasattr(instance, field.name):
      values[field.name] = getattr(instance, field.name)
  if values:
    return values
  for field_name in (
      "encoder_logits",
      "encoder_target",
      "encoder_target_mask",
      "weight",
      "mask",
      "normalize_by",
  ):
    if hasattr(instance, field_name):
      values[field_name] = getattr(instance, field_name)
  return values


@functools.cache
def _encoder_loss_value_cls():
  try:
    from kauldron import kd  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Encoder loss value replacement requires kauldron."
    ) from exc

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class EncoderLossValue(kd.losses.Loss):
    """Reads model-integrated per-example encoder loss values."""

    encoder_loss: kd.kontext.Key = "preds.encoder_loss"

    def get_values(self, encoder_loss):
      return encoder_loss

  EncoderLossValue.__name__ = "EncoderLossValue"
  EncoderLossValue.__annotations__ = {"encoder_loss": kd.kontext.Key}
  return EncoderLossValue


def jnp_sum_over_nonbatch(value: Any):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  if value.ndim <= 1:
    return value
  return jnp.sum(value, axis=tuple(range(1, value.ndim)))


def jnp_maximum(lhs: Any, rhs: Any):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  return jnp.maximum(lhs, rhs)


def _is_qwix_or_official_lora_path(path: str) -> bool:
  leaf = path.rsplit("/", 1)[-1]
  return (
      "/lora/" in path or leaf.endswith("_lora_a") or leaf.endswith("_lora_b")
  )


def _jax_key_path_to_string(path: Sequence[Any]) -> str:
  parts = []
  for part in path:
    if hasattr(part, "key"):
      parts.append(str(part.key))
    elif hasattr(part, "name"):
      parts.append(str(part.name))
    elif hasattr(part, "idx"):
      parts.append(str(part.idx))
    else:
      parts.append(str(part))
  return "/".join(parts)


def _is_qwix_quantized_value(value: Any) -> bool:
  return hasattr(value, "array") and hasattr(value, "how")


def _checkpoint_value_for_model_value(
    model_value: Any,
    checkpoint_value: Any,
) -> Any:
  """Converts checkpoint leaves to the model leaf shape."""
  if not _is_qwix_quantized_value(model_value):
    return checkpoint_value
  try:
    qwix_ptq = importlib.import_module("qwix._src.providers.ptq")
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Qwix quantized checkpoint restore requires qwix."
    ) from exc
  return qwix_ptq.WithAux(
      qwix_ptq.qarray.quantize(checkpoint_value, model_value.how),
      model_value.how,
  )


def _delete_jax_arrays_in_tree(value: Any, jax_module: Any) -> None:
  for leaf in jax_module.tree_util.tree_leaves(value):
    if isinstance(leaf, jax_module.Array):
      leaf.delete()


def _patch_qwix_optax_apply_updates() -> None:
  """Keeps Qwix LoRA base params frozen during Optax updates."""
  global _QWIX_OPTAX_PATCHED
  if _QWIX_OPTAX_PATCHED:
    return
  try:
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import optax  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Qwix LoRA update patching requires jax and optax."
    ) from exc

  if getattr(optax, "_tunix_qwix_apply_updates_patched", False):
    _QWIX_OPTAX_PATCHED = True
    return

  def apply_updates(params, updates):

    def _apply_one(path, param, update):
      if param is None:
        return None
      path_str = _jax_key_path_to_string(path)
      if not _is_qwix_or_official_lora_path(path_str):
        return param
      if update is None:
        return param
      return jnp.asarray(param + update).astype(jnp.asarray(param).dtype)

    return jax.tree_util.tree_map_with_path(_apply_one, params, updates)

  optax._tunix_original_apply_updates = optax.apply_updates
  optax.apply_updates = apply_updates
  optax._tunix_qwix_apply_updates_patched = True
  _QWIX_OPTAX_PATCHED = True
  print(
      _json_dumps({
          "event": "official_backend_qwix_optax_apply_updates_patched",
          "update_predicate": "official_or_qwix_lora_only",
      }),
      flush=True,
  )


def _patch_qwix_checkpoint_loader(config: OfficialSFTConfig) -> None:
  """Extends the official memory-safe loader for Qwix LoRA leaves."""
  del config
  global _QWIX_CHECKPOINTER_PATCHED
  if _QWIX_CHECKPOINTER_PATCHED:
    return

  try:
    gemma_checkpointer = importlib.import_module(
        "gemma.diffusion.hackable_diffusion_adapter.hd.gemma_checkpointer"
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Could not import the official DiffusionGemma checkpointer for Qwix "
        "checkpoint compatibility patching."
    ) from exc

  if getattr(gemma_checkpointer, "_tunix_qwix_patch_applied", False):
    _QWIX_CHECKPOINTER_PATCHED = True
    return

  def _remap_and_match_params(
      model_flat: dict[str, Any],
      ckpt_flat: dict[str, Any],
      lora_init_values: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    if lora_init_values is None:
      lora_init_values = {}

    remapped_ckpt = {}
    for ckpt_path, value in ckpt_flat.items():
      if ckpt_path.endswith("/w"):
        stripped = ckpt_path.rsplit("/w", 1)[0]
        if stripped in model_flat and ckpt_path not in model_flat:
          remapped_ckpt[stripped] = value
          continue
      remapped_ckpt[ckpt_path] = value

    loaded_count = 0
    for path, model_value in model_flat.items():
      if path in remapped_ckpt:
        model_flat[path] = _checkpoint_value_for_model_value(
            model_value, remapped_ckpt[path]
        )
        loaded_count += 1

    for key, value in lora_init_values.items():
      model_flat[key] = value

    ckpt_only = set(remapped_ckpt) - set(model_flat)
    if ckpt_only:
      gemma_checkpointer.logging.warning(
          "Discarding %d checkpoint-only key(s) not present in the model: %s",
          len(ckpt_only),
          sorted(ckpt_only),
      )

    model_only = set(model_flat) - set(remapped_ckpt)
    lora_keys = {
        key for key in model_only if _is_qwix_or_official_lora_path(key)
    }
    non_lora_model_only = model_only - lora_keys
    if lora_keys:
      gemma_checkpointer.logging.info(
          "Keeping %d LoRA key(s) with their initialized values.",
          len(lora_keys),
      )
    if non_lora_model_only:
      raise KeyError(
          f"Found {len(non_lora_model_only)} model-only key(s) "
          f"(excluding LoRA): {sorted(non_lora_model_only)}"
      )

    gemma_checkpointer.logging.info(
        "Checkpoint loading complete: %d params loaded, "
        "%d checkpoint-only (discarded).",
        loaded_count,
        len(ckpt_only),
    )
    return model_flat

  def cheaply_load_params(params_from_state, checkpoint_path):
    existing = params_from_state
    model_param_spec = (
        gemma_checkpointer._convert_to_element_spec_with_sharding(existing)
    )  # pylint: disable=protected-access

    existing_flat_arrays = gemma_checkpointer.flax.traverse_util.flatten_dict(
        existing, sep="/"
    )
    lora_init_values = {
        key: value
        for key, value in existing_flat_arrays.items()
        if _is_qwix_or_official_lora_path(key)
    }

    for key, value in existing_flat_arrays.items():
      if key not in lora_init_values:
        _delete_jax_arrays_in_tree(value, gemma_checkpointer.jax)

    with gemma_checkpointer.jax.default_device(
        gemma_checkpointer.jax.devices("cpu")[0]
    ):

      def _make_empty_cpu_array(spec):
        return gemma_checkpointer.jnp.empty(
            spec.shape,
            spec.dtype,
            device=gemma_checkpointer.jax.devices("cpu")[0],
        )

      ckpt = gemma_checkpointer.ocp.PyTreeCheckpointer()
      metadata = ckpt.metadata(checkpoint_path)
      lparams_empty = gemma_checkpointer.jax.tree.map(
          _make_empty_cpu_array, metadata.item_metadata.tree
      )
      gemma_params = ckpt.restore(checkpoint_path, item=lparams_empty)

    existing_flat = gemma_checkpointer.flax.traverse_util.flatten_dict(
        model_param_spec, sep="/"
    )
    ckpt_flat = gemma_checkpointer.flax.traverse_util.flatten_dict(
        gemma_params, sep="/"
    )
    existing_flat = _remap_and_match_params(
        existing_flat, ckpt_flat, lora_init_values
    )
    merged = gemma_checkpointer.flax.traverse_util.unflatten_dict(
        existing_flat, sep="/"
    )

    return gemma_checkpointer.jax.tree.map(
        lambda x, y: gemma_checkpointer.jax.device_put(
            x.astype(y.dtype), device=y.sharding
        ),
        merged,
        model_param_spec,
    )

  gemma_checkpointer._remap_and_match_params = _remap_and_match_params  # pylint: disable=protected-access
  gemma_checkpointer.cheaply_load_params = cheaply_load_params
  gemma_checkpointer._tunix_qwix_patch_applied = True
  _QWIX_CHECKPOINTER_PATCHED = True
  print(
      _json_dumps({
          "event": "official_backend_qwix_checkpoint_loader_patched",
          "lora_key_predicate": "official_or_qwix",
      }),
      flush=True,
  )


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
  aux = getattr(cfg, "aux", None)
  if aux is not None:
    aux.lora_backend = config.lora_backend
    aux.lora_alpha = config.lora_alpha
    aux.qwix_lora_module_path = config.qwix_lora_module_path
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


def _emit_device_loss_values(
    loss_states: Any,
    *,
    loop_step: int,
    state_step: int,
) -> list[str]:
  """Logs loss scalars via local-shard scalar reads.

  Device callbacks require blocking an XLA token, which can still force NCCL
  collectives on multi-GPU hosts. This mirrors the official safe-writer style
  by reading the first addressable shard instead.
  """
  values = _safe_average_loss_values(loss_states)
  for metric_name, value in values.items():
    print(
        _json_dumps({
            "event": "diffusion_gemma_hybrid_loss",
            "metric": metric_name,
            "step": state_step,
            "loop_step": loop_step,
            "value": value,
        }),
        flush=True,
    )
  return list(values)


def _emit_param_tree_summary(
    tree: Any,
    *,
    label: str,
    workdir: pathlib.Path,
) -> dict[str, Any]:
  """Writes a metadata-only parameter summary for Qwix LoRA runs."""
  summary = _param_tree_memory_summary(tree)
  summary.update({
      "event": "official_backend_param_tree_summary",
      "label": label,
  })
  _write_json(workdir / f"param_tree_summary_{label}.json", summary)
  print(_json_dumps(summary), flush=True)
  return summary


def _param_tree_memory_summary(
    tree: Any,
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
  """Summarizes quantized, LoRA, and dense leaf storage without host reads."""
  try:
    import jax  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Parameter summary requires jax."
    ) from exc

  summary: dict[str, Any] = {
      "total_leaves": 0,
      "quantized_base_leaves": 0,
      "quantized_base_storage_bytes": 0,
      "quantized_base_dense_equivalent_bf16_bytes": 0,
      "quantized_base_qvalue_bytes": 0,
      "quantized_base_scale_bytes": 0,
      "quantized_base_zero_point_bytes": 0,
      "lora_path_leaves": 0,
      "lora_path_storage_bytes": 0,
      "dense_non_lora_leaves": 0,
      "dense_non_lora_storage_bytes": 0,
      "other_leaves": 0,
      "qtypes": {},
      "sample_quantized_base_paths": [],
      "sample_lora_paths": [],
      "sample_dense_non_lora_paths": [],
  }

  leaves = jax.tree_util.tree_flatten_with_path(
      tree,
      is_leaf=_is_param_summary_leaf,
  )[0]
  for path, leaf in leaves:
    path_str = _jax_path_to_string(path)
    summary["total_leaves"] += 1
    if _is_qwix_quantized_value(leaf):
      _add_quantized_leaf_summary(
          summary,
          path_str,
          leaf.array,
          sample_limit=sample_limit,
      )
      continue
    if _is_qwix_qarray(leaf):
      _add_quantized_leaf_summary(
          summary,
          path_str,
          leaf,
          sample_limit=sample_limit,
      )
      continue

    leaf_bytes = _array_nbytes_metadata(leaf)
    if leaf_bytes is None:
      summary["other_leaves"] += 1
      continue
    if _is_qwix_or_official_lora_path(path_str):
      summary["lora_path_leaves"] += 1
      summary["lora_path_storage_bytes"] += leaf_bytes
      _append_sample(summary["sample_lora_paths"], path_str, sample_limit)
    else:
      summary["dense_non_lora_leaves"] += 1
      summary["dense_non_lora_storage_bytes"] += leaf_bytes
      _append_sample(
          summary["sample_dense_non_lora_paths"], path_str, sample_limit
      )

  for key in (
      "quantized_base_storage_bytes",
      "quantized_base_dense_equivalent_bf16_bytes",
      "lora_path_storage_bytes",
      "dense_non_lora_storage_bytes",
  ):
    summary[f"{key}_gib"] = summary[key] / float(1024**3)
  q_storage = summary["quantized_base_storage_bytes"]
  dense_equiv = summary["quantized_base_dense_equivalent_bf16_bytes"]
  summary["quantized_base_vs_bf16_ratio"] = (
      None if dense_equiv == 0 else q_storage / dense_equiv
  )
  return summary


def _is_param_summary_leaf(value: Any) -> bool:
  return _is_qwix_quantized_value(value) or _is_qwix_qarray(value)


def _is_qwix_qarray(value: Any) -> bool:
  return hasattr(value, "qvalue") and hasattr(value, "scale")


def _add_quantized_leaf_summary(
    summary: dict[str, Any],
    path: str,
    qarray: Any,
    *,
    sample_limit: int,
) -> None:
  qvalue_bytes = _array_nbytes_metadata(getattr(qarray, "qvalue", None)) or 0
  scale_bytes = _array_nbytes_metadata(getattr(qarray, "scale", None)) or 0
  zero_point = getattr(qarray, "zero_point", None)
  zero_point_bytes = (
      0 if zero_point is None else (_array_nbytes_metadata(zero_point) or 0)
  )
  original_elements = _shape_num_elements(getattr(qarray, "shape", ()))
  dense_equivalent_bf16_bytes = original_elements * 2
  qtype = str(getattr(qarray, "qtype", None))

  summary["quantized_base_leaves"] += 1
  summary["quantized_base_qvalue_bytes"] += qvalue_bytes
  summary["quantized_base_scale_bytes"] += scale_bytes
  summary["quantized_base_zero_point_bytes"] += zero_point_bytes
  summary["quantized_base_storage_bytes"] += (
      qvalue_bytes + scale_bytes + zero_point_bytes
  )
  summary[
      "quantized_base_dense_equivalent_bf16_bytes"
  ] += dense_equivalent_bf16_bytes
  summary["qtypes"][qtype] = summary["qtypes"].get(qtype, 0) + 1
  _append_sample(summary["sample_quantized_base_paths"], path, sample_limit)


def _append_sample(samples: list[str], value: str, limit: int) -> None:
  if len(samples) < limit:
    samples.append(value)


def _array_nbytes_metadata(value: Any) -> int | None:
  shape = getattr(value, "shape", None)
  dtype = getattr(value, "dtype", None)
  if shape is None or dtype is None:
    return None
  return _shape_num_elements(shape) * _dtype_itemsize(dtype)


def _shape_num_elements(shape: Any) -> int:
  size = 1
  try:
    parts = tuple(shape)
  except TypeError:
    parts = (shape,)
  for dim in parts:
    size *= int(dim)
  return int(size)


def _dtype_itemsize(dtype: Any) -> int:
  itemsize = getattr(dtype, "itemsize", None)
  if itemsize is not None:
    return int(itemsize)
  try:
    import numpy as np  # pylint: disable=g-import-not-at-top

    return int(np.dtype(dtype).itemsize)
  except Exception:
    return 0


@functools.lru_cache(maxsize=None)
def _device_loss_emitter(metric_name: str):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  def emit(total, count, loop_step, state_step):
    total = jnp.asarray(total)
    count = jnp.asarray(count)
    value = jnp.where(count == 0, jnp.zeros_like(total), total / count)
    jax.debug.callback(
        functools.partial(_log_device_loss, metric_name),
        value,
        total,
        count,
        loop_step,
        state_step,
        ordered=False,
        partitioned=True,
    )
    return jnp.zeros_like(total)

  return jax.jit(emit)


def _log_device_loss(
    metric_name: str,
    value: Any,
    total: Any,
    count: Any,
    loop_step: Any,
    state_step: Any,
) -> None:
  try:
    import numpy as np  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Device loss logging requires numpy."
    ) from exc

  step = int(np.asarray(state_step).reshape(-1)[0])
  key = (metric_name, step)
  if key in _LOGGED_DEVICE_LOSS_KEYS:
    return
  _LOGGED_DEVICE_LOSS_KEYS.add(key)
  print(
      _json_dumps({
          "event": "diffusion_gemma_hybrid_loss",
          "metric": metric_name,
          "step": step,
          "loop_step": int(np.asarray(loop_step).reshape(-1)[0]),
          "value": float(np.asarray(value).reshape(-1)[0]),
          "total": float(np.asarray(total).reshape(-1)[0]),
          "count": float(np.asarray(count).reshape(-1)[0]),
      }),
      flush=True,
  )


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
      if getattr(leaf, "addressable_shards", None):
        shard_arrays = [shard.data for shard in leaf.addressable_shards]
      elif hasattr(leaf, "addressable_data"):
        shard_arrays = [
            leaf.addressable_data(i)
            for i in range(len(leaf.addressable_shards))
        ]
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
  return hasattr(value, "count") and (
      hasattr(value, "total") or hasattr(value, "value")
  )


def _block_first_array(tree: Any) -> None:
  import jax  # pylint: disable=g-import-not-at-top

  for leaf in jax.tree_util.tree_leaves(tree):
    if isinstance(leaf, jax.Array):
      leaf.block_until_ready()
      return


def _safe_array_scalar(value: Any) -> float:
  try:
    import jax  # pylint: disable=g-import-not-at-top
    import numpy as np  # pylint: disable=g-import-not-at-top
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise OfficialBackendDependencyError(
        "Scalar extraction requires jax and numpy."
    ) from exc

  if isinstance(value, jax.Array):
    if getattr(value, "addressable_shards", None):
      value = value.addressable_shards[0].data
    elif hasattr(value, "addressable_data"):
      value = value.addressable_data(0)
  host_value = np.asarray(jax.device_get(value))
  if host_value.size == 0:
    return 0.0
  return float(host_value.reshape(-1)[0])


def _safe_train_step(value: Any, fallback: int) -> int:
  try:
    return int(_safe_array_scalar(value))
  except Exception:  # pylint: disable=broad-exception-caught
    return fallback


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
