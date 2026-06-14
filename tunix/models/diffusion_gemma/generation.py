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

"""DiffusionGemma generation helpers for Tunix-native NNX models."""

from __future__ import annotations

import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.diffusion_gemma import sft


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiffusionGemmaGenerationConfig:
  """Configuration for a lightweight DiffusionGemma denoising sampler."""

  prompt_len: int
  canvas_size: int
  num_canvases: int
  vocab_size: int
  pad_token: int = sft.PAD_TOKEN
  start_token: int | None = None
  num_steps: int = 8
  temperature: float = 0.0
  seed: int = 0
  use_self_conditioning: bool = True
  entropy_budget: float | None = None
  renoise_rejected_tokens: bool = False
  canvas_order: str = "sequential"
  decoder_implementation: str = "cached_selected_canvas_slice"

  @property
  def total_canvas_len(self) -> int:
    return self.canvas_size * self.num_canvases

  def as_sft_config(self) -> sft.DiffusionGemmaSFTConfig:
    return sft.DiffusionGemmaSFTConfig(
        prompt_len=self.prompt_len,
        canvas_size=self.canvas_size,
        num_canvases=self.num_canvases,
        vocab_size=self.vocab_size,
        pad_token=self.pad_token,
        self_cond_prob=1.0 if self.use_self_conditioning else 0.0,
        decoder_implementation=self.decoder_implementation,
        encoder_loss_weight=0.0,
    )


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaGenerationTrace:
  """Token-level denoising trace returned by ``generate_tokens``."""

  prompt: jax.Array
  final_tokens: jax.Array
  frames: tuple[jax.Array, ...]
  selected_canvas_idx: jax.Array
  changed_fraction: jax.Array
  accepted_fraction: jax.Array


def _validate_config(config: DiffusionGemmaGenerationConfig) -> None:
  if config.prompt_len <= 0:
    raise ValueError("prompt_len must be positive.")
  if config.canvas_size <= 0:
    raise ValueError("canvas_size must be positive.")
  if config.num_canvases <= 0:
    raise ValueError("num_canvases must be positive.")
  if config.vocab_size <= 1:
    raise ValueError("vocab_size must be greater than 1.")
  if config.num_steps <= 0:
    raise ValueError("num_steps must be positive.")
  if config.temperature < 0.0:
    raise ValueError("temperature must be non-negative.")
  if config.entropy_budget is not None and config.entropy_budget <= 0.0:
    raise ValueError("entropy_budget must be positive when set.")
  if config.canvas_order not in ("sequential", "reverse"):
    raise ValueError("canvas_order must be 'sequential' or 'reverse'.")
  if config.decoder_implementation != "cached_selected_canvas_slice":
    raise ValueError(
        "Generation currently requires decoder_implementation="
        "'cached_selected_canvas_slice'."
    )
  if config.start_token is not None and not (
      0 <= config.start_token < config.vocab_size
  ):
    raise ValueError("start_token must be in [0, vocab_size).")


def _normalize_prompt(
    prompt: jax.Array, config: DiffusionGemmaGenerationConfig
) -> jax.Array:
  prompt = jnp.asarray(prompt, dtype=jnp.int32)
  if prompt.ndim != 2:
    raise ValueError("prompt must have shape [batch, prompt_len].")
  if prompt.shape[1] != config.prompt_len:
    raise ValueError(
        f"prompt has length {prompt.shape[1]}, expected {config.prompt_len}."
    )
  return prompt


def _normalize_canvas_mask(
    canvas_mask: jax.Array | None,
    *,
    batch_size: int,
    config: DiffusionGemmaGenerationConfig,
) -> jax.Array:
  if canvas_mask is None:
    return jnp.ones((batch_size, config.total_canvas_len), dtype=jnp.bool_)
  canvas_mask = jnp.asarray(canvas_mask, dtype=jnp.bool_)
  expected_shape = (batch_size, config.total_canvas_len)
  if canvas_mask.shape != expected_shape:
    raise ValueError(
        f"canvas_mask has shape {canvas_mask.shape}, expected {expected_shape}."
    )
  return canvas_mask


def _initial_canvas(
    rng: jax.Array,
    *,
    batch_size: int,
    canvas_mask: jax.Array,
    config: DiffusionGemmaGenerationConfig,
    initial_canvas: jax.Array | None,
) -> jax.Array:
  expected_shape = (batch_size, config.total_canvas_len)
  if initial_canvas is not None:
    initial_canvas = jnp.asarray(initial_canvas, dtype=jnp.int32)
    if initial_canvas.ndim == 3 and initial_canvas.shape[-1] == 1:
      initial_canvas = initial_canvas[..., 0]
    if initial_canvas.shape != expected_shape:
      raise ValueError(
          f"initial_canvas has shape {initial_canvas.shape}, "
          f"expected {expected_shape}."
      )
    return jnp.where(canvas_mask, initial_canvas, config.pad_token)

  if config.start_token is not None:
    canvas = jnp.full(expected_shape, config.start_token, dtype=jnp.int32)
  else:
    canvas = jax.random.randint(
        rng,
        expected_shape,
        minval=1,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )
  return jnp.where(canvas_mask, canvas, config.pad_token)


def _selected_canvas_idx(
    step: int,
    *,
    batch_size: int,
    config: DiffusionGemmaGenerationConfig,
) -> jax.Array:
  if config.canvas_order == "reverse":
    index = config.num_canvases - 1 - (step % config.num_canvases)
  else:
    index = step % config.num_canvases
  return jnp.full((batch_size,), index, dtype=jnp.int32)


def _sample_tokens(
    rng: jax.Array,
    logits: jax.Array,
    *,
    temperature: float,
    pad_token: int,
) -> jax.Array:
  logits = logits.astype(jnp.float32)
  logits = logits.at[..., pad_token].set(jnp.finfo(jnp.float32).min)
  if temperature == 0.0:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)
  return jax.random.categorical(rng, logits / temperature, axis=-1).astype(
      jnp.int32
  )


def _entropy_bound_acceptance_mask(
    logits: jax.Array,
    selected_mask: jax.Array,
    *,
    entropy_budget: float | None,
) -> jax.Array:
  if entropy_budget is None:
    return selected_mask

  logits = logits.astype(jnp.float32)
  probs = jax.nn.softmax(logits, axis=-1)
  entropy = -jnp.sum(probs * jax.nn.log_softmax(logits, axis=-1), axis=-1)
  valid_entropy = jnp.where(selected_mask, entropy, jnp.inf)
  order = jnp.argsort(valid_entropy, axis=-1)
  sorted_entropy = jnp.take_along_axis(valid_entropy, order, axis=-1)
  sorted_valid = jnp.take_along_axis(selected_mask, order, axis=-1)
  sorted_accept = sorted_valid & (
      jnp.cumsum(sorted_entropy, axis=-1) <= entropy_budget
  )

  first_valid = jnp.argmax(sorted_valid.astype(jnp.int32), axis=-1)
  has_valid = jnp.any(sorted_valid, axis=-1)
  sorted_accept = sorted_accept | (
      has_valid[:, None]
      & (jnp.arange(sorted_accept.shape[1])[None, :] == first_valid[:, None])
  )

  batch_indices = jnp.arange(logits.shape[0])[:, None]
  accepted = jnp.zeros_like(sorted_accept)
  return accepted.at[batch_indices, order].set(sorted_accept)


def _gather_selected_tokens(
    canvas: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaGenerationConfig,
) -> jax.Array:
  selected_positions = (
      selected_canvas_idx[:, None] * config.canvas_size
      + jnp.arange(config.canvas_size, dtype=jnp.int32)[None, :]
  )
  return jnp.take_along_axis(canvas, selected_positions, axis=1)


def _apply_acceptance_mask(
    rng: jax.Array,
    *,
    selected_tokens: jax.Array,
    old_selected: jax.Array,
    accepted_mask: jax.Array,
    config: DiffusionGemmaGenerationConfig,
) -> jax.Array:
  if config.renoise_rejected_tokens:
    fallback = jax.random.randint(
        rng,
        selected_tokens.shape,
        minval=1,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )
  else:
    fallback = old_selected
  return jnp.where(accepted_mask, selected_tokens, fallback)


def _scatter_selected_canvas(
    canvas: jax.Array,
    selected_tokens: jax.Array,
    selected_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaGenerationConfig,
) -> jax.Array:
  selected_positions = (
      selected_canvas_idx[:, None] * config.canvas_size
      + jnp.arange(config.canvas_size, dtype=jnp.int32)[None, :]
  )
  old_selected = jnp.take_along_axis(canvas, selected_positions, axis=1)
  selected_tokens = jnp.where(selected_mask, selected_tokens, old_selected)
  return canvas.at[
      jnp.arange(canvas.shape[0])[:, None], selected_positions
  ].set(selected_tokens)


def _decode_selected_canvas_logits(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    canvas: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    sft_config: sft.DiffusionGemmaSFTConfig,
    sc_logits: jax.Array | None = None,
) -> jax.Array:
  _, kv_cache, positions, prompt_mask = sft.sft_encode(
      model,
      prompt=prompt,
      x0_tokens=canvas,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      config=sft_config,
      return_encoder_logits=False,
      return_encoder_hidden=False,
  )
  end_index = (
      sft_config.prompt_len + selected_canvas_idx * sft_config.canvas_size
  )
  kv_cache = sft.set_cache_end_index(kv_cache, end_index)
  return sft.sft_decode(
      model,
      prompt=prompt,
      xt=canvas,
      kv_cache=kv_cache,
      positions=positions,
      prompt_mask=prompt_mask,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      config=sft_config,
      sc_logits=sc_logits,
  )


def generate_tokens(
    model: nnx.Module,
    prompt: jax.Array,
    config: DiffusionGemmaGenerationConfig,
    *,
    canvas_mask: jax.Array | None = None,
    initial_canvas: jax.Array | None = None,
    rng: jax.Array | None = None,
) -> DiffusionGemmaGenerationTrace:
  """Generates canvas tokens and returns a step-by-step denoising trace.

  This is a Tunix-native lightweight sampler for validation, visualization, and
  evaluator plumbing. It reuses the same NNX encoder prefill, KV cache setup,
  selected-canvas decode, and optional self-conditioning path as SFT.
  """
  _validate_config(config)
  prompt = _normalize_prompt(prompt, config)
  batch_size = prompt.shape[0]
  canvas_mask = _normalize_canvas_mask(
      canvas_mask, batch_size=batch_size, config=config
  )
  if rng is None:
    rng = jax.random.PRNGKey(config.seed)
  rng_init, rng = jax.random.split(rng)
  canvas = _initial_canvas(
      rng_init,
      batch_size=batch_size,
      canvas_mask=canvas_mask,
      config=config,
      initial_canvas=initial_canvas,
  )
  sft_config = config.as_sft_config()

  frames: list[jax.Array] = [canvas]
  selected_steps: list[jax.Array] = []
  changed_steps: list[jax.Array] = []
  accepted_steps: list[jax.Array] = []
  for step in range(config.num_steps):
    rng, rng_sample, rng_renoise = jax.random.split(rng, 3)
    selected_canvas_idx = _selected_canvas_idx(
        step, batch_size=batch_size, config=config
    )
    selected_steps.append(selected_canvas_idx)
    logits = _decode_selected_canvas_logits(
        model,
        prompt=prompt,
        canvas=canvas,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        sft_config=sft_config,
    )
    if config.use_self_conditioning:
      logits = _decode_selected_canvas_logits(
          model,
          prompt=prompt,
          canvas=canvas,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          sft_config=sft_config,
          sc_logits=jax.lax.stop_gradient(logits),
      )
    selected_tokens = _sample_tokens(
        rng_sample,
        logits,
        temperature=config.temperature,
        pad_token=config.pad_token,
    )
    selected_mask = sft._gather_selected_canvas(  # pylint: disable=protected-access
        canvas_mask, selected_canvas_idx, config.canvas_size
    )
    accepted_mask = _entropy_bound_acceptance_mask(
        logits,
        selected_mask,
        entropy_budget=config.entropy_budget,
    )
    old_selected = _gather_selected_tokens(canvas, selected_canvas_idx, config)
    selected_tokens = _apply_acceptance_mask(
        rng_renoise,
        selected_tokens=selected_tokens,
        old_selected=old_selected,
        accepted_mask=accepted_mask,
        config=config,
    )
    old_canvas = canvas
    canvas = _scatter_selected_canvas(
        canvas,
        selected_tokens,
        selected_mask,
        selected_canvas_idx,
        config,
    )
    changed = jnp.mean((canvas != old_canvas).astype(jnp.float32), axis=1)
    changed_steps.append(changed)
    accepted = jnp.sum(accepted_mask.astype(jnp.float32), axis=1) / jnp.maximum(
        jnp.sum(selected_mask.astype(jnp.float32), axis=1), 1.0
    )
    accepted_steps.append(accepted)
    frames.append(canvas)

  return DiffusionGemmaGenerationTrace(
      prompt=prompt,
      final_tokens=canvas,
      frames=tuple(frames),
      selected_canvas_idx=jnp.stack(selected_steps, axis=0),
      changed_fraction=jnp.stack(changed_steps, axis=0),
      accepted_fraction=jnp.stack(accepted_steps, axis=0),
  )


def decode_trace(
    tokenizer: Any,
    trace: DiffusionGemmaGenerationTrace,
    *,
    batch_index: int = 0,
    skip_special_tokens: bool = True,
) -> list[str]:
  """Decodes every frame in a generation trace with a tokenizer-like object."""
  frames = jax.device_get(trace.frames)
  decoded = []
  for frame in frames:
    token_ids = [int(token) for token in frame[batch_index].tolist()]
    try:
      decoded.append(
          tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
      )
    except TypeError:
      del skip_special_tokens
      decoded.append(tokenizer.decode(token_ids))
  return decoded
