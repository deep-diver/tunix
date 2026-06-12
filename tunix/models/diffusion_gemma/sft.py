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

"""DiffusionGemma SFT adapter for Tunix PeftTrainer."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import qwix
from tunix.models.gemma4 import moe as gemma4_moe
from tunix.sft import peft_trainer


PAD_TOKEN = 0
DEFAULT_LORA_MODULE_PATH = (
    r".*q_einsum|.*kv_einsum|.*k_einsum|.*attn_vec_einsum|"
    r".*gate_proj|.*up_proj|.*down_proj|.*moe.*|"
    r".*router_logits|.*gating_einsum|.*linear"
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiffusionGemmaSFTConfig:
  """Configuration for DiffusionGemma supervised fine tuning."""

  prompt_len: int
  canvas_size: int
  num_canvases: int
  vocab_size: int
  pad_token: int = PAD_TOKEN
  self_cond_prob: float = 0.5
  min_time: float = 1e-4
  max_time: float = 1.0 - 1e-4
  decoder_loss_weight: float = 1.0
  encoder_loss_weight: float = 1.0
  stop_gradient_from_denoiser_to_encoder: bool = False
  decoder_implementation: str = "cached_selected_canvas"
  fast_uniform_corruption: bool = False
  encoder_loss_chunk_size: int | None = 64
  force_full_encoder_prefill: bool = False

  @property
  def total_canvas_len(self) -> int:
    return self.canvas_size * self.num_canvases


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaSFTBatch:
  """Batch layout expected by the DiffusionGemma SFT loss."""

  prompt: jax.Array
  canvas: jax.Array
  canvas_id: jax.Array
  canvas_mask: jax.Array
  encoder_target: jax.Array
  encoder_target_mask: jax.Array
  rng: jax.Array


def build_positions_from_mask(mask: jax.Array) -> jax.Array:
  positions = jnp.cumsum(jax.lax.optimization_barrier(mask), axis=-1)
  return positions - (positions >= 1)


def make_causal_prefill_mask(
    token_mask: jax.Array,
    cache_length: int,
) -> jax.Array:
  seq_len = token_mask.shape[-1]
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attn_mask = token_mask[:, None, :] & causal[None, :, :]
  pad_width = cache_length - seq_len
  if pad_width > 0:
    attn_mask = jnp.pad(
        attn_mask,
        ((0, 0), (0, 0), (0, pad_width)),
        constant_values=False,
    )
  return attn_mask


def create_decoder_attention_mask(
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    *,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    num_queries: int,
) -> jax.Array:
  batch_size = prompt_mask.shape[0]
  cache_len = prompt_len + total_canvas_len
  kv_positions = jnp.arange(cache_len)

  prompt_region = kv_positions < prompt_len
  prompt_pad_mask = jnp.zeros((batch_size, cache_len), dtype=jnp.bool_)
  prompt_pad_mask = prompt_pad_mask.at[:, :prompt_len].set(prompt_mask)
  prompt_attention = prompt_region[None, None, :] & prompt_pad_mask[:, None, :]

  in_canvas_region = kv_positions >= prompt_len
  kv_canvas_id = (kv_positions - prompt_len) // canvas_size
  canvas_attention = (
      kv_canvas_id[None, None, :] <= selected_canvas_idx[:, None, None]
  ) & in_canvas_region[None, None, :]

  canvas_valid_mask = jnp.zeros((batch_size, cache_len), dtype=jnp.bool_)
  canvas_valid_mask = canvas_valid_mask.at[
      :, prompt_len : prompt_len + total_canvas_len
  ].set(canvas_mask)
  canvas_attention = canvas_attention & canvas_valid_mask[:, None, :]
  attn_mask = prompt_attention | canvas_attention
  return jnp.broadcast_to(attn_mask, (batch_size, num_queries, cache_len))


def set_cache_end_index(kv_cache: Mapping[str, Any], end_index: jax.Array):
  return {
      name: {
          **layer,
          "end_index": jnp.broadcast_to(end_index, layer["end_index"].shape),
      }
      for name, layer in kv_cache.items()
  }


def _init_cache(model: nnx.Module, batch_size: int, cache_length: int):
  dtype = getattr(getattr(model, "config", None), "dtype", jnp.float32)
  try:
    return model.init_cache(batch_size, cache_length, dtype)
  except TypeError:
    return model.init_cache(batch_size=batch_size, cache_length=cache_length)


def sft_encode(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    x0_tokens: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaSFTConfig,
    return_encoder_logits: bool = True,
    return_encoder_hidden: bool = False,
) -> tuple[jax.Array, Any, jax.Array, jax.Array]:
  del selected_canvas_idx
  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)
  prompt_mask = prompt != config.pad_token
  full_seq_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)
  positions = build_positions_from_mask(full_seq_mask)
  attention_mask = make_causal_prefill_mask(full_seq_mask, full_seq.shape[1])
  kv_cache = _init_cache(model, full_seq.shape[0], full_seq.shape[1])
  if return_encoder_hidden:
    if not hasattr(model, "forward_hidden"):
      raise ValueError(
          "return_encoder_hidden=True requires a model with forward_hidden()."
      )
    encoder_output, kv_cache = model.forward_hidden(
        full_seq,
        positions=positions,
        cache=kv_cache,
        attention_mask=attention_mask,
        decode_only_last_token=not return_encoder_logits,
    )
  else:
    encoder_output, kv_cache = model(
        full_seq,
        positions=positions,
        cache=kv_cache,
        attention_mask=attention_mask,
        decode_only_last_token=not return_encoder_logits,
    )
  if kv_cache is None:
    raise ValueError("KV cache should not be None after SFT prefill.")
  return encoder_output, kv_cache, positions, prompt_mask


def _full_sequence_decoder_attention_mask(
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaSFTConfig,
) -> jax.Array:
  prompt_len = config.prompt_len
  full_len = prompt_len + config.total_canvas_len
  batch_size = prompt_mask.shape[0]

  prompt_causal = make_causal_prefill_mask(prompt_mask, prompt_len)
  prompt_rows = jnp.zeros((batch_size, prompt_len, full_len), dtype=jnp.bool_)
  prompt_rows = prompt_rows.at[:, :, :prompt_len].set(prompt_causal)
  canvas_rows = create_decoder_attention_mask(
      prompt_mask,
      canvas_mask,
      selected_canvas_idx,
      prompt_len=prompt_len,
      total_canvas_len=config.total_canvas_len,
      canvas_size=config.canvas_size,
      num_queries=config.total_canvas_len,
  )
  return jnp.concatenate([prompt_rows, canvas_rows], axis=1)


def sft_decode_full_sequence(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    xt: jax.Array,
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaSFTConfig,
    sc_logits: jax.Array | None = None,
) -> jax.Array:
  full_tokens = jnp.concatenate([prompt, xt], axis=1)
  attention_mask = _full_sequence_decoder_attention_mask(
      prompt_mask, canvas_mask, selected_canvas_idx, config
  )
  positions = build_positions_from_mask(
      jnp.concatenate([prompt_mask, canvas_mask], axis=1)
  )
  if sc_logits is None:
    sc_logits = jnp.zeros(
        (xt.shape[0], xt.shape[1], config.vocab_size), dtype=jnp.float32
    )
  prompt_sc_logits = jnp.zeros(
      (prompt.shape[0], prompt.shape[1], config.vocab_size),
      dtype=sc_logits.dtype,
  )
  full_sc_logits = jnp.concatenate([prompt_sc_logits, sc_logits], axis=1)
  sc_mask = jnp.concatenate(
      [jnp.zeros_like(prompt, dtype=jnp.bool_), canvas_mask], axis=1
  )
  logits, _ = model(
      full_tokens,
      positions=positions,
      cache=None,
      attention_mask=attention_mask,
      sc_logits=full_sc_logits,
      self_conditioning_mask=sc_mask,
  )
  return logits[:, config.prompt_len :, :]


def sft_decode_cached_selected_canvas(
    model: nnx.Module,
    *,
    xt: jax.Array,
    kv_cache: Any,
    positions: jax.Array,
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaSFTConfig,
    sc_logits: jax.Array | None = None,
) -> jax.Array:
  """Runs the official-style decoder over canvas tokens using prefilled cache."""
  attn_mask = create_decoder_attention_mask(
      prompt_mask,
      canvas_mask,
      selected_canvas_idx,
      prompt_len=config.prompt_len,
      total_canvas_len=config.total_canvas_len,
      canvas_size=config.canvas_size,
      num_queries=config.total_canvas_len,
  )
  canvas_positions = positions[:, config.prompt_len :]
  if sc_logits is None:
    sc_logits = jnp.zeros(
        (xt.shape[0], xt.shape[1], config.vocab_size), dtype=jnp.float32
    )
  logits, _ = model(
      xt,
      positions=canvas_positions,
      cache=kv_cache,
      attention_mask=attn_mask,
      sc_logits=sc_logits,
      self_conditioning_mask=canvas_mask,
  )
  return logits


def _selected_canvas_indices(
    selected_canvas_idx: jax.Array, canvas_size: int
) -> jax.Array:
  return (
      selected_canvas_idx[:, None] * canvas_size
      + jnp.arange(canvas_size, dtype=selected_canvas_idx.dtype)[None, :]
  )


def _gather_selected_canvas(
    x: jax.Array, selected_canvas_idx: jax.Array, canvas_size: int
) -> jax.Array:
  indices = _selected_canvas_indices(selected_canvas_idx, canvas_size)
  while indices.ndim < x.ndim:
    indices = indices[..., None]
  indices = jnp.broadcast_to(
      indices, x.shape[:1] + (canvas_size,) + x.shape[2:]
  )
  return jnp.take_along_axis(x, indices, axis=1)


def sft_decode_cached_selected_canvas_slice(
    model: nnx.Module,
    *,
    xt: jax.Array,
    kv_cache: Any,
    positions: jax.Array,
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaSFTConfig,
    sc_logits: jax.Array | None = None,
) -> jax.Array:
  """Runs cached decoding only for the selected canvas loss window."""
  selected_xt = _gather_selected_canvas(
      xt, selected_canvas_idx, config.canvas_size
  )
  selected_positions = _gather_selected_canvas(
      positions[:, config.prompt_len :],
      selected_canvas_idx,
      config.canvas_size,
  )
  selected_canvas_mask = _gather_selected_canvas(
      canvas_mask, selected_canvas_idx, config.canvas_size
  )
  attn_mask = create_decoder_attention_mask(
      prompt_mask,
      canvas_mask,
      selected_canvas_idx,
      prompt_len=config.prompt_len,
      total_canvas_len=config.total_canvas_len,
      canvas_size=config.canvas_size,
      num_queries=config.canvas_size,
  )
  if sc_logits is None:
    sc_logits = jnp.zeros(
        (xt.shape[0], config.canvas_size, config.vocab_size),
        dtype=jnp.float32,
    )
  elif sc_logits.shape[1] == config.total_canvas_len:
    sc_logits = _gather_selected_canvas(
        sc_logits, selected_canvas_idx, config.canvas_size
    )
  logits, _ = model(
      selected_xt,
      positions=selected_positions,
      cache=kv_cache,
      attention_mask=attn_mask,
      sc_logits=sc_logits,
      self_conditioning_mask=selected_canvas_mask,
  )
  return logits


def sft_decode(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    xt: jax.Array,
    kv_cache: Any,
    positions: jax.Array,
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    config: DiffusionGemmaSFTConfig,
    sc_logits: jax.Array | None = None,
) -> jax.Array:
  """Dispatches to the requested DiffusionGemma SFT decoder implementation."""
  if config.decoder_implementation == "cached_selected_canvas":
    return sft_decode_cached_selected_canvas(
        model,
        xt=xt,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=config,
        sc_logits=sc_logits,
    )
  if config.decoder_implementation == "cached_selected_canvas_slice":
    return sft_decode_cached_selected_canvas_slice(
        model,
        xt=xt,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=config,
        sc_logits=sc_logits,
    )
  if config.decoder_implementation == "full_sequence":
    return sft_decode_full_sequence(
        model,
        prompt=prompt,
        xt=xt,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=config,
        sc_logits=sc_logits,
    )
  raise ValueError(
      "Unknown decoder_implementation: "
      f"{config.decoder_implementation!r}. Expected 'cached_selected_canvas' "
      "'cached_selected_canvas_slice', or 'full_sequence'."
  )


def _sample_selected_canvas(
    rng: jax.Array,
    canvas_mask: jax.Array,
    config: DiffusionGemmaSFTConfig,
) -> jax.Array:
  first_token_indices = jnp.arange(config.num_canvases) * config.canvas_size
  canvas_validity = canvas_mask[:, first_token_indices]
  num_valid = jnp.maximum(
      jnp.sum(canvas_validity.astype(jnp.int32), axis=-1), 1
  )
  return jax.random.randint(
      rng,
      shape=num_valid.shape,
      minval=0,
      maxval=num_valid,
      dtype=jnp.int32,
  )


def _corrupt_tokens(
    rng: jax.Array,
    x0_tokens: jax.Array,
    time: jax.Array,
    vocab_size: int,
    *,
    fast_uniform: bool = False,
) -> tuple[jax.Array, jax.Array]:
  rng_mask, rng_noise = jax.random.split(rng)
  if fast_uniform:
    random_tokens = jax.random.randint(
        rng_noise,
        shape=x0_tokens.shape,
        minval=0,
        maxval=vocab_size,
        dtype=x0_tokens.dtype,
    )
  else:
    random_tokens = jax.random.choice(
        rng_noise,
        a=vocab_size,
        shape=x0_tokens.shape,
        p=jnp.full((vocab_size,), 1.0 / vocab_size, dtype=jnp.float32),
        mode="high",
    ).astype(x0_tokens.dtype)
  corrupt_prob = jnp.broadcast_to(time, x0_tokens.shape)
  is_not_corrupted = jax.random.bernoulli(
      rng_mask, p=1.0 - corrupt_prob, shape=x0_tokens.shape, mode="high"
  )
  is_corrupted = jnp.logical_not(is_not_corrupted)
  xt = jnp.where(is_corrupted, random_tokens, x0_tokens)
  return xt, is_corrupted


def _masked_ce_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
) -> jax.Array:
  logits = logits.astype(jnp.float32)
  target_logits = jnp.take_along_axis(
      logits, targets[..., None], axis=-1
  ).squeeze(axis=-1)
  loss = jax.nn.logsumexp(logits, axis=-1) - target_logits
  mask = mask.astype(loss.dtype)
  reduce_axes = tuple(range(1, loss.ndim))
  per_example_loss = jnp.sum(loss * mask, axis=reduce_axes)
  per_example_denom = jnp.maximum(jnp.sum(mask, axis=reduce_axes), 1.0)
  return jnp.mean(per_example_loss / per_example_denom)


def _masked_ce_loss_from_hidden(
    model: nnx.Module,
    hidden: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    *,
    chunk_size: int | None,
) -> jax.Array:
  """Computes CE from hidden states without materializing full-sequence logits."""
  if chunk_size is None or chunk_size <= 0 or chunk_size >= hidden.shape[1]:
    return _masked_ce_loss(model.decode_hidden(hidden), targets, mask)

  seq_len = hidden.shape[1]
  pad_len = (-seq_len) % chunk_size
  if pad_len:
    hidden = jnp.pad(hidden, ((0, 0), (0, pad_len), (0, 0)))
    targets = jnp.pad(targets, ((0, 0), (0, pad_len)))
    mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=False)

  num_chunks = hidden.shape[1] // chunk_size
  hidden_chunks = hidden.reshape(
      hidden.shape[0], num_chunks, chunk_size, hidden.shape[-1]
  )
  target_chunks = targets.reshape(targets.shape[0], num_chunks, chunk_size)
  mask_chunks = mask.reshape(mask.shape[0], num_chunks, chunk_size)

  def scan_body(carry, chunk_inputs):
    total_loss, total_weight = carry
    hidden_chunk, target_chunk, mask_chunk = chunk_inputs
    logits = model.decode_hidden(hidden_chunk)
    logits = logits.astype(jnp.float32)
    target_logits = jnp.take_along_axis(
        logits, target_chunk[..., None], axis=-1
    ).squeeze(axis=-1)
    loss = jax.nn.logsumexp(logits, axis=-1) - target_logits
    weight = mask_chunk.astype(loss.dtype)
    return (
        total_loss + jnp.sum(loss * weight, axis=1),
        total_weight + jnp.sum(weight, axis=1),
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
  per_example_loss = loss_sum / jnp.maximum(weight_sum, 1.0)
  return jnp.mean(per_example_loss)


def diffusion_gemma_sft_self_conditioning_logits(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    canvas: jax.Array,
    canvas_mask: jax.Array,
    rng: jax.Array,
    config: DiffusionGemmaSFTConfig,
) -> tuple[jax.Array, jax.Array]:
  """Computes the stopped first-pass logits used for self-conditioning."""
  prefill = diffusion_gemma_sft_self_conditioning_prefill(
      model,
      prompt=prompt,
      canvas=canvas,
      canvas_mask=canvas_mask,
      rng=rng,
      config=config,
  )
  return diffusion_gemma_sft_self_conditioning_logits_from_prefill(
      model,
      **prefill,
      config=config,
  )


def diffusion_gemma_sft_self_conditioning_prefill(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    canvas: jax.Array,
    canvas_mask: jax.Array,
    rng: jax.Array,
    config: DiffusionGemmaSFTConfig,
) -> dict[str, Any]:
  """Builds the no-grad clean prefill state for self-conditioning."""
  if canvas.ndim == 3:
    x0_tokens = canvas[..., 0]
  else:
    x0_tokens = canvas
  canvas_mask = canvas_mask.astype(jnp.bool_)

  rng_time, rng_corrupt, rng_canvas, rng_self_cond = jax.random.split(rng, 4)
  batch_size = x0_tokens.shape[0]
  time = jax.random.uniform(
      rng_time,
      (batch_size, 1),
      minval=config.min_time,
      maxval=config.max_time,
      dtype=jnp.float32,
  )
  xt, _ = _corrupt_tokens(
      rng_corrupt,
      x0_tokens,
      time,
      config.vocab_size,
      fast_uniform=config.fast_uniform_corruption,
  )
  selected_canvas_idx = _sample_selected_canvas(rng_canvas, canvas_mask, config)

  _, kv_cache, positions, prompt_mask = sft_encode(
      model,
      prompt=prompt,
      x0_tokens=x0_tokens,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      config=config,
      return_encoder_logits=False,
      return_encoder_hidden=False,
  )
  end_index = config.prompt_len + selected_canvas_idx * config.canvas_size
  kv_cache = set_cache_end_index(kv_cache, end_index)
  if config.stop_gradient_from_denoiser_to_encoder:
    kv_cache = jax.lax.stop_gradient(kv_cache)

  do_self_cond = (
      jax.random.uniform(rng_self_cond, (batch_size,)) < config.self_cond_prob
  )
  do_self_cond = do_self_cond.reshape((batch_size, 1, 1))
  return {
      "prompt": prompt,
      "xt": xt,
      "kv_cache": jax.lax.stop_gradient(kv_cache),
      "positions": positions,
      "prompt_mask": prompt_mask,
      "canvas_mask": canvas_mask,
      "selected_canvas_idx": selected_canvas_idx,
      "do_self_cond": do_self_cond,
  }


def diffusion_gemma_sft_self_conditioning_logits_from_prefill(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    xt: jax.Array,
    kv_cache: Any,
    positions: jax.Array,
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    do_self_cond: jax.Array,
    config: DiffusionGemmaSFTConfig,
) -> tuple[jax.Array, jax.Array]:
  """Computes stopped first-pass logits from a precomputed clean KV cache."""
  first_pass_logits = sft_decode(
      model,
      prompt=prompt,
      xt=xt,
      kv_cache=kv_cache,
      positions=positions,
      prompt_mask=prompt_mask,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      config=config,
  )
  first_pass_logits = jax.lax.stop_gradient(first_pass_logits)
  sc_logits = jnp.where(
      do_self_cond, first_pass_logits, jnp.zeros_like(first_pass_logits)
  )
  return jax.lax.stop_gradient(sc_logits), do_self_cond


def diffusion_gemma_sft_loss(
    model: nnx.Module,
    *,
    prompt: jax.Array,
    canvas: jax.Array,
    canvas_id: jax.Array,
    canvas_mask: jax.Array,
    encoder_target: jax.Array,
    encoder_target_mask: jax.Array,
    rng: jax.Array,
    config: DiffusionGemmaSFTConfig,
    precomputed_sc_logits: jax.Array | None = None,
    precomputed_self_conditioning_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes the DiffusionGemma SFT objective used by Tunix."""
  if canvas.ndim == 3:
    x0_tokens = canvas[..., 0]
  else:
    x0_tokens = canvas
  canvas_mask = canvas_mask.astype(jnp.bool_)

  rng_time, rng_corrupt, rng_canvas, rng_self_cond = jax.random.split(rng, 4)
  batch_size = x0_tokens.shape[0]
  time = jax.random.uniform(
      rng_time,
      (batch_size, 1),
      minval=config.min_time,
      maxval=config.max_time,
      dtype=jnp.float32,
  )
  xt, is_corrupted = _corrupt_tokens(
      rng_corrupt,
      x0_tokens,
      time,
      config.vocab_size,
      fast_uniform=config.fast_uniform_corruption,
  )
  selected_canvas_idx = _sample_selected_canvas(rng_canvas, canvas_mask, config)

  use_chunked_encoder_loss = (
      config.encoder_loss_weight != 0.0
      and config.encoder_loss_chunk_size is not None
      and config.encoder_loss_chunk_size > 0
  )
  return_full_encoder_prefill = (
      config.encoder_loss_weight != 0.0 or config.force_full_encoder_prefill
  )
  encoder_output, kv_cache, positions, prompt_mask = sft_encode(
      model,
      prompt=prompt,
      x0_tokens=x0_tokens,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      config=config,
      return_encoder_logits=return_full_encoder_prefill,
      return_encoder_hidden=(
          use_chunked_encoder_loss or config.force_full_encoder_prefill
      ),
  )
  end_index = config.prompt_len + selected_canvas_idx * config.canvas_size
  kv_cache = set_cache_end_index(kv_cache, end_index)
  if config.stop_gradient_from_denoiser_to_encoder:
    kv_cache = jax.lax.stop_gradient(kv_cache)

  target_mask = canvas_mask & (canvas_id == selected_canvas_idx[:, None])
  denoise_loss_mask = is_corrupted & target_mask

  if config.decoder_loss_weight == 0.0:
    decoder_loss = jnp.asarray(0.0, dtype=jnp.float32)
    do_self_cond = jnp.zeros((batch_size, 1, 1), dtype=jnp.bool_)
  else:
    if precomputed_sc_logits is None:
      first_pass_logits = sft_decode(
          model,
          prompt=prompt,
          xt=xt,
          kv_cache=kv_cache,
          positions=positions,
          prompt_mask=prompt_mask,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          config=config,
      )
      first_pass_logits = jax.lax.stop_gradient(first_pass_logits)
      do_self_cond = (
          jax.random.uniform(rng_self_cond, (batch_size,))
          < config.self_cond_prob
      )
      do_self_cond = do_self_cond.reshape((batch_size, 1, 1))
      sc_logits = jnp.where(
          do_self_cond, first_pass_logits, jnp.zeros_like(first_pass_logits)
      )
    else:
      precomputed_sc_logits = jax.lax.stop_gradient(precomputed_sc_logits)
      if precomputed_self_conditioning_mask is None:
        do_self_cond = (
            jax.random.uniform(rng_self_cond, (batch_size,))
            < config.self_cond_prob
        )
        do_self_cond = do_self_cond.reshape((batch_size, 1, 1))
        sc_logits = jnp.where(
            do_self_cond,
            precomputed_sc_logits,
            jnp.zeros_like(precomputed_sc_logits),
        )
      else:
        do_self_cond = precomputed_self_conditioning_mask
        sc_logits = precomputed_sc_logits
    logits = sft_decode(
        model,
        prompt=prompt,
        xt=xt,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=config,
        sc_logits=sc_logits,
    )
    if config.decoder_implementation == "cached_selected_canvas_slice":
      decoder_loss = _masked_ce_loss(
          logits,
          _gather_selected_canvas(
              x0_tokens, selected_canvas_idx, config.canvas_size
          ),
          _gather_selected_canvas(
              denoise_loss_mask, selected_canvas_idx, config.canvas_size
          ),
      )
    else:
      decoder_loss = _masked_ce_loss(logits, x0_tokens, denoise_loss_mask)
  if config.encoder_loss_weight == 0.0:
    encoder_loss = jnp.asarray(0.0, dtype=jnp.float32)
  elif use_chunked_encoder_loss:
    encoder_loss = _masked_ce_loss_from_hidden(
        model,
        encoder_output,
        encoder_target,
        encoder_target_mask,
        chunk_size=config.encoder_loss_chunk_size,
    )
  else:
    encoder_loss = _masked_ce_loss(
        encoder_output, encoder_target, encoder_target_mask
    )
  loss = (
      config.decoder_loss_weight * decoder_loss
      + config.encoder_loss_weight * encoder_loss
  )
  aux = {
      "decoder_loss": decoder_loss,
      "encoder_loss": encoder_loss,
      "corrupted_fraction": jnp.mean(is_corrupted.astype(jnp.float32)),
      "selected_canvas_idx_mean": jnp.mean(
          selected_canvas_idx.astype(jnp.float32)
      ),
      "time_mean": jnp.mean(time),
      "self_conditioning_fraction": jnp.mean(do_self_cond.astype(jnp.float32)),
      "decoder_implementation": jnp.asarray(
          {
              "cached_selected_canvas": 0,
              "cached_selected_canvas_slice": 1,
              "full_sequence": 2,
          }[config.decoder_implementation],
          dtype=jnp.int32,
      ),
  }
  return loss, aux


def make_loss_fn(
    config: DiffusionGemmaSFTConfig,
) -> Callable[..., tuple[jax.Array, dict[str, jax.Array]]]:

  def loss_fn(
      model: nnx.Module,
      prompt: jax.Array,
      canvas: jax.Array,
      canvas_id: jax.Array,
      canvas_mask: jax.Array,
      encoder_target: jax.Array,
      encoder_target_mask: jax.Array,
      rng: jax.Array,
  ):
    return diffusion_gemma_sft_loss(
        model,
        prompt=prompt,
        canvas=canvas,
        canvas_id=canvas_id,
        canvas_mask=canvas_mask,
        encoder_target=encoder_target,
        encoder_target_mask=encoder_target_mask,
        rng=rng,
        config=config,
    )

  return loss_fn


def gen_model_input_fn(batch: Any) -> dict[str, jax.Array]:
  if dataclasses.is_dataclass(batch):
    return dataclasses.asdict(batch)
  if isinstance(batch, Mapping):
    return dict(batch)
  return vars(batch)


def _init_lora_param(
    rngs: nnx.Rngs,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    *,
    zeros: bool = False,
) -> nnx.LoRAParam:
  if zeros:
    value = jnp.zeros(shape, dtype=dtype)
  else:
    value = nnx.initializers.normal(stddev=0.01, dtype=dtype)(
        rngs.params(), shape
    )
  return nnx.LoRAParam(value)


def _ensure_moe_lora_param(
    module: gemma4_moe.MoERagged,
    name: str,
    *,
    rank: int,
    rngs: nnx.Rngs,
) -> None:
  if hasattr(module, f"{name}_lora_a") and hasattr(module, f"{name}_lora_b"):
    return
  weight = getattr(module, name).value
  dtype = getattr(weight, "dtype", jnp.float32)
  setattr(
      module,
      f"{name}_lora_a",
      _init_lora_param(rngs, (*weight.shape[:-1], rank), dtype),
  )
  setattr(
      module,
      f"{name}_lora_b",
      _init_lora_param(rngs, (rank, weight.shape[-1]), dtype, zeros=True),
  )


def apply_moe_lora(
    model: nnx.Module,
    *,
    rank: int,
    alpha: float,
    rng_seed: int,
) -> nnx.Module:
  """Adds LoRA leaves for Gemma4 MoE params used by DiffusionGemma 26B.

  Qwix covers the ordinary Linear/Einsum modules. Gemma4 MoE expert weights are
  bare ``nnx.Param`` leaves consumed by ragged_dot, so they need explicit LoRA
  leaves plus the optional forward hook in ``gemma4.moe``.
  """
  rngs = nnx.Rngs(rng_seed)
  for _path, module in nnx.iter_modules(model):
    if not isinstance(module, gemma4_moe.MoERagged):
      continue
    module.moe_lora_scale = alpha / rank
    for name in ("router_logits", "gating_einsum", "linear"):
      _ensure_moe_lora_param(module, name, rank=rank, rngs=rngs)
  return model


def apply_lora(
    model: nnx.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    module_path: str = DEFAULT_LORA_MODULE_PATH,
    rng_seed: int = 10003,
    materialize_without_remat: bool = True,
    apply_moe: bool = True,
) -> nnx.Module:
  provider = qwix.LoraProvider(
      module_path=module_path,
      rank=rank,
      alpha=alpha,
  )
  original_config = getattr(model, "config", None)
  remat_config = getattr(original_config, "remat_config", None)
  attention_implementation = getattr(
      original_config, "attention_implementation", None
  )
  temporarily_simplify_materialization = (
      materialize_without_remat
      and dataclasses.is_dataclass(original_config)
      and (remat_config is not None or attention_implementation is not None)
  )
  if temporarily_simplify_materialization:
    config_updates = {}
    if remat_config is not None:
      config_updates["remat_config"] = None
    if attention_implementation is not None:
      config_updates["attention_implementation"] = None
    model.set_attributes(
        config=dataclasses.replace(original_config, **config_updates)
    )
  try:
    model = qwix.apply_lora_to_model(
        model,
        provider,
        **model.get_model_input(),
        rngs=nnx.Rngs(rng_seed),
    )
    if apply_moe:
      model = apply_moe_lora(
          model, rank=rank, alpha=alpha, rng_seed=rng_seed + 1
      )
    model.set_attributes(qwix_rngs=nnx.Rngs(rng_seed))
  finally:
    if temporarily_simplify_materialization:
      model.set_attributes(config=original_config)
  return model


class DiffusionGemmaTrainer(peft_trainer.PeftTrainer):
  """PeftTrainer wrapper preconfigured for DiffusionGemma SFT."""

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: peft_trainer.TrainingConfig,
      diffusion_config: DiffusionGemmaSFTConfig,
      *,
      split_loss_gradients: bool = False,
      **kwargs,
  ):
    self.diffusion_config = diffusion_config
    self.split_loss_gradients = split_loss_gradients
    self.decoder_only_config = dataclasses.replace(
        diffusion_config,
        encoder_loss_weight=0.0,
        force_full_encoder_prefill=True,
    )
    self.encoder_only_config = dataclasses.replace(
        diffusion_config, decoder_loss_weight=0.0
    )
    self.last_train_aux = None
    self.last_eval_aux = None
    super().__init__(model, optimizer, training_config, **kwargs)
    self.with_gen_model_input_fn(gen_model_input_fn)
    self.with_loss_fn(make_loss_fn(diffusion_config), has_aux=True)

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> tuple[jax.Array, Any | None, jax.Array]:
    if not self.split_loss_gradients:
      return super()._train_step(model, optimizer, inputs)

    inputs = self.gen_model_input_fn(inputs)
    grad_arg = nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0
    decoder_grad_fn = nnx.value_and_grad(
        make_loss_fn(self.decoder_only_config),
        argnums=grad_arg,
        has_aux=True,
    )
    encoder_grad_fn = nnx.value_and_grad(
        make_loss_fn(self.encoder_only_config),
        argnums=grad_arg,
        has_aux=True,
    )
    (decoder_loss, decoder_aux), decoder_grads = decoder_grad_fn(
        model, **inputs
    )
    (encoder_loss, encoder_aux), encoder_grads = encoder_grad_fn(
        model, **inputs
    )
    grads = jax.tree.map(lambda x, y: x + y, decoder_grads, encoder_grads)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    aux = dict(decoder_aux)
    aux["encoder_loss"] = encoder_aux["encoder_loss"]
    loss = decoder_loss + encoder_loss
    return loss, aux, grad_norm

  def _post_process_train_step(self, aux: Any) -> None:
    self.last_train_aux = aux

  def _post_process_eval_step(self, aux: Any) -> None:
    self.last_eval_aux = aux
