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

"""NNX DiffusionGemma model integration.

This module keeps the DiffusionGemma family separate from the autoregressive
Gemma 4 family while reusing Tunix's existing Gemma4 implementation for the
base transformer. The official DeepMind implementation defines
``DiffusionGemma_A26B_A4B`` as Gemma4 A26B/A4B plus a self-conditioning mixin.
The class below mirrors that shape in NNX so Tunix PEFT/SFT trainers can target
it without pulling in Kauldron or Flax Linen at training time.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
from tunix.models.gemma4 import model as gemma4_model


class ModelConfig:
  """Factory namespace for DiffusionGemma configs."""

  @classmethod
  def diffusion_gemma_a26b_a4b(
      cls,
      sharding_config: gemma4_model.ShardingConfig = (
          gemma4_model.ShardingConfig.get_default_sharding()
      ),
  ) -> gemma4_model.ModelConfig:
    """Returns the official DiffusionGemma A26B/A4B base config."""
    return gemma4_model.ModelConfig.gemma4_26b_a4b(sharding_config)

  @classmethod
  def diffusion_gemma_a26b_a4b_it(
      cls,
      sharding_config: gemma4_model.ShardingConfig = (
          gemma4_model.ShardingConfig.get_default_sharding()
      ),
  ) -> gemma4_model.ModelConfig:
    """Instruction-tuned DiffusionGemma uses the same architecture config."""
    return cls.diffusion_gemma_a26b_a4b(sharding_config)

  @classmethod
  def tiny(
      cls,
      *,
      vocab_size: int = 64,
      num_layers: int = 2,
      embed_dim: int = 32,
      hidden_dim: int = 64,
      num_heads: int = 4,
      head_dim: int = 8,
      num_kv_heads: int = 2,
  ) -> gemma4_model.ModelConfig:
    """Small config for CPU/GPU smoke tests."""
    return gemma4_model.ModelConfig(
        num_layers=num_layers,
        num_embed=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_global_kv_heads=num_kv_heads,
        global_key_size=head_dim,
        sliding_window_size=16,
        attention_pattern=(gemma4_model.AttentionType.LOCAL_SLIDING,),
        use_sliding_window_kv_cache=False,
        final_logit_softcap=None,
    )


class SelfConditioning(nnx.Module):
  """Feed-forward self-conditioning block from the DiffusionGemma paper path."""

  def __init__(
      self,
      config: gemma4_model.ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.pre_norm = gemma4_model.RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.ffw = gemma4_model.FeedForward(
        config,
        hidden_dim=config.hidden_dim,
        rngs=rngs,
    )
    self.post_norm = UnscaledRMSNorm(dtype=config.dtype)

  def __call__(
      self,
      *,
      canvas_embeddings: jaxtyping.Array,
      self_conditioning_signal: jaxtyping.Array,
  ) -> jaxtyping.Array:
    normed = self.pre_norm(self_conditioning_signal)
    sc_signal = self.ffw(normed)
    return self.post_norm(canvas_embeddings + sc_signal)


class UnscaledRMSNorm(nnx.Module):
  """RMSNorm without learned scale, matching official DiffusionGemma."""

  def __init__(self, *, dtype: jnp.dtype):
    self.dtype = dtype

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06).astype(x.dtype)
    return normed_inputs.astype(self.dtype)


class DiffusionGemma_A26B_A4B(gemma4_model.Gemma4):  # pylint: disable=invalid-name
  """DiffusionGemma A26B/A4B NNX module.

  The model behaves like Tunix Gemma4 for ordinary autoregressive calls. Passing
  ``sc_logits`` activates the DiffusionGemma self-conditioning path.
  """

  keep_last_prefill_kv: bool = True

  def __init__(
      self,
      config: gemma4_model.ModelConfig | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    if config is None:
      config = ModelConfig.diffusion_gemma_a26b_a4b()
    super().__init__(config, rngs=rngs)
    self.self_conditioner = SelfConditioning(config, rngs=rngs)

  def encode_logits(self, logits: jaxtyping.Array) -> jaxtyping.Array:
    """Encodes logits as an embedding-weighted probability mixture."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(
        self.config.dtype
    )
    embeddings = jnp.astype(
        self.embedder.input_embedding.value, self.config.dtype
    )
    encoded = jnp.einsum("...v,ve->...e", probs, embeddings)
    encoded *= math.sqrt(self.config.embed_dim)
    return encoded.astype(self.config.dtype)

  def __call__(
      self,
      tokens,
      positions=None,
      cache=None,
      attention_mask=None,
      decode_only_last_token=False,
      segment_ids=None,
      *,
      sc_logits=None,
      self_conditioning_mask=None,
  ):
    if sc_logits is None:
      return super().__call__(
          tokens,
          positions=positions,
          cache=cache,
          attention_mask=attention_mask,
          decode_only_last_token=decode_only_last_token,
          segment_ids=segment_ids,
      )
    return self.call_with_self_conditioning(
        tokens=tokens,
        sc_logits=sc_logits,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        decode_only_last_token=decode_only_last_token,
        segment_ids=segment_ids,
        self_conditioning_mask=self_conditioning_mask,
    )

  def call_with_self_conditioning(
      self,
      *,
      tokens,
      sc_logits,
      positions=None,
      cache=None,
      attention_mask=None,
      decode_only_last_token=False,
      segment_ids=None,
      self_conditioning_mask=None,
  ):
    """Forward pass with DiffusionGemma self-conditioning."""
    del segment_ids
    if positions is None:
      batch_size, seq_len = tokens.shape
      positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))

    if attention_mask is None:
      seq_len = tokens.shape[1]
      causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
      attention_mask = jnp.broadcast_to(
          causal[None, :, :], (tokens.shape[0], seq_len, seq_len)
      )

    new_cache = {}
    x = self.embedder.encode(tokens)
    sc_embeddings = self.encode_logits(sc_logits)
    sc_output = self.self_conditioner(
        canvas_embeddings=x,
        self_conditioning_signal=sc_embeddings,
    )
    if self_conditioning_mask is None:
      x = sc_output
    else:
      x = jnp.where(self_conditioning_mask[..., None], sc_output, x)

    per_layer_inputs = None
    if self.config.per_layer_input_dim > 0:
      per_layer_inputs = self.embedder.encode_per_layer_input(x, tokens)

    transient_kvs: dict[str, tuple[Any, Any]] = {}
    is_prefill = tokens.shape[1] > 1
    for i, layer in enumerate(self.layers):
      layer_name = f"layer_{i}"
      shared_idx = self.kv_cache_sharing_patterns[i]
      is_shared = shared_idx != i
      if is_shared:
        layer_cache = None
        shared_layer_name = f"layer_{shared_idx}"
        if is_prefill:
          shared_k, shared_v = transient_kvs[shared_layer_name]
          kv_shared_cache = {"k": shared_k, "v": shared_v}
        else:
          kv_shared_cache = new_cache.get(shared_layer_name)
      else:
        layer_cache = cache[layer_name] if cache else None
        kv_shared_cache = None

      layer_cache, x, layers_kvs = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
          per_layer_input=per_layer_inputs[:, :, i, :]
          if per_layer_inputs is not None
          else None,
          kv_shared_cache=kv_shared_cache,
      )
      if is_prefill and i in self.shared_layer_origins:
        transient_kvs[layer_name] = layers_kvs
      if not is_shared:
        new_cache[layer_name] = layer_cache

    x = self.final_norm(x)
    if decode_only_last_token:
      x = x[:, -1:, :]
    logits = self.embedder.decode(x).astype(jnp.float32)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return logits, (None if cache is None else new_cache)

  def get_model_input(self):
    model_input = super().get_model_input()
    tokens = model_input["tokens"]
    model_input["sc_logits"] = jnp.zeros(
        (*tokens.shape, self.config.num_embed), dtype=self.config.dtype
    )
    model_input["self_conditioning_mask"] = jnp.ones(
        tokens.shape, dtype=jnp.bool_
    )
    return model_input


DiffusionGemma = DiffusionGemma_A26B_A4B


def create_model(
    config: gemma4_model.ModelConfig | None = None,
    *,
    rngs: nnx.Rngs,
) -> DiffusionGemma_A26B_A4B:
  return DiffusionGemma_A26B_A4B(config, rngs=rngs)


def create_tiny_model(
    *,
    vocab_size: int = 64,
    rng_seed: int = 0,
) -> DiffusionGemma_A26B_A4B:
  return DiffusionGemma_A26B_A4B(
      ModelConfig.tiny(vocab_size=vocab_size),
      rngs=nnx.Rngs(rng_seed),
  )


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaCheckpointInfo:
  """Known public checkpoint metadata."""

  model_name: str = "diffusion-gemma-a26b-a4b-it"
  gcs_path: str = "gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it"
