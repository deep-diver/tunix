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

"""Full-model DiffusionGemma logits parity against the official implementation.

The official DiffusionGemma source is Flax Linen and imports Kauldron utilities.
This script installs tiny import stubs for those utilities so the official
model can run directly from a local google-deepmind/gemma checkout without
pulling in the full Kauldron training stack. The numerical path being compared
is still the official Linen transformer/model code.

The comparison uses a deliberately tiny non-MoE DiffusionGemma config, copies
the official Linen parameters into the Tunix NNX model with explicit layout
mapping, and checks both ordinary transformer logits and self-conditioning
logits over the complete sequence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import types
from typing import Any

from flax import linen as linen
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tunix.models.diffusion_gemma import model as tunix_diffusion_model
from tunix.models.gemma4 import model as tunix_gemma4_model


class _KTypingAlias:

  def __getitem__(self, _):
    return Any


class _KTypingModule(types.ModuleType):

  def __getattr__(self, name: str):
    if name in {"typechecked", "check_type"}:
      return _identity_typechecked
    if name == "KTypeCheckError":
      return TypeError
    alias = _KTypingAlias()
    setattr(self, name, alias)
    return alias


def _identity_typechecked(fn=None, **_):
  if fn is None:
    return lambda wrapped: wrapped
  return fn


def _flatten_unflatten_batch_dim():
  """No-op replacement for standard 2D parity inputs.

  The real helper flattens arbitrary leading batch dimensions. The parity
  inputs in this script use ordinary [B, L] tensors, so flattening would be a
  no-op. Keeping this stub local avoids importing Kauldron's runtime typing
  internals on developer machines where Kauldron cannot be installed.
  """

  def decorator(fn):
    return fn

  return decorator


class _Identity(linen.Module):

  @linen.compact
  def __call__(self, x):
    return x


class _Key(str):
  pass


class _KontextPath(str):

  @classmethod
  def from_jax_path(cls, path):
    parts = []
    for part in path:
      key = getattr(part, "key", part)
      parts.append(str(key))
    return cls("/".join(parts))


def _install_official_import_stubs() -> None:
  """Installs the small Kauldron surface needed for official model imports."""
  kauldron_mod = sys.modules.setdefault(
      "kauldron", types.ModuleType("kauldron")
  )
  kauldron_mod.__path__ = []  # Marks the stub as package-like.

  ktyping_mod = _KTypingModule("kauldron.ktyping")
  for name in (
      "Array",
      "BFloat16",
      "Bool",
      "Complex",
      "Complex64",
      "DType",
      "Float",
      "Float32",
      "Float64",
      "Int",
      "Int8",
      "Int16",
      "Int32",
      "Int64",
      "Num",
      "PRNGKey",
      "PyTree",
      "Scalar",
      "SInt",
      "UInt",
      "UInt8",
      "UInt16",
      "UInt32",
      "UInt64",
  ):
    setattr(ktyping_mod, name, _KTypingAlias())
  ktyping_mod.typechecked = _identity_typechecked
  ktyping_mod.check_type = _identity_typechecked
  ktyping_mod.KTypeCheckError = TypeError
  kauldron_mod.ktyping = ktyping_mod
  sys.modules["kauldron.ktyping"] = ktyping_mod

  typing_mod = _KTypingModule("kauldron.typing")
  typing_mod.PRNGKeyLike = Any
  kauldron_mod.typing = typing_mod
  sys.modules["kauldron.typing"] = typing_mod

  kd_mod = types.ModuleType("kauldron.kd")
  kd_nn_mod = types.ModuleType("kauldron.kd.nn")
  kd_nn_mod.Identity = _Identity
  kd_sharding_mod = types.ModuleType("kauldron.kd.sharding")
  kd_sharding_mod.ShardingTree = object
  kd_sharding_mod.REPLICATED = None
  kd_sharding_mod.FIRST_DIM = None
  kd_sharding_mod.with_sharding_constraint = lambda x, sharding=None: x
  kd_sharding_mod.device_put = lambda x, sharding=None: x
  kd_mod.nn = kd_nn_mod
  kd_mod.sharding = kd_sharding_mod
  kauldron_mod.kd = kd_mod
  sys.modules["kauldron.kd"] = kd_mod
  sys.modules["kauldron.kd.nn"] = kd_nn_mod
  sys.modules["kauldron.kd.sharding"] = kd_sharding_mod

  kontext_mod = types.ModuleType("kauldron.kontext")
  kontext_mod.Key = _Key
  kontext_mod.Path = _KontextPath
  kontext_mod.REQUIRED = _Key("__required__")
  kauldron_mod.kontext = kontext_mod
  kd_mod.kontext = kontext_mod
  sys.modules["kauldron.kontext"] = kontext_mod

  utils_mod = types.ModuleType("kauldron.utils")
  immutabledict_mod = types.ModuleType("kauldron.utils.immutabledict")
  immutabledict_mod.freeze_dict_attrs = lambda obj, attrs: None
  utils_mod.immutabledict = immutabledict_mod
  kauldron_mod.utils = utils_mod
  sys.modules["kauldron.utils"] = utils_mod
  sys.modules["kauldron.utils.immutabledict"] = immutabledict_mod

  jax_utils_mod = types.ModuleType("gemma.gm.utils._jax_utils")
  jax_utils_mod.flatten_unflatten_batch_dim = _flatten_unflatten_batch_dim
  sys.modules["gemma.gm.utils._jax_utils"] = jax_utils_mod


def _add_reference_paths(gemma_ref: str) -> None:
  if gemma_ref and gemma_ref not in sys.path:
    sys.path.insert(0, gemma_ref)


def _git_revision(path: str) -> str | None:
  try:
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "--short", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None


def _official_and_tunix_configs():
  from gemma.gm.nn.gemma4 import _config as official_config
  from gemma.gm.nn.gemma4 import _modules as official_modules

  official_cfg = official_config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=1,
      attention_types=[official_modules.AttentionType.GLOBAL],
      kv_cache_sharing_config=None,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      final_logit_softcap=None,
      num_global_kv_heads=1,
      global_key_size=4,
      global_rope_proportion=1.0,
      local_rope_proportion=1.0,
  )
  tunix_cfg = tunix_gemma4_model.ModelConfig(
      num_layers=1,
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=1,
      num_global_kv_heads=1,
      global_key_size=4,
      attention_pattern=(tunix_gemma4_model.AttentionType.GLOBAL,),
      global_rope_proportion=1.0,
      local_rope_proportion=1.0,
      global_base_frequency=10_000,
      local_base_frequency=10_000,
      use_sliding_window_kv_cache=False,
      final_logit_softcap=None,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  return official_cfg, tunix_cfg


def _make_inputs(vocab_size: int):
  tokens = jnp.array(
      [[1, 2, 3, 4], [5, 6, 7, 8]],
      dtype=jnp.int32,
  )
  batch_size, seq_len = tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(seq_len, dtype=jnp.int32), tokens.shape
  )
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attention_mask = jnp.broadcast_to(causal, (batch_size, seq_len, seq_len))
  sc_logits = jnp.linspace(
      -1.5,
      1.5,
      batch_size * seq_len * vocab_size,
      dtype=jnp.float32,
  ).reshape((batch_size, seq_len, vocab_size))
  return tokens, positions, attention_mask, sc_logits


def _assign(param: nnx.Param, value: jax.Array, name: str) -> dict[str, Any]:
  value = jnp.asarray(value, dtype=param[...].dtype)
  if tuple(param[...].shape) != tuple(value.shape):
    raise ValueError(
        f"Shape mismatch for {name}: Tunix {param[...].shape}, official"
        f" {value.shape}"
    )
  param[...] = value
  max_abs_diff = float(
      np.max(np.abs(np.asarray(param[...]) - np.asarray(value)))
  )
  return {
      "name": name,
      "shape": list(value.shape),
      "max_abs_diff_after_copy": max_abs_diff,
  }


def _copy_feed_forward(
    official_params: dict[str, Any],
    tunix_ffw,
    prefix: str,
    copied: list[dict[str, Any]],
) -> None:
  gating = official_params["gating_einsum"]
  copied.append(
      _assign(tunix_ffw.gate_proj.kernel, gating[0].T, f"{prefix}.gate_proj")
  )
  copied.append(
      _assign(tunix_ffw.up_proj.kernel, gating[1].T, f"{prefix}.up_proj")
  )
  copied.append(
      _assign(
          tunix_ffw.down_proj.kernel,
          official_params["linear"],
          f"{prefix}.down_proj",
      )
  )


def _copy_official_params_to_tunix(
    official_params, tunix_model
) -> list[dict[str, Any]]:
  """Copies official Linen parameters into the equivalent Tunix NNX leaves."""
  copied: list[dict[str, Any]] = []
  copied.append(
      _assign(
          tunix_model.embedder.input_embedding,
          official_params["embedder"]["input_embedding"],
          "embedder.input_embedding",
      )
  )
  copied.append(
      _assign(
          tunix_model.final_norm.scale,
          official_params["final_norm"]["scale"],
          "final_norm.scale",
      )
  )

  for i, tunix_layer in enumerate(tunix_model.layers):
    official_layer = official_params[f"layer_{i}"]
    prefix = f"layers.{i}"
    copied.append(
        _assign(
            tunix_layer.skip_scale,
            official_layer["skip_scale"],
            f"{prefix}.skip_scale",
        )
    )
    copied.append(
        _assign(
            tunix_layer.pre_attention_norm.scale,
            official_layer["pre_attention_norm"]["scale"],
            f"{prefix}.pre_attention_norm.scale",
        )
    )
    copied.append(
        _assign(
            tunix_layer.post_attention_norm.scale,
            official_layer["post_attention_norm"]["scale"],
            f"{prefix}.post_attention_norm.scale",
        )
    )
    copied.append(
        _assign(
            tunix_layer.pre_ffw_norm.scale,
            official_layer["pre_ffw_norm"]["scale"],
            f"{prefix}.pre_ffw_norm.scale",
        )
    )
    copied.append(
        _assign(
            tunix_layer.post_ffw_norm.scale,
            official_layer["post_ffw_norm"]["scale"],
            f"{prefix}.post_ffw_norm.scale",
        )
    )

    official_attn = official_layer["attn"]
    copied.append(
        _assign(
            tunix_layer.attn.q_einsum.w,
            official_attn["q_einsum"]["w"],
            f"{prefix}.attn.q_einsum.w",
        )
    )
    copied.append(
        _assign(
            tunix_layer.attn.kv_einsum.w,
            official_attn["kv_einsum"]["w"],
            f"{prefix}.attn.kv_einsum.w",
        )
    )
    copied.append(
        _assign(
            tunix_layer.attn.attn_vec_einsum.w,
            official_attn["attn_vec_einsum"]["w"],
            f"{prefix}.attn.attn_vec_einsum.w",
        )
    )
    copied.append(
        _assign(
            tunix_layer.attn._query_norm.scale,  # pylint: disable=protected-access
            official_attn["query_norm"]["scale"],
            f"{prefix}.attn.query_norm.scale",
        )
    )
    copied.append(
        _assign(
            tunix_layer.attn._key_norm.scale,  # pylint: disable=protected-access
            official_attn["key_norm"]["scale"],
            f"{prefix}.attn.key_norm.scale",
        )
    )
    _copy_feed_forward(
        official_layer["mlp"], tunix_layer.mlp, f"{prefix}.mlp", copied
    )

  official_sc = official_params["self_conditioner"]
  copied.append(
      _assign(
          tunix_model.self_conditioner.pre_norm.scale,
          official_sc["pre_norm"]["scale"],
          "self_conditioner.pre_norm.scale",
      )
  )
  _copy_feed_forward(
      official_sc["ffw"],
      tunix_model.self_conditioner.ffw,
      "self_conditioner.ffw",
      copied,
  )
  return copied


def _close_metrics(actual, expected) -> dict[str, Any]:
  actual_np = np.asarray(jax.device_get(actual))
  expected_np = np.asarray(jax.device_get(expected))
  diff = actual_np - expected_np
  return {
      "shape": list(actual_np.shape),
      "max_abs_diff": float(np.max(np.abs(diff))),
      "mean_abs_diff": float(np.mean(np.abs(diff))),
  }


def _assert_close(
    name: str,
    actual,
    expected,
    results: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> None:
  metrics = _close_metrics(actual, expected)
  np.testing.assert_allclose(
      np.asarray(jax.device_get(actual)),
      np.asarray(jax.device_get(expected)),
      rtol=rtol,
      atol=atol,
      err_msg=name,
  )
  results[name] = metrics


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--gemma_ref", default="/tmp/gemma-diffusion-reference")
  parser.add_argument("--rtol", type=float, default=1e-5)
  parser.add_argument("--atol", type=float, default=1e-5)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  _install_official_import_stubs()
  _add_reference_paths(args.gemma_ref)

  from gemma.diffusion import _models as official_models
  from gemma.diffusion import _transformer as official_diffusion_transformer

  official_cfg, tunix_cfg = _official_and_tunix_configs()
  tokens, positions, attention_mask, sc_logits = _make_inputs(
      official_cfg.num_embed
  )
  sc_embeddings_for_init = jnp.zeros(
      (*tokens.shape, official_cfg.embed_dim), dtype=jnp.float32
  )

  official_model = official_models.DiffusionGemma_A26B_A4B(
      config=official_cfg,
      dtype=jnp.float32,
      self_conditioning_config=(
          official_diffusion_transformer.SelfConditioningConfig(
              features=official_cfg.embed_dim,
              hidden_dim=official_cfg.hidden_dim,
          )
      ),
  )
  variables = official_model.init(
      jax.random.PRNGKey(0),
      tokens=tokens,
      positions=positions,
      attention_mask=attention_mask,
      sc_embeddings=sc_embeddings_for_init,
      method=official_model.call_with_self_conditioning,
  )

  tunix_model = tunix_diffusion_model.DiffusionGemma_A26B_A4B(
      tunix_cfg,
      rngs=nnx.Rngs(0),
  )
  copied = _copy_official_params_to_tunix(variables["params"], tunix_model)

  def official_encode_logits(model, logits):
    return model.embedder.encode_logits(logits)

  official_encoded_sc = official_model.apply(
      variables,
      sc_logits,
      method=official_encode_logits,
  )
  tunix_encoded_sc = tunix_model.encode_logits(sc_logits)

  official_plain = official_model.apply(
      variables,
      tokens=tokens,
      positions=positions,
      attention_mask=attention_mask,
  ).logits
  tunix_plain, _ = tunix_model(
      tokens,
      positions=positions,
      attention_mask=attention_mask,
  )

  official_sc = official_model.apply(
      variables,
      tokens=tokens,
      positions=positions,
      attention_mask=attention_mask,
      sc_embeddings=official_encoded_sc,
      method=official_model.call_with_self_conditioning,
  ).logits
  tunix_sc, _ = tunix_model(
      tokens,
      positions=positions,
      attention_mask=attention_mask,
      sc_logits=sc_logits,
  )

  results: dict[str, Any] = {}
  _assert_close(
      "encode_logits",
      tunix_encoded_sc,
      official_encoded_sc,
      results,
      rtol=args.rtol,
      atol=args.atol,
  )
  _assert_close(
      "plain_logits",
      tunix_plain,
      official_plain,
      results,
      rtol=args.rtol,
      atol=args.atol,
  )
  _assert_close(
      "self_conditioned_logits",
      tunix_sc,
      official_sc,
      results,
      rtol=args.rtol,
      atol=args.atol,
  )

  print(
      json.dumps(
          {
              "event": "official_logits_parity_passed",
              "gemma_ref": args.gemma_ref,
              "gemma_ref_revision": _git_revision(args.gemma_ref),
              "rtol": args.rtol,
              "atol": args.atol,
              "copied_parameter_count": len(copied),
              "copied_parameters": copied,
              "results": results,
          },
          sort_keys=True,
      )
  )


if __name__ == "__main__":
  main()
