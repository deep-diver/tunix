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

"""Tests for the DiffusionGemma Tunix integration."""

import dataclasses
import sys
import types
from unittest import mock

from absl.testing import absltest
import flax.traverse_util
from flax import linen as linen_nn
from flax.core import freeze
from flax.core import unfreeze
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.models import automodel
from tunix.models import low_peak_params
from tunix.models import naming
from tunix.models.diffusion_gemma import hackable_adapter
from tunix.models.diffusion_gemma import linen_qwix_lora
from tunix.models.diffusion_gemma import lora_inventory
from tunix.models.diffusion_gemma import model as diffusion_model
from tunix.models.diffusion_gemma import params as diffusion_params
from tunix.models.diffusion_gemma import sft as diffusion_sft
from tunix.models.gemma4 import model as gemma4_model
from tunix.sft import peft_trainer


def _make_batch(
    *,
    seed: int = 0,
    batch_size: int = 2,
    prompt_len: int = 4,
    canvas_size: int = 4,
    num_canvases: int = 2,
    vocab_size: int = 32,
) -> diffusion_sft.DiffusionGemmaSFTBatch:
  total_canvas_len = canvas_size * num_canvases
  key = jax.random.PRNGKey(seed)
  key_prompt, key_canvas = jax.random.split(key)
  prompt = jax.random.randint(
      key_prompt, (batch_size, prompt_len), 1, vocab_size, dtype=jnp.int32
  )
  canvas = jax.random.randint(
      key_canvas,
      (batch_size, total_canvas_len),
      1,
      vocab_size,
      dtype=jnp.int32,
  )
  canvas_id = jnp.broadcast_to(
      jnp.repeat(jnp.arange(num_canvases, dtype=jnp.int32), canvas_size),
      (batch_size, total_canvas_len),
  )
  canvas_mask = jnp.ones((batch_size, total_canvas_len), dtype=jnp.bool_)
  full_seq = jnp.concatenate([prompt, canvas], axis=1)
  encoder_target = jnp.roll(full_seq, shift=-1, axis=1)
  encoder_target = encoder_target.at[:, -1].set(0)
  encoder_target_mask = jnp.ones_like(full_seq, dtype=jnp.float32)
  encoder_target_mask = encoder_target_mask.at[:, -1].set(0.0)
  return diffusion_sft.DiffusionGemmaSFTBatch(
      prompt=prompt,
      canvas=canvas,
      canvas_id=canvas_id,
      canvas_mask=canvas_mask,
      encoder_target=encoder_target,
      encoder_target_mask=encoder_target_mask,
      rng=jax.random.PRNGKey(seed + 100),
  )


def _all_equal(before, after):
  return all(
      bool(jax.device_get(x))
      for x in jax.tree.leaves(
          jax.tree.map(lambda a, b: jnp.all(a == b), before, after)
      )
  )


def _any_changed(before, after):
  return any(
      bool(jax.device_get(x))
      for x in jax.tree.leaves(
          jax.tree.map(lambda a, b: jnp.any(a != b), before, after)
      )
  )


def _official_reference_corrupt_tokens(rng, x0_tokens, time, vocab_size):
  rng_mask, rng_noise = jax.random.split(rng)
  random_tokens = jax.random.choice(
      rng_noise,
      a=vocab_size,
      shape=x0_tokens.shape,
      p=jnp.full((vocab_size,), 1.0 / vocab_size, dtype=jnp.float32),
      mode="high",
  ).astype(x0_tokens.dtype)
  is_not_corrupted = jax.random.bernoulli(
      rng_mask,
      p=jnp.broadcast_to(1.0 - time, x0_tokens.shape),
      shape=x0_tokens.shape,
      mode="high",
  )
  is_corrupted = jnp.logical_not(is_not_corrupted)
  return jnp.where(is_corrupted, random_tokens, x0_tokens), is_corrupted


def _official_reference_masked_ce(logits, targets, mask):
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
  mask = mask.astype(loss.dtype)
  reduce_axes = tuple(range(1, loss.ndim))
  per_example_loss = jnp.sum(loss * mask, axis=reduce_axes)
  per_example_denom = jnp.maximum(jnp.sum(mask, axis=reduce_axes), 1.0)
  return jnp.mean(per_example_loss / per_example_denom)


def _official_reference_masked_ce_from_token_loss(loss, mask):
  mask = mask.astype(loss.dtype)
  reduce_axes = tuple(range(1, loss.ndim))
  per_example_loss = jnp.sum(loss * mask, axis=reduce_axes)
  per_example_denom = jnp.maximum(jnp.sum(mask, axis=reduce_axes), 1.0)
  return jnp.mean(per_example_loss / per_example_denom)


class _ToyLinenAttention(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    q = linen_nn.Dense(4, use_bias=False, name="q_einsum")(x)
    kv = linen_nn.Dense(4, use_bias=False, name="kv_einsum")(x)
    out = linen_nn.Dense(4, use_bias=False, name="attn_vec_einsum")(x)
    return q + kv + out


class _ToyLinenMLP(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    gate = linen_nn.Dense(4, use_bias=False, name="gating_einsum")(x)
    down = linen_nn.Dense(4, use_bias=False, name="linear")(x)
    router = linen_nn.Dense(4, use_bias=False, name="router_logits")(x)
    return gate + down + router


class _ToyLinenDirectParamMLP(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    gate = self.param(
        "gating_einsum",
        linen_nn.initializers.ones,
        (x.shape[-1], 4),
    )
    down = self.param(
        "linear",
        linen_nn.initializers.ones,
        (x.shape[-1], 4),
    )
    return jnp.einsum("...d,df->...f", x, gate) + jnp.einsum(
        "...d,df->...f", x, down
    )


class _Weight(linen_nn.Module):
  shape: tuple[int, ...]
  weight_name: str = "w"

  @linen_nn.compact
  def __call__(self):
    return self.param(self.weight_name, linen_nn.initializers.ones, self.shape)


class _ToyLinenRaggedMLP(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    gate = _Weight(
        shape=(2, 2, 3, x.shape[-1]),
        name="gating_einsum",
    )()
    down = _Weight(
        shape=(2, 3, x.shape[-1]),
        name="linear",
    )()
    group_sizes = jnp.array([x.shape[0], 0], dtype=jnp.int32)
    gate = jnp.transpose(gate, (0, 3, 1, 2)).reshape(2, x.shape[-1], 6)
    gate_out = jax.lax.ragged_dot(x, gate, group_sizes=group_sizes)
    gate_out = gate_out.reshape(x.shape[0], 2, 3)
    activation = jax.nn.gelu(gate_out[:, 0, :]) * gate_out[:, 1, :]
    return jax.lax.ragged_dot(activation, down, group_sizes=group_sizes)


class _ToyLinenSelfConditioner(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    ffw = _ToyLinenFFW(name="ffw")
    return ffw(x)


class _ToyLinenFFW(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    gate = linen_nn.Dense(4, use_bias=False, name="gating_einsum")(x)
    down = linen_nn.Dense(4, use_bias=False, name="linear")(x)
    return gate + down


class _ToyLinenLayer(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    attn = _ToyLinenAttention(name="attn")
    mlp = _ToyLinenMLP(name="mlp")
    return attn(x) + mlp(x)


class _ToyLinenDiffusionGemmaMethods(linen_nn.Module):
  """Tiny Linen surface with DiffusionGemma method names and target scopes."""

  @linen_nn.compact
  def __call__(self, x):
    embed = linen_nn.Dense(4, use_bias=False, name="embedder")(x)
    layer = _ToyLinenLayer(name="layer_0")
    conditioner = _ToyLinenSelfConditioner(name="self_conditioner")
    return embed + layer(x) + conditioner(x)

  @linen_nn.compact
  def encoder_call(self, x):
    layer = _ToyLinenLayer(name="layer_1")
    return layer(x)

  @linen_nn.compact
  def init_cache(self, x):
    attn = _ToyLinenAttention(name="attn")
    return attn(x)


class _ToyLinenDirectParamMethods(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    mlp = _ToyLinenDirectParamMLP(name="mlp")
    return mlp(x)

  @linen_nn.compact
  def encoder_call(self, x):
    mlp = _ToyLinenDirectParamMLP(name="mlp")
    return mlp(x)

  @linen_nn.compact
  def init_cache(self, x):
    mlp = _ToyLinenDirectParamMLP(name="mlp")
    return mlp(x)


class _ToyLinenRaggedMethods(linen_nn.Module):

  @linen_nn.compact
  def __call__(self, x):
    mlp = _ToyLinenRaggedMLP(name="mlp")
    return mlp(x)

  @linen_nn.compact
  def encoder_call(self, x):
    mlp = _ToyLinenRaggedMLP(name="mlp")
    return mlp(x)

  @linen_nn.compact
  def init_cache(self, x):
    mlp = _ToyLinenRaggedMLP(name="mlp")
    return mlp(x)


def _set_lora_b_leaves_to_constant(variables, value: float):
  mutable = unfreeze(variables)
  flat_params = flax.traverse_util.flatten_dict(mutable["params"])
  for path, leaf in flat_params.items():
    if str(path[-1]).endswith("_lora_b"):
      flat_params[path] = jnp.ones_like(leaf) * value
  mutable["params"] = flax.traverse_util.unflatten_dict(flat_params)
  return freeze(mutable)


class DiffusionGemmaTest(absltest.TestCase):

  def test_low_peak_restore_target_excludes_adapter_leaves(self):
    initialized = {
        "layers": {
            0: {
                "attn": {
                    "q_einsum": {"kernel": jnp.zeros((2, 3))},
                    "q_einsum_lora": {
                        "lora_a": jnp.ones((2, 1)),
                        "lora_b": jnp.ones((1, 3)),
                    },
                }
            }
        },
        "self_conditioner": {
            "ffw": {"down_proj": {"kernel": jnp.zeros((3, 2))}}
        },
    }

    info = low_peak_params.build_restore_target(
        initialized,
        preserve_predicate=low_peak_params.path_contains_lora,
        dtype=jnp.bfloat16,
    )
    flat_target = jax.tree_util.tree_flatten_with_path(info.target)[0]
    target_paths = {tuple(part.key for part in path) for path, _ in flat_target}

    self.assertIn(("layers", 0, "attn", "q_einsum", "kernel"), target_paths)
    self.assertIn(
        ("self_conditioner", "ffw", "down_proj", "kernel"), target_paths
    )
    self.assertNotIn(
        ("layers", 0, "attn", "q_einsum_lora", "lora_a"), target_paths
    )
    self.assertEqual(
        info.preserved_paths,
        (
            ("layers", 0, "attn", "q_einsum_lora", "lora_a"),
            ("layers", 0, "attn", "q_einsum_lora", "lora_b"),
        ),
    )
    self.assertEqual(
        info.target["layers"][0]["attn"]["q_einsum"]["kernel"].dtype,
        jnp.bfloat16,
    )

  def test_low_peak_merge_restores_base_and_preserves_adapters(self):
    initialized = {
        "base": {"kernel": jnp.zeros((2, 2))},
        "block_lora": {
            "lora_a": jnp.full((2, 1), 7.0),
            "lora_b": jnp.full((1, 2), 9.0),
        },
    }
    restored = {
        "base": {"kernel": jnp.full((2, 2), 3.0)},
        "block_lora": {
            "lora_a": jnp.full((2, 1), 100.0),
        },
        "unused": {"kernel": jnp.ones((1,))},
    }

    result = low_peak_params.merge_restored_state(
        initialized,
        restored,
        preserve_predicate=low_peak_params.path_contains_lora,
        strict=False,
    )

    np.testing.assert_array_equal(
        result.tree["base"]["kernel"], jnp.full((2, 2), 3.0)
    )
    np.testing.assert_array_equal(
        result.tree["block_lora"]["lora_a"], jnp.full((2, 1), 7.0)
    )
    np.testing.assert_array_equal(
        result.tree["block_lora"]["lora_b"], jnp.full((1, 2), 9.0)
    )
    self.assertEqual(result.report.restored_paths, (("base", "kernel"),))
    self.assertEqual(
        result.report.skipped_preserved_restore_paths,
        (("block_lora", "lora_a"),),
    )
    self.assertEqual(result.report.extra_paths, (("unused", "kernel"),))

  def test_low_peak_checkpoint_mapper_rejects_collisions(self):
    checkpoint = {
        "transformer/layer_0/attn/q_einsum": {"w": jnp.ones((2, 2))},
        "layer_0": {"attn": {"q_einsum": {"w": jnp.zeros((2, 2))}}},
    }

    def mapper(path, _value):
      parts = []
      for part in path:
        parts.extend(str(part).split("/"))
      if parts[0] == "transformer":
        parts = parts[1:]
      return tuple(parts)

    with self.assertRaisesRegex(ValueError, "Multiple checkpoint leaves map"):
      low_peak_params.map_checkpoint_tree(checkpoint, mapper)

  def test_diffusion_gemma_merge_can_preserve_lora_leaves(self):
    initialized = {
        "layers": {
            0: {
                "mlp": {
                    "gate_proj": {"kernel": jnp.zeros((2, 3))},
                    "gate_proj_lora": {"lora_a": jnp.full((2, 1), 5.0)},
                }
            }
        }
    }
    mapped = {
        "layers": {
            0: {
                "mlp": {
                    "gate_proj": {"kernel": jnp.ones((2, 3))},
                    "gate_proj_lora": {"lora_a": jnp.full((2, 1), 99.0)},
                }
            }
        }
    }

    merged = diffusion_params._merge_with_initialized_state(  # pylint: disable=protected-access
        initialized,
        mapped,
        preserve_predicate=low_peak_params.path_contains_lora,
    )

    np.testing.assert_array_equal(
        merged["layers"][0]["mlp"]["gate_proj"]["kernel"],
        jnp.ones((2, 3)),
    )
    np.testing.assert_array_equal(
        merged["layers"][0]["mlp"]["gate_proj_lora"]["lora_a"],
        jnp.full((2, 1), 5.0),
    )

  def test_official_backend_dependency_probe_is_non_throwing(self):
    report = hackable_adapter.check_dependencies()
    self.assertIn("available", report)
    self.assertIn("missing", report)
    self.assertIn("versions", report)

  def test_official_chunked_encoder_ce_matches_optax(self):
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (2, 7, 19), dtype=jnp.float32)
    targets = jax.random.randint(
        jax.random.PRNGKey(1), (2, 7), 0, 19, dtype=jnp.int32
    )
    mask = jnp.array(
        [[1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 1, 1, 0]],
        dtype=jnp.float32,
    )

    full_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets
    )
    chunked_token_loss = (
        hackable_adapter.chunked_softmax_cross_entropy_with_integer_labels(
            logits, targets, token_chunk_size=3
        )
    )
    np.testing.assert_allclose(
        chunked_token_loss, full_token_loss, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        _official_reference_masked_ce_from_token_loss(chunked_token_loss, mask),
        _official_reference_masked_ce(logits, targets, mask),
        rtol=1e-6,
        atol=1e-6,
    )

  def test_official_backend_builds_fake_recipe_with_overrides(self):
    module_name = hackable_adapter.recipe_module_name("pubmedqa")
    fake_module = types.ModuleType(module_name)
    fake_module.CHECKPOINT_PATH = "old_checkpoint"
    fake_module._LORA_RANK = 4
    seen_dataset_batches = []

    def make_pubmedqa_ds(**kwargs):
      seen_dataset_batches.append(kwargs["batch_size"])
      return kwargs

    fake_module.pubmedqa_data = types.SimpleNamespace(
        make_pubmedqa_ds=make_pubmedqa_ds
    )

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class FakeConfigArgs:
      use_early_stopping: bool = True

    def get_config(args=FakeConfigArgs()):
      train_ds = fake_module.pubmedqa_data.make_pubmedqa_ds(batch_size=2)
      return types.SimpleNamespace(
          aux=types.SimpleNamespace(
              checkpoint_every_n_steps=1000,
              eval_num_batches=None,
          ),
          checkpointer=types.SimpleNamespace(save_interval_steps=1000),
          evals={"sample": object()},
          train_ds=train_ds,
          schedules={
              "learning_rate": types.SimpleNamespace(
                  warmup_steps=100,
                  decay_steps=200,
              )
          },
          seen_checkpoint=fake_module.CHECKPOINT_PATH,
          seen_lora_rank=fake_module._LORA_RANK,
          seen_early_stopping=args.use_early_stopping,
      )

    fake_module.ConfigArgs = FakeConfigArgs
    fake_module.get_config = get_config

    with mock.patch.dict(sys.modules, {module_name: fake_module}):
      cfg = hackable_adapter.build_official_sft_config(
          hackable_adapter.OfficialSFTConfig(
              recipe="pubmedqa",
              workdir="/tmp/dg-workdir",
              checkpoint_path="/tmp/dg-checkpoint",
              num_train_steps=1,
              checkpoint_every_n_steps=1,
              lora_rank=8,
              dataset_batch_size=1,
              use_early_stopping=False,
              disable_evals=True,
              config_overrides={
                  "aux.eval_num_batches": 2,
                  "schedules.learning_rate.warmup_steps": 0,
              },
          )
      )

    self.assertEqual(cfg.seen_checkpoint, "/tmp/dg-checkpoint")
    self.assertEqual(cfg.seen_lora_rank, 8)
    self.assertFalse(cfg.seen_early_stopping)
    self.assertEqual(cfg.workdir, "/tmp/dg-workdir")
    self.assertEqual(cfg.num_train_steps, 1)
    self.assertEqual(cfg.aux.checkpoint_every_n_steps, 1)
    self.assertEqual(cfg.aux.eval_num_batches, 2)
    self.assertEqual(cfg.train_ds["batch_size"], 1)
    self.assertEqual(seen_dataset_batches, [1])
    self.assertEqual(cfg.schedules["learning_rate"].warmup_steps, 0)
    self.assertEqual(cfg.schedules["learning_rate"].decay_steps, 200)
    self.assertEqual(cfg.checkpointer.save_interval_steps, 1)
    self.assertEmpty(cfg.evals)
    self.assertEqual(fake_module.CHECKPOINT_PATH, "old_checkpoint")
    self.assertEqual(fake_module._LORA_RANK, 4)

  def test_official_backend_can_skip_step_metrics(self):
    calls = []

    class FakeWriter:

      def write_step_metrics(self, *args, **kwargs):
        calls.append(("original", args, kwargs))

    writer = FakeWriter()
    trainer = types.SimpleNamespace(writer=writer)
    hackable_adapter._patch_trainer_to_skip_step_metrics(trainer)

    writer.write_step_metrics(step=7)

    self.assertEmpty(calls)

  def test_official_backend_checksum_uses_addressable_leaves(self):
    tree = {
        "lora": {"a": jnp.array([1.0, 2.0])},
        "base": {"w": jnp.array([4.0, 8.0])},
    }

    lora = hackable_adapter._addressable_param_checksum(
        tree, lambda path: "lora" in path
    )
    base = hackable_adapter._addressable_param_checksum(
        tree, lambda path: "base" in path
    )

    self.assertEqual(lora["num_leaves"], 1)
    self.assertEqual(lora["num_elements"], 2)
    self.assertEqual(lora["checksum"], 3.0)
    self.assertEqual(base["num_leaves"], 1)
    self.assertEqual(base["num_elements"], 2)
    self.assertEqual(base["checksum"], 12.0)

  def test_model_naming_and_config(self):
    info = naming.ModelNaming(model_name="diffusion-gemma-a26b-a4b-it")
    self.assertEqual(info.model_family, "diffusion_gemma")
    self.assertEqual(info.model_config_category, "diffusion_gemma")
    cfg = automodel.call_model_config("diffusion-gemma-a26b-a4b-it")
    self.assertEqual(cfg.num_embed, 262144)
    self.assertIsNone(cfg.remat_config)

  def test_decoder_remat_preserves_qwix_lora_coverage(self):
    vocab_size = 32
    no_remat = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    no_remat = diffusion_sft.apply_lora(no_remat, rank=4, alpha=8.0)
    remat_cfg = dataclasses.replace(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        remat_config=gemma4_model.RematConfig.DECODER,
    )
    remat = diffusion_model.DiffusionGemma_A26B_A4B(
        remat_cfg,
        rngs=nnx.Rngs(0),
    )
    remat = diffusion_sft.apply_lora(remat, rank=4, alpha=8.0)

    no_remat_leaves = jax.tree.leaves(nnx.state(no_remat, nnx.LoRAParam))
    remat_leaves = jax.tree.leaves(nnx.state(remat, nnx.LoRAParam))
    self.assertEqual(len(no_remat_leaves), len(remat_leaves))

  def test_lora_targets_official_ragged_moe_router_by_default(self):
    base_cfg = diffusion_model.ModelConfig.tiny(
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
    )
    cfg = dataclasses.replace(
        base_cfg,
        enable_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        expert_dim=8,
        moe_dense_hidden_dim=16,
    )
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        cfg,
        rngs=nnx.Rngs(0),
    )
    model = diffusion_sft.apply_lora(model, rank=2, alpha=4.0)
    lora_paths = {
        "/".join(str(part.key) for part in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(
            nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam))
        )[0]
    }

    for suffix in ("router_logits_lora_a", "router_logits_lora_b"):
      self.assertIn(f"layers/0/moe/{suffix}", lora_paths)
    for suffix in (
        "gating_einsum_lora_a",
        "gating_einsum_lora_b",
        "linear_lora_a",
        "linear_lora_b",
    ):
      self.assertNotIn(f"layers/0/moe/{suffix}", lora_paths)
    self.assertFalse(any("_lora_a_lora_" in path for path in lora_paths))
    self.assertFalse(any("_lora_b_lora_" in path for path in lora_paths))

    moe = model.layers[0].moe
    x = jnp.ones((1, 3, cfg.embed_dim), dtype=jnp.float32)
    before = moe(x)
    moe.router_logits_lora_a.value = (
        jnp.ones_like(moe.router_logits_lora_a.value) * 0.5
    )
    moe.router_logits_lora_b.value = jnp.arange(
        moe.router_logits_lora_b.value.size,
        dtype=moe.router_logits_lora_b.value.dtype,
    ).reshape(moe.router_logits_lora_b.value.shape)
    after = moe(x)
    self.assertTrue(bool(jnp.any(before != after)))

  def test_lora_inventory_matches_official_all_linear_default(self):
    base_cfg = diffusion_model.ModelConfig.tiny(
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
    )
    cfg = dataclasses.replace(
        base_cfg,
        enable_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        expert_dim=8,
        moe_dense_hidden_dim=16,
    )
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        cfg,
        rngs=nnx.Rngs(0),
    )
    model = diffusion_sft.apply_lora(model, rank=2, alpha=4.0)

    comparison = lora_inventory.compare_model_to_official_all_linear(model)

    self.assertEmpty(comparison.missing_from_inventory)
    self.assertEmpty(comparison.extra_in_inventory)
    self.assertEmpty(comparison.inventory.unknown_leaf_paths)
    self.assertTrue(comparison.matches_official)

  def test_lora_raw_expert_targets_are_opt_in(self):
    base_cfg = diffusion_model.ModelConfig.tiny(
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
    )
    cfg = dataclasses.replace(
        base_cfg,
        enable_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        expert_dim=8,
        moe_dense_hidden_dim=16,
    )
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        cfg,
        rngs=nnx.Rngs(0),
    )
    model = diffusion_sft.apply_lora(
        model,
        rank=2,
        alpha=4.0,
        moe_target_names=("router_logits", "gating_einsum", "linear"),
    )
    lora_paths = {
        "/".join(str(part.key) for part in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(
            nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam))
        )[0]
    }

    for suffix in (
        "router_logits_lora_a",
        "router_logits_lora_b",
        "gating_einsum_lora_a",
        "gating_einsum_lora_b",
        "linear_lora_a",
        "linear_lora_b",
    ):
      self.assertIn(f"layers/0/moe/{suffix}", lora_paths)

    moe = model.layers[0].moe
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 3, cfg.embed_dim))
    before = moe(x)
    moe.gating_einsum_lora_b.value = (
        jnp.ones_like(moe.gating_einsum_lora_b.value) * 0.01
    )
    moe.linear_lora_b.value = jnp.ones_like(moe.linear_lora_b.value) * 0.01
    after = moe(x)
    self.assertTrue(bool(jnp.any(before != after)))

    comparison = lora_inventory.compare_model_to_official_all_linear(model)
    self.assertEqual(
        comparison.extra_in_inventory,
        frozenset({"moe.gating_einsum", "moe.linear"}),
    )
    self.assertEmpty(comparison.missing_from_inventory)

  def test_linen_qwix_lora_bridge_targets_diffusion_methods(self):
    model = linen_qwix_lora.apply_lora_to_linen_model(
        _ToyLinenDiffusionGemmaMethods(),
        rank=2,
        alpha=4.0,
    )
    x = jnp.ones((1, 4), dtype=jnp.float32)

    call_variables = model.init(jax.random.PRNGKey(0), x)
    call_inventory = linen_qwix_lora.inventory_from_linen_params(
        call_variables["params"]
    )
    call_comparison = lora_inventory.LoRATargetComparison(call_inventory)

    self.assertTrue(call_comparison.matches_official)
    self.assertFalse(
        any("embedder" in path for path in call_inventory.leaf_paths)
    )

    encoder_variables = model.init(
        jax.random.PRNGKey(1), x, method=model.encoder_call
    )
    encoder_inventory = linen_qwix_lora.inventory_from_linen_params(
        encoder_variables["params"]
    )
    self.assertEqual(
        encoder_inventory.families,
        frozenset({
            "attention.attn_vec_einsum",
            "attention.kv_einsum",
            "attention.q_einsum",
            "ffw.gating_einsum",
            "ffw.linear",
            "moe.router_logits",
        }),
    )

    cache_variables = model.init(
        jax.random.PRNGKey(2), x, method=model.init_cache
    )
    cache_inventory = linen_qwix_lora.inventory_from_linen_params(
        cache_variables["params"]
    )
    self.assertEqual(
        cache_inventory.families,
        frozenset({
            "attention.attn_vec_einsum",
            "attention.kv_einsum",
            "attention.q_einsum",
        }),
    )

  def test_linen_qwix_lora_bridge_affects_outputs(self):
    model = linen_qwix_lora.apply_lora_to_linen_model(
        _ToyLinenDiffusionGemmaMethods(),
        rank=2,
        alpha=4.0,
    )
    x = jnp.ones((1, 4), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x)
    changed_variables = _set_lora_b_leaves_to_constant(variables, 0.01)

    before = model.apply(variables, x)
    after = model.apply(changed_variables, x)

    self.assertTrue(bool(jnp.any(before != after)))

  def test_qwix_lora_select_patterns_include_official_and_qwix_names(self):
    self.assertEqual(
        hackable_adapter._qwix_lora_select_patterns("lora"),  # pylint: disable=protected-access
        ("lora", r".*_lora_a", r".*_lora_b"),
    )
    self.assertEqual(
        hackable_adapter._qwix_lora_select_patterns(  # pylint: disable=protected-access
            ("lora", "head")
        ),
        ("lora", r".*_lora_a", r".*_lora_b", "head"),
    )

  def test_qwix_lora_paths_are_checkpoint_lora_paths(self):
    self.assertTrue(
        hackable_adapter._is_qwix_or_official_lora_path(  # pylint: disable=protected-access
            "layer_0/attn/q_einsum/w_lora_a"
        )
    )
    self.assertTrue(
        hackable_adapter._is_qwix_or_official_lora_path(  # pylint: disable=protected-access
            "layer_0/attn/q_einsum/w_lora_b"
        )
    )
    self.assertTrue(
        hackable_adapter._is_qwix_or_official_lora_path(  # pylint: disable=protected-access
            "layer_0/attn/q_einsum/lora/a"
        )
    )
    self.assertFalse(
        hackable_adapter._is_qwix_or_official_lora_path(  # pylint: disable=protected-access
            "layer_0/attn/q_einsum/w"
        )
    )

  def test_qwix_lora_backend_rejects_qlora(self):
    config = hackable_adapter.OfficialSFTConfig(lora_backend="qwix_qlora")

    with self.assertRaisesRegex(ValueError, "lora_backend"):
      hackable_adapter._validate_lora_backend(config)  # pylint: disable=protected-access

  def test_qwix_apply_updates_keeps_base_frozen(self):
    original_apply_updates = optax.apply_updates
    original_patch_flag = hackable_adapter._QWIX_OPTAX_PATCHED  # pylint: disable=protected-access
    original_optax_marker = getattr(
        optax,
        "_tunix_qwix_apply_updates_patched",
        None,
    )
    original_optax_apply_updates = getattr(
        optax,
        "_tunix_original_apply_updates",
        None,
    )
    if hasattr(optax, "_tunix_qwix_apply_updates_patched"):
      delattr(optax, "_tunix_qwix_apply_updates_patched")
    if hasattr(optax, "_tunix_original_apply_updates"):
      delattr(optax, "_tunix_original_apply_updates")
    hackable_adapter._QWIX_OPTAX_PATCHED = False  # pylint: disable=protected-access
    try:
      hackable_adapter._patch_qwix_optax_apply_updates()  # pylint: disable=protected-access
      int_param = jnp.array([1, 2], dtype=jnp.int32)

      def _constant_loss(value):
        del value
        return jnp.asarray(0.0, dtype=jnp.float32)

      float0_update = jax.grad(_constant_loss, allow_int=True)(int_param)
      params = {
          "layer_0": {
              "attn": {
                  "q_einsum": {
                      "kernel": int_param,
                      "kernel_lora_a": jnp.ones((2,), dtype=jnp.float32),
                  }
              }
          }
      }
      updates = {
          "layer_0": {
              "attn": {
                  "q_einsum": {
                      "kernel": float0_update,
                      "kernel_lora_a": jnp.ones(
                          (2,),
                          dtype=jnp.float32,
                      ),
                  }
              }
          }
      }

      updated = optax.apply_updates(params, updates)

      np.testing.assert_array_equal(
          np.asarray(updated["layer_0"]["attn"]["q_einsum"]["kernel"]),
          np.asarray(int_param),
      )
      np.testing.assert_allclose(
          np.asarray(updated["layer_0"]["attn"]["q_einsum"]["kernel_lora_a"]),
          np.asarray([2.0, 2.0], dtype=np.float32),
      )
    finally:
      optax.apply_updates = original_apply_updates
      hackable_adapter._QWIX_OPTAX_PATCHED = original_patch_flag  # pylint: disable=protected-access
      if original_optax_marker is None:
        if hasattr(optax, "_tunix_qwix_apply_updates_patched"):
          delattr(optax, "_tunix_qwix_apply_updates_patched")
      else:
        optax._tunix_qwix_apply_updates_patched = original_optax_marker
      if original_optax_apply_updates is None:
        if hasattr(optax, "_tunix_original_apply_updates"):
          delattr(optax, "_tunix_original_apply_updates")
      else:
        optax._tunix_original_apply_updates = original_optax_apply_updates

  def test_sft_loss_is_finite(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
        self_cond_prob=1.0,
    )
    batch = _make_batch(vocab_size=vocab_size)
    loss, aux = diffusion_sft.make_loss_fn(cfg)(
        model, **diffusion_sft.gen_model_input_fn(batch)
    )
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertIn("decoder_loss", aux)
    self.assertIn("encoder_loss", aux)

  def test_split_loss_gradients_match_full_loss_gradients(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    model = diffusion_sft.apply_lora(model, rank=2, alpha=4.0)
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
        self_cond_prob=1.0,
        decoder_implementation="cached_selected_canvas_slice",
        encoder_loss_chunk_size=3,
    )
    batch = _make_batch(vocab_size=vocab_size)
    inputs = diffusion_sft.gen_model_input_fn(batch)

    grad_arg = nnx.DiffState(0, nnx.LoRAParam)
    decoder_config = dataclasses.replace(
        cfg, encoder_loss_weight=0.0, force_full_encoder_prefill=True
    )
    full_grad_fn = nnx.value_and_grad(
        diffusion_sft.make_loss_fn(cfg),
        argnums=grad_arg,
        has_aux=True,
    )
    decoder_grad_fn = nnx.value_and_grad(
        diffusion_sft.make_loss_fn(decoder_config),
        argnums=grad_arg,
        has_aux=True,
    )
    encoder_grad_fn = nnx.value_and_grad(
        diffusion_sft.make_loss_fn(
            dataclasses.replace(cfg, decoder_loss_weight=0.0)
        ),
        argnums=grad_arg,
        has_aux=True,
    )
    sc_logits, do_self_cond = (
        diffusion_sft.diffusion_gemma_sft_self_conditioning_logits(
            model,
            prompt=inputs["prompt"],
            canvas=inputs["canvas"],
            canvas_mask=inputs["canvas_mask"],
            rng=inputs["rng"],
            config=decoder_config,
        )
    )

    def precomputed_decoder_loss_fn(
        model,
        prompt,
        canvas,
        canvas_id,
        canvas_mask,
        encoder_target,
        encoder_target_mask,
        rng,
    ):
      return diffusion_sft.diffusion_gemma_sft_loss(
          model,
          prompt=prompt,
          canvas=canvas,
          canvas_id=canvas_id,
          canvas_mask=canvas_mask,
          encoder_target=encoder_target,
          encoder_target_mask=encoder_target_mask,
          rng=rng,
          config=decoder_config,
          precomputed_sc_logits=sc_logits,
          precomputed_self_conditioning_mask=do_self_cond,
      )

    precomputed_decoder_grad_fn = nnx.value_and_grad(
        precomputed_decoder_loss_fn,
        argnums=grad_arg,
        has_aux=True,
    )
    (full_loss, _), full_grads = full_grad_fn(model, **inputs)
    (decoder_loss, _), decoder_grads = decoder_grad_fn(model, **inputs)
    (precomputed_decoder_loss, _), precomputed_decoder_grads = (
        precomputed_decoder_grad_fn(model, **inputs)
    )
    (encoder_loss, _), encoder_grads = encoder_grad_fn(model, **inputs)
    split_grads = jax.tree.map(lambda x, y: x + y, decoder_grads, encoder_grads)

    np.testing.assert_allclose(
        full_loss, decoder_loss + encoder_loss, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        decoder_loss, precomputed_decoder_loss, rtol=2e-5, atol=2e-5
    )
    for decoder_grad, precomputed_grad in zip(
        jax.tree.leaves(decoder_grads),
        jax.tree.leaves(precomputed_decoder_grads),
    ):
      np.testing.assert_allclose(
          decoder_grad, precomputed_grad, rtol=2e-5, atol=2e-5
      )
    for full_grad, split_grad in zip(
        jax.tree.leaves(full_grads), jax.tree.leaves(split_grads)
    ):
      np.testing.assert_allclose(full_grad, split_grad, rtol=2e-5, atol=2e-5)

  def test_masked_ce_matches_optax_reference(self):
    logits = jnp.array(
        [
            [[1.0, -1.0, 0.5], [0.25, 0.5, -0.75]],
            [[-0.5, 1.5, 0.0], [0.75, -0.25, 0.125]],
        ],
        dtype=jnp.float32,
    )
    targets = jnp.array([[0, 2], [1, 0]], dtype=jnp.int32)
    mask = jnp.array([[True, False], [True, True]], dtype=jnp.bool_)

    actual = diffusion_sft._masked_ce_loss(  # pylint: disable=protected-access
        logits, targets, mask
    )
    expected = _official_reference_masked_ce(logits, targets, mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  def test_cached_decoder_updates_cache_at_per_example_end_index(self):
    vocab_size = 32
    config = diffusion_model.ModelConfig.tiny(vocab_size=vocab_size)
    config = dataclasses.replace(
        config,
        attention_pattern=(gemma4_model.AttentionType.GLOBAL,),
        use_sliding_window_kv_cache=False,
    )
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        config,
        rngs=nnx.Rngs(0),
    )
    cache = model.init_cache(batch_size=2, max_seq_len=6, dtype=jnp.float32)
    cache = diffusion_sft.set_cache_end_index(
        cache, jnp.array([1, 3], dtype=jnp.int32)
    )
    _, new_cache = model(
        jnp.array([[5, 6], [7, 8]], dtype=jnp.int32),
        positions=jnp.array([[10, 11], [20, 21]], dtype=jnp.int32),
        cache=cache,
        attention_mask=jnp.ones((2, 2, 6), dtype=jnp.bool_),
    )
    self.assertIsNotNone(new_cache)
    for layer_cache in new_cache.values():
      np.testing.assert_array_equal(layer_cache["end_index"], [3, 5])
      np.testing.assert_array_equal(layer_cache["positions"][0, 1:3], [10, 11])
      np.testing.assert_array_equal(layer_cache["positions"][1, 3:5], [20, 21])

  def test_cached_decoder_returns_canvas_only_logits(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
        self_cond_prob=1.0,
    )
    batch = _make_batch(vocab_size=vocab_size)
    x0_tokens = batch.canvas
    selected_canvas_idx = jnp.array([0, 1], dtype=jnp.int32)
    encoder_logits, kv_cache, positions, prompt_mask = diffusion_sft.sft_encode(
        model,
        prompt=batch.prompt,
        x0_tokens=x0_tokens,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )
    del encoder_logits
    kv_cache = diffusion_sft.set_cache_end_index(
        kv_cache, cfg.prompt_len + selected_canvas_idx * cfg.canvas_size
    )
    logits = diffusion_sft.sft_decode_cached_selected_canvas(
        model,
        xt=x0_tokens,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )
    self.assertEqual(logits.shape, (2, cfg.total_canvas_len, vocab_size))

  def test_sft_encode_can_skip_full_encoder_logits(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
    )
    batch = _make_batch(vocab_size=vocab_size)
    selected_canvas_idx = jnp.array([0, 1], dtype=jnp.int32)
    full_logits, full_cache, full_positions, full_prompt_mask = (
        diffusion_sft.sft_encode(
            model,
            prompt=batch.prompt,
            x0_tokens=batch.canvas,
            canvas_mask=batch.canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            config=cfg,
        )
    )
    last_logits, last_cache, last_positions, last_prompt_mask = (
        diffusion_sft.sft_encode(
            model,
            prompt=batch.prompt,
            x0_tokens=batch.canvas,
            canvas_mask=batch.canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            config=cfg,
            return_encoder_logits=False,
        )
    )

    self.assertEqual(
        full_logits.shape,
        (2, cfg.prompt_len + cfg.total_canvas_len, vocab_size),
    )
    self.assertEqual(last_logits.shape, (2, 1, vocab_size))
    self.assertEqual(full_cache.keys(), last_cache.keys())
    np.testing.assert_array_equal(full_positions, last_positions)
    np.testing.assert_array_equal(full_prompt_mask, last_prompt_mask)

  def test_forward_hidden_matches_logits_call(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
    )
    batch = _make_batch(vocab_size=vocab_size)
    tokens = jnp.concatenate([batch.prompt, batch.canvas], axis=1)
    token_mask = jnp.ones_like(tokens, dtype=jnp.bool_)
    positions = diffusion_sft.build_positions_from_mask(token_mask)
    attention_mask = diffusion_sft.make_causal_prefill_mask(
        token_mask, tokens.shape[1]
    )
    cache = model.init_cache(
        batch_size=tokens.shape[0],
        max_seq_len=tokens.shape[1],
        dtype=jnp.float32,
    )

    logits, _ = model(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
    )
    hidden, _ = model.forward_hidden(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
    )
    np.testing.assert_allclose(
        model.decode_hidden(hidden),
        logits,
        rtol=0,
        atol=0,
    )

  def test_chunked_encoder_loss_matches_full_logits_loss(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    batch = _make_batch(vocab_size=vocab_size)
    selected_canvas_idx = jnp.array([0, 1], dtype=jnp.int32)
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
    )
    logits, _, _, _ = diffusion_sft.sft_encode(
        model,
        prompt=batch.prompt,
        x0_tokens=batch.canvas,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )
    hidden, _, _, _ = diffusion_sft.sft_encode(
        model,
        prompt=batch.prompt,
        x0_tokens=batch.canvas,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
        return_encoder_hidden=True,
    )

    full_loss = diffusion_sft._masked_ce_loss(  # pylint: disable=protected-access
        logits, batch.encoder_target, batch.encoder_target_mask
    )
    chunked_loss = diffusion_sft._masked_ce_loss_from_hidden(  # pylint: disable=protected-access
        model,
        hidden,
        batch.encoder_target,
        batch.encoder_target_mask,
        chunk_size=3,
    )
    np.testing.assert_allclose(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)

  def test_cached_slice_decoder_matches_full_for_first_canvas(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
        self_cond_prob=1.0,
    )
    batch = _make_batch(vocab_size=vocab_size)
    selected_canvas_idx = jnp.array([0, 0], dtype=jnp.int32)
    encoder_logits, kv_cache, positions, prompt_mask = diffusion_sft.sft_encode(
        model,
        prompt=batch.prompt,
        x0_tokens=batch.canvas,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )
    del encoder_logits
    kv_cache = diffusion_sft.set_cache_end_index(
        kv_cache, cfg.prompt_len + selected_canvas_idx * cfg.canvas_size
    )
    full_logits = diffusion_sft.sft_decode_cached_selected_canvas(
        model,
        xt=batch.canvas,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )

    _, kv_cache, positions, prompt_mask = diffusion_sft.sft_encode(
        model,
        prompt=batch.prompt,
        x0_tokens=batch.canvas,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )
    kv_cache = diffusion_sft.set_cache_end_index(
        kv_cache, cfg.prompt_len + selected_canvas_idx * cfg.canvas_size
    )
    slice_logits = diffusion_sft.sft_decode_cached_selected_canvas_slice(
        model,
        xt=batch.canvas,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=batch.canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        config=cfg,
    )
    np.testing.assert_allclose(
        full_logits[:, : cfg.canvas_size],
        slice_logits,
        rtol=0,
        atol=0,
    )

  def test_slice_decoder_sft_loss_is_finite(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
        self_cond_prob=1.0,
        decoder_implementation="cached_selected_canvas_slice",
    )
    batch = _make_batch(vocab_size=vocab_size)
    loss, aux = diffusion_sft.make_loss_fn(cfg)(
        model, **diffusion_sft.gen_model_input_fn(batch)
    )
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertEqual(
        int(jax.device_get(aux["decoder_implementation"])),
        1,
    )

  def test_official_helper_parity(self):
    prompt_mask = jnp.array(
        [[True, True, False], [True, False, False]], dtype=jnp.bool_
    )
    canvas_mask = jnp.array(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=jnp.bool_,
    )
    selected_canvas_idx = jnp.array([1, 0], dtype=jnp.int32)

    np.testing.assert_array_equal(
        diffusion_sft.build_positions_from_mask(prompt_mask),
        jnp.array([[0, 1, 1], [0, 0, 0]], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        diffusion_sft.make_causal_prefill_mask(prompt_mask, cache_length=5),
        jnp.array([
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
            ],
            [
                [True, False, False, False, False],
                [True, False, False, False, False],
                [True, False, False, False, False],
            ],
        ]),
    )

    mask = diffusion_sft.create_decoder_attention_mask(
        prompt_mask,
        canvas_mask,
        selected_canvas_idx,
        prompt_len=3,
        total_canvas_len=6,
        canvas_size=2,
        num_queries=6,
    )
    expected_row_0 = jnp.array(
        [True, True, False, True, True, True, True, False, False]
    )
    expected_row_1 = jnp.array(
        [True, False, False, True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(mask[0, 0], expected_row_0)
    np.testing.assert_array_equal(mask[1, 0], expected_row_1)

    fake_cache = {
        "layer_0": {
            "k": jnp.ones((2, 4, 1, 2)),
            "v": jnp.zeros((2, 4, 1, 2)),
            "end_index": jnp.zeros((2,), dtype=jnp.int32),
        }
    }
    updated_cache = diffusion_sft.set_cache_end_index(
        fake_cache, jnp.array([3, 5], dtype=jnp.int32)
    )
    np.testing.assert_array_equal(updated_cache["layer_0"]["end_index"], [3, 5])
    np.testing.assert_array_equal(
        updated_cache["layer_0"]["k"], fake_cache["layer_0"]["k"]
    )

  def test_official_sampling_corruption_and_loss_parity(self):
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=3,
        num_canvases=3,
        vocab_size=17,
    )
    canvas_mask = jnp.array(
        [
            [True, True, True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False, False, False],
        ],
        dtype=jnp.bool_,
    )
    selected = diffusion_sft._sample_selected_canvas(  # pylint: disable=protected-access
        jax.random.PRNGKey(12), canvas_mask, cfg
    )
    num_valid = jnp.array([2, 1], dtype=jnp.int32)
    expected_selected = jax.random.randint(
        jax.random.PRNGKey(12),
        shape=num_valid.shape,
        minval=0,
        maxval=num_valid,
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(selected, expected_selected)

    x0_tokens = jnp.array(
        [[1, 2, 3, 4, 5, 6, 0, 0, 0], [6, 5, 4, 0, 0, 0, 0, 0, 0]],
        dtype=jnp.int32,
    )
    time = jnp.array([[0.2], [0.8]], dtype=jnp.float32)
    actual_xt, actual_is_corrupted = diffusion_sft._corrupt_tokens(  # pylint: disable=protected-access
        jax.random.PRNGKey(23), x0_tokens, time, cfg.vocab_size
    )
    expected_xt, expected_is_corrupted = _official_reference_corrupt_tokens(
        jax.random.PRNGKey(23), x0_tokens, time, cfg.vocab_size
    )
    np.testing.assert_array_equal(actual_xt, expected_xt)
    np.testing.assert_array_equal(actual_is_corrupted, expected_is_corrupted)

    fast_xt, fast_is_corrupted = diffusion_sft._corrupt_tokens(  # pylint: disable=protected-access
        jax.random.PRNGKey(23),
        x0_tokens,
        time,
        cfg.vocab_size,
        fast_uniform=True,
    )
    self.assertEqual(fast_xt.shape, x0_tokens.shape)
    self.assertEqual(fast_is_corrupted.shape, x0_tokens.shape)
    self.assertTrue(bool(jnp.all(fast_xt >= 0)))
    self.assertTrue(bool(jnp.all(fast_xt < cfg.vocab_size)))

    logits = jnp.arange(
        x0_tokens.size * cfg.vocab_size, dtype=jnp.float32
    ).reshape(x0_tokens.shape + (cfg.vocab_size,))
    varied_mask = jnp.array(
        [
            [True, True, False, False, False, False, False, False, False],
            [True, True, True, True, False, False, False, False, False],
        ],
        dtype=jnp.bool_,
    )
    actual_loss = diffusion_sft._masked_ce_loss(  # pylint: disable=protected-access
        logits, x0_tokens, varied_mask
    )
    expected_loss = _official_reference_masked_ce(
        logits, x0_tokens, varied_mask
    )
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)

  def test_self_conditioning_matches_official_unscaled_post_norm(self):
    vocab_size = 11
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    self.assertFalse(hasattr(model.self_conditioner.post_norm, "scale"))

    logits = jnp.arange(2 * 3 * vocab_size, dtype=jnp.float32).reshape(
        2, 3, vocab_size
    )
    embeddings = model.embedder.input_embedding.value
    expected = jnp.einsum(
        "...v,ve->...e", jax.nn.softmax(logits, axis=-1), embeddings
    )
    expected *= jnp.sqrt(model.config.embed_dim).astype(expected.dtype)
    np.testing.assert_allclose(
        model.encode_logits(logits), expected, rtol=1e-5, atol=1e-5
    )

  def test_lora_trainer_updates_only_lora_params(self):
    vocab_size = 32
    model = diffusion_model.DiffusionGemma_A26B_A4B(
        diffusion_model.ModelConfig.tiny(vocab_size=vocab_size),
        rngs=nnx.Rngs(0),
    )
    model = diffusion_sft.apply_lora(
        model,
        rank=2,
        alpha=4.0,
        module_path=r".*gate_proj|.*up_proj|.*down_proj",
    )
    cfg = diffusion_sft.DiffusionGemmaSFTConfig(
        prompt_len=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=vocab_size,
        self_cond_prob=1.0,
    )
    train_cfg = peft_trainer.TrainingConfig(
        eval_every_n_steps=1,
        max_steps=1,
        max_inflight_computations=1,
        pbar_description=None,
    )
    before_base = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
    )
    before_lora = jax.tree.map(jnp.copy, nnx.state(model, nnx.LoRAParam))
    trainer = diffusion_sft.DiffusionGemmaTrainer(
        model, optax.adamw(1e-3), train_cfg, cfg
    )
    trainer.train(
        [diffusion_sft.gen_model_input_fn(_make_batch(vocab_size=vocab_size))],
        cache_nnx_graph=False,
    )
    after_base = nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
    after_lora = nnx.state(model, nnx.LoRAParam)
    self.assertTrue(_all_equal(before_base, after_base))
    self.assertTrue(_any_changed(before_lora, after_lora))


if __name__ == "__main__":
  absltest.main()
