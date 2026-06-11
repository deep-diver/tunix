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

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.models import automodel
from tunix.models import naming
from tunix.models.diffusion_gemma import model as diffusion_model
from tunix.models.diffusion_gemma import sft as diffusion_sft
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


class DiffusionGemmaTest(absltest.TestCase):

  def test_model_naming_and_config(self):
    info = naming.ModelNaming(model_name="diffusion-gemma-a26b-a4b-it")
    self.assertEqual(info.model_family, "diffusion_gemma")
    self.assertEqual(info.model_config_category, "diffusion_gemma")
    cfg = automodel.call_model_config("diffusion-gemma-a26b-a4b-it")
    self.assertEqual(cfg.num_embed, 262144)

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
