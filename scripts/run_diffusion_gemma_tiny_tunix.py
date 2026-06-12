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

"""Tiny DiffusionGemma SFT validation test for CPU/GPU/JarvisLabs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.models.diffusion_gemma import model as diffusion_model
from tunix.models.diffusion_gemma import sft as diffusion_sft
from tunix.sft import peft_trainer


def _tree_any_changed(before: Any, after: Any) -> bool:
  leaves = jax.tree.leaves(
      jax.tree.map(lambda x, y: jnp.any(x != y), before, after)
  )
  return any(bool(jax.device_get(x)) for x in leaves)


def _tree_all_equal(before: Any, after: Any) -> bool:
  leaves = jax.tree.leaves(
      jax.tree.map(lambda x, y: jnp.all(x == y), before, after)
  )
  return all(bool(jax.device_get(x)) for x in leaves)


def make_batch(
    *,
    seed: int,
    batch_size: int,
    prompt_len: int,
    canvas_size: int,
    num_canvases: int,
    vocab_size: int,
) -> diffusion_sft.DiffusionGemmaSFTBatch:
  rng = np.random.default_rng(seed)
  total_canvas_len = canvas_size * num_canvases
  prompt = rng.integers(
      1, vocab_size, size=(batch_size, prompt_len), dtype=np.int32
  )
  canvas = rng.integers(
      1, vocab_size, size=(batch_size, total_canvas_len), dtype=np.int32
  )
  canvas_id = np.tile(
      np.repeat(np.arange(num_canvases, dtype=np.int32), canvas_size),
      (batch_size, 1),
  )
  canvas_mask = np.ones((batch_size, total_canvas_len), dtype=np.bool_)
  full_seq = np.concatenate([prompt, canvas], axis=1)
  encoder_target = np.roll(full_seq, shift=-1, axis=1).astype(np.int32)
  encoder_target[:, -1] = 0
  encoder_target_mask = np.ones_like(full_seq, dtype=np.float32)
  encoder_target_mask[:, -1] = 0.0
  return diffusion_sft.DiffusionGemmaSFTBatch(
      prompt=jnp.asarray(prompt),
      canvas=jnp.asarray(canvas),
      canvas_id=jnp.asarray(canvas_id),
      canvas_mask=jnp.asarray(canvas_mask),
      encoder_target=jnp.asarray(encoder_target),
      encoder_target_mask=jnp.asarray(encoder_target_mask),
      rng=jax.random.PRNGKey(seed + 10_000),
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--prompt_len", type=int, default=8)
  parser.add_argument("--canvas_size", type=int, default=8)
  parser.add_argument("--num_canvases", type=int, default=2)
  parser.add_argument("--vocab_size", type=int, default=96)
  parser.add_argument(
      "--use_lora", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--checkpoint_dir", type=str, default="")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  devices = [str(d) for d in jax.devices()]
  print(json.dumps({"event": "devices", "devices": devices}))

  model_config = diffusion_model.ModelConfig.tiny(vocab_size=args.vocab_size)
  model = diffusion_model.DiffusionGemma_A26B_A4B(
      model_config, rngs=nnx.Rngs(0)
  )
  if args.use_lora:
    model = diffusion_sft.apply_lora(
        model,
        rank=4,
        alpha=8.0,
        module_path=r".*gate_proj|.*up_proj|.*down_proj",
    )

  diffusion_config = diffusion_sft.DiffusionGemmaSFTConfig(
      prompt_len=args.prompt_len,
      canvas_size=args.canvas_size,
      num_canvases=args.num_canvases,
      vocab_size=args.vocab_size,
      self_cond_prob=1.0,
  )
  batch = make_batch(
      seed=0,
      batch_size=args.batch_size,
      prompt_len=args.prompt_len,
      canvas_size=args.canvas_size,
      num_canvases=args.num_canvases,
      vocab_size=args.vocab_size,
  )
  loss, aux = diffusion_sft.make_loss_fn(diffusion_config)(
      model, **diffusion_sft.gen_model_input_fn(batch)
  )
  loss.block_until_ready()
  if not bool(jnp.isfinite(loss)):
    raise RuntimeError(f"Initial loss is not finite: {loss}")
  print(
      json.dumps({
          "event": "initial_loss",
          "loss": float(jax.device_get(loss)),
          "decoder_loss": float(jax.device_get(aux["decoder_loss"])),
          "encoder_loss": float(jax.device_get(aux["encoder_loss"])),
      })
  )

  ckpt_dir = args.checkpoint_dir or tempfile.mkdtemp(
      prefix="diffusion_gemma_tunix_ckpt_"
  )
  before_base = None
  before_lora = None
  if args.use_lora:
    before_base = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
    )
    before_lora = jax.tree.map(jnp.copy, nnx.state(model, nnx.LoRAParam))

  train_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=max(1, args.steps),
      max_steps=args.steps,
      checkpoint_root_directory=ckpt_dir,
      max_inflight_computations=1,
      pbar_description=None,
  )
  trainer = diffusion_sft.DiffusionGemmaTrainer(
      model,
      optax.adamw(1e-3),
      train_config,
      diffusion_config,
  )
  train_ds = [
      diffusion_sft.gen_model_input_fn(
          make_batch(
              seed=i + 1,
              batch_size=args.batch_size,
              prompt_len=args.prompt_len,
              canvas_size=args.canvas_size,
              num_canvases=args.num_canvases,
              vocab_size=args.vocab_size,
          )
      )
      for i in range(args.steps)
  ]
  trainer.train(train_ds, cache_nnx_graph=False)

  result = {
      "event": "train_complete",
      "steps": trainer.train_steps,
      "checkpoint_dir": ckpt_dir,
      "checkpoint_dir_exists": pathlib.Path(ckpt_dir).exists(),
      "jax_platform": jax.default_backend(),
  }
  if args.use_lora:
    after_base = nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
    after_lora = nnx.state(model, nnx.LoRAParam)
    result["base_params_unchanged"] = _tree_all_equal(before_base, after_base)
    result["lora_params_changed"] = _tree_any_changed(before_lora, after_lora)
    if not result["base_params_unchanged"]:
      raise RuntimeError("Non-LoRA parameters changed in LoRA validation run.")
    if not result["lora_params_changed"]:
      raise RuntimeError("LoRA parameters did not change in LoRA validation run.")
  if not os.path.exists(ckpt_dir):
    raise RuntimeError(f"Checkpoint directory was not created: {ckpt_dir}")
  print(json.dumps(result))


if __name__ == "__main__":
  main()
