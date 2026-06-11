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

"""Tunix DiffusionGemma no-tuning text generation demo.

This script loads DiffusionGemma weights, runs a minimal diffusion denoising
sampler on top of the Tunix NNX model, and prints each denoising step so the
canvas refinement is visible.

The sampler intentionally uses a full-sequence no-cache denoising path. That is
the path covered by the official-vs-Tunix logits parity check. It is slower
than the official cached sampler, but it avoids claiming cache-decoder parity
that the MVP integration does not yet implement.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time
from typing import Any

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tunix.generate import tokenizer_adapter
from tunix.models.diffusion_gemma import model as diffusion_model
from tunix.models.diffusion_gemma import params as diffusion_params
from tunix.models.gemma4 import model as gemma4_model


DEFAULT_CHECKPOINT = diffusion_params.DIFFUSIONGEMMA_A26B_A4B_IT
DEFAULT_TOKENIZER = "gs://gemma-data/tokenizers/tokenizer_gemma4.model"
END_TOKENS = (1, 106, 50)  # EOS, Gemma4 end-of-turn, tool-response boundary.
VISIBLE_SPECIAL_TOKENS = {
    0: "<pad>",
    1: "<eos>",
    2: "<bos>",
    3: "<unk>",
    4: "<mask>",
    50: "<|tool_response>",
    105: "<|turn>",
    106: "<turn|>",
}


@dataclasses.dataclass(frozen=True)
class StepTrace:
  step: int
  noise: float
  changed_tokens: int
  accepted_tokens: int
  text: str
  token_ids: list[int]


def _dtype_from_name(name: str):
  return {
      "bfloat16": jnp.bfloat16,
      "float16": jnp.float16,
      "float32": jnp.float32,
  }[name]


def _mesh(fsdp: int, tp: int) -> jax.sharding.Mesh:
  return jax.make_mesh(
      (fsdp, tp),
      ("fsdp", "tp"),
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
  )


def _format_prompt(prompt: str, *, chat_template: bool) -> str:
  if not chat_template:
    return prompt
  return f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"


def _load_tokenizer(path: str, *, add_bos: bool = False, add_eos: bool = False):
  return tokenizer_adapter.Tokenizer(
      "sentencepiece",
      path,
      add_bos=add_bos,
      add_eos=add_eos,
  )


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
  visible = []
  normal = []
  for token_id in token_ids:
    if token_id in VISIBLE_SPECIAL_TOKENS:
      if normal:
        visible.append(tokenizer.decode(normal))
        normal = []
      visible.append(VISIBLE_SPECIAL_TOKENS[token_id])
    else:
      normal.append(token_id)
  if normal:
    visible.append(tokenizer.decode(normal))
  return "".join(visible)


def _truncate_at_stop(token_ids: np.ndarray) -> np.ndarray:
  stop = np.isin(token_ids, np.asarray(END_TOKENS, dtype=np.int32))
  if not np.any(stop):
    return token_ids
  return token_ids[: int(np.argmax(stop)) + 1]


def _make_attention_mask(
    batch_size: int,
    context_len: int,
    canvas_len: int,
) -> jax.Array:
  full_len = context_len + canvas_len
  context_causal = jnp.tril(
      jnp.ones((context_len, context_len), dtype=jnp.bool_)
  )
  mask = jnp.zeros((batch_size, full_len, full_len), dtype=jnp.bool_)
  if context_len:
    mask = mask.at[:, :context_len, :context_len].set(context_causal)
  canvas_rows = jnp.ones((batch_size, canvas_len, full_len), dtype=jnp.bool_)
  mask = mask.at[:, context_len:, :].set(canvas_rows)
  return mask


def _temperature_shape(
    logits: jax.Array,
    noise_proportion: float,
    *,
    min_temperature: float,
    max_temperature: float,
    exponent: float,
) -> jax.Array:
  noise = jnp.asarray(noise_proportion, dtype=logits.dtype)
  temp_fraction = 1.0 - (1.0 - noise) ** exponent
  temperature = (
      temp_fraction * (max_temperature - min_temperature) + min_temperature
  )
  return logits / temperature


def _sample_from_predictions(
    rng: jax.Array,
    logits: jax.Array,
    canvas: jax.Array,
    *,
    entropy_bound: float,
    vocab_size: int,
) -> tuple[jax.Array, jax.Array]:
  categorical_rng, noise_rng = jax.random.split(rng)
  denoised = jax.random.categorical(
      categorical_rng, logits.astype(jnp.float32)
  ).astype(canvas.dtype)

  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
  probs = jnp.exp(log_probs)
  safe_log_probs = jnp.where(probs == 0, 0.0, log_probs)
  token_entropy = -jnp.sum(safe_log_probs * probs, axis=-1)

  sorted_idx = jnp.argsort(token_entropy, axis=-1)
  sorted_entropy = jnp.take_along_axis(token_entropy, sorted_idx, axis=-1)
  accumulated_entropy = jnp.cumsum(sorted_entropy, axis=-1)
  sorted_selection = accumulated_entropy - sorted_entropy <= entropy_bound
  selection = (
      jnp.zeros_like(sorted_idx, dtype=jnp.bool_)
      .at[jnp.arange(canvas.shape[0])[:, None], sorted_idx]
      .set(sorted_selection)
  )

  random_tokens = jax.random.randint(
      noise_rng,
      shape=canvas.shape,
      minval=0,
      maxval=vocab_size,
      dtype=canvas.dtype,
  )
  sampled = jnp.where(selection, denoised, random_tokens)
  return sampled, selection


@nnx.jit
def _full_sequence_logits(
    model,
    tokens,
    positions,
    attention_mask,
    sc_logits,
    self_conditioning_mask,
):
  logits, _ = model(
      tokens,
      positions=positions,
      attention_mask=attention_mask,
      sc_logits=sc_logits,
      self_conditioning_mask=self_conditioning_mask,
  )
  return logits


def _load_real_model(args):
  dtype = _dtype_from_name(args.dtype)
  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError(
        "mesh_fsdp * mesh_tp must equal jax.device_count(). Got "
        f"{args.mesh_fsdp} * {args.mesh_tp} != {jax.device_count()}."
    )
  mesh = _mesh(args.mesh_fsdp, args.mesh_tp)
  with mesh:
    config = diffusion_model.ModelConfig.diffusion_gemma_a26b_a4b()
    return diffusion_params.create_model_from_checkpoint(
        args.checkpoint,
        config,
        mesh=mesh,
        dtype=dtype,
    )


def _load_tiny_model(args, vocab_size: int):
  config = diffusion_model.ModelConfig.tiny(
      vocab_size=vocab_size,
      num_layers=1,
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      num_kv_heads=1,
  )
  config = dataclasses.replace(
      config,
      attention_pattern=(gemma4_model.AttentionType.GLOBAL,),
      num_global_kv_heads=1,
      global_key_size=8,
      use_sliding_window_kv_cache=False,
      final_logit_softcap=None,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  del args
  return diffusion_model.DiffusionGemma_A26B_A4B(config, rngs=nnx.Rngs(0))


def generate(args) -> dict[str, Any]:
  t0 = time.time()
  tokenizer = _load_tokenizer(args.tokenizer)
  vocab_size = tokenizer.tokenizer.GetPieceSize()
  prompt_text = _format_prompt(
      args.prompt, chat_template=not args.no_chat_template
  )
  prompt_ids = [tokenizer.bos_id()] + tokenizer.encode(prompt_text)
  if args.tiny:
    # Tiny mode is just a local sampler sanity check; keep token IDs in range.
    prompt_ids = [token_id % args.tiny_vocab_size for token_id in prompt_ids]
    vocab_size = args.tiny_vocab_size

  print(
      json.dumps(
          {
              "event": "prompt",
              "prompt": args.prompt,
              "formatted_prompt": prompt_text,
              "prompt_token_count": len(prompt_ids),
              "vocab_size": vocab_size,
          },
          ensure_ascii=False,
      ),
      flush=True,
  )

  if args.tiny:
    model = _load_tiny_model(args, vocab_size)
  else:
    print(
        json.dumps({
            "event": "loading_model",
            "checkpoint": args.checkpoint,
            "dtype": args.dtype,
        }),
        flush=True,
    )
    model = _load_real_model(args)
  print(
      json.dumps({
          "event": "model_loaded",
          "seconds": round(time.time() - t0, 3),
          "jax_devices": [str(device) for device in jax.devices()],
      }),
      flush=True,
  )

  rng = jax.random.PRNGKey(args.seed)
  context = jnp.asarray(prompt_ids, dtype=jnp.int32)[None, :]
  traces: list[StepTrace] = []
  generated: list[int] = []
  done = False

  num_canvases = int(np.ceil(args.max_new_tokens / args.canvas_length))
  for canvas_idx in range(num_canvases):
    remaining = args.max_new_tokens - len(generated)
    if remaining <= 0 or done:
      break
    canvas_len = min(args.canvas_length, remaining)
    rng, canvas_rng = jax.random.split(rng)
    canvas = jax.random.randint(
        canvas_rng,
        shape=(1, canvas_len),
        minval=0,
        maxval=vocab_size,
        dtype=jnp.int32,
    )
    sc_canvas_logits = jnp.zeros((1, canvas_len, vocab_size), dtype=jnp.float32)
    context_len = context.shape[1]
    full_len = context_len + canvas_len
    positions = jnp.broadcast_to(
        jnp.arange(full_len, dtype=jnp.int32)[None, :], (1, full_len)
    )
    attention_mask = _make_attention_mask(1, context_len, canvas_len)
    context_sc_logits = jnp.zeros(
        (1, context_len, vocab_size), dtype=sc_canvas_logits.dtype
    )
    sc_mask = jnp.concatenate(
        [
            jnp.zeros((1, context_len), dtype=jnp.bool_),
            jnp.ones((1, canvas_len), dtype=jnp.bool_),
        ],
        axis=1,
    )

    initial_text = _decode_tokens(
        tokenizer, _truncate_at_stop(np.asarray(canvas[0])).tolist()
    )
    print(
        json.dumps(
            {
                "event": "canvas_initial",
                "canvas": canvas_idx,
                "text": initial_text,
                "token_ids": np.asarray(canvas[0]).tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for step in range(args.denoising_steps):
      rng, step_rng = jax.random.split(rng)
      noise = 1.0 - step / args.denoising_steps
      full_tokens = jnp.concatenate([context, canvas], axis=1)
      full_sc_logits = jnp.concatenate(
          [context_sc_logits, sc_canvas_logits], axis=1
      )
      logits = _full_sequence_logits(
          model,
          full_tokens,
          positions,
          attention_mask,
          full_sc_logits,
          sc_mask,
      )[:, context_len:, :]
      shaped_logits = _temperature_shape(
          logits,
          noise,
          min_temperature=args.min_temperature,
          max_temperature=args.max_temperature,
          exponent=args.temperature_exponent,
      )
      new_canvas, selected = _sample_from_predictions(
          step_rng,
          shaped_logits,
          canvas,
          entropy_bound=args.entropy_bound,
          vocab_size=vocab_size,
      )
      changed = int(np.sum(np.asarray(new_canvas != canvas)))
      accepted = int(np.sum(np.asarray(selected)))
      canvas = new_canvas
      sc_canvas_logits = jax.lax.stop_gradient(shaped_logits)

      visible_ids = _truncate_at_stop(np.asarray(canvas[0])).tolist()
      text = _decode_tokens(tokenizer, visible_ids)
      trace = StepTrace(
          step=step + 1,
          noise=float(noise),
          changed_tokens=changed,
          accepted_tokens=accepted,
          text=text,
          token_ids=visible_ids,
      )
      traces.append(trace)
      print(
          json.dumps(
              {
                  "event": "denoise_step",
                  "canvas": canvas_idx,
                  **dataclasses.asdict(trace),
              },
              ensure_ascii=False,
          ),
          flush=True,
      )

    final_canvas = _truncate_at_stop(np.asarray(canvas[0]).astype(np.int32))
    generated.extend(final_canvas.tolist())
    context = jnp.concatenate(
        [context, jnp.asarray(final_canvas, dtype=jnp.int32)[None, :]], axis=1
    )
    if np.any(np.isin(final_canvas, np.asarray(END_TOKENS, dtype=np.int32))):
      done = True

  output_ids = _truncate_at_stop(np.asarray(generated, dtype=np.int32)).tolist()
  output_text = _decode_tokens(tokenizer, output_ids)
  report = {
      "event": "generation_complete",
      "tiny": args.tiny,
      "checkpoint": None if args.tiny else args.checkpoint,
      "prompt": args.prompt,
      "output_text": output_text,
      "output_token_ids": output_ids,
      "num_traces": len(traces),
      "seconds": round(time.time() - t0, 3),
  }
  print(json.dumps(report, ensure_ascii=False), flush=True)
  return report


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
  parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
  parser.add_argument("--prompt", default="What are diffusion LLMs?")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max_new_tokens", type=int, default=32)
  parser.add_argument("--canvas_length", type=int, default=32)
  parser.add_argument("--denoising_steps", type=int, default=8)
  parser.add_argument("--entropy_bound", type=float, default=0.1)
  parser.add_argument("--min_temperature", type=float, default=0.4)
  parser.add_argument("--max_temperature", type=float, default=0.8)
  parser.add_argument("--temperature_exponent", type=float, default=1.0)
  parser.add_argument(
      "--dtype",
      choices=("bfloat16", "float16", "float32"),
      default="bfloat16",
  )
  parser.add_argument("--mesh_fsdp", type=int, default=1)
  parser.add_argument("--mesh_tp", type=int, default=1)
  parser.add_argument("--no_chat_template", action="store_true")
  parser.add_argument("--tiny", action="store_true")
  parser.add_argument("--tiny_vocab_size", type=int, default=256)
  return parser.parse_args()


def main() -> None:
  generate(parse_args())


if __name__ == "__main__":
  main()
