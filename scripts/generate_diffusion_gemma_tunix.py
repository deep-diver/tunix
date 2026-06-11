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
import html as html_lib
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
  canvas: int
  step: int
  phase: str
  noise: float
  target_noise: float
  changed_tokens: int
  accepted_tokens: int
  mean_entropy: float | None
  stable_tokens: bool
  low_entropy: bool
  early_stop: bool
  text: str
  token_ids: list[int]
  token_texts: list[str]
  selected_mask: list[bool]
  changed_mask: list[bool]


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


def _token_texts(tokenizer, token_ids: list[int]) -> list[str]:
  pieces = []
  for token_id in token_ids:
    if token_id in VISIBLE_SPECIAL_TOKENS:
      pieces.append(VISIBLE_SPECIAL_TOKENS[token_id])
    else:
      piece = tokenizer.decode([token_id])
      pieces.append(piece if piece else f"<id:{token_id}>")
  return pieces


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


def _token_entropy(logits: jax.Array) -> jax.Array:
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
  probs = jnp.exp(log_probs)
  safe_log_probs = jnp.where(probs == 0, 0.0, log_probs)
  return -jnp.sum(safe_log_probs * probs, axis=-1)


def _sample_from_predictions(
    rng: jax.Array,
    logits: jax.Array,
    canvas: jax.Array,
    *,
    entropy_bound: float,
    vocab_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  categorical_rng, noise_rng = jax.random.split(rng)
  denoised = jax.random.categorical(
      categorical_rng, logits.astype(jnp.float32)
  ).astype(canvas.dtype)

  token_entropy = _token_entropy(logits)

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
  return sampled, selection, denoised, token_entropy


def _early_stop_flags(
    logits: jax.Array,
    previous_canvas: jax.Array,
    *,
    entropy_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  most_likely = jnp.argmax(logits.astype(jnp.float32), axis=-1).astype(
      previous_canvas.dtype
  )
  stable_tokens = jnp.all(most_likely == previous_canvas, axis=-1)
  low_entropy = jnp.mean(_token_entropy(logits), axis=-1) <= entropy_threshold
  return stable_tokens, low_entropy, jnp.logical_and(stable_tokens, low_entropy)


def _make_trace(
    *,
    tokenizer,
    canvas_idx: int,
    step: int,
    phase: str,
    noise: float,
    target_noise: float,
    canvas: np.ndarray,
    previous_canvas: np.ndarray | None,
    selected_mask: np.ndarray | None,
    mean_entropy: float | None,
    stable_tokens: bool = False,
    low_entropy: bool = False,
    early_stop: bool = False,
) -> StepTrace:
  token_ids = canvas.astype(np.int32).tolist()
  if previous_canvas is None:
    changed_mask = np.zeros_like(canvas, dtype=np.bool_)
  else:
    changed_mask = canvas != previous_canvas
  if selected_mask is None:
    selected_mask = np.zeros_like(canvas, dtype=np.bool_)
  return StepTrace(
      canvas=canvas_idx,
      step=step,
      phase=phase,
      noise=float(noise),
      target_noise=float(target_noise),
      changed_tokens=int(np.sum(changed_mask)),
      accepted_tokens=int(np.sum(selected_mask)),
      mean_entropy=mean_entropy,
      stable_tokens=stable_tokens,
      low_entropy=low_entropy,
      early_stop=early_stop,
      text=_decode_tokens(tokenizer, token_ids),
      token_ids=token_ids,
      token_texts=_token_texts(tokenizer, token_ids),
      selected_mask=selected_mask.astype(np.bool_).tolist(),
      changed_mask=changed_mask.astype(np.bool_).tolist(),
  )


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


def _write_json(path: str | None, payload: dict[str, Any]) -> str | None:
  if not path:
    return None
  output_path = pathlib.Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
  )
  return str(output_path)


def _render_animation_html(payload: dict[str, Any]) -> str:
  data_json = html_lib.escape(json.dumps(payload, ensure_ascii=False))
  title = html_lib.escape(f"DiffusionGemma trace: {payload['prompt']}")
  return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f7f5ef;
  --panel: #ffffff;
  --ink: #1c2430;
  --muted: #627084;
  --line: #d8d4c8;
  --accepted: #2f8f6b;
  --changed: #c85739;
  --token: #f2eee4;
  --token-dark: #232b36;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #111418;
    --panel: #171c22;
    --ink: #eef2f4;
    --muted: #aab4bf;
    --line: #323943;
    --token: #222832;
    --token-dark: #2b3340;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
  width: min(1120px, calc(100vw - 32px));
  margin: 28px auto;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 24px;
  letter-spacing: 0;
}}
.subhead, .meta {{
  color: var(--muted);
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin-top: 14px;
}}
.controls {{
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  align-items: center;
  gap: 10px;
}}
button {{
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
}}
input[type="range"] {{
  width: 100%;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
  margin-top: 12px;
}}
.stat {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
}}
.stat b {{
  display: block;
  font-size: 18px;
}}
.canvas-text {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  min-height: 62px;
  border-left: 4px solid var(--accepted);
  padding-left: 12px;
  margin-top: 12px;
}}
.tokens {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(86px, 1fr));
  gap: 8px;
  margin-top: 14px;
}}
.token {{
  min-height: 54px;
  border: 1px solid var(--line);
  background: var(--token);
  border-radius: 6px;
  padding: 7px;
  overflow: hidden;
  transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
}}
.token.changed {{
  border-color: var(--changed);
  transform: translateY(-2px);
}}
.token.accepted {{
  box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--accepted) 70%, transparent);
}}
.piece {{
  display: block;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
}}
.id {{
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 11px;
}}
.legend {{
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 10px;
  color: var(--muted);
}}
.swatch {{
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 5px;
  vertical-align: -1px;
}}
.swatch.accepted {{ border: 2px solid var(--accepted); }}
.swatch.changed {{ border: 2px solid var(--changed); }}
@media (max-width: 720px) {{
  .controls {{ grid-template-columns: 1fr 1fr; }}
  .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<main>
  <h1>DiffusionGemma Denoising Trace</h1>
  <div class="subhead"></div>
  <section class="panel">
    <div class="controls">
      <button id="play" type="button">Play</button>
      <button id="prev" type="button">Prev</button>
      <input id="slider" type="range" min="0" value="0" step="1">
      <button id="next" type="button">Next</button>
    </div>
    <div class="stats">
      <div class="stat"><span>Frame</span><b id="frame"></b></div>
      <div class="stat"><span>Canvas</span><b id="canvas"></b></div>
      <div class="stat"><span>Noise</span><b id="noise"></b></div>
      <div class="stat"><span>Accepted</span><b id="accepted"></b></div>
      <div class="stat"><span>Entropy</span><b id="entropy"></b></div>
    </div>
    <div class="legend">
      <span><span class="swatch accepted"></span>selected by confidence</span>
      <span><span class="swatch changed"></span>changed this step</span>
      <span id="stop"></span>
    </div>
    <div id="text" class="canvas-text"></div>
    <div id="tokens" class="tokens"></div>
  </section>
  <section class="panel">
    <div class="meta" id="meta"></div>
  </section>
</main>
<script type="application/json" id="trace-data">{data_json}</script>
<script>
const payload = JSON.parse(document.getElementById('trace-data').textContent);
const frames = payload.frames || [];
const slider = document.getElementById('slider');
const play = document.getElementById('play');
let index = 0;
let timer = null;

function esc(s) {{
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}}

function fmt(v, digits = 3) {{
  return v === null || v === undefined ? '-' : Number(v).toFixed(digits);
}}

function render(i) {{
  if (!frames.length) return;
  index = Math.max(0, Math.min(i, frames.length - 1));
  const f = frames[index];
  slider.value = String(index);
  document.getElementById('frame').textContent = `${{index + 1}}/${{frames.length}}`;
  document.getElementById('canvas').textContent = `${{f.canvas}}:${{f.step}}`;
  document.getElementById('noise').textContent = `${{fmt(f.noise, 2)}} -> ${{fmt(f.target_noise, 2)}}`;
  document.getElementById('accepted').textContent = `${{f.accepted_tokens}}/${{f.token_ids.length}}`;
  document.getElementById('entropy').textContent = fmt(f.mean_entropy, 4);
  document.getElementById('stop').textContent = f.early_stop
    ? 'early stop: token-stable and low entropy'
    : '';
  document.getElementById('text').textContent = f.text || '';
  const tokens = document.getElementById('tokens');
  tokens.innerHTML = '';
  f.token_ids.forEach((id, pos) => {{
    const cell = document.createElement('div');
    cell.className = 'token';
    if (f.selected_mask[pos]) cell.classList.add('accepted');
    if (f.changed_mask[pos]) cell.classList.add('changed');
    const piece = f.token_texts[pos] ?? '';
    cell.innerHTML = `<span class="piece">${{esc(piece)}}</span><span class="id">#${{pos}} - ${{id}}</span>`;
    tokens.appendChild(cell);
  }});
}}

function stopTimer() {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
    play.textContent = 'Play';
  }}
}}

slider.max = String(Math.max(frames.length - 1, 0));
document.querySelector('.subhead').textContent = payload.prompt || '';
document.getElementById('meta').textContent =
  `output: ${{payload.output_text || ''}} | frames: ${{frames.length}} | settings: ${{JSON.stringify(payload.settings || {{}})}}`;
document.getElementById('prev').addEventListener('click', () => {{ stopTimer(); render(index - 1); }});
document.getElementById('next').addEventListener('click', () => {{ stopTimer(); render(index + 1); }});
slider.addEventListener('input', (event) => {{ stopTimer(); render(Number(event.target.value)); }});
play.addEventListener('click', () => {{
  if (timer) {{ stopTimer(); return; }}
  play.textContent = 'Pause';
  timer = setInterval(() => {{
    if (index >= frames.length - 1) {{
      stopTimer();
      return;
    }}
    render(index + 1);
  }}, 900);
}});
render(0);
</script>
</body>
</html>
"""


def _write_animation(path: str | None, payload: dict[str, Any]) -> str | None:
  if not path:
    return None
  output_path = pathlib.Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(_render_animation_html(payload), encoding="utf-8")
  return str(output_path)


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
  frames: list[StepTrace] = []
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

    initial_canvas = np.asarray(canvas[0])
    initial_trace = _make_trace(
        tokenizer=tokenizer,
        canvas_idx=canvas_idx,
        step=0,
        phase="initial",
        noise=1.0,
        target_noise=1.0,
        canvas=initial_canvas,
        previous_canvas=None,
        selected_mask=None,
        mean_entropy=None,
    )
    frames.append(initial_trace)
    print(
        json.dumps(
            {
                "event": "canvas_initial",
                "canvas": canvas_idx,
                **dataclasses.asdict(initial_trace),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for step in range(args.denoising_steps):
      rng, step_rng = jax.random.split(rng)
      noise = 1.0 - step / args.denoising_steps
      target_noise = 1.0 - (step + 1) / args.denoising_steps
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
      previous_canvas = canvas
      new_canvas, selected, _, token_entropy = _sample_from_predictions(
          step_rng,
          shaped_logits,
          canvas,
          entropy_bound=args.entropy_bound,
          vocab_size=vocab_size,
      )
      stable_tokens, low_entropy, early_stop = _early_stop_flags(
          shaped_logits,
          previous_canvas,
          entropy_threshold=args.entropy_stop_threshold,
      )
      early_stop = (
          early_stop
          if not args.disable_early_stop
          else jnp.zeros_like(early_stop)
      )
      canvas = new_canvas
      sc_canvas_logits = jax.lax.stop_gradient(shaped_logits)

      trace = StepTrace(
          canvas=canvas_idx,
          step=step + 1,
          phase="denoise",
          noise=float(noise),
          target_noise=float(target_noise),
          changed_tokens=int(
              np.sum(np.asarray(canvas[0]) != np.asarray(previous_canvas[0]))
          ),
          accepted_tokens=int(np.sum(np.asarray(selected[0]))),
          mean_entropy=float(np.mean(np.asarray(token_entropy[0]))),
          stable_tokens=bool(np.asarray(stable_tokens[0])),
          low_entropy=bool(np.asarray(low_entropy[0])),
          early_stop=bool(np.asarray(early_stop[0])),
          text=_decode_tokens(tokenizer, np.asarray(canvas[0]).tolist()),
          token_ids=np.asarray(canvas[0]).astype(np.int32).tolist(),
          token_texts=_token_texts(
              tokenizer, np.asarray(canvas[0]).astype(np.int32).tolist()
          ),
          selected_mask=np.asarray(selected[0]).astype(np.bool_).tolist(),
          changed_mask=(np.asarray(canvas[0]) != np.asarray(previous_canvas[0]))
          .astype(np.bool_)
          .tolist(),
      )
      frames.append(trace)
      print(
          json.dumps(
              {
                  "event": "denoise_step",
                  **dataclasses.asdict(trace),
              },
              ensure_ascii=False,
          ),
          flush=True,
      )
      if bool(np.asarray(early_stop[0])):
        break

    final_canvas = _truncate_at_stop(np.asarray(canvas[0]).astype(np.int32))
    generated.extend(final_canvas.tolist())
    context = jnp.concatenate(
        [context, jnp.asarray(final_canvas, dtype=jnp.int32)[None, :]], axis=1
    )
    if np.any(np.isin(final_canvas, np.asarray(END_TOKENS, dtype=np.int32))):
      done = True

  output_ids = _truncate_at_stop(np.asarray(generated, dtype=np.int32)).tolist()
  output_text = _decode_tokens(tokenizer, output_ids)
  animation_payload = {
      "prompt": args.prompt,
      "formatted_prompt": prompt_text,
      "tiny": args.tiny,
      "checkpoint": None if args.tiny else args.checkpoint,
      "output_text": output_text,
      "output_token_ids": output_ids,
      "settings": {
          "max_new_tokens": args.max_new_tokens,
          "canvas_length": args.canvas_length,
          "denoising_steps": args.denoising_steps,
          "entropy_bound": args.entropy_bound,
          "min_temperature": args.min_temperature,
          "max_temperature": args.max_temperature,
          "temperature_exponent": args.temperature_exponent,
          "early_stop": not args.disable_early_stop,
          "entropy_stop_threshold": args.entropy_stop_threshold,
          "seed": args.seed,
      },
      "frames": [dataclasses.asdict(frame) for frame in frames],
  }
  trace_json_path = _write_json(args.trace_json, animation_payload)
  animation_path = _write_animation(args.animation_output, animation_payload)
  if trace_json_path:
    print(
        json.dumps({"event": "trace_json_written", "path": trace_json_path}),
        flush=True,
    )
  if animation_path:
    print(
        json.dumps({"event": "animation_written", "path": animation_path}),
        flush=True,
    )
  report = {
      "event": "generation_complete",
      "tiny": args.tiny,
      "checkpoint": None if args.tiny else args.checkpoint,
      "prompt": args.prompt,
      "output_text": output_text,
      "output_token_ids": output_ids,
      "num_frames": len(frames),
      "num_denoise_steps": sum(frame.phase == "denoise" for frame in frames),
      "trace_json": trace_json_path,
      "animation_output": animation_path,
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
  parser.add_argument("--entropy_stop_threshold", type=float, default=0.005)
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
  parser.add_argument("--animation_output")
  parser.add_argument("--trace_json")
  parser.add_argument("--disable_early_stop", action="store_true")
  parser.add_argument("--no_chat_template", action="store_true")
  parser.add_argument("--tiny", action="store_true")
  parser.add_argument("--tiny_vocab_size", type=int, default=256)
  return parser.parse_args()


def main() -> None:
  generate(parse_args())


if __name__ == "__main__":
  main()
