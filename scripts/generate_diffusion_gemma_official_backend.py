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

"""Generate text from a Tunix-wrapped official DiffusionGemma SFT checkpoint.

This is intentionally an official-backend generation path. It restores the
Kauldron checkpoint written by `OfficialDiffusionGemmaTrainer` and invokes the
official Hackable Diffusion AR sampling evaluator for one evaluation batch.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

_HACKABLE_ADAPTER_PATH = (
    REPO_ROOT / "tunix" / "models" / "diffusion_gemma" / "hackable_adapter.py"
)
_HACKABLE_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "_tunix_diffusion_gemma_hackable_adapter", _HACKABLE_ADAPTER_PATH
)
if _HACKABLE_ADAPTER_SPEC is None or _HACKABLE_ADAPTER_SPEC.loader is None:
  raise ImportError(f"Could not load {_HACKABLE_ADAPTER_PATH}")
hackable_adapter = importlib.util.module_from_spec(_HACKABLE_ADAPTER_SPEC)
sys.modules[_HACKABLE_ADAPTER_SPEC.name] = hackable_adapter
_HACKABLE_ADAPTER_SPEC.loader.exec_module(hackable_adapter)


DEFAULT_TOKENIZER = "gs://gemma-data/tokenizers/tokenizer_gemma4.model"
END_TOKENS = (1, 106, 50)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--recipe", choices=["pubmedqa"], default="pubmedqa")
  parser.add_argument("--gemma_ref", default=None)
  parser.add_argument("--hackable_diffusion_ref", default=None)
  parser.add_argument("--workdir", required=True)
  parser.add_argument("--checkpoint_path", default=None)
  parser.add_argument(
      "--step",
      default=None,
      help=(
          "Checkpoint step to restore. Use 'latest' to select the newest ckpt."
      ),
  )
  parser.add_argument("--num_train_steps", type=int, default=2000)
  parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000)
  parser.add_argument("--lora_rank", type=int, default=4)
  parser.add_argument("--lora_alpha", type=float, default=None)
  parser.add_argument(
      "--lora_backend",
      choices=["official", "qwix_lora"],
      default="official",
  )
  parser.add_argument("--qwix_lora_module_path", default=None)
  parser.add_argument("--encoder_loss_token_chunk_size", type=int, default=None)
  parser.add_argument("--dataset_batch_size", type=int, default=2)
  parser.add_argument("--eval_num_batches", type=int, default=1)
  parser.add_argument("--denoising_steps", type=int, default=8)
  parser.add_argument("--max_num_canvases", type=int, default=1)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
  parser.add_argument("--output_json", required=True)
  parser.add_argument("--trace_json", default=None)
  parser.add_argument(
      "--trace_mode",
      choices=["visual", "measured"],
      default="visual",
      help=(
          "'visual' renders a deterministic non-prefix reveal. 'measured' "
          "runs the official canvas sampler with store_trajectory=True and "
          "derives masks from the actual denoising xt trajectory."
      ),
  )
  parser.add_argument("--max_trace_tokens", type=int, default=36)
  parser.add_argument("--module_override", action="append", default=[])
  parser.add_argument("--config_override", action="append", default=[])
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  extra_paths = [
      path for path in (args.hackable_diffusion_ref, args.gemma_ref) if path
  ]
  hackable_adapter.initialize_jax_before_tensorflow(extra_paths)
  config = _build_official_config(args)
  cfg = _build_generation_config(args, config)

  with _temporary_sys_path(extra_paths):
    import jax  # pylint: disable=g-import-not-at-top
    from jax.experimental import checkify  # pylint: disable=g-import-not-at-top
    from kauldron import konfig  # pylint: disable=g-import-not-at-top
    from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-import-not-at-top

    trainer = konfig.resolve(cfg)
    hackable_adapter._replace_resolved_lora_backend(  # pylint: disable=protected-access
        trainer, config
    )
    state = trainer.init_state()
    resolved_step = hackable_adapter.resolve_official_checkpoint_step(
        args.workdir, args.step
    )
    if resolved_step is None:
      state = trainer.checkpointer.restore(state)
      restored_step = int(_scalar_to_host(getattr(state, "step", -1)))
    else:
      state = trainer.checkpointer.restore(state, step=resolved_step)
      restored_step = resolved_step

    evaluator_name = next(iter(trainer.evals))
    evaluator = trainer.evals[evaluator_name]
    backend_info = _generation_backend_info(
        evaluator,
        lora_backend=args.lora_backend,
    )
    batch = next(iter(evaluator.ds))
    batch = sharding_lib.device_put(batch, trainer.sharding.batch)
    step_nr = sharding_lib.device_put(1, sharding_lib.REPLICATED)
    measured_trajectory = None
    if args.trace_json and args.trace_mode == "measured":
      measured_trajectory = _run_measured_official_trajectory(
          evaluator=evaluator,
          state=state,
          batch=batch,
          step_nr=step_nr,
      )
      prompt_ids = measured_trajectory["prompt_tokens"]
      response_ids = measured_trajectory["final"]
    else:
      aux = evaluator.step(step_nr=step_nr, state=state, batch=batch)
      with jax.transfer_guard("allow"):
        if aux.error is not None:
          checkify.check_error(aux.error)
        aux = aux.finalize()
      prompt_ids, response_ids = _extract_text_sample_tokens(aux)
    prompt_texts, response_texts, response_token_texts = _decode_samples(
        prompt_ids, response_ids, args.tokenizer_path
    )

  result = {
      "event": "official_backend_generation_complete",
      "workdir": str(args.workdir),
      "checkpoint_step": restored_step,
      "checkpoint_info": (
          hackable_adapter.get_official_checkpoint_info(
              args.workdir, step=restored_step
          ).as_dict()
      ),
      "evaluator": evaluator_name,
      "backend": backend_info,
      "denoising_steps": args.denoising_steps,
      "max_num_canvases": args.max_num_canvases,
      "num_samples": len(response_ids),
      "samples": [
          {
              "index": i,
              "prompt_text": prompt_texts[i],
              "response_text": response_texts[i],
              "prompt_token_ids": _trim_ids(prompt_ids[i], stop_at_end=False),
              "response_token_ids": _trim_ids(
                  response_ids[i], stop_at_end=True
              ),
              "response_token_texts": response_token_texts[i],
          }
          for i in range(len(response_ids))
      ],
  }
  _write_json(pathlib.Path(args.output_json), result)

  if args.trace_json:
    if args.trace_mode == "measured":
      trace = _make_measured_visual_trace(
          result["samples"][0],
          measured_trajectory,
          denoising_steps=args.denoising_steps,
          max_tokens=args.max_trace_tokens,
      )
    else:
      trace = _make_visual_trace(
          result["samples"][0],
          denoising_steps=args.denoising_steps,
          max_tokens=args.max_trace_tokens,
      )
    result["trace_token_parity"] = _trace_token_parity(
        result["samples"][0],
        trace,
    )
    _write_json(pathlib.Path(args.output_json), result)
    _write_json(pathlib.Path(args.trace_json), trace)
    print(
        json.dumps({"event": "trace_json_written", "path": args.trace_json}),
        flush=True,
    )

  print(json.dumps(result, ensure_ascii=False), flush=True)


def _build_official_config(args: argparse.Namespace):
  module_overrides = _parse_key_value(args.module_override)
  config_overrides = _parse_key_value(args.config_override)
  config_overrides.setdefault("aux.eval_num_batches", args.eval_num_batches)
  return hackable_adapter.OfficialSFTConfig(
      recipe=args.recipe,
      gemma_ref=args.gemma_ref,
      hackable_diffusion_ref=args.hackable_diffusion_ref,
      workdir=args.workdir,
      checkpoint_path=args.checkpoint_path,
      num_train_steps=args.num_train_steps,
      checkpoint_every_n_steps=args.checkpoint_every_n_steps,
      lora_rank=args.lora_rank,
      lora_alpha=args.lora_alpha,
      lora_backend=args.lora_backend,
      qwix_lora_module_path=args.qwix_lora_module_path,
      encoder_loss_token_chunk_size=args.encoder_loss_token_chunk_size,
      dataset_batch_size=args.dataset_batch_size,
      use_early_stopping=False,
      disable_evals=True,
      module_overrides=module_overrides,
      config_overrides=config_overrides,
  )


def _build_generation_config(
    args: argparse.Namespace,
    config: Any,
):
  cfg = hackable_adapter.build_official_sft_config(config)
  cfg.evals = {}
  cfg.init_transform = None
  with _temporary_sys_path(
      [path for path in (args.hackable_diffusion_ref, args.gemma_ref) if path]
  ):
    from gemma.diffusion.hackable_diffusion_adapter.eval import ar_eval  # pylint: disable=g-import-not-at-top

    cfg.evals.update(
        ar_eval.make_ar_evals(
            cfg,
            gemma_network_ref=cfg.ref.model.gemma_network,
            corruption_process_ref=cfg.ref.aux.corruption_process,
            canvas_size_ref=cfg.ref.aux.canvas_size,
            metrics={},
            max_num_canvases=args.max_num_canvases,
            denoising_steps=[args.denoising_steps],
            use_early_stopping=False,
        )
    )
  return cfg


def _extract_text_sample_tokens(aux: Any) -> tuple[np.ndarray, np.ndarray]:
  summaries = aux.summary_states
  text_state = summaries.get("text_samples_ar")
  if text_state is None:
    raise ValueError(
        "Generation auxiliary state did not contain 'text_samples_ar'. "
        f"Available summaries: {list(summaries.keys())}"
    )
  prompt = np.asarray(text_state.prompt)
  response = np.asarray(text_state.response)
  if prompt.ndim == 3:
    prompt = prompt[..., 0]
  if response.ndim == 3:
    response = response[..., 0]
  return prompt.astype(np.int32), response.astype(np.int32)


def _generation_backend_info(
    evaluator: Any,
    *,
    lora_backend: str,
) -> dict[str, Any]:
  sampler = getattr(evaluator, "ar_diffusion_sampler", None)
  sampler_handler = getattr(sampler, "state_handler", None)
  return {
      "path": "official_hackable_diffusion_ar_evaluator",
      "lora_backend": lora_backend,
      "evaluator_class": _qualified_class_name(evaluator),
      "sampler_class": _qualified_class_name(sampler),
      "sampler_state_handler_class": _qualified_class_name(sampler_handler),
      "uses_official_text_samples_ar": True,
  }


def _qualified_class_name(value: Any) -> str | None:
  if value is None:
    return None
  cls = value.__class__
  return f"{cls.__module__}.{cls.__qualname__}"


def _decode_samples(
    prompt_ids: np.ndarray,
    response_ids: np.ndarray,
    tokenizer_path: str,
) -> tuple[list[str], list[str], list[list[str]]]:
  vocab = _load_vocab(tokenizer_path)
  prompt_texts = []
  response_texts = []
  response_token_texts = []
  for prompt, response in zip(prompt_ids, response_ids):
    prompt_trimmed = _trim_ids(prompt, stop_at_end=False)
    response_trimmed = _trim_ids(response, stop_at_end=True)
    prompt_texts.append(_decode(vocab, prompt_trimmed))
    response_texts.append(_decode(vocab, response_trimmed))
    response_token_texts.append([
        _decode(vocab, [token_id]) or f"<{token_id}>"
        for token_id in response_trimmed
    ])
  return prompt_texts, response_texts, response_token_texts


def _load_vocab(tokenizer_path: str):
  try:
    import seqio  # pylint: disable=g-import-not-at-top

    return seqio.vocabularies.SentencePieceVocabulary(tokenizer_path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(
        json.dumps({
            "event": "tokenizer_load_failed",
            "tokenizer_path": tokenizer_path,
            "error": repr(exc),
        }),
        flush=True,
    )
    return None


def _decode(vocab: Any, token_ids: list[int]) -> str:
  if not token_ids:
    return ""
  if vocab is None:
    return " ".join(str(token_id) for token_id in token_ids)
  try:
    return vocab.decode(token_ids)
  except Exception:  # pylint: disable=broad-exception-caught
    return " ".join(str(token_id) for token_id in token_ids)


def _trim_ids(ids: np.ndarray | list[int], *, stop_at_end: bool) -> list[int]:
  values = [int(x) for x in np.asarray(ids).reshape(-1).tolist()]
  trimmed = []
  for token_id in values:
    if token_id == 0:
      continue
    if stop_at_end and token_id in END_TOKENS:
      break
    trimmed.append(token_id)
  return trimmed


def _make_visual_trace(
    sample: dict[str, Any],
    *,
    denoising_steps: int,
    max_tokens: int,
) -> dict[str, Any]:
  token_ids = sample["response_token_ids"][:max_tokens]
  token_texts = sample["response_token_texts"][:max_tokens]
  if not token_ids:
    token_ids = [0]
    token_texts = ["<empty>"]
  frames = []
  total_steps = max(2, denoising_steps)
  reveal_steps = _denoising_visual_reveal_steps(token_ids, total_steps)
  previous_selected = [False for _ in token_ids]
  for step in range(total_steps + 1):
    progress = step / total_steps
    selected = [step >= reveal_step for reveal_step in reveal_steps]
    if step == total_steps:
      selected = [True for _ in token_ids]
    changed = [
        is_selected and not was_selected
        for is_selected, was_selected in zip(selected, previous_selected)
    ]
    visible_text = _masked_visual_text(token_texts, selected)
    frames.append({
        "canvas": 0,
        "step": step,
        "phase": "official-sample-visualization",
        "noise": float(1.0 - progress),
        "target_noise": float(max(0.0, 1.0 - ((step + 1) / total_steps))),
        "changed_tokens": int(sum(changed)),
        "accepted_tokens": int(sum(selected)),
        "mean_entropy": None,
        "stable_tokens": step == total_steps,
        "low_entropy": step == total_steps,
        "early_stop": False,
        "text": visible_text,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "selected_mask": selected,
        "changed_mask": changed,
    })
    previous_selected = selected
  return {
      "prompt": sample["prompt_text"],
      "formatted_prompt": sample["prompt_text"],
      "output_text": sample["response_text"],
      "output_token_ids": sample["response_token_ids"],
      "settings": {
          "source": "official_backend_tuned_checkpoint",
          "visualization": (
              "denoising-style non-prefix reveal from generated token ids"
          ),
          "acceptance_trace": "synthetic_visualization_not_sampler_mask",
          "denoising_steps": denoising_steps,
          "max_trace_tokens": max_tokens,
      },
      "frames": frames,
  }


def _run_measured_official_trajectory(
    *,
    evaluator: Any,
    state: Any,
    batch: Any,
    step_nr: Any,
) -> dict[str, Any]:
  """Runs the official AR canvas sampler once and returns measured trajectory.

  The public official evaluator currently returns only final text samples. This
  helper keeps the same official inference function, state handler, corruption
  process, and canvas sampler, but flips `DiffusionSampler.store_trajectory` for
  the one-canvas trace path so the GIF can be derived from real `xt` states.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from kauldron import kd  # pylint: disable=g-import-not-at-top
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model  # pylint: disable=g-import-not-at-top

  sampler = evaluator.ar_diffusion_sampler
  if sampler is None:
    raise ValueError("Evaluator does not expose ar_diffusion_sampler.")
  if int(sampler.max_num_canvases) != 1:
    raise ValueError(
        "Measured trace currently supports max_num_canvases=1, got "
        f"{sampler.max_num_canvases}."
    )
  if not hasattr(sampler.canvas_sampler, "store_trajectory"):
    raise ValueError(
        "Measured trace requires the non-early-stopping DiffusionSampler "
        "with a store_trajectory field."
    )

  base_context = kd.train.Context.from_state_and_batch(
      state=state,
      batch=batch,
  )
  context = sft_model.SamplingContext(**base_context.__dict__)
  inference_fn = evaluator._make_inference_fn(  # pylint: disable=protected-access
      evaluator.model, context
  )
  sampler.update_from_context(context)
  rngs = evaluator.base_cfg.rng_streams.eval_rngs(step_nr)
  _, sample_rng = jax.random.split(rngs[evaluator.rng_stream], 2)
  _, kwargs = kd.data.utils.get_model_inputs(evaluator.model, context)
  prompt_tokens = kwargs["prompt"]
  prompt_lengths = jnp.sum(prompt_tokens != evaluator.pad_token, axis=-1)
  conditioning = {
      "prompt_tokens": prompt_tokens,
      "prompt_lengths": prompt_lengths,
  }

  sampler_state = sampler.state_handler.init_ar_state(
      batch_size=len(prompt_tokens),
      conditioning=conditioning,
      canvas_length=sampler.canvas_length,
      max_num_canvases=sampler.max_num_canvases,
  )
  _, canvas_init_rng, canvas_sampler_rng = jax.random.split(sample_rng, 3)
  initial_canvas = sampler.diffusion_process.sample_from_invariant(
      key=canvas_init_rng,
      data_spec=jnp.zeros(
          (len(prompt_tokens), sampler.canvas_length) + sampler.data_shape,
          dtype=sampler.data_dtype,
      ),
  )
  canvas_sampler = dataclasses.replace(
      sampler.canvas_sampler,
      store_trajectory=True,
  )
  canvas_last_step, trajectory = canvas_sampler(
      inference_fn=inference_fn,
      rng=canvas_sampler_rng,
      initial_noise=initial_canvas,
      conditioning=sampler.state_handler.create_conditioning_from_state(
          sampler_state=sampler_state
      ),
  )
  sampler_state = sampler.state_handler.update_ar_state(
      canvas_last_step=canvas_last_step,
      sampler_state=sampler_state,
  )
  final = sampler.state_handler.finalize_ar_state(
      sampler_state=sampler_state
  )
  return {
      "trajectory_xt": jax.device_get(trajectory.xt),
      "trajectory_step": jax.device_get(trajectory.step_info.step),
      "trajectory_time": jax.device_get(trajectory.step_info.time),
      "prompt_tokens": jax.device_get(prompt_tokens),
      "final": jax.device_get(final),
  }


def _make_measured_visual_trace(
    sample: dict[str, Any],
    measured_trajectory: dict[str, Any] | None,
    *,
    denoising_steps: int,
    max_tokens: int,
) -> dict[str, Any]:
  """Builds a GIF trace from actual official diffusion trajectory arrays."""
  if measured_trajectory is None:
    raise ValueError("measured_trajectory is required for trace_mode=measured.")
  token_ids = sample["response_token_ids"][:max_tokens]
  token_texts = sample["response_token_texts"][:max_tokens]
  if not token_ids:
    token_ids = [0]
    token_texts = ["<empty>"]
  trajectory_xt = np.asarray(measured_trajectory["trajectory_xt"])
  if trajectory_xt.ndim == 4:
    trajectory_xt = trajectory_xt[:, :, :, 0]
  sample_xt = trajectory_xt[:, 0, : len(token_ids)].astype(np.int32)
  final_ids = np.asarray(token_ids, dtype=np.int32)
  stable_from = np.zeros_like(sample_xt, dtype=np.bool_)
  future_stable = np.ones((sample_xt.shape[1],), dtype=np.bool_)
  for idx in range(sample_xt.shape[0] - 1, -1, -1):
    future_stable = future_stable & (sample_xt[idx] == final_ids)
    stable_from[idx] = future_stable

  step_values = np.asarray(measured_trajectory["trajectory_step"]).reshape(-1)
  time_values = np.asarray(measured_trajectory["trajectory_time"])
  while time_values.ndim > 1:
    time_values = time_values[:, 0]
  time_values = time_values.reshape(-1)

  frames = []
  previous_ids = None
  total_frames = sample_xt.shape[0]
  for frame_idx in range(total_frames):
    current_ids = sample_xt[frame_idx]
    selected = [bool(value) for value in stable_from[frame_idx].tolist()]
    if previous_ids is None:
      changed = [False for _ in selected]
    else:
      changed = [
          bool(current != previous)
          for current, previous in zip(current_ids.tolist(), previous_ids)
      ]
    previous_ids = current_ids.tolist()
    progress = frame_idx / max(1, total_frames - 1)
    visible_text = _masked_visual_text(token_texts, selected)
    frames.append({
        "canvas": 0,
        "step": int(step_values[frame_idx])
        if frame_idx < len(step_values)
        else frame_idx,
        "phase": "official-measured-diffusion-trajectory",
        "noise": float(1.0 - progress),
        "target_noise": float(max(0.0, 1.0 - ((frame_idx + 1) / total_frames))),
        "changed_tokens": int(sum(changed)),
        "accepted_tokens": int(sum(selected)),
        "mean_entropy": None,
        "stable_tokens": frame_idx == total_frames - 1,
        "low_entropy": frame_idx == total_frames - 1,
        "early_stop": False,
        "text": visible_text,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "selected_mask": selected,
        "changed_mask": changed,
        "measured_token_ids": current_ids.tolist(),
        "time": float(time_values[frame_idx])
        if frame_idx < len(time_values)
        else None,
    })

  return {
      "prompt": sample["prompt_text"],
      "formatted_prompt": sample["prompt_text"],
      "output_text": sample["response_text"],
      "output_token_ids": sample["response_token_ids"],
      "settings": {
          "source": "official_backend_tuned_checkpoint",
          "visualization": "measured denoising trajectory from official sampler",
          "acceptance_trace": (
              "measured_stability_from_official_diffusion_trajectory"
          ),
          "measurement": "DiffusionSampler(store_trajectory=True).trajectory.xt",
          "denoising_steps": denoising_steps,
          "measured_frames": total_frames,
          "max_trace_tokens": max_tokens,
      },
      "frames": frames,
  }


def _denoising_visual_reveal_steps(
    token_ids: list[int],
    total_steps: int,
) -> list[int]:
  """Builds a deterministic non-prefix reveal order for visual traces.

  The official text-sample evaluator returns final token ids but not the
  internal per-position acceptance masks. This order is therefore only a
  visualization aid: it makes the GIF look like denoising across the canvas
  instead of left-to-right autoregressive decoding.
  """
  if not token_ids:
    return []
  keyed_positions = []
  for idx, token_id in enumerate(token_ids):
    digest = hashlib.blake2s(
        f"{idx}:{token_id}".encode("utf-8"), digest_size=4
    ).digest()
    jitter = int.from_bytes(digest, "little") / 2**32
    center_bias = abs((idx + 0.5) / len(token_ids) - 0.5)
    score = 0.72 * jitter + 0.28 * center_bias
    keyed_positions.append((score, idx))
  order = [idx for _, idx in sorted(keyed_positions)]
  reveal_steps = [total_steps for _ in token_ids]
  for rank, idx in enumerate(order):
    progress = (rank + 1) / len(order)
    step = max(1, int(round(total_steps * (progress**1.18))))
    reveal_steps[idx] = min(total_steps, step)
  return reveal_steps


def _masked_visual_text(token_texts: list[str], selected: list[bool]) -> str:
  pieces = [
      piece if keep else "·"
      for piece, keep in zip(token_texts, selected)
  ]
  text = "".join(pieces).strip()
  return text if text else "..."


def _trace_token_parity(
    sample: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
  sample_tokens = sample["response_token_ids"]
  trace_tokens = trace["output_token_ids"]
  equal = sample_tokens == trace_tokens
  return {
      "source": "official_evaluator_text_samples_ar",
      "trace": "rendered_generation_trace",
      "equal": equal,
      "num_sample_tokens": len(sample_tokens),
      "num_trace_tokens": len(trace_tokens),
  }


def _scalar_to_host(value: Any) -> float:
  import jax  # pylint: disable=g-import-not-at-top

  return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _parse_key_value(items: list[str]) -> dict[str, Any]:
  values: dict[str, Any] = {}
  for item in items:
    key, sep, raw_value = item.partition("=")
    if not sep or not key:
      raise ValueError(f"Override {item!r} must use dotted.path=value syntax.")
    try:
      values[key] = json.loads(raw_value)
    except json.JSONDecodeError:
      values[key] = raw_value
  return values


@contextlib.contextmanager
def _temporary_sys_path(paths):
  old_path = list(sys.path)
  try:
    for path in reversed([str(pathlib.Path(p).expanduser()) for p in paths]):
      if path and path not in sys.path:
        sys.path.insert(0, path)
    yield
  finally:
    sys.path[:] = old_path


def _write_json(path: pathlib.Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


if __name__ == "__main__":
  main()
