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
  parser.add_argument("--step", type=int, default=None)
  parser.add_argument("--num_train_steps", type=int, default=2000)
  parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000)
  parser.add_argument("--lora_rank", type=int, default=4)
  parser.add_argument("--dataset_batch_size", type=int, default=2)
  parser.add_argument("--eval_num_batches", type=int, default=1)
  parser.add_argument("--denoising_steps", type=int, default=8)
  parser.add_argument("--max_num_canvases", type=int, default=1)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
  parser.add_argument("--output_json", required=True)
  parser.add_argument("--trace_json", default=None)
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
  cfg = _build_generation_config(args)

  with _temporary_sys_path(extra_paths):
    import jax  # pylint: disable=g-import-not-at-top
    from jax.experimental import checkify  # pylint: disable=g-import-not-at-top
    from kauldron import konfig  # pylint: disable=g-import-not-at-top
    from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-import-not-at-top

    trainer = konfig.resolve(cfg)
    state = trainer.init_state()
    if args.step is None:
      state = trainer.checkpointer.restore(state)
      restored_step = int(_scalar_to_host(getattr(state, "step", -1)))
    else:
      state = trainer.checkpointer.restore(state, step=args.step)
      restored_step = args.step

    evaluator_name = next(iter(trainer.evals))
    evaluator = trainer.evals[evaluator_name]
    batch = next(iter(evaluator.ds))
    batch = sharding_lib.device_put(batch, trainer.sharding.batch)
    step_nr = sharding_lib.device_put(1, sharding_lib.REPLICATED)
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
      "evaluator": evaluator_name,
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
    trace = _make_visual_trace(
        result["samples"][0],
        denoising_steps=args.denoising_steps,
        max_tokens=args.max_trace_tokens,
    )
    _write_json(pathlib.Path(args.trace_json), trace)
    print(
        json.dumps({"event": "trace_json_written", "path": args.trace_json}),
        flush=True,
    )

  print(json.dumps(result, ensure_ascii=False), flush=True)


def _build_generation_config(args: argparse.Namespace):
  module_overrides = _parse_key_value(args.module_override)
  config_overrides = _parse_key_value(args.config_override)
  config_overrides.setdefault("aux.eval_num_batches", args.eval_num_batches)
  config = hackable_adapter.OfficialSFTConfig(
      recipe=args.recipe,
      gemma_ref=args.gemma_ref,
      hackable_diffusion_ref=args.hackable_diffusion_ref,
      workdir=args.workdir,
      checkpoint_path=args.checkpoint_path,
      num_train_steps=args.num_train_steps,
      checkpoint_every_n_steps=args.checkpoint_every_n_steps,
      lora_rank=args.lora_rank,
      dataset_batch_size=args.dataset_batch_size,
      use_early_stopping=False,
      disable_evals=True,
      module_overrides=module_overrides,
      config_overrides=config_overrides,
  )
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
  previous_revealed = 0
  total_steps = max(2, denoising_steps)
  for step in range(total_steps + 1):
    progress = step / total_steps
    revealed = int(round(len(token_ids) * (progress ** 1.35)))
    if step == total_steps:
      revealed = len(token_ids)
    selected = [idx < revealed for idx in range(len(token_ids))]
    changed = [
        previous_revealed <= idx < revealed for idx in range(len(token_ids))
    ]
    visible_text = "".join(token_texts[:revealed])
    if revealed < len(token_ids):
      visible_text += " ..."
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
    previous_revealed = revealed
  return {
      "prompt": sample["prompt_text"],
      "formatted_prompt": sample["prompt_text"],
      "output_text": sample["response_text"],
      "output_token_ids": sample["response_token_ids"],
      "settings": {
          "source": "official_backend_tuned_checkpoint",
          "visualization": "denoising-style reveal from generated token ids",
          "denoising_steps": denoising_steps,
          "max_trace_tokens": max_tokens,
      },
      "frames": frames,
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
