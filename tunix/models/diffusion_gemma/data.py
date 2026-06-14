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

"""Dataset helpers for DiffusionGemma SFT.

DiffusionGemma SFT does not consume the usual causal-LM `input_ids`/`labels`
pair directly. The response is represented as a fixed number of denoising
canvases, and the encoder objective is represented by a shifted full sequence
target. This module provides the small conversion layer between Tunix-style
text records and the `DiffusionGemmaSFTBatch` expected by `sft.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.diffusion_gemma import sft as diffusion_sft


class TokenizerLike(Protocol):
  """Minimal tokenizer surface needed by the DiffusionGemma data adapter."""

  def encode(self, text: str) -> Sequence[int]:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiffusionGemmaTextBatchConfig:
  """Configuration for text-to-DiffusionGemma SFT batch conversion."""

  prompt_len: int
  canvas_size: int
  num_canvases: int
  pad_id: int
  eos_id: int
  bos_id: int | None = None
  add_bos_to_prompt: bool = True
  add_canvas_feature_axis: bool = True
  tiny_vocab_size: int | None = None

  @property
  def total_canvas_len(self) -> int:
    return self.canvas_size * self.num_canvases

  def __post_init__(self):
    if self.prompt_len <= 0:
      raise ValueError(f"prompt_len must be positive, got {self.prompt_len}.")
    if self.canvas_size <= 0:
      raise ValueError(f"canvas_size must be positive, got {self.canvas_size}.")
    if self.num_canvases <= 0:
      raise ValueError(
          f"num_canvases must be positive, got {self.num_canvases}."
      )
    if self.add_bos_to_prompt and self.bos_id is None:
      raise ValueError("bos_id is required when add_bos_to_prompt=True.")
    if self.tiny_vocab_size is not None and self.tiny_vocab_size <= 0:
      raise ValueError(
          "tiny_vocab_size must be positive when set, got "
          f"{self.tiny_vocab_size}."
      )


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaTextExample:
  """Simple prompt/response record for DiffusionGemma SFT."""

  prompt: str
  response: str


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaBatchSummary:
  """Small, host-side description of a DiffusionGemma SFT batch."""

  batch_size: int
  prompt_shape: tuple[int, ...]
  canvas_shape: tuple[int, ...]
  canvas_id_shape: tuple[int, ...]
  canvas_mask_shape: tuple[int, ...]
  encoder_target_shape: tuple[int, ...]
  encoder_target_mask_shape: tuple[int, ...]
  valid_canvas_tokens: int
  valid_canvas_fraction: float
  rng_shape: tuple[int, ...]

  def as_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


PromptFn = Callable[[Any], str]
ResponseFn = Callable[[Any], str]


def tokenizer_special_id(
    tokenizer: Any,
    method_name: str,
    attr_name: str,
    *,
    required: bool = True,
) -> int | None:
  """Reads a special token id from Tunix or HF-style tokenizers."""
  method = getattr(tokenizer, method_name, None)
  if callable(method):
    value = method()
  else:
    value = getattr(tokenizer, attr_name, None)
  if value is None:
    if required:
      raise ValueError(
          f"Tokenizer does not expose {method_name}() or {attr_name}."
      )
    return None
  return int(value)


def config_from_tokenizer(
    tokenizer: Any,
    *,
    prompt_len: int,
    canvas_size: int,
    num_canvases: int,
    add_bos_to_prompt: bool = True,
    add_canvas_feature_axis: bool = True,
    tiny_vocab_size: int | None = None,
) -> DiffusionGemmaTextBatchConfig:
  """Builds a text batch config from tokenizer special-token ids."""
  pad_id = tokenizer_special_id(tokenizer, "pad_id", "pad_token_id")
  eos_id = tokenizer_special_id(tokenizer, "eos_id", "eos_token_id")
  bos_id = tokenizer_special_id(
      tokenizer, "bos_id", "bos_token_id", required=add_bos_to_prompt
  )
  if tiny_vocab_size is not None:
    pad_id %= tiny_vocab_size
    eos_id %= tiny_vocab_size
    if bos_id is not None:
      bos_id %= tiny_vocab_size
  return DiffusionGemmaTextBatchConfig(
      prompt_len=prompt_len,
      canvas_size=canvas_size,
      num_canvases=num_canvases,
      pad_id=pad_id,
      eos_id=eos_id,
      bos_id=bos_id,
      add_bos_to_prompt=add_bos_to_prompt,
      add_canvas_feature_axis=add_canvas_feature_axis,
      tiny_vocab_size=tiny_vocab_size,
  )


def encode_text(tokenizer: Any, text: str) -> list[int]:
  """Tokenizes text with either a simple `encode` or HF tokenizer surface."""
  encode = getattr(tokenizer, "encode", None)
  if callable(encode):
    return [int(token_id) for token_id in encode(text)]
  call = getattr(tokenizer, "__call__", None)
  if callable(call):
    encoded = call(text, add_special_tokens=False)
    if isinstance(encoded, Mapping) and "input_ids" in encoded:
      return [int(token_id) for token_id in encoded["input_ids"]]
  raise ValueError("Tokenizer must expose encode(text) or __call__(text).")


def pad_or_truncate(ids: Sequence[int], length: int, pad_id: int) -> np.ndarray:
  """Pads or truncates token ids to a fixed length."""
  if length <= 0:
    raise ValueError(f"length must be positive, got {length}.")
  ids = list(ids[:length])
  out = np.full((length,), pad_id, dtype=np.int32)
  out[: len(ids)] = np.asarray(ids, dtype=np.int32)
  return out


def make_canvas(
    response_ids: Sequence[int],
    *,
    num_canvases: int,
    canvas_size: int,
    eos_id: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Converts response token ids into DiffusionGemma canvas fields."""
  total_capacity = num_canvases * canvas_size
  response = np.asarray(list(response_ids[:total_capacity]), dtype=np.int32)
  flat = np.full((total_capacity,), pad_id, dtype=np.int32)
  flat[: len(response)] = response
  if len(response) > 0:
    last_canvas_idx = (len(response) - 1) // canvas_size
    canvas_end = (last_canvas_idx + 1) * canvas_size
    flat[len(response) : canvas_end] = eos_id
    num_valid_canvases = last_canvas_idx + 1
  else:
    num_valid_canvases = 0
  canvas_id = np.repeat(np.arange(num_canvases, dtype=np.int32), canvas_size)
  canvas_mask = np.zeros((total_capacity,), dtype=np.bool_)
  canvas_mask[: num_valid_canvases * canvas_size] = True
  return flat, canvas_id, canvas_mask


def make_encoder_shift(
    prompt: np.ndarray,
    canvas: np.ndarray,
    canvas_mask: np.ndarray,
    *,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Builds the encoder AR target and mask from prompt plus clean canvas."""
  prompt_valid = prompt != pad_id
  full_seq = np.concatenate([prompt, canvas])
  full_valid = np.concatenate([prompt_valid, canvas_mask])
  target = np.roll(full_seq, -1).astype(np.int32)
  target[-1] = pad_id
  shifted_valid = np.roll(full_valid, -1)
  shifted_valid[-1] = False
  return target, full_valid & shifted_valid


def default_prompt_fn(example: Any) -> str:
  if isinstance(example, str):
    return example
  if isinstance(example, DiffusionGemmaTextExample):
    return example.prompt
  if isinstance(example, Mapping):
    for key in ("prompt", "prompts", "input", "question"):
      if key in example:
        return str(example[key])
  if hasattr(example, "prompt"):
    return str(example.prompt)
  raise ValueError(
      "Could not find a prompt field. Provide prompt_fn for this dataset."
  )


def default_response_fn(example: Any) -> str:
  if isinstance(example, DiffusionGemmaTextExample):
    return example.response
  if isinstance(example, Mapping):
    for key in ("response", "completion", "target", "targets", "answer"):
      if key in example:
        return str(example[key])
  if hasattr(example, "response"):
    return str(example.response)
  raise ValueError(
      "Could not find a response field. Provide response_fn for this dataset."
  )


def make_text_example(
    example: Any,
    *,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
) -> DiffusionGemmaTextExample:
  """Normalizes a user record into a prompt/response example."""
  prompt_fn = prompt_fn or default_prompt_fn
  response_fn = response_fn or default_response_fn
  return DiffusionGemmaTextExample(
      prompt=prompt_fn(example),
      response=response_fn(example),
  )


def make_text_examples(
    examples: Iterable[Any],
    *,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
) -> list[DiffusionGemmaTextExample]:
  """Normalizes user records into DiffusionGemma prompt/response examples."""
  return [
      make_text_example(
          example,
          prompt_fn=prompt_fn,
          response_fn=response_fn,
      )
      for example in examples
  ]


def load_text_examples_jsonl(
    path: str | pathlib.Path,
    *,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
) -> list[DiffusionGemmaTextExample]:
  """Loads prompt/response examples from a JSONL file.

  The default field lookup accepts common Tunix/SFT names such as `prompt`,
  `input`, `question`, `response`, `completion`, `target`, and `answer`.
  Custom datasets can pass `prompt_fn` and `response_fn` to format records.
  """
  path = pathlib.Path(path)
  examples = []
  with path.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {line_number}: {exc}"
        ) from exc
      examples.append(
          make_text_example(
              record,
              prompt_fn=prompt_fn,
              response_fn=response_fn,
          )
      )
  return examples


def make_sft_batch_from_token_ids(
    prompt_ids_batch: Sequence[Sequence[int]],
    response_ids_batch: Sequence[Sequence[int]],
    *,
    config: DiffusionGemmaTextBatchConfig,
    rng_seed: int,
) -> diffusion_sft.DiffusionGemmaSFTBatch:
  """Builds a DiffusionGemma SFT batch from tokenized prompt/response ids."""
  if len(prompt_ids_batch) != len(response_ids_batch):
    raise ValueError(
        "prompt_ids_batch and response_ids_batch must have the same length, "
        f"got {len(prompt_ids_batch)} and {len(response_ids_batch)}."
    )
  if not prompt_ids_batch:
    raise ValueError("At least one example is required to build a batch.")

  prompts = []
  canvases = []
  canvas_ids = []
  canvas_masks = []
  encoder_targets = []
  encoder_target_masks = []
  for prompt_ids, response_ids in zip(prompt_ids_batch, response_ids_batch):
    if config.add_bos_to_prompt:
      prompt_ids = [config.bos_id] + list(prompt_ids)
    if config.tiny_vocab_size is not None:
      prompt_ids = [
          int(token_id) % config.tiny_vocab_size for token_id in prompt_ids
      ]
      response_ids = [
          int(token_id) % config.tiny_vocab_size for token_id in response_ids
      ]

    prompt = pad_or_truncate(prompt_ids, config.prompt_len, config.pad_id)
    canvas, canvas_id, canvas_mask = make_canvas(
        response_ids,
        num_canvases=config.num_canvases,
        canvas_size=config.canvas_size,
        eos_id=config.eos_id,
        pad_id=config.pad_id,
    )
    if not canvas_mask.any():
      raise ValueError(
          "DiffusionGemma SFT requires each response to produce at least one "
          "valid canvas token."
      )
    encoder_target, encoder_target_mask = make_encoder_shift(
        prompt, canvas, canvas_mask, pad_id=config.pad_id
    )
    prompts.append(prompt)
    canvases.append(
        canvas[:, None] if config.add_canvas_feature_axis else canvas
    )
    canvas_ids.append(canvas_id)
    canvas_masks.append(canvas_mask)
    encoder_targets.append(encoder_target)
    encoder_target_masks.append(encoder_target_mask.astype(np.float32))

  return diffusion_sft.DiffusionGemmaSFTBatch(
      prompt=jnp.asarray(np.stack(prompts), dtype=jnp.int32),
      canvas=jnp.asarray(np.stack(canvases), dtype=jnp.int32),
      canvas_id=jnp.asarray(np.stack(canvas_ids), dtype=jnp.int32),
      canvas_mask=jnp.asarray(np.stack(canvas_masks), dtype=jnp.bool_),
      encoder_target=jnp.asarray(np.stack(encoder_targets), dtype=jnp.int32),
      encoder_target_mask=jnp.asarray(
          np.stack(encoder_target_masks), dtype=jnp.float32
      ),
      rng=jax.random.PRNGKey(rng_seed),
  )


def make_sft_batch_from_text_examples(
    examples: Sequence[Any],
    *,
    tokenizer: Any,
    config: DiffusionGemmaTextBatchConfig,
    rng_seed: int,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
) -> diffusion_sft.DiffusionGemmaSFTBatch:
  """Builds a DiffusionGemma SFT batch from text examples."""
  prompt_fn = prompt_fn or default_prompt_fn
  response_fn = response_fn or default_response_fn
  prompt_ids_batch = [
      encode_text(tokenizer, prompt_fn(example)) for example in examples
  ]
  response_ids_batch = [
      encode_text(tokenizer, response_fn(example)) for example in examples
  ]
  return make_sft_batch_from_token_ids(
      prompt_ids_batch,
      response_ids_batch,
      config=config,
      rng_seed=rng_seed,
  )


def make_sft_batches_from_text_examples(
    examples: Sequence[Any],
    *,
    tokenizer: Any,
    config: DiffusionGemmaTextBatchConfig,
    batch_size: int,
    rng_seed: int = 0,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
    drop_remainder: bool = False,
    as_model_inputs: bool = False,
) -> Iterable[diffusion_sft.DiffusionGemmaSFTBatch | dict[str, jax.Array]]:
  """Yields DiffusionGemma SFT batches from an in-memory text dataset."""
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}.")
  total = len(examples)
  for start in range(0, total, batch_size):
    batch_examples = examples[start : start + batch_size]
    if len(batch_examples) < batch_size and drop_remainder:
      continue
    batch = make_sft_batch_from_text_examples(
        batch_examples,
        tokenizer=tokenizer,
        config=config,
        rng_seed=rng_seed + start // batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
    )
    if as_model_inputs:
      yield diffusion_sft.gen_model_input_fn(batch)
    else:
      yield batch


def make_sft_dataset(
    examples: Sequence[Any],
    *,
    tokenizer: Any,
    config: DiffusionGemmaTextBatchConfig,
    batch_size: int,
    rng_seed: int = 0,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
    drop_remainder: bool = False,
    as_model_inputs: bool = True,
) -> Iterable[diffusion_sft.DiffusionGemmaSFTBatch | dict[str, jax.Array]]:
  """Tunix-style dataset helper for in-memory prompt/response records."""
  return make_sft_batches_from_text_examples(
      examples,
      tokenizer=tokenizer,
      config=config,
      batch_size=batch_size,
      rng_seed=rng_seed,
      prompt_fn=prompt_fn,
      response_fn=response_fn,
      drop_remainder=drop_remainder,
      as_model_inputs=as_model_inputs,
  )


def make_sft_dataset_from_jsonl(
    path: str | pathlib.Path,
    *,
    tokenizer: Any,
    config: DiffusionGemmaTextBatchConfig,
    batch_size: int,
    rng_seed: int = 0,
    prompt_fn: PromptFn | None = None,
    response_fn: ResponseFn | None = None,
    drop_remainder: bool = False,
    as_model_inputs: bool = True,
) -> Iterable[diffusion_sft.DiffusionGemmaSFTBatch | dict[str, jax.Array]]:
  """Loads a JSONL file and yields DiffusionGemma SFT batches."""
  examples = load_text_examples_jsonl(
      path,
      prompt_fn=prompt_fn,
      response_fn=response_fn,
  )
  return make_sft_dataset(
      examples,
      tokenizer=tokenizer,
      config=config,
      batch_size=batch_size,
      rng_seed=rng_seed,
      drop_remainder=drop_remainder,
      as_model_inputs=as_model_inputs,
  )


def describe_sft_batch(
    batch: diffusion_sft.DiffusionGemmaSFTBatch | Mapping[str, Any],
) -> DiffusionGemmaBatchSummary:
  """Returns a compact shape/mask summary for logs and runbooks."""
  if isinstance(batch, Mapping):
    get = batch.__getitem__
  else:
    get = lambda key: getattr(batch, key)

  prompt = np.asarray(get("prompt"))
  canvas = np.asarray(get("canvas"))
  canvas_id = np.asarray(get("canvas_id"))
  canvas_mask = np.asarray(get("canvas_mask"))
  encoder_target = np.asarray(get("encoder_target"))
  encoder_target_mask = np.asarray(get("encoder_target_mask"))
  rng = np.asarray(get("rng"))
  valid_canvas_tokens = int(np.sum(canvas_mask))
  return DiffusionGemmaBatchSummary(
      batch_size=int(prompt.shape[0]),
      prompt_shape=tuple(int(dim) for dim in prompt.shape),
      canvas_shape=tuple(int(dim) for dim in canvas.shape),
      canvas_id_shape=tuple(int(dim) for dim in canvas_id.shape),
      canvas_mask_shape=tuple(int(dim) for dim in canvas_mask.shape),
      encoder_target_shape=tuple(int(dim) for dim in encoder_target.shape),
      encoder_target_mask_shape=tuple(
          int(dim) for dim in encoder_target_mask.shape
      ),
      valid_canvas_tokens=valid_canvas_tokens,
      valid_canvas_fraction=float(valid_canvas_tokens / canvas_mask.size),
      rng_shape=tuple(int(dim) for dim in rng.shape),
  )
