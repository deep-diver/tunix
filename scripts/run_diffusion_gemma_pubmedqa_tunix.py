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

"""PubMedQA LoRA SFT validation test for Tunix DiffusionGemma.

This script mirrors the official DeepMind
``hackable_diffusion_adapter/data/pubmedqa`` preprocessing path closely enough
to exercise Tunix on real PubMedQA examples without importing Kauldron/Grain.
It can run with a tiny randomly initialized model for local checks or with the
public DiffusionGemma checkpoint for a GPU validation run.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tunix.generate import tokenizer_adapter
from tunix.models.diffusion_gemma import model as diffusion_model
from tunix.models.diffusion_gemma import params as diffusion_params
from tunix.models.diffusion_gemma import sft as diffusion_sft
from tunix.models.gemma4 import model as gemma4_model
from tunix.sft import peft_trainer


DEFAULT_TOKENIZER = "gs://gemma-data/tokenizers/tokenizer_gemma4.model"
PUBMEDQA_REPO = "https://github.com/pubmedqa/pubmedqa.git"
SYSTEM_PROMPT = (
    "<|turn>system\n"
    "You are a medical research assistant. Given a medical research context "
    "and a question, provide a brief explanation and then state your final "
    "answer.\n\n"
    'When you show the final answer, use the expression "The answer is: " '
    "followed by yes, no, or maybe. Use the expression without any "
    "modification.\n\n"
    "For example:\n"
    "The study shows a statistically significant improvement in outcomes for "
    "the treatment group compared to the control group (p < 0.05). The answer "
    "is: yes<turn|>\n"
)


@dataclasses.dataclass(frozen=True)
class PreparedExample:
  pubmed_id: str
  prompt: str
  response: str
  long_response: str
  short_answer: str


def _log(event: str, **kwargs: Any) -> None:
  print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


class _GpuMemoryMonitor:
  """Polls nvidia-smi so GPU validation runs leave explicit VRAM evidence."""

  def __init__(self, poll_seconds: float):
    self._poll_seconds = poll_seconds
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    self._samples = 0
    self._peak_used_mib: dict[str, int] = {}
    self._total_mib: dict[str, int] = {}
    self._last_error = ""
    self._lock = threading.Lock()

  def start(self) -> None:
    if self._poll_seconds <= 0:
      return
    if not shutil.which("nvidia-smi"):
      _log("gpu_memory_monitor_unavailable", reason="nvidia-smi not found")
      return
    self._thread = threading.Thread(
        target=self._poll_loop,
        name="gpu-memory-monitor",
        daemon=True,
    )
    self._thread.start()
    atexit.register(self.stop)
    _log("gpu_memory_monitor_started", poll_seconds=self._poll_seconds)

  def stop(self) -> dict[str, Any] | None:
    if self._thread is None:
      return None
    self._stop_event.set()
    self._thread.join(timeout=max(1.0, self._poll_seconds + 1.0))
    with self._lock:
      if not self._peak_used_mib:
        return {
            "samples": self._samples,
            "last_error": self._last_error,
            "observed": False,
        }
      free_mib = {
          index: self._total_mib[index] - used
          for index, used in self._peak_used_mib.items()
      }
      return {
          "samples": self._samples,
          "observed": True,
          "total_mib_by_gpu": dict(sorted(self._total_mib.items())),
          "peak_used_mib_by_gpu": dict(sorted(self._peak_used_mib.items())),
          "min_free_mib_by_gpu": dict(sorted(free_mib.items())),
          "max_peak_used_mib": max(self._peak_used_mib.values()),
          "min_headroom_mib": min(free_mib.values()),
          "last_error": self._last_error,
      }

  def _poll_loop(self) -> None:
    while not self._stop_event.is_set():
      self._sample_once()
      self._stop_event.wait(self._poll_seconds)
    self._sample_once()

  def _sample_once(self) -> None:
    try:
      proc = subprocess.run(
          [
              "nvidia-smi",
              "--query-gpu=index,memory.used,memory.total",
              "--format=csv,noheader,nounits",
          ],
          check=True,
          capture_output=True,
          text=True,
          timeout=10,
      )
      with self._lock:
        self._samples += 1
        for line in proc.stdout.strip().splitlines():
          if not line.strip():
            continue
          index, used, total = [part.strip() for part in line.split(",")]
          used_mib = int(used)
          total_mib = int(total)
          self._total_mib[index] = total_mib
          self._peak_used_mib[index] = max(
              self._peak_used_mib.get(index, 0), used_mib
          )
        self._last_error = ""
    except Exception as exc:  # pylint: disable=broad-exception-caught
      with self._lock:
        self._last_error = repr(exc)


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


def _resolve_mesh_axes(args: argparse.Namespace) -> tuple[int, int]:
  """Resolves mesh axes, defaulting to official-style FSDP first."""
  if args.tiny:
    return 1, 1
  device_count = jax.device_count()
  fsdp = args.mesh_fsdp
  tp = args.mesh_tp
  if fsdp is None and tp is None:
    return device_count, 1
  if fsdp is None:
    if device_count % tp != 0:
      raise ValueError(
          f"jax.device_count()={device_count} must be divisible by"
          f" mesh_tp={tp}."
      )
    return device_count // tp, tp
  if tp is None:
    if device_count % fsdp != 0:
      raise ValueError(
          f"jax.device_count()={device_count} must be divisible by"
          f" mesh_fsdp={fsdp}."
      )
    return fsdp, device_count // fsdp
  if fsdp * tp != device_count:
    raise ValueError(
        "mesh_fsdp * mesh_tp must equal jax.device_count(). Got "
        f"{fsdp} * {tp} != {device_count}."
    )
  return fsdp, tp


def _format_pubmedqa_prompt(question: str, contexts: list[str], max_chars: int):
  context_text = "\n".join(contexts)
  if len(context_text) > max_chars:
    context_text = context_text[:max_chars] + "..."
  return f"Context:\n{context_text}\n\nQuestion: {question}"


def _convert_pubmedqa_json(
    data: dict[str, Any],
    *,
    max_context_chars: int,
) -> list[PreparedExample]:
  examples = []
  for pubmed_id, entry in data.items():
    prompt = _format_pubmedqa_prompt(
        entry.get("QUESTION", ""),
        entry.get("CONTEXTS", []),
        max_context_chars,
    )
    answer = entry.get("final_decision", "").strip().lower()
    examples.append(
        PreparedExample(
            pubmed_id=str(pubmed_id),
            prompt=prompt,
            response=answer,
            long_response=entry.get("LONG_ANSWER", ""),
            short_answer=answer,
        )
    )
  return examples


def _write_jsonl(examples: list[PreparedExample], path: pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for example in examples:
      f.write(json.dumps(dataclasses.asdict(example), ensure_ascii=False))
      f.write("\n")


def _prepare_pubmedqa(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, pathlib.Path]:
  data_dir = pathlib.Path(args.data_dir)
  train_path = pathlib.Path(
      args.train_jsonl or data_dir / "pubmedqa_train.jsonl"
  )
  test_path = pathlib.Path(args.test_jsonl or data_dir / "pubmedqa_test.jsonl")
  if train_path.exists() and test_path.exists():
    return train_path, test_path

  repo_dir = pathlib.Path(args.pubmedqa_repo_dir)
  if not repo_dir.exists():
    if not shutil.which("git"):
      raise RuntimeError("git is required to clone PubMedQA.")
    _log("clone_pubmedqa", repo=PUBMEDQA_REPO, dest=str(repo_dir))
    subprocess.run(
        ["git", "clone", "--depth", "1", PUBMEDQA_REPO, str(repo_dir)],
        check=True,
    )

  raw_data_dir = repo_dir / "data"
  pqal_path = raw_data_dir / "ori_pqal.json"
  test_gt_path = raw_data_dir / "test_ground_truth.json"
  pqal = json.loads(pqal_path.read_text(encoding="utf-8"))
  test_gt = json.loads(test_gt_path.read_text(encoding="utf-8"))
  test_ids = set(test_gt.keys())
  train_data = {k: v for k, v in pqal.items() if k not in test_ids}
  test_data = {k: v for k, v in pqal.items() if k in test_ids}
  train_examples = _convert_pubmedqa_json(
      train_data, max_context_chars=args.max_context_chars
  )
  test_examples = _convert_pubmedqa_json(
      test_data, max_context_chars=args.max_context_chars
  )
  _write_jsonl(train_examples, train_path)
  _write_jsonl(test_examples, test_path)
  _log(
      "pubmedqa_prepared",
      train_path=str(train_path),
      test_path=str(test_path),
      train_examples=len(train_examples),
      test_examples=len(test_examples),
      official_split="ori_pqal minus test_ground_truth ids",
  )
  return train_path, test_path


def _read_jsonl(path: pathlib.Path) -> list[PreparedExample]:
  examples = []
  for line in path.read_text(encoding="utf-8").strip().split("\n"):
    if line:
      examples.append(PreparedExample(**json.loads(line)))
  return examples


def _load_tokenizer(path: str):
  return tokenizer_adapter.Tokenizer(
      "sentencepiece", path, add_bos=False, add_eos=False
  )


def _pad_or_truncate(ids: list[int], length: int, pad_id: int) -> np.ndarray:
  ids = ids[:length]
  out = np.full((length,), pad_id, dtype=np.int32)
  out[: len(ids)] = np.asarray(ids, dtype=np.int32)
  return out


def _canvas_chunk(
    response_ids: list[int],
    *,
    num_canvases: int,
    canvas_size: int,
    eos_id: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  total_capacity = num_canvases * canvas_size
  response = np.asarray(response_ids[:total_capacity], dtype=np.int32)
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


def _encoder_shift(
    prompt: np.ndarray,
    canvas: np.ndarray,
    canvas_mask: np.ndarray,
    *,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
  prompt_valid = prompt != pad_id
  canvas_valid = canvas_mask
  full_seq = np.concatenate([prompt, canvas])
  full_valid = np.concatenate([prompt_valid, canvas_valid])
  target = np.roll(full_seq, -1).astype(np.int32)
  target[-1] = pad_id
  shifted_valid = np.roll(full_valid, -1)
  shifted_valid[-1] = False
  return target, full_valid & shifted_valid


def _model_prompt(example: PreparedExample) -> str:
  return f"{SYSTEM_PROMPT}<|turn>user\n{example.prompt}<turn|>\n<|turn>model\n"


def _model_response(example: PreparedExample, *, use_long_answer: bool) -> str:
  response = example.long_response if use_long_answer else example.response
  return f"{response.strip()} The answer is: {example.short_answer}<turn|>"


def _make_batch(
    examples: list[PreparedExample],
    *,
    tokenizer,
    prompt_len: int,
    canvas_size: int,
    num_canvases: int,
    use_long_answer: bool,
    rng_seed: int,
    tiny_vocab_size: int | None = None,
) -> diffusion_sft.DiffusionGemmaSFTBatch:
  pad_id = tokenizer.pad_id()
  eos_id = tokenizer.eos_id()
  bos_id = tokenizer.bos_id()
  prompts = []
  canvases = []
  canvas_ids = []
  canvas_masks = []
  encoder_targets = []
  encoder_target_masks = []
  for example in examples:
    prompt_ids = [bos_id] + tokenizer.encode(_model_prompt(example))
    response_ids = tokenizer.encode(
        _model_response(example, use_long_answer=use_long_answer)
    )
    if tiny_vocab_size is not None:
      prompt_ids = [x % tiny_vocab_size for x in prompt_ids]
      response_ids = [x % tiny_vocab_size for x in response_ids]
      pad_id = pad_id % tiny_vocab_size
      eos_id = eos_id % tiny_vocab_size

    prompt = _pad_or_truncate(prompt_ids, prompt_len, pad_id)
    canvas, canvas_id, canvas_mask = _canvas_chunk(
        response_ids,
        num_canvases=num_canvases,
        canvas_size=canvas_size,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    encoder_target, encoder_target_mask = _encoder_shift(
        prompt, canvas, canvas_mask, pad_id=pad_id
    )
    prompts.append(prompt)
    canvases.append(canvas[:, None])
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


def _load_tiny_model(args: argparse.Namespace, vocab_size: int):
  config = diffusion_model.ModelConfig.tiny(
      vocab_size=vocab_size,
      num_layers=args.tiny_layers,
      embed_dim=args.tiny_embed_dim,
      hidden_dim=args.tiny_hidden_dim,
      num_heads=2,
      head_dim=max(4, args.tiny_embed_dim // 2),
      num_kv_heads=1,
  )
  config = dataclasses.replace(
      config,
      attention_pattern=(gemma4_model.AttentionType.GLOBAL,) * args.tiny_layers,
      num_global_kv_heads=1,
      global_key_size=max(4, args.tiny_embed_dim // 2),
      use_sliding_window_kv_cache=False,
      final_logit_softcap=None,
      remat_config=(
          gemma4_model.RematConfig.DECODER if args.remat_decoder else None
      ),
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  return diffusion_model.DiffusionGemma_A26B_A4B(config, rngs=nnx.Rngs(0))


def _load_real_model(args: argparse.Namespace):
  mesh_fsdp, mesh_tp = _resolve_mesh_axes(args)
  mesh = _mesh(mesh_fsdp, mesh_tp)
  remat_config = (
      gemma4_model.RematConfig.DECODER if args.remat_decoder else None
  )
  with mesh:
    return diffusion_params.create_model_from_checkpoint(
        args.checkpoint,
        diffusion_model.ModelConfig.diffusion_gemma_a26b_a4b(
            remat_config=remat_config
        ),
        mesh=mesh,
        dtype=_dtype_from_name(args.dtype),
        restore_concurrent_gb=args.restore_concurrent_gb,
    )


def _tree_l2_norm(state: Any) -> jax.Array:
  leaves = jax.tree.leaves(state)
  if not leaves:
    return jnp.asarray(0.0, dtype=jnp.float32)
  total = jnp.asarray(0.0, dtype=jnp.float32)
  for leaf in leaves:
    if hasattr(leaf, "shape"):
      value = leaf.astype(jnp.float32)
      total = total + jnp.sum(value * value)
  return jnp.sqrt(total)


def _small_leaf_checksums(
    state: Any,
    *,
    limit: int = 8,
    max_size: int = 32768,
) -> dict[tuple[Any, ...], jax.Array]:
  pure = nnx.to_pure_dict(state)
  flat = flax.traverse_util.flatten_dict(pure)
  checksums = {}
  for path, value in sorted(flat.items(), key=lambda item: str(item[0])):
    if not hasattr(value, "shape") or value.size > max_size:
      continue
    try:
      checksums[path] = jnp.sum(value.astype(jnp.float32))
    except (TypeError, NotImplementedError):
      continue
    if len(checksums) >= limit:
      break
  return checksums


def _max_checksum_delta(
    before: dict[tuple[Any, ...], jax.Array],
    after_state: Any,
) -> float:
  pure = nnx.to_pure_dict(after_state)
  flat = flax.traverse_util.flatten_dict(pure)
  deltas = []
  for path, before_value in before.items():
    after_value = flat[tuple(path)]
    deltas.append(
        jnp.abs(jnp.sum(after_value.astype(jnp.float32)) - before_value)
    )
  if not deltas:
    return 0.0
  return float(jax.device_get(jnp.max(jnp.stack(deltas))))


def _state_summary(state: Any) -> dict[str, int]:
  leaves = [leaf for leaf in jax.tree.leaves(state) if hasattr(leaf, "shape")]
  num_elements = sum(int(leaf.size) for leaf in leaves)
  num_bytes = sum(int(leaf.size * leaf.dtype.itemsize) for leaf in leaves)
  return {
      "leaves": len(leaves),
      "elements": num_elements,
      "bytes": num_bytes,
  }


def _block_until_ready_state(state: Any) -> None:
  for leaf in jax.tree.leaves(state):
    if hasattr(leaf, "block_until_ready"):
      leaf.block_until_ready()


def _block_until_ready_first(value: Any) -> None:
  leaves = jax.tree.leaves(value)
  if leaves and hasattr(leaves[0], "block_until_ready"):
    leaves[0].block_until_ready()


def _checkpoint_dir(args: argparse.Namespace, *, prefix: str) -> str:
  ckpt_dir = args.checkpoint_dir or tempfile.mkdtemp(prefix=prefix)
  pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
  return ckpt_dir


def _write_result(
    result: dict[str, Any],
    *,
    event: str,
    gpu_memory_monitor: _GpuMemoryMonitor,
) -> dict[str, Any]:
  gpu_memory = gpu_memory_monitor.stop()
  if gpu_memory is not None:
    result["gpu_memory"] = gpu_memory
    _log("gpu_memory_summary", **gpu_memory)
  pathlib.Path(result["minimal_state_path"]).write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  _log(event, **result)
  return result


def _train_with_separate_loss_jits(
    trainer: diffusion_sft.DiffusionGemmaTrainer,
    train_batches: list[dict[str, jax.Array]],
    diffusion_config: diffusion_sft.DiffusionGemmaSFTConfig,
    max_runtime_seconds: float = 0.0,
) -> bool:
  """Runs exact split-gradient training with separate JAX executables."""
  if trainer.config.gradient_accumulation_steps not in (None, 1):
    raise ValueError(
        "--separate_loss_jits does not support gradient accumulation."
    )

  grad_arg = nnx.DiffState(0, nnx.LoRAParam) if trainer._lora_enabled else 0
  decoder_config = dataclasses.replace(
      diffusion_config,
      encoder_loss_weight=0.0,
      force_full_encoder_prefill=True,
  )
  encoder_config = dataclasses.replace(
      diffusion_config, decoder_loss_weight=0.0
  )
  encoder_loss_fn = diffusion_sft.make_loss_fn(encoder_config)

  def precompute_self_conditioning_prefill_step(
      model: nnx.Module, batch: dict[str, jax.Array]
  ):
    inputs = diffusion_sft.gen_model_input_fn(batch)
    return diffusion_sft.diffusion_gemma_sft_self_conditioning_prefill(
        model,
        prompt=inputs["prompt"],
        canvas=inputs["canvas"],
        canvas_mask=inputs["canvas_mask"],
        rng=inputs["rng"],
        config=decoder_config,
    )

  def precompute_self_conditioning_decode_step(
      model: nnx.Module, prefill: dict[str, Any]
  ):
    return (
        diffusion_sft.diffusion_gemma_sft_self_conditioning_logits_from_prefill(
            model,
            **prefill,
            config=decoder_config,
        )
    )

  def decoder_loss_fn(
      model: nnx.Module,
      prompt: jax.Array,
      canvas: jax.Array,
      canvas_id: jax.Array,
      canvas_mask: jax.Array,
      encoder_target: jax.Array,
      encoder_target_mask: jax.Array,
      rng: jax.Array,
      precomputed_sc_logits: jax.Array,
      precomputed_self_conditioning_mask: jax.Array,
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
        precomputed_sc_logits=precomputed_sc_logits,
        precomputed_self_conditioning_mask=precomputed_self_conditioning_mask,
    )

  def decoder_grad_step(
      model: nnx.Module,
      batch: dict[str, jax.Array],
      precomputed_sc_logits: jax.Array,
      precomputed_self_conditioning_mask: jax.Array,
  ):
    grad_fn = nnx.value_and_grad(
        decoder_loss_fn, argnums=grad_arg, has_aux=True
    )
    return grad_fn(
        model,
        **diffusion_sft.gen_model_input_fn(batch),
        precomputed_sc_logits=precomputed_sc_logits,
        precomputed_self_conditioning_mask=precomputed_self_conditioning_mask,
    )

  def encoder_grad_step(model: nnx.Module, batch: dict[str, jax.Array]):
    grad_fn = nnx.value_and_grad(
        encoder_loss_fn, argnums=grad_arg, has_aux=True
    )
    return grad_fn(model, **diffusion_sft.gen_model_input_fn(batch))

  def apply_split_grad_step(
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      decoder_grads: Any,
      encoder_grads: Any,
  ) -> jax.Array:
    grads = jax.tree.map(jnp.add, decoder_grads, encoder_grads)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return grad_norm

  precompute_self_conditioning_prefill_step = nnx.jit(
      precompute_self_conditioning_prefill_step
  )
  precompute_self_conditioning_decode_step = nnx.jit(
      precompute_self_conditioning_decode_step
  )
  decoder_grad_step = nnx.jit(decoder_grad_step)
  encoder_grad_step = nnx.jit(encoder_grad_step)
  apply_split_grad_step = nnx.jit(
      apply_split_grad_step, donate_argnames=("optimizer",)
  )

  max_steps = trainer.config.max_steps or len(train_batches)
  start_time = time.monotonic()
  timed_out = False
  for step in range(max_steps):
    batch = train_batches[step % len(train_batches)]
    step_started = time.monotonic()
    sc_prefill = precompute_self_conditioning_prefill_step(trainer.model, batch)
    jax.tree.leaves(sc_prefill)[0].block_until_ready()
    if step == 0 or (step + 1) % 50 == 0:
      _log(
          "separate_loss_jit_phase",
          step=step + 1,
          phase="self_conditioning_prefill",
          elapsed_seconds=time.monotonic() - step_started,
      )
    phase_started = time.monotonic()
    sc_logits, do_self_cond = precompute_self_conditioning_decode_step(
        trainer.model, sc_prefill
    )
    sc_logits.block_until_ready()
    if step == 0 or (step + 1) % 50 == 0:
      _log(
          "separate_loss_jit_phase",
          step=step + 1,
          phase="self_conditioning_decode",
          elapsed_seconds=time.monotonic() - phase_started,
      )
    phase_started = time.monotonic()
    (decoder_loss, decoder_aux), decoder_grads = decoder_grad_step(
        trainer.model, batch, sc_logits, do_self_cond
    )
    _block_until_ready_first(decoder_loss)
    del sc_prefill, sc_logits, do_self_cond
    if step == 0 or (step + 1) % 50 == 0:
      _log(
          "separate_loss_jit_phase",
          step=step + 1,
          phase="decoder_grad",
          elapsed_seconds=time.monotonic() - phase_started,
      )
    phase_started = time.monotonic()
    (encoder_loss, encoder_aux), encoder_grads = encoder_grad_step(
        trainer.model, batch
    )
    _block_until_ready_first(encoder_loss)
    if step == 0 or (step + 1) % 50 == 0:
      _log(
          "separate_loss_jit_phase",
          step=step + 1,
          phase="encoder_grad",
          elapsed_seconds=time.monotonic() - phase_started,
      )
    phase_started = time.monotonic()
    grad_norm = apply_split_grad_step(
        trainer.model, trainer.optimizer, decoder_grads, encoder_grads
    )
    del decoder_grads, encoder_grads
    loss = decoder_loss + encoder_loss
    loss.block_until_ready()
    grad_norm.block_until_ready()
    if step == 0 or (step + 1) % 50 == 0:
      _log(
          "separate_loss_jit_phase",
          step=step + 1,
          phase="apply_split_grad",
          elapsed_seconds=time.monotonic() - phase_started,
      )
    aux = dict(decoder_aux)
    aux["encoder_loss"] = encoder_aux["encoder_loss"]
    trainer.last_train_aux = aux
    trainer.last_train_loss = loss
    trainer._iter_steps += 1
    trainer._train_steps += 1
    total_value = float(jax.device_get(loss))
    decoder_value = float(jax.device_get(decoder_aux["decoder_loss"]))
    encoder_value = float(jax.device_get(encoder_aux["encoder_loss"]))
    _log(
        "separate_loss_jit_step",
        step=step + 1,
        loss=total_value,
        decoder_loss=decoder_value,
        encoder_loss=encoder_value,
        grad_norm=float(jax.device_get(grad_norm)),
    )
    for metric, value in (
        ("losses/diffusion_loss", decoder_value),
        ("losses/encoder_loss", encoder_value),
        ("losses/total", total_value),
    ):
      _log(
          "diffusion_gemma_native_loss",
          metric=metric,
          value=value,
          step=trainer.train_steps,
          loop_step=step,
      )
    if max_runtime_seconds and time.monotonic() - start_time >= max_runtime_seconds:
      timed_out = True
      _log(
          "native_max_runtime_reached",
          step=trainer.train_steps,
          max_runtime_seconds=max_runtime_seconds,
      )
      break
  return timed_out


def _make_train_batches(
    examples: list[PreparedExample],
    *,
    tokenizer,
    args: argparse.Namespace,
    vocab_size: int,
) -> list[dict[str, jax.Array]]:
  tiny_vocab_size = vocab_size if args.tiny else None
  batches = []
  accumulation_steps = args.gradient_accumulation_steps or 1
  num_micro_batches = args.steps * accumulation_steps
  for micro_step in range(num_micro_batches):
    start = (micro_step * args.batch_size) % len(examples)
    batch_examples = [
        examples[(start + offset) % len(examples)]
        for offset in range(args.batch_size)
    ]
    batch = _make_batch(
        batch_examples,
        tokenizer=tokenizer,
        prompt_len=args.prompt_len,
        canvas_size=args.canvas_size,
        num_canvases=args.num_canvases,
        use_long_answer=args.use_long_answer,
        rng_seed=args.seed + 10_000 + micro_step,
        tiny_vocab_size=tiny_vocab_size,
    )
    batches.append(diffusion_sft.gen_model_input_fn(batch))
  return batches


def run(args: argparse.Namespace) -> dict[str, Any]:
  mesh_fsdp, mesh_tp = _resolve_mesh_axes(args)
  _log(
      "devices",
      devices=[str(d) for d in jax.devices()],
      backend=jax.default_backend(),
      mesh_fsdp=mesh_fsdp,
      mesh_tp=mesh_tp,
  )
  gpu_memory_monitor = _GpuMemoryMonitor(args.gpu_memory_poll_seconds)
  gpu_memory_monitor.start()

  if args.load_only:
    if args.tiny:
      vocab_size = args.tiny_vocab_size
      model = _load_tiny_model(args, vocab_size)
    else:
      vocab_size = (
          diffusion_model.ModelConfig.diffusion_gemma_a26b_a4b().num_embed
      )
      _log("loading_model", checkpoint=args.checkpoint, dtype=args.dtype)
      model = _load_real_model(args)

    model = diffusion_sft.apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        module_path=args.lora_module_path,
    )
    _block_until_ready_state(nnx.state(model))
    _log(
        "model_ready",
        tiny=args.tiny,
        lora_rank=args.lora_rank,
        lora_module_path=args.lora_module_path,
        remat_decoder=args.remat_decoder,
        load_only=True,
    )
    trainable_summary = _state_summary(nnx.state(model, nnx.LoRAParam))
    frozen_summary = _state_summary(
        nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
    )
    _log(
        "trainable_state",
        lora=trainable_summary,
        frozen=frozen_summary,
        lora_fraction=(
            trainable_summary["elements"]
            / max(
                trainable_summary["elements"] + frozen_summary["elements"], 1
            )
        ),
    )
    ckpt_dir = _checkpoint_dir(
        args, prefix="diffusion_gemma_pubmedqa_load_only_"
    )
    result = {
        "mode": "load_only",
        "steps": 0,
        "checkpoint_dir": ckpt_dir,
        "checkpoint_dir_exists": pathlib.Path(ckpt_dir).exists(),
        "minimal_state_path": str(pathlib.Path(ckpt_dir) / "minimal_state.json"),
        "checkpoint": args.checkpoint,
        "tiny": args.tiny,
        "dtype": args.dtype,
        "vocab_size": vocab_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_module_path": args.lora_module_path,
        "remat_decoder": args.remat_decoder,
        "mesh_fsdp": mesh_fsdp,
        "mesh_tp": mesh_tp,
        "restore_concurrent_gb": args.restore_concurrent_gb,
        "trainable_state": trainable_summary,
        "frozen_state": frozen_summary,
    }
    if not result["checkpoint_dir_exists"]:
      raise RuntimeError(f"Checkpoint directory was not created: {ckpt_dir}")
    return _write_result(
        result,
        event="load_only_complete",
        gpu_memory_monitor=gpu_memory_monitor,
    )

  train_path, test_path = _prepare_pubmedqa(args)
  all_examples = _read_jsonl(train_path)
  examples = all_examples[args.slice_start :]
  if args.max_examples:
    examples = examples[: args.max_examples]
  if len(examples) < args.batch_size:
    raise ValueError(
        f"Need at least batch_size={args.batch_size} examples, got"
        f" {len(examples)}"
    )
  _log(
      "dataset_loaded",
      train_path=str(train_path),
      test_path=str(test_path),
      selected_examples=len(examples),
      slice_start=args.slice_start,
      use_long_answer=args.use_long_answer,
      first_pubmed_id=examples[0].pubmed_id,
  )

  tokenizer = _load_tokenizer(args.tokenizer)
  tokenizer_vocab_size = tokenizer.tokenizer.GetPieceSize()
  vocab_size = args.tiny_vocab_size if args.tiny else tokenizer_vocab_size
  _log(
      "tokenizer_loaded",
      tokenizer=args.tokenizer,
      tokenizer_vocab_size=tokenizer_vocab_size,
      training_vocab_size=vocab_size,
      bos_id=tokenizer.bos_id(),
      eos_id=tokenizer.eos_id(),
      pad_id=tokenizer.pad_id(),
  )

  if args.tiny:
    model = _load_tiny_model(args, vocab_size)
  else:
    _log("loading_model", checkpoint=args.checkpoint, dtype=args.dtype)
    model = _load_real_model(args)

  model = diffusion_sft.apply_lora(
      model,
      rank=args.lora_rank,
      alpha=args.lora_alpha,
      module_path=args.lora_module_path,
  )
  _log(
      "model_ready",
      tiny=args.tiny,
      lora_rank=args.lora_rank,
      lora_module_path=args.lora_module_path,
      remat_decoder=args.remat_decoder,
  )
  trainable_summary = _state_summary(nnx.state(model, nnx.LoRAParam))
  frozen_summary = _state_summary(
      nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
  )
  _log(
      "trainable_state",
      lora=trainable_summary,
      frozen=frozen_summary,
      lora_fraction=(
          trainable_summary["elements"]
          / max(trainable_summary["elements"] + frozen_summary["elements"], 1)
      ),
  )

  diffusion_config = diffusion_sft.DiffusionGemmaSFTConfig(
      prompt_len=args.prompt_len,
      canvas_size=args.canvas_size,
      num_canvases=args.num_canvases,
      vocab_size=vocab_size,
      self_cond_prob=args.self_cond_prob,
      decoder_loss_weight=args.decoder_loss_weight,
      encoder_loss_weight=args.encoder_loss_weight,
      stop_gradient_from_denoiser_to_encoder=(
          args.stop_gradient_from_denoiser_to_encoder
      ),
      decoder_implementation=args.decoder_implementation,
      fast_uniform_corruption=args.fast_uniform_corruption,
      encoder_loss_chunk_size=args.encoder_loss_chunk_size,
  )
  train_batches = _make_train_batches(
      examples, tokenizer=tokenizer, args=args, vocab_size=vocab_size
  )
  initial_loss, initial_aux = diffusion_sft.make_loss_fn(diffusion_config)(
      model, **train_batches[0]
  )
  initial_loss.block_until_ready()
  if not bool(jnp.isfinite(initial_loss)):
    raise RuntimeError(f"Initial PubMedQA loss is not finite: {initial_loss}")
  _log(
      "initial_loss",
      loss=float(jax.device_get(initial_loss)),
      decoder_loss=float(jax.device_get(initial_aux["decoder_loss"])),
      encoder_loss=float(jax.device_get(initial_aux["encoder_loss"])),
      corrupted_fraction=float(
          jax.device_get(initial_aux["corrupted_fraction"])
      ),
      time_mean=float(jax.device_get(initial_aux["time_mean"])),
  )
  trainable_summary = _state_summary(nnx.state(model, nnx.LoRAParam))
  frozen_summary = _state_summary(
      nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
  )
  _log(
      "trainable_state_after_materialize",
      lora=trainable_summary,
      frozen=frozen_summary,
      lora_fraction=(
          trainable_summary["elements"]
          / max(trainable_summary["elements"] + frozen_summary["elements"], 1)
      ),
  )

  if args.initial_loss_only:
    ckpt_dir = _checkpoint_dir(
        args, prefix="diffusion_gemma_pubmedqa_initial_loss_"
    )
    result = {
        "mode": "initial_loss_only",
        "steps": 0,
        "checkpoint_dir": ckpt_dir,
        "checkpoint_dir_exists": pathlib.Path(ckpt_dir).exists(),
        "minimal_state_path": str(pathlib.Path(ckpt_dir) / "minimal_state.json"),
        "initial_loss": float(jax.device_get(initial_loss)),
        "initial_decoder_loss": float(jax.device_get(initial_aux["decoder_loss"])),
        "initial_encoder_loss": float(jax.device_get(initial_aux["encoder_loss"])),
        "corrupted_fraction": float(
            jax.device_get(initial_aux["corrupted_fraction"])
        ),
        "time_mean": float(jax.device_get(initial_aux["time_mean"])),
        "first_pubmed_id": examples[0].pubmed_id,
        "prompt_len": args.prompt_len,
        "canvas_size": args.canvas_size,
        "num_canvases": args.num_canvases,
        "batch_size": args.batch_size,
        "decoder_implementation": args.decoder_implementation,
        "remat_decoder": args.remat_decoder,
        "encoder_loss_chunk_size": args.encoder_loss_chunk_size,
        "mesh_fsdp": mesh_fsdp,
        "mesh_tp": mesh_tp,
        "trainable_state": trainable_summary,
        "frozen_state": frozen_summary,
    }
    if not result["checkpoint_dir_exists"]:
      raise RuntimeError(f"Checkpoint directory was not created: {ckpt_dir}")
    return _write_result(
        result,
        event="initial_loss_only_complete",
        gpu_memory_monitor=gpu_memory_monitor,
    )

  lora_before_norm = _tree_l2_norm(nnx.state(model, nnx.LoRAParam))
  lora_before_checksums = _small_leaf_checksums(
      nnx.state(model, nnx.LoRAParam), limit=32, max_size=262144
  )
  base_before_checksums = _small_leaf_checksums(
      nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
  )
  ckpt_dir = _checkpoint_dir(args, prefix="diffusion_gemma_pubmedqa_ckpt_")
  optimizer = optax.chain(
      optax.clip_by_global_norm(args.max_grad_norm),
      optax.adamw(
          learning_rate=args.learning_rate,
          b1=0.95,
          b2=0.99,
          eps=1e-8,
          weight_decay=args.weight_decay,
      ),
  )
  train_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=max(1, args.steps),
      max_steps=args.steps,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      checkpoint_root_directory=ckpt_dir if args.orbax_checkpoint else None,
      max_inflight_computations=1,
      pbar_description=None,
  )
  trainer = diffusion_sft.DiffusionGemmaTrainer(
      model,
      optimizer,
      train_config,
      diffusion_config,
      split_loss_gradients=args.split_loss_gradients,
  )
  timed_out = False
  if args.max_runtime_seconds and not args.separate_loss_jits:
    raise ValueError("--max_runtime_seconds requires --separate_loss_jits.")
  if args.separate_loss_jits:
    if not args.split_loss_gradients:
      raise ValueError("--separate_loss_jits requires --split_loss_gradients.")
    timed_out = _train_with_separate_loss_jits(
        trainer,
        train_batches,
        diffusion_config,
        max_runtime_seconds=args.max_runtime_seconds,
    )
  else:
    trainer.train(train_batches, cache_nnx_graph=False)
  final_loss, final_aux = diffusion_sft.make_loss_fn(diffusion_config)(
      model, **train_batches[0]
  )
  final_loss.block_until_ready()
  if not bool(jnp.isfinite(final_loss)):
    raise RuntimeError(f"Final PubMedQA loss is not finite: {final_loss}")
  lora_after_norm = _tree_l2_norm(nnx.state(model, nnx.LoRAParam))
  lora_norm_delta = float(
      jax.device_get(jnp.abs(lora_after_norm - lora_before_norm))
  )
  lora_checksum_delta = _max_checksum_delta(
      lora_before_checksums, nnx.state(model, nnx.LoRAParam)
  )
  base_checksum_delta = _max_checksum_delta(
      base_before_checksums, nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam))
  )
  result = {
      "steps": trainer.train_steps,
      "checkpoint_dir": ckpt_dir,
      "checkpoint_dir_exists": pathlib.Path(ckpt_dir).exists(),
      "orbax_checkpoint": args.orbax_checkpoint,
      "minimal_state_path": str(pathlib.Path(ckpt_dir) / "minimal_state.json"),
      "initial_loss": float(jax.device_get(initial_loss)),
      "final_loss": float(jax.device_get(final_loss)),
      "final_decoder_loss": float(jax.device_get(final_aux["decoder_loss"])),
      "final_encoder_loss": float(jax.device_get(final_aux["encoder_loss"])),
      "lora_norm_before": float(jax.device_get(lora_before_norm)),
      "lora_norm_after": float(jax.device_get(lora_after_norm)),
      "lora_norm_delta": lora_norm_delta,
      "lora_checksum_delta": lora_checksum_delta,
      "base_checksum_delta": base_checksum_delta,
      "first_pubmed_id": examples[0].pubmed_id,
      "prompt_len": args.prompt_len,
      "canvas_size": args.canvas_size,
      "num_canvases": args.num_canvases,
      "batch_size": args.batch_size,
      "gradient_accumulation_steps": args.gradient_accumulation_steps or 1,
      "effective_batch_size": (
          args.batch_size * (args.gradient_accumulation_steps or 1)
      ),
      "decoder_implementation": args.decoder_implementation,
      "remat_decoder": args.remat_decoder,
      "prefill_decode_only_last_token": args.encoder_loss_weight == 0.0,
      "encoder_loss_chunk_size": args.encoder_loss_chunk_size,
      "split_loss_gradients": args.split_loss_gradients,
      "separate_loss_jits": args.separate_loss_jits,
      "precomputed_self_conditioning": args.separate_loss_jits,
      "max_runtime_seconds": args.max_runtime_seconds,
      "timed_out": timed_out,
      "mesh_fsdp": mesh_fsdp,
      "mesh_tp": mesh_tp,
      "trainable_state": trainable_summary,
      "frozen_state": frozen_summary,
  }
  if trainer.train_steps < args.steps and not timed_out:
    raise RuntimeError(
        f"Expected {args.steps} train steps, got {trainer.train_steps}"
    )
  if not result["checkpoint_dir_exists"]:
    raise RuntimeError(f"Checkpoint directory was not created: {ckpt_dir}")
  if lora_norm_delta <= 0 and lora_checksum_delta <= 0:
    raise RuntimeError("LoRA parameters did not change during training.")
  if base_checksum_delta != 0:
    raise RuntimeError(
        "Sampled non-LoRA parameter checksums changed during LoRA training."
    )
  return _write_result(
      result,
      event="train_complete",
      gpu_memory_monitor=gpu_memory_monitor,
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--tiny", action=argparse.BooleanOptionalAction, default=False
  )
  parser.add_argument(
      "--load_only",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Load the DiffusionGemma checkpoint, attach LoRA params, write a "
          "minimal_state.json proof artifact, and exit before dataset, loss, "
          "or trainer construction. This isolates checkpoint/LoRA peak memory "
          "from train-step peak memory."
      ),
  )
  parser.add_argument(
      "--initial_loss_only",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Load model/data, attach LoRA params, compute the initial "
          "DiffusionGemma SFT loss once, write minimal_state.json, and exit "
          "before optimizer/trainer construction."
      ),
  )
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument(
      "--max_runtime_seconds",
      type=float,
      default=0.0,
      help=(
          "Stop cleanly after this many seconds in --separate_loss_jits mode. "
          "Use with a large --steps value for timed comparison runs."
      ),
  )
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
  parser.add_argument("--prompt_len", type=int, default=1024)
  parser.add_argument("--canvas_size", type=int, default=128)
  parser.add_argument("--num_canvases", type=int, default=2)
  parser.add_argument(
      "--use_long_answer", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--self_cond_prob", type=float, default=1.0)
  parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
  parser.add_argument("--encoder_loss_weight", type=float, default=1.0)
  parser.add_argument(
      "--encoder_loss_chunk_size",
      type=int,
      default=64,
      help=(
          "Chunk size for encoder AR loss projection. Set to 0 to materialize "
          "full encoder logits before CE."
      ),
  )
  parser.add_argument(
      "--decoder_implementation",
      choices=[
          "cached_selected_canvas",
          "cached_selected_canvas_slice",
          "full_sequence",
      ],
      default="cached_selected_canvas",
  )
  parser.add_argument(
      "--fast_uniform_corruption",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--stop_gradient_from_denoiser_to_encoder",
      action=argparse.BooleanOptionalAction,
      default=False,
  )
  parser.add_argument(
      "--split_loss_gradients",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Compute decoder and encoder gradients in separate exact passes and "
          "sum them before the optimizer update. This is slower but lowers the "
          "peak memory of 26B LoRA validation runs on 2x80GB GPUs."
      ),
  )
  parser.add_argument(
      "--separate_loss_jits",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "When --split_loss_gradients is enabled, compile decoder-gradient, "
          "encoder-gradient, and optimizer-update steps as separate JAX "
          "executables. This is slower but further lowers peak memory for "
          "2x80GB GPU validation runs."
      ),
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--data_dir", default="/tmp/pubmedqa_jsonl")
  parser.add_argument("--pubmedqa_repo_dir", default="/tmp/pubmedqa_repo")
  parser.add_argument("--train_jsonl", default="")
  parser.add_argument("--test_jsonl", default="")
  parser.add_argument("--slice_start", type=int, default=50)
  parser.add_argument("--max_examples", type=int, default=8)
  parser.add_argument("--max_context_chars", type=int, default=4000)
  parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
  parser.add_argument(
      "--checkpoint", default=diffusion_params.DIFFUSIONGEMMA_A26B_A4B_IT
  )
  parser.add_argument("--checkpoint_dir", default="")
  parser.add_argument(
      "--orbax_checkpoint",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Use Tunix/Orbax checkpointing. Disabled by default for public 26B "
          "GPU validation runs because the optimizer checkpoint can exceed memory; "
          "a minimal_state.json proof artifact is always written instead."
      ),
  )
  parser.add_argument(
      "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
  )
  parser.add_argument(
      "--remat_decoder",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Enable decoder rematerialization. DiffusionGemma LoRA is"
          " materialized with remat temporarily disabled and then restored, so"
          " backbone LoRA coverage is preserved."
      ),
  )
  parser.add_argument("--restore_concurrent_gb", type=int, default=16)
  parser.add_argument(
      "--gpu_memory_poll_seconds",
      type=float,
      default=0.0,
      help=(
          "Poll nvidia-smi at this interval and write peak VRAM/headroom into "
          "the final train_complete JSON. Disabled when set to 0."
      ),
  )
  parser.add_argument(
      "--mesh_fsdp",
      type=int,
      default=None,
      help=(
          "FSDP mesh axis. Defaults to jax.device_count() when --mesh_tp is "
          "also omitted, matching the official PubMedQA recipe's FSDP-first "
          "layout. If only --mesh_tp is set, inferred from device count."
      ),
  )
  parser.add_argument(
      "--mesh_tp",
      type=int,
      default=None,
      help=(
          "Tensor-parallel mesh axis. Defaults to 1 when --mesh_fsdp is also "
          "omitted. If only --mesh_fsdp is set, inferred from device count."
      ),
  )
  parser.add_argument("--lora_rank", type=int, default=4)
  parser.add_argument("--lora_alpha", type=float, default=8.0)
  parser.add_argument(
      "--lora_module_path",
      default=diffusion_sft.DEFAULT_LORA_MODULE_PATH,
  )
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--tiny_vocab_size", type=int, default=256)
  parser.add_argument("--tiny_layers", type=int, default=1)
  parser.add_argument("--tiny_embed_dim", type=int, default=16)
  parser.add_argument("--tiny_hidden_dim", type=int, default=32)
  return parser.parse_args()


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
