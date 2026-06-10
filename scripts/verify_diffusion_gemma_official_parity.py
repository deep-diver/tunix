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

"""Parity checks against official DiffusionGemma/Hackable Diffusion helpers.

This script imports the official source trees directly and compares Tunix helper
outputs against the real official implementations for the deterministic pieces
that are portable without constructing the full Kauldron trainer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tunix.models.diffusion_gemma import sft as diffusion_sft


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


def _install_kauldron_ktyping_stub() -> None:
  """Installs a tiny kauldron.ktyping stub for official helper imports."""
  if "kauldron.ktyping" in sys.modules:
    return
  kauldron_mod = sys.modules.setdefault(
      "kauldron", types.ModuleType("kauldron")
  )
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


def _add_reference_paths(gemma_ref: str, hackable_ref: str) -> None:
  for path in (hackable_ref, gemma_ref):
    if path and path not in sys.path:
      sys.path.insert(0, path)


def _assert_equal(name: str, actual, expected, results: dict[str, Any]) -> None:
  np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
  results[name] = {
      "shape": list(np.asarray(actual).shape),
      "max_abs_diff": 0.0,
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
  actual_np = np.asarray(actual)
  expected_np = np.asarray(expected)
  np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol)
  results[name] = {
      "shape": list(actual_np.shape),
      "max_abs_diff": float(np.max(np.abs(actual_np - expected_np))),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--gemma_ref", default="/tmp/gemma-diffusion-reference")
  parser.add_argument(
      "--hackable_ref", default="/tmp/hackable-diffusion-reference"
  )
  parser.add_argument("--rtol", type=float, default=1e-6)
  parser.add_argument("--atol", type=float, default=1e-6)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  _install_kauldron_ktyping_stub()
  _add_reference_paths(args.gemma_ref, args.hackable_ref)

  from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
  from hackable_diffusion.lib import jax_helpers
  from hackable_diffusion.lib.corruption import discrete
  from hackable_diffusion.lib.corruption import schedules
  from hackable_diffusion.lib.training import discrete_loss
  from hackable_diffusion.lib.training import time_sampling

  results: dict[str, Any] = {}

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

  _assert_equal(
      "positions",
      diffusion_sft.build_positions_from_mask(prompt_mask),
      mask_helpers.build_positions_from_mask(prompt_mask),
      results,
  )
  _assert_equal(
      "causal_prefill_mask",
      diffusion_sft.make_causal_prefill_mask(prompt_mask, cache_length=9),
      mask_helpers.make_causal_prefill_mask(prompt_mask, cache_length=9),
      results,
  )
  _assert_equal(
      "decoder_attention_mask",
      diffusion_sft.create_decoder_attention_mask(
          prompt_mask,
          canvas_mask,
          selected_canvas_idx,
          prompt_len=3,
          total_canvas_len=6,
          canvas_size=2,
          num_queries=6,
      ),
      mask_helpers.create_decoder_attention_mask(
          prompt_mask=prompt_mask,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          prompt_len=3,
          total_canvas_len=6,
          canvas_size=2,
          num_queries=6,
      ),
      results,
  )

  cache = {
      "layer_0": {
          "k": jnp.ones((2, 9, 1, 2)),
          "v": jnp.zeros((2, 9, 1, 2)),
          "end_index": jnp.zeros((2,), dtype=jnp.int32),
      }
  }
  end_index = jnp.array([3, 5], dtype=jnp.int32)
  _assert_equal(
      "cache_end_index",
      diffusion_sft.set_cache_end_index(cache, end_index)["layer_0"][
          "end_index"
      ],
      mask_helpers.set_cache_end_index(cache, end_index)["layer_0"][
          "end_index"
      ],
      results,
  )

  x0_tokens = jnp.array(
      [[1, 2, 3, 4, 5, 6], [6, 5, 4, 0, 0, 0]], dtype=jnp.int32
  )
  x0 = x0_tokens[..., None]
  time_sampler = time_sampling.UniformTimeSampler(
      span=jax_helpers.SafeSpan(safety_epsilon=1e-4)
  )
  official_time = time_sampler(jax.random.PRNGKey(7), x0)
  tunix_time = jax.random.uniform(
      jax.random.PRNGKey(7),
      (x0_tokens.shape[0], 1, 1),
      minval=1e-4,
      maxval=1.0 - 1e-4,
      dtype=jnp.float32,
  )
  _assert_close(
      "uniform_time_sampler",
      tunix_time,
      official_time,
      results,
      rtol=args.rtol,
      atol=args.atol,
  )

  vocab_size = 17
  process = discrete.CategoricalProcess.uniform_process(
      num_categories=vocab_size,
      schedule=schedules.RFSchedule(),
  )
  official_xt, official_target = process.corrupt(
      jax.random.PRNGKey(23), x0, official_time
  )
  tunix_xt, tunix_is_corrupted = diffusion_sft._corrupt_tokens(  # pylint: disable=protected-access
      jax.random.PRNGKey(23),
      x0_tokens,
      official_time[..., 0],
      vocab_size,
  )
  _assert_equal("corrupt_xt", tunix_xt, official_xt[..., 0], results)
  _assert_equal(
      "corrupt_is_corrupted",
      tunix_is_corrupted,
      official_target["is_corrupted"][..., 0],
      results,
  )

  logits = jnp.arange(x0_tokens.size * vocab_size, dtype=jnp.float32).reshape(
      x0_tokens.shape + (vocab_size,)
  )
  target_mask = jnp.array(
      [
          [True, True, False, False, False, False],
          [True, True, True, True, False, False],
      ],
      dtype=jnp.bool_,
  )
  official_loss = discrete_loss.NoWeightDiscreteLoss(
      use_mask=True, mask_key="target_mask"
  )(
      preds={"logits": logits},
      targets={"x0": x0, "target_mask": target_mask[..., None]},
      time=official_time,
  )
  tunix_loss = diffusion_sft._masked_ce_loss(  # pylint: disable=protected-access
      logits, x0_tokens, target_mask
  )
  _assert_close(
      "no_weight_discrete_loss_mean",
      tunix_loss,
      jnp.mean(official_loss),
      results,
      rtol=args.rtol,
      atol=args.atol,
  )

  print(json.dumps({"event": "official_parity_passed", "results": results}))


if __name__ == "__main__":
  main()
