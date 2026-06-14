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

"""Checks that a rendered DiffusionGemma trace matches official generation."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--generation_json", required=True)
  parser.add_argument("--trace_json", required=True)
  parser.add_argument("--output_json", default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  generation = _read_json(args.generation_json)
  trace = _read_json(args.trace_json)
  sample_index = 0
  sample = generation["samples"][sample_index]
  sample_tokens = [int(token_id) for token_id in sample["response_token_ids"]]
  trace_tokens = [int(token_id) for token_id in trace["output_token_ids"]]
  backend = generation.get("backend", {})
  result = {
      "event": "diffusion_gemma_generation_trace_parity",
      "generation_json": str(args.generation_json),
      "trace_json": str(args.trace_json),
      "sample_index": sample_index,
      "official_backend_path": backend.get("path"),
      "evaluator_class": backend.get("evaluator_class"),
      "sampler_class": backend.get("sampler_class"),
      "lora_backend": backend.get("lora_backend"),
      "token_ids_equal": sample_tokens == trace_tokens,
      "num_generation_tokens": len(sample_tokens),
      "num_trace_tokens": len(trace_tokens),
      "output_text_equal": (
          sample.get("response_text") == trace.get("output_text")
      ),
  }
  if not result["token_ids_equal"]:
    raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
  if args.output_json:
    _write_json(args.output_json, result)
  print(json.dumps(result, indent=2, sort_keys=True))


def _read_json(path: str | pathlib.Path) -> Any:
  return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | pathlib.Path, payload: Any) -> None:
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )


if __name__ == "__main__":
  main()
