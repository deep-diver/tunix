#!/usr/bin/env bash
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

set -euxo pipefail

GEMMA_REF="${GEMMA_REF:-${HOME}/gemma}"
HACKABLE_DIFFUSION_REF="${HACKABLE_DIFFUSION_REF:-${HOME}/hackable_diffusion}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.13)"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

rm -rf "${GEMMA_REF}" "${HACKABLE_DIFFUSION_REF}"
git clone --depth=1 https://github.com/google/hackable_diffusion.git \
  "${HACKABLE_DIFFUSION_REF}"
git clone --depth=1 https://github.com/google-deepmind/gemma.git \
  "${GEMMA_REF}"

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e "${HACKABLE_DIFFUSION_REF}"
"${PYTHON_BIN}" -m pip install -e "${GEMMA_REF}"

# The official DiffusionGemma README asks for the CUDA 13 JAX wheel. H100 VMs
# have also been validated with CUDA 12 JAX wheels, which can be selected with
# JAX_CUDA_EXTRA=cuda12. Remove stale plugins before installing the requested
# wheel.
"${PYTHON_BIN}" -m pip uninstall -y \
  jax-cuda12-plugin \
  jax-cuda12-pjrt \
  jax-cuda13-plugin \
  jax-cuda13-pjrt \
  nvidia-nccl-cu12 \
  nvidia-nccl-cu13 || true
"${PYTHON_BIN}" -m pip install -U "jax[${JAX_CUDA_EXTRA}]"

(
  cd "${GEMMA_REF}/gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa"
  bash prepare_pubmedqa_dataset.sh
)
