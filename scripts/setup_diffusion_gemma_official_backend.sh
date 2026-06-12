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

GEMMA_REF="${GEMMA_REF:-/root/gemma}"
HACKABLE_DIFFUSION_REF="${HACKABLE_DIFFUSION_REF:-/root/hackable_diffusion}"

rm -rf "${GEMMA_REF}" "${HACKABLE_DIFFUSION_REF}"
git clone --depth=1 https://github.com/google/hackable_diffusion.git \
  "${HACKABLE_DIFFUSION_REF}"
git clone --depth=1 https://github.com/google-deepmind/gemma.git \
  "${GEMMA_REF}"

python -m pip install -U pip
python -m pip install -e "${HACKABLE_DIFFUSION_REF}"
python -m pip install -e "${GEMMA_REF}"

# The official DiffusionGemma README asks for the CUDA 13 JAX wheel. Remove
# CUDA 12 plugins if the base image or another setup path pulled them in first.
python -m pip uninstall -y \
  jax-cuda12-plugin \
  jax-cuda12-pjrt \
  nvidia-nccl-cu12 || true
python -m pip install -U "jax[cuda13]"

(
  cd "${GEMMA_REF}/gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa"
  bash prepare_pubmedqa_dataset.sh
)
