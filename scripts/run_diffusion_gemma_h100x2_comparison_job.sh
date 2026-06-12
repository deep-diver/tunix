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

set -euo pipefail

MODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "upstream" && "${MODE}" != "tunix" ]]; then
  echo "Usage: $0 --mode upstream|tunix" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${HOME:-/home/ubuntu}"
RUN_NAME="${RUN_NAME:-${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_BASE="${WORK_BASE:-${HOME_DIR}/diffusion_gemma_compare}"
WORKDIR="${WORKDIR:-${WORK_BASE}/${RUN_NAME}/workdir}"
TRAIN_LOG="${TRAIN_LOG:-${WORKDIR}/train.log}"
GPU_CSV="${GPU_CSV:-${WORKDIR}/gpu_memory.csv}"
RESULT_JSON="${RESULT_JSON:-${WORKDIR}/job_result.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORKDIR}/run_summary.json}"

GEMMA_REF="${GEMMA_REF:-${HOME_DIR}/gemma_official_reference}"
HACKABLE_DIFFUSION_REF="${HACKABLE_DIFFUSION_REF:-${HOME_DIR}/hackable_diffusion_reference}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${HOME_DIR}/checkpoints/diffusiongemma-26B-A4B-it}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-10800}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2000}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-1000}"
DATASET_BATCH_SIZE="${DATASET_BATCH_SIZE:-2}"
LORA_RANK="${LORA_RANK:-4}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12}"

python_version_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
}

if [[ -n "${PYTHON_BIN:-}" ]] && ! python_version_ok "${PYTHON_BIN}"; then
  PYTHON_BIN=""
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.13 >/dev/null 2>&1 && python_version_ok "$(command -v python3.13)"; then
    PYTHON_BIN="$(command -v python3.13)"
  elif command -v python3.12 >/dev/null 2>&1 && python_version_ok "$(command -v python3.12)"; then
    PYTHON_BIN="$(command -v python3.12)"
  elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]] && python_version_ok "${VIRTUAL_ENV}/bin/python"; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif command -v uv >/dev/null 2>&1; then
    uv python install 3.13
    PYTHON_BIN="$(uv python find 3.13)"
  else
    echo "Python >=3.12 is required for official DiffusionGemma packages." >&2
    exit 1
  fi
fi

if [[ -z "${VENV:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ "${PYTHON_BIN}" == "${VIRTUAL_ENV}/bin/python" ]]; then
    VENV="${VIRTUAL_ENV}"
  else
    VENV="${HOME_DIR}/.venvs/diffusion_gemma_${MODE}"
  fi
fi
mkdir -p "${WORKDIR}" "$(dirname "${RESULT_JSON}")"

json_event() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import json
import sys
payload = {"event": sys.argv[1]}
for item in sys.argv[2:]:
  key, _, value = item.partition("=")
  payload[key] = value
print(json.dumps(payload, sort_keys=True), flush=True)
PY
}

json_event comparison_job_start \
  mode="${MODE}" \
  workdir="${WORKDIR}" \
  max_runtime_seconds="${MAX_RUNTIME_SECONDS}" \
  num_train_steps="${NUM_TRAIN_STEPS}" \
  dataset_batch_size="${DATASET_BATCH_SIZE}" \
  lora_rank="${LORA_RANK}"

if [[ ! -x "${VENV}/bin/python" ]] || ! python_version_ok "${VENV}/bin/python"; then
  rm -rf "${VENV}"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_BIN}" --seed "${VENV}"
  else
    "${PYTHON_BIN}" -m venv "${VENV}"
  fi
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install -U pip setuptools wheel

if [[ ! -d "${HACKABLE_DIFFUSION_REF}/.git" ]]; then
  rm -rf "${HACKABLE_DIFFUSION_REF}"
  git clone --depth=1 https://github.com/google/hackable_diffusion.git \
    "${HACKABLE_DIFFUSION_REF}"
fi
if [[ ! -d "${GEMMA_REF}/.git" ]]; then
  rm -rf "${GEMMA_REF}"
  git clone --depth=1 https://github.com/google-deepmind/gemma.git \
    "${GEMMA_REF}"
fi

python -m pip install -e "${HACKABLE_DIFFUSION_REF}"
python -m pip install -e "${GEMMA_REF}"
python -m pip uninstall -y \
  jax-cuda12-plugin \
  jax-cuda12-pjrt \
  jax-cuda13-plugin \
  jax-cuda13-pjrt \
  nvidia-nccl-cu12 \
  nvidia-nccl-cu13 || true
python -m pip install -U "jax[${JAX_CUDA_EXTRA}]" tensorboard

(
  cd "${GEMMA_REF}/gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa"
  bash prepare_pubmedqa_dataset.sh
)

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  python "${REPO_ROOT}/scripts/download_public_gcs_prefix.py" \
    --bucket gemma-data \
    --prefix checkpoints/diffusiongemma-26B-A4B-it/ \
    --dest "${CHECKPOINT_PATH}" \
    --workers 16 \
    --slices_per_large_object 4
fi

python - <<'PY'
import json
import jax
from jax import lax
print(json.dumps({
    "event": "jax_runtime",
    "devices": [str(device) for device in jax.devices()],
    "device_count": jax.device_count(),
}), flush=True)
result = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")(
    jax.numpy.ones((jax.local_device_count(),), dtype=jax.numpy.float32)
)
print(json.dumps({"event": "jax_pmap_psum", "result": result.tolist()}), flush=True)
PY

{
  echo "timestamp,index,name,memory_used_mib,memory_free_mib,utilization_gpu_percent,power_draw_w"
} > "${GPU_CSV}"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu,power.draw \
      --format=csv,noheader,nounits >> "${GPU_CSV}" || true
    sleep "${GPU_POLL_SECONDS}"
  done
) &
GPU_MONITOR_PID="$!"

cleanup() {
  if kill -0 "${GPU_MONITOR_PID}" >/dev/null 2>&1; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

COMMON_ARGS=(
  --recipe pubmedqa
  --gemma_ref "${GEMMA_REF}"
  --hackable_diffusion_ref "${HACKABLE_DIFFUSION_REF}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --workdir "${WORKDIR}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS}"
  --dataset_batch_size "${DATASET_BATCH_SIZE}"
  --lora_rank "${LORA_RANK}"
  --no-use_early_stopping
  --disable_evals
)

if [[ "${MODE}" == "upstream" ]]; then
  RUNNER=(
    python "${REPO_ROOT}/scripts/run_diffusion_gemma_official_reference.py"
    "${COMMON_ARGS[@]}"
  )
else
  RUNNER=(
    python "${REPO_ROOT}/scripts/run_diffusion_gemma_official_backend.py"
    "${COMMON_ARGS[@]}"
    --train_loop kauldron
  )
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-LL128}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

set +e
timeout --preserve-status --signal=TERM "${MAX_RUNTIME_SECONDS}" \
  "${RUNNER[@]}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e

cleanup
trap - EXIT

python "${REPO_ROOT}/scripts/summarize_diffusion_gemma_training.py" \
  --workdir "${WORKDIR}" \
  --log "${TRAIN_LOG}" \
  --gpu_csv "${GPU_CSV}" \
  --output "${SUMMARY_JSON}" || true

TIMED_OUT="false"
JOB_EXIT="${TRAIN_STATUS}"
if [[ "${TRAIN_STATUS}" == "124" || "${TRAIN_STATUS}" == "143" ]]; then
  TIMED_OUT="true"
  JOB_EXIT="0"
fi

python - "${RESULT_JSON}" <<PY
import json
import pathlib
payload = {
    "event": "comparison_job_complete",
    "mode": "${MODE}",
    "train_status": int("${TRAIN_STATUS}"),
    "timed_out": "${TIMED_OUT}" == "true",
    "workdir": "${WORKDIR}",
    "train_log": "${TRAIN_LOG}",
    "gpu_csv": "${GPU_CSV}",
    "summary_json": "${SUMMARY_JSON}",
    "gemma_revision": "$(git -C "${GEMMA_REF}" rev-parse HEAD)",
    "hackable_diffusion_revision": "$(git -C "${HACKABLE_DIFFUSION_REF}" rev-parse HEAD)",
}
path = pathlib.Path("${RESULT_JSON}")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, sort_keys=True), flush=True)
PY

exit "${JOB_EXIT}"
