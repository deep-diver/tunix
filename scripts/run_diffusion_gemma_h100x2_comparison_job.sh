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
CLI_MAX_RUNTIME_SECONDS=""
CLI_NUM_TRAIN_STEPS=""
CLI_RUN_STEPS=""
CLI_DATASET_BATCH_SIZE=""
CLI_LOG_LOSSES=""
CLI_JAX_PACKAGE_SPEC=""
CLI_SYNC_AFTER_STEP=""
CLI_RUN_NAME=""
CLI_LORA_BACKEND=""
CLI_OFFICIAL_REMAT_BLOCKS=""
CLI_STOP_GRADIENT_FROM_DENOISER_TO_ENCODER=""
CLI_LORA_ALPHA=""
CLI_QWIX_LORA_MODULE_PATH=""
CLI_LOG_PARAM_SUMMARY=""
CLI_ENCODER_LOSS_TOKEN_CHUNK_SIZE=""
CLI_ENCODER_LOSS_VOCAB_CHUNK_SIZE=""
CLI_GPU_POLL_SECONDS=""
CLI_XLA_PYTHON_CLIENT_MEM_FRACTION=""
CLI_XLA_PYTHON_CLIENT_PREALLOCATE=""
CLI_TF_GPU_ALLOCATOR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --max_runtime_seconds)
      CLI_MAX_RUNTIME_SECONDS="$2"
      shift 2
      ;;
    --num_train_steps)
      CLI_NUM_TRAIN_STEPS="$2"
      shift 2
      ;;
    --run_steps)
      CLI_RUN_STEPS="$2"
      shift 2
      ;;
    --dataset_batch_size)
      CLI_DATASET_BATCH_SIZE="$2"
      shift 2
      ;;
    --log_losses)
      CLI_LOG_LOSSES="$2"
      shift 2
      ;;
    --jax_package_spec)
      CLI_JAX_PACKAGE_SPEC="$2"
      shift 2
      ;;
    --sync_after_step)
      CLI_SYNC_AFTER_STEP="$2"
      shift 2
      ;;
    --run_name)
      CLI_RUN_NAME="$2"
      shift 2
      ;;
    --lora_backend)
      CLI_LORA_BACKEND="$2"
      shift 2
      ;;
    --official_remat_blocks)
      CLI_OFFICIAL_REMAT_BLOCKS="$2"
      shift 2
      ;;
    --stop_gradient_from_denoiser_to_encoder)
      CLI_STOP_GRADIENT_FROM_DENOISER_TO_ENCODER="$2"
      shift 2
      ;;
    --lora_alpha)
      CLI_LORA_ALPHA="$2"
      shift 2
      ;;
    --qwix_lora_module_path)
      CLI_QWIX_LORA_MODULE_PATH="$2"
      shift 2
      ;;
    --log_param_summary)
      CLI_LOG_PARAM_SUMMARY="$2"
      shift 2
      ;;
    --encoder_loss_token_chunk_size)
      CLI_ENCODER_LOSS_TOKEN_CHUNK_SIZE="$2"
      shift 2
      ;;
    --encoder_loss_vocab_chunk_size)
      CLI_ENCODER_LOSS_VOCAB_CHUNK_SIZE="$2"
      shift 2
      ;;
    --gpu_poll_seconds)
      CLI_GPU_POLL_SECONDS="$2"
      shift 2
      ;;
    --xla_python_client_mem_fraction)
      CLI_XLA_PYTHON_CLIENT_MEM_FRACTION="$2"
      shift 2
      ;;
    --xla_python_client_preallocate)
      CLI_XLA_PYTHON_CLIENT_PREALLOCATE="$2"
      shift 2
      ;;
    --tf_gpu_allocator)
      CLI_TF_GPU_ALLOCATOR="$2"
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
RUN_NAME="${CLI_RUN_NAME:-${RUN_NAME:-${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}}"
WORK_BASE="${WORK_BASE:-${HOME_DIR}/diffusion_gemma_compare}"
WORKDIR="${WORKDIR:-${WORK_BASE}/${RUN_NAME}/workdir}"
TRAIN_LOG="${TRAIN_LOG:-${WORKDIR}/train.log}"
GPU_CSV="${GPU_CSV:-${WORKDIR}/gpu_memory.csv}"
RESULT_JSON="${RESULT_JSON:-${WORKDIR}/job_result.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORKDIR}/run_summary.json}"

GEMMA_REF="${GEMMA_REF:-${HOME_DIR}/gemma_official_reference}"
HACKABLE_DIFFUSION_REF="${HACKABLE_DIFFUSION_REF:-${HOME_DIR}/hackable_diffusion_reference}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${HOME_DIR}/checkpoints/diffusiongemma-26B-A4B-it}"
MAX_RUNTIME_SECONDS="${CLI_MAX_RUNTIME_SECONDS:-${MAX_RUNTIME_SECONDS:-10800}}"
NUM_TRAIN_STEPS="${CLI_NUM_TRAIN_STEPS:-${NUM_TRAIN_STEPS:-2000}}"
RUN_STEPS="${CLI_RUN_STEPS:-${RUN_STEPS:-}}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-1000}"
DATASET_BATCH_SIZE="${CLI_DATASET_BATCH_SIZE:-${DATASET_BATCH_SIZE:-2}}"
LORA_RANK="${LORA_RANK:-4}"
LORA_BACKEND="${CLI_LORA_BACKEND:-${LORA_BACKEND:-official}}"
OFFICIAL_REMAT_BLOCKS="${CLI_OFFICIAL_REMAT_BLOCKS:-${OFFICIAL_REMAT_BLOCKS:-false}}"
STOP_GRADIENT_FROM_DENOISER_TO_ENCODER="${CLI_STOP_GRADIENT_FROM_DENOISER_TO_ENCODER:-${STOP_GRADIENT_FROM_DENOISER_TO_ENCODER:-}}"
LORA_ALPHA="${CLI_LORA_ALPHA:-${LORA_ALPHA:-}}"
QWIX_LORA_MODULE_PATH="${CLI_QWIX_LORA_MODULE_PATH:-${QWIX_LORA_MODULE_PATH:-}}"
GPU_POLL_SECONDS="${CLI_GPU_POLL_SECONDS:-${GPU_POLL_SECONDS:-60}}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda13}"
JAX_PACKAGE_SPEC="${CLI_JAX_PACKAGE_SPEC:-${JAX_PACKAGE_SPEC:-jax[${JAX_CUDA_EXTRA}]}}"
XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=constant_folding}"
TRAIN_LOOP="${TRAIN_LOOP:-hybrid}"
LOG_LOSSES="${CLI_LOG_LOSSES:-${LOG_LOSSES:-true}}"
SYNC_AFTER_STEP="${CLI_SYNC_AFTER_STEP:-${SYNC_AFTER_STEP:-state}}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-false}"
LOG_PARAM_SUMMARY="${CLI_LOG_PARAM_SUMMARY:-${LOG_PARAM_SUMMARY:-false}}"
ENCODER_LOSS_TOKEN_CHUNK_SIZE="${CLI_ENCODER_LOSS_TOKEN_CHUNK_SIZE:-${ENCODER_LOSS_TOKEN_CHUNK_SIZE:-}}"
ENCODER_LOSS_VOCAB_CHUNK_SIZE="${CLI_ENCODER_LOSS_VOCAB_CHUNK_SIZE:-${ENCODER_LOSS_VOCAB_CHUNK_SIZE:-8192}}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${CLI_XLA_PYTHON_CLIENT_MEM_FRACTION:-${XLA_PYTHON_CLIENT_MEM_FRACTION:-}}"
XLA_PYTHON_CLIENT_PREALLOCATE="${CLI_XLA_PYTHON_CLIENT_PREALLOCATE:-${XLA_PYTHON_CLIENT_PREALLOCATE:-}}"
TF_GPU_ALLOCATOR="${CLI_TF_GPU_ALLOCATOR:-${TF_GPU_ALLOCATOR:-}}"

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
  if command -v python3.12 >/dev/null 2>&1 && python_version_ok "$(command -v python3.12)"; then
    PYTHON_BIN="$(command -v python3.12)"
  elif command -v python3.13 >/dev/null 2>&1 && python_version_ok "$(command -v python3.13)"; then
    PYTHON_BIN="$(command -v python3.13)"
  elif command -v uv >/dev/null 2>&1 && uv python find 3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(uv python find 3.12)"
  elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]] && python_version_ok "${VIRTUAL_ENV}/bin/python"; then
    VENV_PY_VERSION="$("${VIRTUAL_ENV}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    if [[ "${VENV_PY_VERSION}" == "3.12" || "${VENV_PY_VERSION}" == "3.13" ]]; then
      PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    fi
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.12
    PYTHON_BIN="$(uv python find 3.12)"
  else
    if ! command -v uv >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
      if ! python3 -m pip --version >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-venv
      fi
      python3 -m pip install --user uv
      export PATH="${HOME_DIR}/.local/bin:${PATH}"
    fi
    if ! command -v uv >/dev/null 2>&1; then
      echo "Python >=3.12 is required and uv is unavailable to install it." >&2
      exit 1
    fi
    uv python install 3.12
    PYTHON_BIN="$(uv python find 3.12)"
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
  run_steps="${RUN_STEPS:-}" \
  dataset_batch_size="${DATASET_BATCH_SIZE}" \
  lora_rank="${LORA_RANK}" \
  lora_backend="${LORA_BACKEND}" \
  lora_alpha="${LORA_ALPHA:-}" \
  train_loop="${TRAIN_LOOP}" \
  log_losses="${LOG_LOSSES}" \
  sync_after_step="${SYNC_AFTER_STEP}" \
  log_param_summary="${LOG_PARAM_SUMMARY}" \
  encoder_loss_token_chunk_size="${ENCODER_LOSS_TOKEN_CHUNK_SIZE:-}" \
  encoder_loss_vocab_chunk_size="${ENCODER_LOSS_VOCAB_CHUNK_SIZE:-}" \
  gpu_poll_seconds="${GPU_POLL_SECONDS}" \
  jax_package_spec="${JAX_PACKAGE_SPEC}" \
  xla_flags="${XLA_FLAGS}" \
  xla_python_client_mem_fraction="${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" \
  xla_python_client_preallocate="${XLA_PYTHON_CLIENT_PREALLOCATE:-}" \
  tf_gpu_allocator="${TF_GPU_ALLOCATOR:-}"

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
python -m pip install -e "${REPO_ROOT}"
python -m pip uninstall -y \
  jax \
  jaxlib \
  jax-cuda12-plugin \
  jax-cuda12-pjrt \
  jax-cuda13-plugin \
  jax-cuda13-pjrt \
  nvidia-cublas-cu12 \
  nvidia-cuda-cccl-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvcc-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 || true
python -m pip install -U "${JAX_PACKAGE_SPEC}" tensorboard

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-LL128}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export XLA_FLAGS
if [[ -n "${XLA_PYTHON_CLIENT_MEM_FRACTION}" ]]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION
fi
if [[ -n "${XLA_PYTHON_CLIENT_PREALLOCATE}" ]]; then
  export XLA_PYTHON_CLIENT_PREALLOCATE
fi
if [[ -n "${TF_GPU_ALLOCATOR}" ]]; then
  export TF_GPU_ALLOCATOR
fi

rm -rf /tmp/pubmedqa_repo
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
  --module_override "PUBMEDQA_TRAIN_PATH=${GEMMA_REF}/gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa/pubmedqa_train.jsonl"
  --module_override "PUBMEDQA_TEST_PATH=${GEMMA_REF}/gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa/pubmedqa_test.jsonl"
)

COMMON_ARGS+=(--lora_backend "${LORA_BACKEND}")
if [[ "${OFFICIAL_REMAT_BLOCKS}" == "true" || "${OFFICIAL_REMAT_BLOCKS}" == "1" ]]; then
  COMMON_ARGS+=(--official_remat_blocks)
else
  COMMON_ARGS+=(--no-official_remat_blocks)
fi
if [[ -n "${STOP_GRADIENT_FROM_DENOISER_TO_ENCODER}" ]]; then
  if [[ "${STOP_GRADIENT_FROM_DENOISER_TO_ENCODER}" == "true" || "${STOP_GRADIENT_FROM_DENOISER_TO_ENCODER}" == "1" ]]; then
    COMMON_ARGS+=(--stop_gradient_from_denoiser_to_encoder)
  else
    COMMON_ARGS+=(--no-stop_gradient_from_denoiser_to_encoder)
  fi
fi
if [[ -n "${LORA_ALPHA}" ]]; then
  COMMON_ARGS+=(--lora_alpha "${LORA_ALPHA}")
fi
if [[ -n "${QWIX_LORA_MODULE_PATH}" ]]; then
  COMMON_ARGS+=(--qwix_lora_module_path "${QWIX_LORA_MODULE_PATH}")
fi

if [[ -n "${RUN_STEPS}" ]]; then
  COMMON_ARGS+=(--run_steps "${RUN_STEPS}")
fi

if [[ "${LOG_LOSSES}" == "false" || "${LOG_LOSSES}" == "0" ]]; then
  COMMON_ARGS+=(--no-log_losses)
else
  COMMON_ARGS+=(--log_losses)
fi

COMMON_ARGS+=(--sync_after_step "${SYNC_AFTER_STEP}")

if [[ "${MODE}" == "tunix" ]] && [[ "${LOG_PARAM_SUMMARY}" == "true" || "${LOG_PARAM_SUMMARY}" == "1" ]]; then
  COMMON_ARGS+=(--log_param_summary)
fi

if [[ "${MODE}" == "tunix" ]] && [[ -n "${ENCODER_LOSS_TOKEN_CHUNK_SIZE}" ]]; then
  COMMON_ARGS+=(--encoder_loss_token_chunk_size "${ENCODER_LOSS_TOKEN_CHUNK_SIZE}")
fi
if [[ "${MODE}" == "tunix" ]] && [[ -n "${ENCODER_LOSS_VOCAB_CHUNK_SIZE}" ]]; then
  COMMON_ARGS+=(--encoder_loss_vocab_chunk_size "${ENCODER_LOSS_VOCAB_CHUNK_SIZE}")
fi

if [[ "${MODE}" == "tunix" ]] && [[ "${SAVE_FINAL_CHECKPOINT}" == "true" || "${SAVE_FINAL_CHECKPOINT}" == "1" ]]; then
  COMMON_ARGS+=(--save_final_checkpoint)
fi

if [[ "${MODE}" == "upstream" ]]; then
  RUNNER=(
    python "${REPO_ROOT}/scripts/run_diffusion_gemma_official_reference.py"
    "${COMMON_ARGS[@]}"
    --train_loop "${TRAIN_LOOP}"
  )
else
  RUNNER=(
    python "${REPO_ROOT}/scripts/run_diffusion_gemma_official_backend.py"
    "${COMMON_ARGS[@]}"
    --train_loop "${TRAIN_LOOP}"
  )
fi

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
    "train_loop": "${TRAIN_LOOP}",
    "lora_backend": "${LORA_BACKEND}",
    "official_remat_blocks": "${OFFICIAL_REMAT_BLOCKS}",
    "stop_gradient_from_denoiser_to_encoder": "${STOP_GRADIENT_FROM_DENOISER_TO_ENCODER}",
    "lora_alpha": "${LORA_ALPHA}",
    "log_losses": "${LOG_LOSSES}",
    "sync_after_step": "${SYNC_AFTER_STEP}",
    "log_param_summary": "${LOG_PARAM_SUMMARY}",
    "encoder_loss_vocab_chunk_size": "${ENCODER_LOSS_VOCAB_CHUNK_SIZE}",
    "jax_package_spec": "${JAX_PACKAGE_SPEC}",
    "xla_flags": "${XLA_FLAGS}",
    "xla_python_client_mem_fraction": "${XLA_PYTHON_CLIENT_MEM_FRACTION}",
    "xla_python_client_preallocate": "${XLA_PYTHON_CLIENT_PREALLOCATE}",
    "tf_gpu_allocator": "${TF_GPU_ALLOCATOR}",
}
path = pathlib.Path("${RESULT_JSON}")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, sort_keys=True), flush=True)
PY

exit "${JOB_EXIT}"
