#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:?setup or cleanup is required}"
INDEX="${2:?PA index is required}"
GPU="${3:?GPU index is required}"
RUN_DIR="${4:?run directory is required}"
ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PIPE_DIR="/tmp/pap-cross-host-mps-${USER:-user}-${INDEX}"
LOG_DIR="${RUN_DIR}/mps-log/pa-${INDEX}"
STATE_PATH="${RUN_DIR}/mps_pa_${INDEX}.env"

mps_control() {
  local command="${1:?MPS command is required}"
  timeout 10 env CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
    nvidia-cuda-mps-control <<<"${command}"
}

partition_id() {
  local chunks="${1:?chunk count is required}"
  local gpu_uuid="${2:?GPU UUID is required}"
  local response line value
  response="$(mps_control "sm_partition add ${gpu_uuid} ${chunks}")"
  while IFS= read -r line; do
    line="${line%$'\r'}"
    if [[ "${line}" == Partition\ *\ created ]]; then
      value="${line#Partition }"
      value="${value% created}"
      [[ -n "${value}" ]]
      printf '%s\n' "${value}"
      return 0
    fi
  done <<<"${response}"
  echo "Unexpected MPS partition response: ${response}" >&2
  return 1
}

case "${ACTION}" in
  setup)
    prefill_partition=""
    attention_partition=""
    cleanup_partial_setup() {
      local code=$?
      trap - EXIT
      set +e
      mps_control quit >/dev/null 2>&1
      exit "${code}"
    }
    trap cleanup_partial_setup EXIT
    [[ -x "${PYTHON_BIN}" ]]
    [[ ! -e "${STATE_PATH}" ]]
    mkdir -p "${PIPE_DIR}" "${LOG_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
      nvidia-cuda-mps-control -d -S
    gpu_uuid="$(
      nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader
    )"
    gpu_uuid="${gpu_uuid//[[:space:]]/}"
    [[ "${gpu_uuid}" == GPU-* ]]
    prefill_partition="$(partition_id 20 "${gpu_uuid}")"
    attention_partition="$(partition_id 3 "${gpu_uuid}")"
    prefill_sms="$(
      env \
        CUDA_VISIBLE_DEVICES=0 \
        CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
        CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
        CUDA_MPS_SM_PARTITION="${prefill_partition}" \
        "${PYTHON_BIN}" -c \
          'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)'
    )"
    attention_sms="$(
      env \
        CUDA_VISIBLE_DEVICES=0 \
        CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
        CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
        CUDA_MPS_SM_PARTITION="${attention_partition}" \
        "${PYTHON_BIN}" -c \
          'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)'
    )"
    [[ "${prefill_sms}" == "80" && "${attention_sms}" == "12" ]]
    {
      printf 'GPU=%q\n' "${GPU}"
      printf 'GPU_UUID=%q\n' "${gpu_uuid}"
      printf 'MPS_PIPE_DIR=%q\n' "${PIPE_DIR}"
      printf 'MPS_LOG_DIR=%q\n' "${LOG_DIR}"
      printf 'PREFILL_PARTITION=%q\n' "${prefill_partition}"
      printf 'ATTENTION_PARTITION=%q\n' "${attention_partition}"
      printf 'PREFILL_VISIBLE_SMS=%q\n' "${prefill_sms}"
      printf 'ATTENTION_VISIBLE_SMS=%q\n' "${attention_sms}"
    } >"${STATE_PATH}"
    trap - EXIT
    ;;
  cleanup)
    [[ -f "${STATE_PATH}" ]] || exit 0
    # shellcheck disable=SC1090
    source "${STATE_PATH}"
    mps_control quit
    deadline=$((SECONDS + 30))
    while [[ -n "$(
      nvidia-smi --id="${GPU}" --query-compute-apps=pid \
        --format=csv,noheader,nounits
    )" ]]; do
      ((SECONDS < deadline)) || {
        echo "MPS GPU context remained after controller quit: GPU ${GPU}" >&2
        exit 1
      }
      sleep 1
    done
    ;;
  *)
    echo "Unknown MPS action: ${ACTION}" >&2
    exit 2
    ;;
esac
