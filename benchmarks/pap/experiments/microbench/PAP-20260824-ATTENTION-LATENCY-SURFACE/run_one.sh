#!/usr/bin/env bash
set -euo pipefail

# Production 12-SM PAP Attention scaling and GPU-counter probe.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PROBE="${EXPERIMENT_DIR}/probe.py"
METRICS="${ROOT_DIR}/benchmarks/pap/tooling/component_gpu_metrics.py"
MODEL_CONFIG="${MODEL_CONFIG:-/data/ssd1/llm-models/Qwen3-8B/config.json}"
GPU_INDEX="${PAP_ATTENTION_SCALING_GPU:-6}"
MPS_CHUNKS="${PAP_ATTENTION_SCALING_MPS_CHUNKS:-3}"
EXPECTED_SMS="${PAP_ATTENTION_SCALING_EXPECTED_SMS:-12}"
RUN_TIMING="${PAP_ATTENTION_SCALING_RUN_TIMING:-1}"
RUN_NSYS="${PAP_ATTENTION_SCALING_RUN_NSYS:-1}"
SHAPE_GROUPS="${PAP_ATTENTION_SCALING_GROUPS:-context,iso_total}"
KERNEL_SET="${PAP_ATTENTION_SCALING_KERNEL_SET:-sweep}"
SHARD_COUNT="${PAP_ATTENTION_SCALING_SHARD_COUNT:-1}"
SHARD_INDEX="${PAP_ATTENTION_SCALING_SHARD_INDEX:-0}"
OUTPUT_ROOT="${PAP_ATTENTION_SCALING_OUTPUT_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/microbench/_runs/$(date +%Y%m%d_%H%M%S)_attention_scaling}"
NSYS_IMPORTER="${PAP_NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}"
MPS_PIPE_DIR="${PAP_ATTENTION_SCALING_MPS_PIPE_DIR:-/tmp/pap-attention-scaling-${USER:-user}-$$}"
MPS_LOG_DIR="${OUTPUT_ROOT}/mps/log"
MPS_STARTED=0
GPU_UUID=""
PARTITION=""

die() {
  echo "ERROR: $*" >&2
  exit 2
}

mps_control() {
  timeout 10 env CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    nvidia-cuda-mps-control <<< "$1"
}

cleanup() {
  local code=$?
  set +e
  if [[ -n "${PARTITION}" && -n "${GPU_UUID}" ]]; then
    mps_control \
      "sm_partition rm ${GPU_UUID} ${PARTITION#"${GPU_UUID}/"}" \
      >/dev/null 2>&1 || true
  fi
  if (( MPS_STARTED )); then
    mps_control quit >/dev/null 2>&1 || true
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

require_paths() {
  [[ -x "${PYTHON_BIN}" ]] || die "missing Python environment"
  [[ -f "${PROBE}" && -f "${METRICS}" && -f "${MODEL_CONFIG}" ]] \
    || die "missing Attention probe input"
  command -v nvidia-cuda-mps-control >/dev/null \
    || die "nvidia-cuda-mps-control is unavailable"
  if [[ "${RUN_NSYS}" == "1" ]]; then
    command -v nsys >/dev/null || die "nsys is unavailable"
  fi
}

ensure_gpu_idle() {
  local pids
  pids="$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${pids//[[:space:]]/}" ]] \
    || die "GPU ${GPU_INDEX} has active compute processes: ${pids}"
}

start_mps() {
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    nvidia-cuda-mps-control -d -S
  MPS_STARTED=1
  GPU_UUID="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=uuid \
    --format=csv,noheader,nounits | tr -d '[:space:]')"
  local response line
  response="$(mps_control "sm_partition add ${GPU_UUID} ${MPS_CHUNKS}")"
  while IFS= read -r line; do
    if [[ "${line}" == Partition\ *\ created ]]; then
      PARTITION="${line#Partition }"
      PARTITION="${PARTITION% created}"
    fi
  done <<< "${response}"
  [[ -n "${PARTITION}" ]] || die "unexpected MPS response: ${response}"
}

client_env() {
  env \
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    CUDA_MPS_SM_PARTITION="${PARTITION}" \
    "$@"
}

import_nsys() {
  local prefix="$1"
  if [[ ! -f "${prefix}.nsys-rep" ]]; then
    [[ -x "${NSYS_IMPORTER}" && -f "${prefix}.qdstrm" ]] \
      || die "nsys report is missing: ${prefix}"
    "${NSYS_IMPORTER}" --force-overwrite \
      --input-file "${prefix}.qdstrm" --output-file "${prefix}.nsys-rep"
  fi
  nsys export --force-overwrite=true --type=sqlite \
    --output="${prefix}.sqlite" "${prefix}.nsys-rep"
}

run_nsys_case() {
  local label="$1"
  local shape="$2"
  local prefix="${OUTPUT_ROOT}/nsys/${label}"
  client_env nsys profile \
    --force-overwrite=true \
    --sample=none \
    --trace=nvtx \
    --gpu-metrics-device="${GPU_INDEX}" \
    --gpu-metrics-set=5 \
    --gpu-metrics-frequency=10000 \
    --output="${prefix}" \
    "${PYTHON_BIN}" "${PROBE}" profile \
      --model-config "${MODEL_CONFIG}" \
      --shape "${shape}" \
      --expected-sms "${EXPECTED_SMS}" \
      --output "${prefix}_metadata.json"
  import_nsys "${prefix}"
  "${PYTHON_BIN}" "${METRICS}" "${prefix}.sqlite" \
    --component attention --output "${prefix}_metrics.json"
}

main() {
  require_paths
  ensure_gpu_idle
  mkdir -p "${OUTPUT_ROOT}/timing" "${OUTPUT_ROOT}/nsys"
  start_mps
  local visible_sms
  visible_sms="$(client_env "${PYTHON_BIN}" -c \
    'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)')"
  [[ "${visible_sms}" == "${EXPECTED_SMS}" ]] \
    || die "MPS exposed ${visible_sms} SMs; expected ${EXPECTED_SMS}"
  local tracked_worktree_dirty=0
  git -C "${ROOT_DIR}" diff --quiet || tracked_worktree_dirty=1
  git -C "${ROOT_DIR}" diff --cached --quiet || tracked_worktree_dirty=1
  {
    printf 'RUN_SCRIPT=%q\n' "${BASH_SOURCE[0]}"
    printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'GIT_TRACKED_WORKTREE_DIRTY=%q\n' "${tracked_worktree_dirty}"
    printf 'MODEL_CONFIG_SHA256=%q\n' "$(sha256sum "${MODEL_CONFIG}" | cut -d' ' -f1)"
    printf 'GPU_INDEX=%q\n' "${GPU_INDEX}"
    printf 'GPU_UUID=%q\n' "${GPU_UUID}"
    printf 'GPU_NAME=%q\n' "$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader)"
    printf 'DRIVER_VERSION=%q\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    printf 'MPS_PARTITION=%q\n' "${PARTITION}"
    printf 'VISIBLE_SMS=%q\n' "${visible_sms}"
    printf 'RUN_TIMING=%q\n' "${RUN_TIMING}"
    printf 'RUN_NSYS=%q\n' "${RUN_NSYS}"
    printf 'SHAPE_GROUPS=%q\n' "${SHAPE_GROUPS}"
    printf 'KERNEL_SET=%q\n' "${KERNEL_SET}"
    printf 'SHARD_COUNT=%q\n' "${SHARD_COUNT}"
    printf 'SHARD_INDEX=%q\n' "${SHARD_INDEX}"
    printf 'PYTORCH_CUDA=%q\n' "$(client_env "${PYTHON_BIN}" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    if [[ "${RUN_NSYS}" == "1" ]]; then
      printf 'NSYS_VERSION=%q\n' "$(nsys --version)"
    fi
  } > "${OUTPUT_ROOT}/run.env"

  if [[ "${RUN_TIMING}" == "1" ]]; then
    client_env "${PYTHON_BIN}" "${PROBE}" timing \
      --model-config "${MODEL_CONFIG}" \
      --groups "${SHAPE_GROUPS}" \
      --kernel-set "${KERNEL_SET}" \
      --shard-count "${SHARD_COUNT}" \
      --shard-index "${SHARD_INDEX}" \
      --expected-sms "${EXPECTED_SMS}" \
      --output "${OUTPUT_ROOT}/timing/result.json"
  fi
  if [[ "${RUN_NSYS}" == "1" ]]; then
    run_nsys_case short_b1_l1k "1,1024"
    run_nsys_case total128k_b4 "4,32768"
    run_nsys_case total128k_b16 "16,8192"
    run_nsys_case total128k_b64 "64,2048"
  fi
  printf 'STATUS=passed\n' > "${OUTPUT_ROOT}/status.env"
  echo "PAP_ATTENTION_SCALING_OUTPUT_ROOT=${OUTPUT_ROOT}"
}

main "$@"
