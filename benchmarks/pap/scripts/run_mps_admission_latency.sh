#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
GPU_INDEX="${PAP_MPS_ADMISSION_GPU:-0}"
PREFILL_CHUNKS="${PAP_MPS_ADMISSION_PREFILL_CHUNKS:-20}"
ATTENTION_CHUNKS="${PAP_MPS_ADMISSION_ATTENTION_CHUNKS:-3}"
PREFILL_SMS="${PAP_MPS_ADMISSION_PREFILL_SMS:-80}"
ATTENTION_SMS="${PAP_MPS_ADMISSION_ATTENTION_SMS:-12}"
PRODUCER_DURATION_MS="${PAP_MPS_ADMISSION_DURATION_MS:-15000}"
PRODUCER_KERNEL_MS="${PAP_MPS_ADMISSION_KERNEL_MS:-8}"
PRODUCER_QUEUE_DEPTH="${PAP_MPS_ADMISSION_QUEUE_DEPTH:-36}"
MARKER_ITERATIONS="${PAP_MPS_ADMISSION_ITERATIONS:-64}"
MARKER_NODES="${PAP_MPS_ADMISSION_GRAPH_NODES:-4}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_mps_admission_gpu${GPU_INDEX}"
RUN_ROOT="${PAP_MPS_ADMISSION_RUN_ROOT:-/tmp/${RUN_ID}}"
SOURCE="${ROOT_DIR}/benchmarks/pap/microbench/mps_admission_latency.cu"
BINARY="${RUN_ROOT}/mps_admission_latency"
MPS_PIPE_DIR="${RUN_ROOT}/mps/pipe"
MPS_LOG_DIR="${RUN_ROOT}/mps/log"
SUMMARY="${RUN_ROOT}/summary.jsonl"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

for value in "${GPU_INDEX}" "${PREFILL_CHUNKS}" "${ATTENTION_CHUNKS}" \
  "${PREFILL_SMS}" "${ATTENTION_SMS}" "${PRODUCER_DURATION_MS}" \
  "${PRODUCER_KERNEL_MS}" "${PRODUCER_QUEUE_DEPTH}" \
  "${MARKER_ITERATIONS}" "${MARKER_NODES}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || die "expected integer, got ${value}"
done
[[ -f "${SOURCE}" ]] || die "missing source: ${SOURCE}"

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
: > "${SUMMARY}"

compute_pids="$(
  nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${compute_pids//[[:space:]]/}" ]]; then
  die "GPU ${GPU_INDEX} has active compute processes: ${compute_pids}"
fi

nvcc -O3 -std=c++17 -lineinfo "${SOURCE}" -o "${BINARY}"

MPS_STARTED=0
GPU_UUID=""
PREFILL_PARTITION=""
ATTENTION_PARTITION=""

mps_control() {
  local command="$1"
  timeout 10 env CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    nvidia-cuda-mps-control <<< "${command}"
}

remove_partition() {
  local partition="$1"
  if [[ -n "${partition}" && -n "${GPU_UUID}" ]]; then
    mps_control \
      "sm_partition rm ${GPU_UUID} ${partition#"${GPU_UUID}/"}" \
      >/dev/null 2>&1 || true
  fi
}

cleanup() {
  remove_partition "${ATTENTION_PARTITION}"
  remove_partition "${PREFILL_PARTITION}"
  if (( MPS_STARTED )); then
    mps_control quit >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  nvidia-cuda-mps-control -d -S
MPS_STARTED=1

GPU_UUID="$(
  nvidia-smi --id="${GPU_INDEX}" --query-gpu=uuid \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
[[ -n "${GPU_UUID}" ]] || die "failed to resolve GPU UUID"

create_partition() {
  local chunks="$1"
  local response
  local partition
  response="$(mps_control "sm_partition add ${GPU_UUID} ${chunks}" 2>&1)"
  if [[ "${response}" == Partition\ *\ created ]]; then
    partition="${response#Partition }"
    partition="${partition% created}"
    printf '%s\n' "${partition}"
    return
  fi
  die "unexpected static MPS response: ${response}"
}

PREFILL_PARTITION="$(create_partition "${PREFILL_CHUNKS}")"
ATTENTION_PARTITION="$(create_partition "${ATTENTION_CHUNKS}")"
mps_control lspart | tee "${RUN_ROOT}/partitions.txt"

run_marker() {
  local label="$1"
  local mode="$2"
  local connections="$3"
  local csv="${RUN_ROOT}/${label}.csv"
  local raw
  raw="$(
    env CUDA_VISIBLE_DEVICES=0 \
      CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
      CUDA_MPS_SM_PARTITION="${ATTENTION_PARTITION}" \
      CUDA_DEVICE_MAX_CONNECTIONS="${connections}" \
      CUDA_MPS_CLIENT_PRIORITY=0 \
      "${BINARY}" \
        --role marker \
        --mode "${mode}" \
        --stream-priority high \
        --expected-sms "${ATTENTION_SMS}" \
        --iterations "${MARKER_ITERATIONS}" \
        --warmup 10 \
        --interval-us 1000 \
        --graph-nodes "${MARKER_NODES}" \
        --output "${csv}"
  )"
  printf '%s\n' "${raw}" | jq -c --arg label "${label}" \
    '. + {label: $label}' | tee -a "${SUMMARY}"
}

run_loaded() {
  local mode="$1"
  local connections="$2"
  local producer_priority="$3"
  local priority_name="normal"
  if [[ "${producer_priority}" == "1" ]]; then
    priority_name="below_normal"
  fi
  local label="load_${mode}_conn${connections}_prefill_${priority_name}"
  local producer_log="${RUN_ROOT}/${label}_producer.json"

  env CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    CUDA_MPS_SM_PARTITION="${PREFILL_PARTITION}" \
    CUDA_DEVICE_MAX_CONNECTIONS="${connections}" \
    CUDA_MPS_CLIENT_PRIORITY="${producer_priority}" \
    "${BINARY}" \
      --role producer \
      --mode eager \
      --stream-priority normal \
      --expected-sms "${PREFILL_SMS}" \
      --duration-ms "${PRODUCER_DURATION_MS}" \
      --kernel-ms "${PRODUCER_KERNEL_MS}" \
      --queue-depth "${PRODUCER_QUEUE_DEPTH}" \
      --dynamic-smem-kib 64 \
      > "${producer_log}" &
  local producer_pid=$!
  sleep 0.5
  run_marker "${label}" "${mode}" "${connections}"
  wait "${producer_pid}"
  jq -c --arg label "${label}_producer" \
    '. + {label: $label}' "${producer_log}" | tee -a "${SUMMARY}"
}

for connections in 2 8; do
  for mode in eager graph; do
    run_marker "idle_${mode}_conn${connections}" "${mode}" "${connections}"
  done
done

for producer_priority in 0 1; do
  for connections in 2 8; do
    for mode in eager graph; do
      run_loaded "${mode}" "${connections}" "${producer_priority}"
    done
  done
done

{
  printf 'RUN_ID=%q\n' "${RUN_ID}"
  printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  printf 'GPU_INDEX=%q\n' "${GPU_INDEX}"
  printf 'GPU_UUID=%q\n' "${GPU_UUID}"
  printf 'GPU_NAME=%q\n' "$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader,nounits)"
  printf 'DRIVER_VERSION=%q\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)"
  printf 'NVCC_VERSION=%q\n' "$(nvcc --version | tail -n1)"
  printf 'PREFILL_PARTITION=%q\n' "${PREFILL_PARTITION}"
  printf 'ATTENTION_PARTITION=%q\n' "${ATTENTION_PARTITION}"
  printf 'PRODUCER_KERNEL_MS=%q\n' "${PRODUCER_KERNEL_MS}"
  printf 'PRODUCER_QUEUE_DEPTH=%q\n' "${PRODUCER_QUEUE_DEPTH}"
  printf 'MARKER_NODES=%q\n' "${MARKER_NODES}"
} > "${RUN_ROOT}/effective_config.env"

echo "MPS admission benchmark completed: ${RUN_ROOT}"
