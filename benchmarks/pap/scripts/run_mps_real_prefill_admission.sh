#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
GPU_INDEX="${PAP_MPS_ADMISSION_GPU:-0}"
PEER_GPU_INDEX="${PAP_MPS_ADMISSION_PEER_GPU:-1}"
MARKER_MODE="${PAP_MPS_ADMISSION_MARKER_MODE:-graph}"
CONNECTIONS="${PAP_MPS_ADMISSION_CONNECTIONS:-2}"
PREFILL_PRIORITY="${PAP_MPS_ADMISSION_PREFILL_PRIORITY:-0}"
PREFILL_REQUESTS="${PAP_MPS_ADMISSION_PREFILL_REQUESTS:-128}"
MARKER_ITERATIONS="${PAP_MPS_ADMISSION_ITERATIONS:-32}"
MARKER_NODES="${PAP_MPS_ADMISSION_GRAPH_NODES:-4}"
MARKER_COOPERATIVE="${PAP_MPS_ADMISSION_COOPERATIVE:-0}"
READY_DELAY_SECONDS="${PAP_MPS_ADMISSION_READY_DELAY_SECONDS:-1}"
MARKER_INTERVAL_US="${PAP_MPS_ADMISSION_INTERVAL_US:-1000}"
MARKER_NVSHMEM="${PAP_MPS_ADMISSION_NVSHMEM:-0}"
MARKER_COMPUTE_US="${PAP_MPS_ADMISSION_COMPUTE_US:-0}"
MARKER_PREPARE_EVENT="${PAP_MPS_ADMISSION_PREPARE_EVENT:-0}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_mps_real_prefill_gpu${GPU_INDEX}"
RUN_ROOT="${PAP_MPS_ADMISSION_RUN_ROOT:-/tmp/${RUN_ID}}"
CUDA_SOURCE="${ROOT_DIR}/benchmarks/pap/microbench/mps_admission_latency.cu"
PRODUCER="${ROOT_DIR}/benchmarks/pap/microbench/mps_real_prefill_producer.py"
BINARY="${RUN_ROOT}/mps_admission_latency"
NVSHMEM_BINARY="${RUN_ROOT}/nvshmem_admission_marker"
NVSHMEM_PREFIX="${PAP_NVSHMEM_PREFIX:-${ROOT_DIR}/.local/nvshmem-3.3.24-cuda13}"
MPS_PIPE_DIR="${RUN_ROOT}/mps/pipe"
MPS_LOG_DIR="${RUN_ROOT}/mps/log"
READY_FILE="${RUN_ROOT}/prefill.ready"
RANK_WRAPPER="${ROOT_DIR}/benchmarks/pap/scripts/exec_nvshmem_rank_partition.sh"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

for required in "${PYTHON_BIN}" "${MODEL_PATH}" "${CUDA_SOURCE}" \
  "${PRODUCER}"; do
  [[ -e "${required}" ]] || die "missing required path: ${required}"
done
[[ "${MARKER_MODE}" == "eager" || "${MARKER_MODE}" == "graph" ]] \
  || die "marker mode must be eager or graph"
[[ "${CONNECTIONS}" == "2" || "${CONNECTIONS}" == "8" ]] \
  || die "connections must be 2 or 8"
[[ "${PREFILL_PRIORITY}" == "0" || "${PREFILL_PRIORITY}" == "1" ]] \
  || die "Prefill priority must be 0 or 1"
[[ "${MARKER_COOPERATIVE}" == "0" || "${MARKER_COOPERATIVE}" == "1" ]] \
  || die "cooperative must be 0 or 1"
[[ "${MARKER_NVSHMEM}" == "0" || "${MARKER_NVSHMEM}" == "1" ]] \
  || die "NVSHMEM marker must be 0 or 1"

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
compute_pids="$(
  nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
[[ -z "${compute_pids//[[:space:]]/}" ]] \
  || die "GPU ${GPU_INDEX} has active compute processes: ${compute_pids}"
if [[ "${MARKER_NVSHMEM}" == "1" ]]; then
  peer_compute_pids="$(
    nvidia-smi --id="${PEER_GPU_INDEX}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  [[ -z "${peer_compute_pids//[[:space:]]/}" ]] \
    || die "GPU ${PEER_GPU_INDEX} has active processes: ${peer_compute_pids}"
fi
nvcc -O3 -std=c++17 -lineinfo "${CUDA_SOURCE}" -o "${BINARY}"
if [[ "${MARKER_NVSHMEM}" == "1" ]]; then
  PAP_NVSHMEM_PREFIX="${NVSHMEM_PREFIX}" \
    "${ROOT_DIR}/benchmarks/pap/scripts/build_nvshmem_device_bridge.sh"
  nvcc -std=c++17 -O3 -lineinfo -rdc=true -arch=sm_89 \
    -I"${NVSHMEM_PREFIX}/include" \
    "${ROOT_DIR}/benchmarks/pap/microbench/nvshmem_admission_marker.cu" \
    -L"${NVSHMEM_PREFIX}/lib" \
    -lpap_nvshmem_device -lnvshmem_host -lnvshmem_device \
    -o "${NVSHMEM_BINARY}"
fi

GPU_UUID=""
PEER_GPU_UUID=""
PREFILL_PARTITION=""
ATTENTION_PARTITION=""
PEER_PARTITION=""
MPS_STARTED=0

mps_control() {
  local command="$1"
  timeout 10 env CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    nvidia-cuda-mps-control <<< "${command}"
}

cleanup() {
  local partition
  for partition in "${ATTENTION_PARTITION}" "${PREFILL_PARTITION}" \
    "${PEER_PARTITION}"; do
    local partition_gpu="${GPU_UUID}"
    if [[ -n "${PEER_GPU_UUID}" && "${partition}" == "${PEER_GPU_UUID}/"* ]]; then
      partition_gpu="${PEER_GPU_UUID}"
    fi
    if [[ -n "${partition}" && -n "${partition_gpu}" ]]; then
      mps_control \
        "sm_partition rm ${partition_gpu} ${partition#"${partition_gpu}/"}" \
        >/dev/null 2>&1 || true
    fi
  done
  if (( MPS_STARTED )); then
    mps_control quit >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

server_visible_devices="${GPU_INDEX}"
if [[ "${MARKER_NVSHMEM}" == "1" ]]; then
  server_visible_devices="${GPU_INDEX},${PEER_GPU_INDEX}"
fi
env CUDA_VISIBLE_DEVICES="${server_visible_devices}" \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  nvidia-cuda-mps-control -d -S
MPS_STARTED=1
GPU_UUID="$(
  nvidia-smi --id="${GPU_INDEX}" --query-gpu=uuid \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
if [[ "${MARKER_NVSHMEM}" == "1" ]]; then
  PEER_GPU_UUID="$(
    nvidia-smi --id="${PEER_GPU_INDEX}" --query-gpu=uuid \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
fi

create_partition() {
  local gpu_uuid="$1"
  local chunks="$2"
  local response
  local partition
  response="$(mps_control "sm_partition add ${gpu_uuid} ${chunks}" 2>&1)"
  [[ "${response}" == Partition\ *\ created ]] \
    || die "unexpected static MPS response: ${response}"
  partition="${response#Partition }"
  printf '%s\n' "${partition% created}"
}

PREFILL_PARTITION="$(create_partition "${GPU_UUID}" 20)"
ATTENTION_PARTITION="$(create_partition "${GPU_UUID}" 3)"
if [[ "${MARKER_NVSHMEM}" == "1" ]]; then
  PEER_PARTITION="$(create_partition "${PEER_GPU_UUID}" 3)"
fi
mps_control lspart | tee "${RUN_ROOT}/partitions.txt"

run_marker() {
  local label="$1"
  local marker_iterations="${MARKER_ITERATIONS}"
  local marker_interval_us="${MARKER_INTERVAL_US}"
  if [[ "${label}" == "idle" ]]; then
    marker_iterations=64
    marker_interval_us=1000
  fi
  if [[ "${MARKER_NVSHMEM}" == "1" ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
      CUDA_DEVICE_MAX_CONNECTIONS="${CONNECTIONS}" \
      CUDA_MPS_CLIENT_PRIORITY=0 \
      PAP_NVSHMEM_GPU_0="${GPU_INDEX}" \
      PAP_NVSHMEM_GPU_1="${PEER_GPU_INDEX}" \
      PAP_NVSHMEM_PARTITION_0="${ATTENTION_PARTITION}" \
      PAP_NVSHMEM_PARTITION_1="${PEER_PARTITION}" \
      LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
      NVSHMEM_BOOTSTRAP=PMI \
      NVSHMEM_REMOTE_TRANSPORT=none \
      NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1 \
      NVSHMEM_SYMMETRIC_SIZE=16M \
      "${NVSHMEM_PREFIX}/bin/nvshmrun.hydra" -launcher fork -np 2 \
      "${RANK_WRAPPER}" "${NVSHMEM_BINARY}" \
        --expected-sms 12 \
        --iterations "${marker_iterations}" \
        --warmup 10 \
        --interval-us "${marker_interval_us}" \
        --layers 36 \
        --compute-us "${MARKER_COMPUTE_US}" \
        --output "${RUN_ROOT}/${label}.csv" \
      | jq -c --arg label "${label}" '. + {label: $label}' \
      | tee -a "${RUN_ROOT}/summary.jsonl"
    return
  fi
  env CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    CUDA_MPS_SM_PARTITION="${ATTENTION_PARTITION}" \
    CUDA_DEVICE_MAX_CONNECTIONS="${CONNECTIONS}" \
    CUDA_MPS_CLIENT_PRIORITY=0 \
    "${BINARY}" \
      --role marker \
      --mode "${MARKER_MODE}" \
      --stream-priority high \
      --expected-sms 12 \
      --iterations "${marker_iterations}" \
      --warmup 10 \
      --interval-us "${marker_interval_us}" \
      --graph-nodes "${MARKER_NODES}" \
      --cooperative "${MARKER_COOPERATIVE}" \
      --prepare-event "${MARKER_PREPARE_EVENT}" \
      --output "${RUN_ROOT}/${label}.csv" \
    | jq -c --arg label "${label}" '. + {label: $label}' \
    | tee -a "${RUN_ROOT}/summary.jsonl"
}

run_marker idle

env CUDA_VISIBLE_DEVICES=0 \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  CUDA_MPS_SM_PARTITION="${PREFILL_PARTITION}" \
  CUDA_DEVICE_MAX_CONNECTIONS="${CONNECTIONS}" \
  CUDA_MPS_CLIENT_PRIORITY="${PREFILL_PRIORITY}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "${PYTHON_BIN}" "${PRODUCER}" \
    --model "${MODEL_PATH}" \
    --ready-file "${READY_FILE}" \
    --output "${RUN_ROOT}/producer_result.json" \
    --requests "${PREFILL_REQUESTS}" \
    --prompt-tokens 2048 \
    --max-batched-tokens 2048 \
    > "${RUN_ROOT}/producer.log" 2>&1 &
producer_pid=$!

for _ in $(seq 1 1800); do
  [[ -e "${READY_FILE}" ]] && break
  if ! kill -0 "${producer_pid}" 2>/dev/null; then
    wait "${producer_pid}" || true
    die "real Prefill producer exited before readiness; see producer.log"
  fi
  sleep 0.1
done
[[ -e "${READY_FILE}" ]] || die "timed out waiting for real Prefill readiness"
sleep "${READY_DELAY_SECONDS}"
nvidia-smi -i "${GPU_INDEX}" \
  --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader,nounits | tee "${RUN_ROOT}/gpu_before_marker.csv"
mps_control lspart | tee "${RUN_ROOT}/partitions_during_load.txt"
run_marker loaded
wait "${producer_pid}"
jq -c '. + {label: "producer"}' "${RUN_ROOT}/producer_result.json" \
  | tee -a "${RUN_ROOT}/summary.jsonl"

{
  printf 'RUN_ID=%q\n' "${RUN_ID}"
  printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  printf 'GPU_INDEX=%q\n' "${GPU_INDEX}"
  printf 'GPU_UUID=%q\n' "${GPU_UUID}"
  printf 'PEER_GPU_INDEX=%q\n' "${PEER_GPU_INDEX}"
  printf 'PEER_GPU_UUID=%q\n' "${PEER_GPU_UUID}"
  printf 'MARKER_MODE=%q\n' "${MARKER_MODE}"
  printf 'CONNECTIONS=%q\n' "${CONNECTIONS}"
  printf 'PREFILL_PRIORITY=%q\n' "${PREFILL_PRIORITY}"
  printf 'PREFILL_REQUESTS=%q\n' "${PREFILL_REQUESTS}"
  printf 'MARKER_NODES=%q\n' "${MARKER_NODES}"
  printf 'MARKER_COOPERATIVE=%q\n' "${MARKER_COOPERATIVE}"
  printf 'MARKER_INTERVAL_US=%q\n' "${MARKER_INTERVAL_US}"
  printf 'READY_DELAY_SECONDS=%q\n' "${READY_DELAY_SECONDS}"
  printf 'MARKER_NVSHMEM=%q\n' "${MARKER_NVSHMEM}"
  printf 'MARKER_COMPUTE_US=%q\n' "${MARKER_COMPUTE_US}"
  printf 'MARKER_PREPARE_EVENT=%q\n' "${MARKER_PREPARE_EVENT}"
} > "${RUN_ROOT}/effective_config.env"

echo "Real-Prefill MPS admission benchmark completed: ${RUN_ROOT}"
