#!/usr/bin/env bash

set -euo pipefail

ROLE="${1:-coordinator}"
ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REMOTE_SSH="${PAP_CROSS_HOST_REMOTE_SSH:-L20T}"
REMOTE_ROOT="${PAP_CROSS_HOST_REMOTE_ROOT:-${ROOT_DIR}}"
LOCAL_IP="${PAP_CROSS_HOST_LOCAL_IP:-10.128.7.24}"
REMOTE_IP="${PAP_CROSS_HOST_REMOTE_IP:-10.128.7.25}"
GPU_ID="${PAP_CROSS_HOST_GPU_ID:-0}"
UID_PORT="${PAP_CROSS_HOST_UID_PORT:-29508}"
PAYLOAD_BYTES="${PAP_CROSS_HOST_PAYLOAD_BYTES:-16384}"
ITERATIONS="${PAP_CROSS_HOST_ITERATIONS:-100}"
TIMEOUT_SECONDS="${PAP_CROSS_HOST_TIMEOUT_SECONDS:-180}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SMOKE_SCRIPT="benchmarks/pap/microbench/nvshmem_cross_host_graph_smoke.py"

run_rank() {
  local rank="${1:?rank is required}"
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-128M}"
  source "${ROOT_DIR}/benchmarks/pap/scripts/configure_nvshmem.sh"
  pap_configure_nvshmem "${ROOT_DIR}" ibgda
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export PAP_NVSHMEM_RANK="${rank}"
  export PAP_NVSHMEM_WORLD_SIZE=2
  export PAP_NVSHMEM_ROOT_RANK=0
  export PAP_NVSHMEM_UID_HOST="${LOCAL_IP}"
  export PAP_NVSHMEM_UID_PORT="${UID_PORT}"
  export PAP_NVSHMEM_INIT_TIMEOUT="${TIMEOUT_SECONDS}"
  export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-INFO}"
  if [[ "${rank}" == "0" ]]; then
    export PAP_NVSHMEM_UID_BIND_HOST=0.0.0.0
  fi
  exec "${PYTHON_BIN}" "${ROOT_DIR}/${SMOKE_SCRIPT}" \
    --bytes "${PAYLOAD_BYTES}" \
    --iterations "${ITERATIONS}"
}

case "${ROLE}" in
  rank0) run_rank 0 ;;
  rank1) run_rank 1 ;;
  coordinator) ;;
  *)
    echo "Unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

command -v ssh >/dev/null
command -v timeout >/dev/null
command -v jq >/dev/null
command -v grep >/dev/null
command -v cmp >/dev/null
command -v sha256sum >/dev/null
command -v nvidia-smi >/dev/null
[[ -x "${PYTHON_BIN}" ]]
[[ -f "${ROOT_DIR}/${SMOKE_SCRIPT}" ]]
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "test -x '${REMOTE_ROOT}/.venv/bin/python' && test -f '${REMOTE_ROOT}/${SMOKE_SCRIPT}'"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PAP_CROSS_HOST_RUN_DIR:-${ROOT_DIR}/benchmarks/pap/experiments/_staging/cross_host_nvshmem_graph/${RUN_ID}}"
mkdir -p "${RUN_DIR}"

{
  printf 'export PAP_CROSS_HOST_REMOTE_SSH=%q\n' "${REMOTE_SSH}"
  printf 'export PAP_CROSS_HOST_REMOTE_ROOT=%q\n' "${REMOTE_ROOT}"
  printf 'export PAP_CROSS_HOST_LOCAL_IP=%q\n' "${LOCAL_IP}"
  printf 'export PAP_CROSS_HOST_REMOTE_IP=%q\n' "${REMOTE_IP}"
  printf 'export PAP_CROSS_HOST_GPU_ID=%q\n' "${GPU_ID}"
  printf 'export PAP_CROSS_HOST_UID_PORT=%q\n' "${UID_PORT}"
  printf 'export PAP_CROSS_HOST_PAYLOAD_BYTES=%q\n' "${PAYLOAD_BYTES}"
  printf 'export PAP_CROSS_HOST_ITERATIONS=%q\n' "${ITERATIONS}"
  printf 'export PAP_CROSS_HOST_TIMEOUT_SECONDS=%q\n' "${TIMEOUT_SECONDS}"
  printf 'export PAP_NVSHMEM_HCA=%q\n' "${PAP_NVSHMEM_HCA:-mlx5_0:1}"
  printf 'export NVSHMEM_SYMMETRIC_SIZE=%q\n' \
    "${NVSHMEM_SYMMETRIC_SIZE:-128M}"
} >"${RUN_DIR}/effective_config.env"

SOURCE_FILES=(
  vllm/pap/transport/nvshmem/world.py
  vllm/pap/transport/nvshmem/runtime.py
  vllm/pap/transport/nvshmem/transport.py
  vllm/pap/transport/nvshmem/device_bridge.cu
  vllm/pap/transport/nvshmem/device_bridge.map
  benchmarks/pap/scripts/configure_nvshmem.sh
  benchmarks/pap/scripts/build_nvshmem_device_bridge.sh
  benchmarks/pap/microbench/nvshmem_cross_host_graph_smoke.py
  benchmarks/pap/scripts/run_cross_host_nvshmem_graph_smoke.sh
)
git rev-parse HEAD >"${RUN_DIR}/local_git_revision.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "cd '${REMOTE_ROOT}' && git rev-parse HEAD" \
  >"${RUN_DIR}/remote_git_revision.txt"
cmp "${RUN_DIR}/local_git_revision.txt" "${RUN_DIR}/remote_git_revision.txt"
sha256sum "${SOURCE_FILES[@]}" >"${RUN_DIR}/local_source_checksums.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "cd '${REMOTE_ROOT}' && sha256sum ${SOURCE_FILES[*]}" \
  >"${RUN_DIR}/remote_source_checksums.txt"
cmp "${RUN_DIR}/local_source_checksums.txt" \
  "${RUN_DIR}/remote_source_checksums.txt"
nvidia-smi --id="${GPU_ID}" \
  --query-compute-apps=pid,gpu_bus_id,used_memory \
  --format=csv,noheader,nounits >"${RUN_DIR}/gpu_processes_before_local.csv"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "nvidia-smi --id='${GPU_ID}' --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits" \
  >"${RUN_DIR}/gpu_processes_before_remote.csv"
[[ ! -s "${RUN_DIR}/gpu_processes_before_local.csv" ]]
[[ ! -s "${RUN_DIR}/gpu_processes_before_remote.csv" ]]

cleanup() {
  trap - EXIT INT TERM
  [[ -z "${LOCAL_PID:-}" ]] || kill "${LOCAL_PID}" 2>/dev/null || true
  [[ -z "${REMOTE_PID:-}" ]] || kill "${REMOTE_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env \
  PAP_ROOT="${ROOT_DIR}" \
  PAP_CROSS_HOST_LOCAL_IP="${LOCAL_IP}" \
  PAP_CROSS_HOST_GPU_ID="${GPU_ID}" \
  PAP_CROSS_HOST_UID_PORT="${UID_PORT}" \
  PAP_CROSS_HOST_PAYLOAD_BYTES="${PAYLOAD_BYTES}" \
  PAP_CROSS_HOST_ITERATIONS="${ITERATIONS}" \
  PAP_CROSS_HOST_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
  timeout "${TIMEOUT_SECONDS}s" \
  bash "$0" rank0 >"${RUN_DIR}/rank0.log" 2>&1 &
LOCAL_PID=$!

ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "cd '${REMOTE_ROOT}' && env PAP_ROOT='${REMOTE_ROOT}' PAP_CROSS_HOST_LOCAL_IP='${LOCAL_IP}' PAP_CROSS_HOST_GPU_ID='${GPU_ID}' PAP_CROSS_HOST_UID_PORT='${UID_PORT}' PAP_CROSS_HOST_PAYLOAD_BYTES='${PAYLOAD_BYTES}' PAP_CROSS_HOST_ITERATIONS='${ITERATIONS}' PAP_CROSS_HOST_TIMEOUT_SECONDS='${TIMEOUT_SECONDS}' timeout '${TIMEOUT_SECONDS}s' bash 'benchmarks/pap/scripts/run_cross_host_nvshmem_graph_smoke.sh' rank1" \
  >"${RUN_DIR}/rank1.log" 2>&1 &
REMOTE_PID=$!

local_status=0
remote_status=0
wait "${LOCAL_PID}" || local_status=$?
wait "${REMOTE_PID}" || remote_status=$?
LOCAL_PID=
REMOTE_PID=

printf 'LOCAL_STATUS=%s\nREMOTE_STATUS=%s\n' \
  "${local_status}" "${remote_status}" >"${RUN_DIR}/exit_status.env"
if [[ "${local_status}" != "0" || "${remote_status}" != "0" ]]; then
  echo "Cross-host NVSHMEM Graph smoke failed: ${RUN_DIR}" >&2
  exit 1
fi

for log_path in "${RUN_DIR}/rank0.log" "${RUN_DIR}/rank1.log"; do
  grep -Fq "Successfully initialized the transport: IBGDA" "${log_path}"
  grep -Fq "NIC handler will be GPU" "${log_path}"
done

jq -e 'select(.rank == 0 and .mode == "ibgda" and .valid == true)' \
  < <(sed -n '/^{/p' "${RUN_DIR}/rank0.log") >/dev/null
jq -e 'select(.rank == 1 and .mode == "ibgda" and .valid == true)' \
  < <(sed -n '/^{/p' "${RUN_DIR}/rank1.log") >/dev/null
rank0_host="$(sed -n '/^{/p' "${RUN_DIR}/rank0.log" | jq -r '.hostname')"
rank1_host="$(sed -n '/^{/p' "${RUN_DIR}/rank1.log" | jq -r '.hostname')"
jq -n \
  --arg rank0_host "${rank0_host}" \
  --arg rank1_host "${rank1_host}" \
  --argjson gpu_id "${GPU_ID}" \
  '{
    nvshmem_world_size: 2,
    ranks: [
      {rank: 0, host: $rank0_host, gpu: $gpu_id},
      {rank: 1, host: $rank1_host, gpu: $gpu_id}
    ]
  }' >"${RUN_DIR}/topology_manifest.json"
printf 'STATUS=passed\nLOCAL_HOST=%q\nREMOTE_HOST=%q\n' \
  "${LOCAL_IP}" "${REMOTE_IP}" >"${RUN_DIR}/summary.env"
echo "Cross-host NVSHMEM Graph smoke passed: ${RUN_DIR}"
