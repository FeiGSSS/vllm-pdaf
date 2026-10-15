#!/usr/bin/env bash

set -euo pipefail

ROLE="${1:-coordinator}"
ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REMOTE_SSH="${PAP_CROSS_HOST_REMOTE_SSH:-L20T}"
REMOTE_ROOT="${PAP_CROSS_HOST_REMOTE_ROOT:-${ROOT_DIR}}"
LOCAL_IP="${PAP_CROSS_HOST_LOCAL_IP:-10.128.7.24}"
REMOTE_IP="${PAP_CROSS_HOST_REMOTE_IP:-10.128.7.25}"
GPU_ID="${PAP_CROSS_HOST_GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
VLLM_BIN="${ROOT_DIR}/.venv/bin/vllm"

PREFILL_PORT="${PAP_CROSS_HOST_PREFILL_PORT:-8100}"
PROJECTION_PORT="${PAP_CROSS_HOST_PROJECTION_PORT:-8200}"
ATTENTION_PORT="${PAP_CROSS_HOST_ATTENTION_PORT:-8300}"
KV_EVENT_PORT="${PAP_CROSS_HOST_KV_EVENT_PORT:-8400}"
ATTENTION_TCP_PORT="${PAP_CROSS_HOST_ATTENTION_TCP_PORT:-9300}"
GATEWAY_PORT="${PAP_CROSS_HOST_GATEWAY_PORT:-9460}"
VLLM_PREFILL_PORT="${PAP_CROSS_HOST_VLLM_PREFILL_PORT:-50000}"
VLLM_PROJECTION_PORT="${PAP_CROSS_HOST_VLLM_PROJECTION_PORT:-50020}"
UID_PORT="${PAP_CROSS_HOST_UID_PORT:-29508}"

MAX_MODEL_LEN="${PAP_CROSS_HOST_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${PAP_CROSS_HOST_MAX_NUM_SEQS:-8}"
BUFFER_BYTES="${PAP_NVSHMEM_BUFFER_BYTES:-2097152}"
CONTROL_BYTES="${PAP_NVSHMEM_CONTROL_BYTES:-32768}"
START_TIMEOUT="${PAP_CROSS_HOST_START_TIMEOUT:-300}"
REQUEST_TIMEOUT="${PAP_CROSS_HOST_REQUEST_TIMEOUT:-180}"
PREFILL_MEMORY="${PAP_CROSS_HOST_PREFILL_MEMORY:-0.80}"
PROJECTION_MEMORY="${PAP_CROSS_HOST_PROJECTION_MEMORY:-0.45}"
SCRIPT_RELATIVE="benchmarks/pap/scripts/run_cross_host_nvshmem_1pa1p.sh"

set_common_env() {
  local direct_hosts="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}"
  export NO_PROXY="${direct_hosts}"
  export no_proxy="${direct_hosts}"
  export PAP_TOPOLOGY=1pa1p
  export PAP_PA_COUNT=1
  export PAP_PROJECTION_COUNT=1
  export PAP_BLOCK_SIZE=16
  export VLLM_PLUGINS=pap
  export PAP_MODEL_HOOKS=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
}

set_nvshmem_env() {
  local rank="${1:?rank is required}"
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-128M}"
  source "${ROOT_DIR}/benchmarks/pap/scripts/configure_nvshmem.sh"
  pap_configure_nvshmem "${ROOT_DIR}" ibgda
  export PAP_NVSHMEM_RANK="${rank}"
  export PAP_NVSHMEM_WORLD_SIZE=2
  export PAP_NVSHMEM_ROOT_RANK=0
  export PAP_NVSHMEM_UID_HOST="${LOCAL_IP}"
  export PAP_NVSHMEM_UID_PORT="${UID_PORT}"
  export PAP_NVSHMEM_INIT_TIMEOUT="${START_TIMEOUT}"
  export PAP_NVSHMEM_BUFFER_BYTES="${BUFFER_BYTES}"
  export PAP_NVSHMEM_CONTROL_BYTES="${CONTROL_BYTES}"
  export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-INFO}"
}

run_attention() {
  set_common_env
  set_nvshmem_env 1
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export PAP_OFFLOAD_EXEC_LOCAL_RANK=0
  export PAP_OFFLOAD_EXEC_Q_SIZE=4096
  export PAP_OFFLOAD_EXEC_KV_SIZE=1024
  export PAP_OFFLOAD_EXEC_NUM_HEADS=32
  export PAP_OFFLOAD_EXEC_NUM_KV_HEADS=8
  export PAP_OFFLOAD_EXEC_HEAD_DIM=128
  export PAP_ATTENTION_KERNEL_POLICY=triton
  export PAP_DECODE_COMMIT_FAIL_CLOSED=1
  export PAP_DECODE_COMMIT_ENDPOINT="http://127.0.0.1:${PREFILL_PORT}/v1/pap/prefill/decode-commit"
  export PAP_LEASE_RELEASE_ENDPOINT="http://127.0.0.1:${PREFILL_PORT}/v1/pap/prefill/lease-release"
  exec "${PYTHON_BIN}" -m vllm.pap.service \
    --host 0.0.0.0 \
    --port "${ATTENTION_PORT}" \
    --tcp-port "${ATTENTION_TCP_PORT}"
}

run_prefill() {
  set_common_env
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export VLLM_PORT="${VLLM_PREFILL_PORT}"
  export PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=64
  exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PREFILL_PORT}" \
    --enforce-eager \
    --generation-config vllm \
    --dtype float16 \
    --enable-request-id-headers \
    --enable-prompt-tokens-details \
    --enable-prefix-caching \
    --kv-events-config \
      "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${KV_EVENT_PORT}\",\"topic\":\"pap-pa-0\",\"hwm\":100000,\"max_queue_size\":0}" \
    --enable-chunked-prefill \
    --block-size 16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_MODEL_LEN}" \
    --no-async-scheduling \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${PREFILL_MEMORY}" \
    --kv-transfer-config \
      '{"kv_connector":"PAPPrefillConnector","kv_connector_module_path":"vllm.pap.kv_connector","kv_role":"kv_producer"}'
}

run_projection() {
  set_common_env
  set_nvshmem_env 0
  export PAP_NVSHMEM_UID_BIND_HOST=0.0.0.0
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export VLLM_PORT="${VLLM_PROJECTION_PORT}"
  export PAP_TP_SIZE=1
  export PAP_PROJECTION_KV_UNAWARE=1
  exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PROJECTION_PORT}" \
    --enforce-eager \
    --generation-config vllm \
    --dtype float16 \
    --enable-request-id-headers \
    --block-size 16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_SEQS}" \
    --async-scheduling \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${PROJECTION_MEMORY}"
}

run_gateway() {
  set_common_env
  exec "${PYTHON_BIN}" -m vllm.pap.gateway.app \
    --host 127.0.0.1 \
    --port "${GATEWAY_PORT}" \
    --model "${MODEL_PATH}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --generation-config vllm \
    --pap-groups \
      "${REMOTE_IP}:${PREFILL_PORT}:${REMOTE_IP}:${ATTENTION_PORT}:${ATTENTION_TCP_PORT}" \
    --projections "127.0.0.1:${PROJECTION_PORT}" \
    --kv-event-endpoints "tcp://${REMOTE_IP}:${KV_EVENT_PORT}" \
    --prefix-block-size 16 \
    --dynamo-site-packages "${ROOT_DIR}/.local/pap-dynamo-router" \
    --dynamo-max-num-batched-tokens "${MAX_MODEL_LEN}" \
    --routing-policy dynamo
}

case "${ROLE}" in
  attention) run_attention ;;
  prefill) run_prefill ;;
  projection) run_projection ;;
  gateway) run_gateway ;;
  coordinator) ;;
  *)
    echo "Unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

command -v ssh >/dev/null
command -v curl >/dev/null
command -v jq >/dev/null
command -v grep >/dev/null
command -v cmp >/dev/null
command -v sha256sum >/dev/null
command -v nvidia-smi >/dev/null
[[ -x "${PYTHON_BIN}" && -x "${VLLM_BIN}" ]]
[[ -d "${MODEL_PATH}" ]]
[[ -f "${ROOT_DIR}/${SCRIPT_RELATIVE}" ]]

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PAP_CROSS_HOST_RUN_DIR:-${ROOT_DIR}/benchmarks/pap/experiments/_staging/cross_host_nvshmem_1pa1p/${RUN_ID}}"
REMOTE_RUN_DIR="/tmp/pap-cross-host-nvshmem-${RUN_ID}"
mkdir -p "${RUN_DIR}"

{
  printf 'export MODEL_PATH=%q\n' "${MODEL_PATH}"
  printf 'export PAP_CROSS_HOST_REMOTE_SSH=%q\n' "${REMOTE_SSH}"
  printf 'export PAP_CROSS_HOST_REMOTE_ROOT=%q\n' "${REMOTE_ROOT}"
  printf 'export PAP_CROSS_HOST_LOCAL_IP=%q\n' "${LOCAL_IP}"
  printf 'export PAP_CROSS_HOST_REMOTE_IP=%q\n' "${REMOTE_IP}"
  printf 'export PAP_CROSS_HOST_GPU_ID=%q\n' "${GPU_ID}"
  printf 'export PAP_CROSS_HOST_PREFILL_PORT=%q\n' "${PREFILL_PORT}"
  printf 'export PAP_CROSS_HOST_PROJECTION_PORT=%q\n' "${PROJECTION_PORT}"
  printf 'export PAP_CROSS_HOST_ATTENTION_PORT=%q\n' "${ATTENTION_PORT}"
  printf 'export PAP_CROSS_HOST_KV_EVENT_PORT=%q\n' "${KV_EVENT_PORT}"
  printf 'export PAP_CROSS_HOST_ATTENTION_TCP_PORT=%q\n' "${ATTENTION_TCP_PORT}"
  printf 'export PAP_CROSS_HOST_GATEWAY_PORT=%q\n' "${GATEWAY_PORT}"
  printf 'export PAP_CROSS_HOST_VLLM_PREFILL_PORT=%q\n' "${VLLM_PREFILL_PORT}"
  printf 'export PAP_CROSS_HOST_VLLM_PROJECTION_PORT=%q\n' \
    "${VLLM_PROJECTION_PORT}"
  printf 'export PAP_CROSS_HOST_UID_PORT=%q\n' "${UID_PORT}"
  printf 'export PAP_CROSS_HOST_MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN}"
  printf 'export PAP_CROSS_HOST_MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS}"
  printf 'export PAP_NVSHMEM_BUFFER_BYTES=%q\n' "${BUFFER_BYTES}"
  printf 'export PAP_NVSHMEM_CONTROL_BYTES=%q\n' "${CONTROL_BYTES}"
  printf 'export PAP_CROSS_HOST_START_TIMEOUT=%q\n' "${START_TIMEOUT}"
  printf 'export PAP_CROSS_HOST_REQUEST_TIMEOUT=%q\n' "${REQUEST_TIMEOUT}"
  printf 'export PAP_CROSS_HOST_PREFILL_MEMORY=%q\n' "${PREFILL_MEMORY}"
  printf 'export PAP_CROSS_HOST_PROJECTION_MEMORY=%q\n' "${PROJECTION_MEMORY}"
  printf 'export PAP_NVSHMEM_HCA=%q\n' "${PAP_NVSHMEM_HCA:-mlx5_0:1}"
  printf 'export NVSHMEM_SYMMETRIC_SIZE=%q\n' \
    "${NVSHMEM_SYMMETRIC_SIZE:-128M}"
} >"${RUN_DIR}/effective_config.env"

SOURCE_FILES=(
  vllm/pap/config.py
  vllm/pap/transport/nvshmem/world.py
  vllm/pap/transport/nvshmem/runtime.py
  vllm/pap/transport/nvshmem/transport.py
  vllm/pap/transport/nvshmem/device_bridge.cu
  vllm/pap/transport/nvshmem/device_bridge.map
  benchmarks/pap/scripts/configure_nvshmem.sh
  benchmarks/pap/scripts/build_nvshmem_device_bridge.sh
  benchmarks/pap/scripts/run_cross_host_nvshmem_1pa1p.sh
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

LOCAL_PIDS=()
REMOTE_PIDS=()

stop_local_group() {
  local pid="$1"
  kill -TERM -- "-${pid}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

stop_remote_group() {
  local pid="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "kill -TERM -- -${pid} 2>/dev/null || true; for _ in \$(seq 1 10); do kill -0 ${pid} 2>/dev/null || exit 0; sleep 1; done; kill -KILL -- -${pid} 2>/dev/null || true" \
    >/dev/null 2>&1 || true
}

cleanup() {
  trap - EXIT INT TERM
  set +e
  for (( index=${#LOCAL_PIDS[@]}-1; index>=0; index-- )); do
    stop_local_group "${LOCAL_PIDS[index]}"
  done
  for (( index=${#REMOTE_PIDS[@]}-1; index>=0; index-- )); do
    stop_remote_group "${REMOTE_PIDS[index]}"
  done
  mkdir -p "${RUN_DIR}/remote_logs"
  rsync -a \
    "${REMOTE_SSH}:${REMOTE_RUN_DIR}/" "${RUN_DIR}/remote_logs/" \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

remote_start() {
  local role="${1:?role is required}"
  local command
  # shellcheck disable=SC2016  # Expanded by the remote shell after printf.
  printf -v command \
    'mkdir -p %q; cd %q || exit 1; nohup setsid env PAP_ROOT=%q MODEL_PATH=%q PAP_CROSS_HOST_LOCAL_IP=%q PAP_CROSS_HOST_REMOTE_IP=%q PAP_CROSS_HOST_GPU_ID=%q PAP_CROSS_HOST_PREFILL_PORT=%q PAP_CROSS_HOST_ATTENTION_PORT=%q PAP_CROSS_HOST_KV_EVENT_PORT=%q PAP_CROSS_HOST_ATTENTION_TCP_PORT=%q PAP_CROSS_HOST_VLLM_PREFILL_PORT=%q PAP_CROSS_HOST_UID_PORT=%q PAP_CROSS_HOST_MAX_MODEL_LEN=%q PAP_CROSS_HOST_MAX_NUM_SEQS=%q PAP_CROSS_HOST_START_TIMEOUT=%q PAP_NVSHMEM_BUFFER_BYTES=%q PAP_NVSHMEM_CONTROL_BYTES=%q PAP_CROSS_HOST_PREFILL_MEMORY=%q bash %q %q > %q 2>&1 < /dev/null & pid=$!; echo "${pid}"' \
    "${REMOTE_RUN_DIR}" "${REMOTE_ROOT}" "${REMOTE_ROOT}" "${MODEL_PATH}" \
    "${LOCAL_IP}" "${REMOTE_IP}" "${GPU_ID}" "${PREFILL_PORT}" \
    "${ATTENTION_PORT}" "${KV_EVENT_PORT}" "${ATTENTION_TCP_PORT}" \
    "${VLLM_PREFILL_PORT}" "${UID_PORT}" "${MAX_MODEL_LEN}" \
    "${MAX_NUM_SEQS}" "${START_TIMEOUT}" "${BUFFER_BYTES}" \
    "${CONTROL_BYTES}" "${PREFILL_MEMORY}" "${SCRIPT_RELATIVE}" \
    "${role}" "${REMOTE_RUN_DIR}/${role}.log"
  local pid
  pid="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" "${command}")"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Failed to start remote ${role}: ${pid}" >&2
    return 1
  }
  REMOTE_PIDS+=("${pid}")
  echo "Started remote ${role}: pid=${pid}"
}

local_start() {
  local role="${1:?role is required}"
  setsid env \
    PAP_ROOT="${ROOT_DIR}" \
    MODEL_PATH="${MODEL_PATH}" \
    PAP_CROSS_HOST_LOCAL_IP="${LOCAL_IP}" \
    PAP_CROSS_HOST_REMOTE_IP="${REMOTE_IP}" \
    PAP_CROSS_HOST_GPU_ID="${GPU_ID}" \
    PAP_CROSS_HOST_PREFILL_PORT="${PREFILL_PORT}" \
    PAP_CROSS_HOST_PROJECTION_PORT="${PROJECTION_PORT}" \
    PAP_CROSS_HOST_ATTENTION_PORT="${ATTENTION_PORT}" \
    PAP_CROSS_HOST_KV_EVENT_PORT="${KV_EVENT_PORT}" \
    PAP_CROSS_HOST_ATTENTION_TCP_PORT="${ATTENTION_TCP_PORT}" \
    PAP_CROSS_HOST_GATEWAY_PORT="${GATEWAY_PORT}" \
    PAP_CROSS_HOST_VLLM_PREFILL_PORT="${VLLM_PREFILL_PORT}" \
    PAP_CROSS_HOST_VLLM_PROJECTION_PORT="${VLLM_PROJECTION_PORT}" \
    PAP_CROSS_HOST_UID_PORT="${UID_PORT}" \
    PAP_CROSS_HOST_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    PAP_CROSS_HOST_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_CROSS_HOST_START_TIMEOUT="${START_TIMEOUT}" \
    PAP_NVSHMEM_BUFFER_BYTES="${BUFFER_BYTES}" \
    PAP_NVSHMEM_CONTROL_BYTES="${CONTROL_BYTES}" \
    PAP_CROSS_HOST_PROJECTION_MEMORY="${PROJECTION_MEMORY}" \
    bash "${ROOT_DIR}/${SCRIPT_RELATIVE}" "${role}" \
    >"${RUN_DIR}/${role}.log" 2>&1 < /dev/null &
  LOCAL_PIDS+=("$!")
  echo "Started local ${role}: pid=$!"
}

wait_for_health() {
  local endpoint="${1:?endpoint is required}"
  local label="${2:?label is required}"
  for _ in $(seq 1 "${START_TIMEOUT}"); do
    if curl --noproxy '*' -fsS "${endpoint}" >/dev/null 2>&1; then
      echo "Ready: ${label}"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${label}: ${endpoint}" >&2
  return 1
}

gpu_process_snapshot() {
  nvidia-smi \
    --id="${GPU_ID}" \
    --query-compute-apps=pid,gpu_bus_id,used_memory \
    --format=csv,noheader,nounits
}

remote_gpu_process_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "nvidia-smi --id='${GPU_ID}' --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits"
}

ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "test -x '${REMOTE_ROOT}/.venv/bin/vllm' && test -d '${MODEL_PATH}' && test -f '${REMOTE_ROOT}/${SCRIPT_RELATIVE}'"
[[ -f "${ROOT_DIR}/.local/pap-dynamo-router/pap_dynamo_router.abi3.so" ]]

gpu_process_snapshot >"${RUN_DIR}/gpu_processes_before_local.csv"
remote_gpu_process_snapshot >"${RUN_DIR}/gpu_processes_before_remote.csv"
[[ ! -s "${RUN_DIR}/gpu_processes_before_local.csv" ]] || {
  echo "Local GPUs are already in use; refusing a contaminated smoke" >&2
  exit 1
}
[[ ! -s "${RUN_DIR}/gpu_processes_before_remote.csv" ]] || {
  echo "Remote GPUs are already in use; refusing a contaminated smoke" >&2
  exit 1
}

remote_start attention
local_start projection
wait_for_health "http://${REMOTE_IP}:${ATTENTION_PORT}/health" Attention
remote_start prefill
wait_for_health "http://${REMOTE_IP}:${PREFILL_PORT}/health" Prefill
wait_for_health "http://127.0.0.1:${PROJECTION_PORT}/health" Projection
grep -Fq "Successfully initialized the transport: IBGDA" \
  "${RUN_DIR}/projection.log"
grep -Fq "NIC handler will be GPU" "${RUN_DIR}/projection.log"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "grep -Fq 'Successfully initialized the transport: IBGDA' '${REMOTE_RUN_DIR}/attention.log' && grep -Fq 'NIC handler will be GPU' '${REMOTE_RUN_DIR}/attention.log'"
curl --noproxy '*' -fsS "http://${REMOTE_IP}:${PREFILL_PORT}/openapi.json" \
  | jq -e '
      .paths
      | has("/v1/pap/prefill/kv-load")
        and has("/v1/pap/prefill/decode-commit")
        and has("/v1/pap/prefill/lease-release")
    ' >/dev/null \
  || {
    echo "Remote Prefill PAP control routes are missing" >&2
    exit 1
  }
curl --noproxy '*' -fsS \
  "http://127.0.0.1:${PROJECTION_PORT}/openapi.json" \
  | jq -e '.paths | has("/v1/pap/projection/quiesce")' >/dev/null \
  || {
    echo "Local Projection PAP control route is missing" >&2
    exit 1
  }
local_start gateway
wait_for_health "http://127.0.0.1:${GATEWAY_PORT}/health" Gateway

gpu_process_snapshot >"${RUN_DIR}/gpu_processes_after_local.csv"
remote_gpu_process_snapshot >"${RUN_DIR}/gpu_processes_after_remote.csv"
[[ "$(wc -l < "${RUN_DIR}/gpu_processes_after_local.csv")" -ge 1 ]]
[[ "$(wc -l < "${RUN_DIR}/gpu_processes_after_remote.csv")" -ge 2 ]]
local_gpu_processes="$(wc -l < "${RUN_DIR}/gpu_processes_after_local.csv")"
remote_gpu_processes="$(wc -l < "${RUN_DIR}/gpu_processes_after_remote.csv")"
jq -n \
  --arg local_host "${LOCAL_IP}" \
  --arg remote_host "${REMOTE_IP}" \
  --argjson gpu_id "${GPU_ID}" \
  --argjson local_gpu_processes "${local_gpu_processes}" \
  --argjson remote_gpu_processes "${remote_gpu_processes}" \
  '{
    topology: "1pa1p",
    nvshmem_world_size: 2,
    roles: {
      projection: {host: $local_host, nvshmem_rank: 0, gpu: $gpu_id},
      gateway: {host: $local_host},
      attention: {host: $remote_host, nvshmem_rank: 1, gpu: $gpu_id},
      prefill: {host: $remote_host, gpu: $gpu_id}
    },
    observed_gpu_processes: {
      local: $local_gpu_processes,
      remote: $remote_gpu_processes
    }
  }' >"${RUN_DIR}/topology_manifest.json"

REQUEST_ID="pap-cross-host-${RUN_ID}"
curl --noproxy '*' -fsS --max-time "${REQUEST_TIMEOUT}" \
  -D "${RUN_DIR}/response_headers.txt" \
  -o "${RUN_DIR}/response.json" \
  -H 'Content-Type: application/json' \
  -H "X-Request-Id: ${REQUEST_ID}" \
  "http://127.0.0.1:${GATEWAY_PORT}/v1/completions" \
  --data-binary \
    "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"Explain why the sky is blue in one short sentence.\",\"max_tokens\":8,\"temperature\":0,\"stream\":false}"
sleep 2
curl --noproxy '*' -fsS \
  "http://${REMOTE_IP}:${ATTENTION_PORT}/v1/pap/attention/stats" \
  -o "${RUN_DIR}/attention_stats.json"
curl --noproxy '*' -fsS \
  "http://${REMOTE_IP}:${ATTENTION_PORT}/v1/pap/attention/sessions" \
  -o "${RUN_DIR}/attention_sessions.json"
curl --noproxy '*' -fsS \
  "http://127.0.0.1:${GATEWAY_PORT}/v1/pap/topology/stats" \
  -o "${RUN_DIR}/topology_stats.json"

jq -e '.choices | length == 1' "${RUN_DIR}/response.json" >/dev/null
jq -e '
  .attention_dispatch_mode == "nvshmem_graph"
  and .attention_bound_source_ids == ["projection-r0"]
  and .slot_topology_mismatches == 0
  and .decode_token_mismatches == 0
  and .decode_token_dispatch_failures == 0
  and .offload_exec_compute_calls > 0
  and .offload_exec_peer_batches > 0
' "${RUN_DIR}/attention_stats.json" >/dev/null
jq -e '.active_sessions == 0' "${RUN_DIR}/attention_sessions.json" >/dev/null
jq -e '
  .pa_count == 1
  and .projection_count == 1
  and .total_requests == 1
  and .pending_route_requests == 0
  and .dynamo_router.failures == 0
  and .dynamo_router.active_reservations == 0
  and .lifecycle.completed == 1
  and .lifecycle.failed == 0
' "${RUN_DIR}/topology_stats.json" >/dev/null

{
  printf 'STATUS=passed\nLOCAL_HOST=%q\nREMOTE_HOST=%q\nREQUEST_ID=%q\n' \
    "${LOCAL_IP}" "${REMOTE_IP}" "${REQUEST_ID}"
  printf 'OFFLOAD_EXEC_COMPUTE_CALLS=%q\nDECODE_TOKEN_MATCHED=%q\n' \
    "$(jq -r '.offload_exec_compute_calls' "${RUN_DIR}/attention_stats.json")" \
    "$(jq -r '.decode_token_matched' "${RUN_DIR}/attention_stats.json")"
} >"${RUN_DIR}/summary.env"
echo "Cross-host NVSHMEM Graph 1PA1P smoke passed: ${RUN_DIR}"
