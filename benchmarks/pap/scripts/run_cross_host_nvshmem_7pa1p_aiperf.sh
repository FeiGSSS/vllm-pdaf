#!/usr/bin/env bash

set -euo pipefail

ROLE="${1:-coordinator}"
INDEX="${2:-}"
ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REMOTE_SSH="${PAP_CROSS_HOST_REMOTE_SSH:-L20T}"
REMOTE_ROOT="${PAP_CROSS_HOST_REMOTE_ROOT:-${ROOT_DIR}}"
LOCAL_IP="${PAP_CROSS_HOST_LOCAL_IP:-10.128.7.24}"
REMOTE_IP="${PAP_CROSS_HOST_REMOTE_IP:-10.128.7.25}"
LOCAL_GPU="${PAP_CROSS_HOST_PROJECTION_GPU:-0}"
LOCAL_NVSHMEM_GPUS="${PAP_CROSS_HOST_LOCAL_NVSHMEM_GPUS:-0,1,2,3,4,5,6}"
REMOTE_GPUS="${PAP_CROSS_HOST_PA_GPUS:-0,1,2,3,4,5,6}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET="${PAP_AIPERF_INPUT_FILE:-${ROOT_DIR}/benchmarks/pap/datasets/agentic-code/s60-t3-native32k-stratified-seed42/dataset.jsonl}"
EXPECTED_DATASET_SHA256="5d4357a9a037e9e1a3c25acf402ba06afd1c9e7643678da225b9f5cc63cb257c"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
VLLM_BIN="${ROOT_DIR}/.venv/bin/vllm"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_aiperf_profile.sh"
AIPERF_BIN="${ROOT_DIR}/.venv-aiperf/bin/aiperf"
MPS_HELPER="benchmarks/pap/scripts/cross_host_mps.sh"
SCRIPT_RELATIVE="benchmarks/pap/scripts/run_cross_host_nvshmem_7pa1p_aiperf.sh"

PA_COUNT=7
WORLD_SIZE=14
ATTENTION_RANK_BASE=7
PREFILL_PORT_BASE="${PAP_CROSS_HOST_PREFILL_PORT_BASE:-8100}"
PROJECTION_PORT="${PAP_CROSS_HOST_PROJECTION_PORT:-8200}"
ATTENTION_PORT_BASE="${PAP_CROSS_HOST_ATTENTION_PORT_BASE:-8300}"
KV_EVENT_PORT_BASE="${PAP_CROSS_HOST_KV_EVENT_PORT_BASE:-8400}"
KV_EVENT_REPLAY_PORT_BASE="${PAP_CROSS_HOST_KV_EVENT_REPLAY_PORT_BASE:-8500}"
ATTENTION_TCP_PORT_BASE="${PAP_CROSS_HOST_ATTENTION_TCP_PORT_BASE:-9300}"
GATEWAY_PORT="${PAP_CROSS_HOST_GATEWAY_PORT:-9460}"
VLLM_PREFILL_PORT_BASE="${PAP_CROSS_HOST_VLLM_PREFILL_PORT_BASE:-50000}"
VLLM_PROJECTION_PORT="${PAP_CROSS_HOST_VLLM_PROJECTION_PORT:-50200}"
UID_PORT="${PAP_CROSS_HOST_UID_PORT:-29508}"
MAX_MODEL_LEN=32768
PREFILL_BUDGET=2048
PROJECTION_BUDGET=256
MAX_NUM_SEQS=256
BUFFER_BYTES=4194304
CONTROL_BYTES=32768
START_TIMEOUT="${PAP_CROSS_HOST_START_TIMEOUT:-900}"
BENCH_TIMEOUT="${PAP_CROSS_HOST_BENCH_TIMEOUT:-21600}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${PAP_CROSS_HOST_RUN_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/_staging/cross_host_nvshmem_7pa1p_c30/${RUN_ID}}"
REMOTE_RUN_ROOT="${PAP_CROSS_HOST_REMOTE_RUN_ROOT:-/tmp/pap-cross-host-7pa1p-${RUN_ID}}"

IFS=, read -r -a PA_GPUS <<<"${REMOTE_GPUS}"
[[ "${#PA_GPUS[@]}" == "${PA_COUNT}" ]]

set_common_env() {
  local direct_hosts="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}"
  export NO_PROXY="${direct_hosts}"
  export no_proxy="${direct_hosts}"
  export PAP_TOPOLOGY=7pa1p
  export PAP_PA_COUNT=7
  export PAP_PROJECTION_COUNT=1
  export PAP_PREFILL_GPUS=0,1,2,3,4,5,6
  export PAP_ATTENTION_GPUS=0,1,2,3,4,5,6
  export PAP_PROJECTION_GPUS=7
  export PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc
  export PAP_BLOCK_SIZE=16
  export PYTHONHASHSEED=123
  export PAP_ENABLE_DYNAMO_KV_FENCE="${PAP_CROSS_HOST_ENABLE_KV_FENCE:-1}"
  if [[ "${PAP_ENABLE_DYNAMO_KV_FENCE}" != "0" \
    && "${PAP_ENABLE_DYNAMO_KV_FENCE}" != "1" ]]; then
    echo "PAP_CROSS_HOST_ENABLE_KV_FENCE must be 0 or 1" >&2
    return 2
  fi
  export PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=256
  export PAP_DECODE_COMMIT_FAIL_CLOSED=1
  export PAP_DECODE_COMMIT_TIMEOUT=5.0
  export PAP_DECODE_COMMIT_QUEUE_SIZE=1024
  export PAP_DECODE_COMMIT_MAX_ATTEMPTS=8
  export PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS=0.05
  export PAP_DECODE_COMMIT_RETRY_MAX_SECONDS=0.5
  export PAP_DECODE_COMMIT_FLUSH_TIMEOUT=15.0
  export PAP_DECODE_TOKEN_TIMEOUT=0.2
  export PAP_DECODE_TOKEN_QUEUE_SIZE=1024
  export PAP_DECODE_TOKEN_MAX_ATTEMPTS=8
  export PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS=0.05
  export PAP_DECODE_TOKEN_RETRY_MAX_SECONDS=0.5
  export PAP_DECODE_TOKEN_FLUSH_TIMEOUT=5.0
  export PAP_LEASE_RELEASE_TIMEOUT=5.0
  export PAP_LEASE_RELEASE_MAX_ATTEMPTS=5
  export PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS=0.05
  export PAP_LEASE_RELEASE_RETRY_MAX_SECONDS=0.5
  export PAP_KV_LEASE_TTL_SECONDS=300
  export PAP_DECODE_SLOT_PLAN_CACHE_LIMIT=256
  export PAP_PAT_PLAN_CACHE_ENTRIES=32
  export PAP_ATTENTION_GPU_RESIDENT_DISPATCH=0
  export PAP_ATTENTION_KERNEL_POLICY=auto
  export DISABLE_STREAM=1
  export VLLM_PLUGINS=pap
  export PAP_MODEL_HOOKS=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
}

set_nvshmem_env() {
  local rank="${1:?rank is required}"
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-256M}"
  source "${ROOT_DIR}/benchmarks/pap/scripts/configure_nvshmem.sh"
  pap_configure_nvshmem "${ROOT_DIR}" ibgda
  export PAP_NVSHMEM_RANK="${rank}"
  export PAP_NVSHMEM_WORLD_SIZE="${WORLD_SIZE}"
  export PAP_NVSHMEM_ROOT_RANK=0
  export PAP_NVSHMEM_UID_HOST="${LOCAL_IP}"
  export PAP_NVSHMEM_UID_PORT="${UID_PORT}"
  export PAP_NVSHMEM_INIT_TIMEOUT="${START_TIMEOUT}"
  export PAP_NVSHMEM_BUFFER_BYTES="${BUFFER_BYTES}"
  export PAP_NVSHMEM_CONTROL_BYTES="${CONTROL_BYTES}"
  export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-INFO}"
}

load_mps_state() {
  local index="${1:?PA index is required}"
  local state="${REMOTE_RUN_ROOT}/mps_pa_${index}.env"
  [[ -f "${state}" ]]
  # shellcheck disable=SC1090
  source "${state}"
}

run_attention() {
  local index="${INDEX:?Attention index is required}"
  set_common_env
  load_mps_state "${index}"
  set_nvshmem_env "$((ATTENTION_RANK_BASE + index))"
  export CUDA_VISIBLE_DEVICES=0
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  export CUDA_MPS_SM_PARTITION="${ATTENTION_PARTITION}"
  export PAP_OFFLOAD_EXEC_LOCAL_RANK=0
  export PAP_OFFLOAD_EXEC_Q_SIZE=4096
  export PAP_OFFLOAD_EXEC_KV_SIZE=1024
  export PAP_OFFLOAD_EXEC_NUM_HEADS=32
  export PAP_OFFLOAD_EXEC_NUM_KV_HEADS=8
  export PAP_OFFLOAD_EXEC_HEAD_DIM=128
  export PAP_ATTENTION_DECODE_KV_INITIAL_CAPACITY=128
  export PAP_DECODE_COMMIT_ENDPOINT="http://127.0.0.1:$((PREFILL_PORT_BASE + index))/v1/pap/prefill/decode-commit"
  export PAP_LEASE_RELEASE_ENDPOINT="http://127.0.0.1:$((PREFILL_PORT_BASE + index))/v1/pap/prefill/lease-release"
  export PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH="${REMOTE_RUN_ROOT}/runtime_cuda_context_attention_${index}.json"
  exec "${PYTHON_BIN}" -m vllm.pap.service \
    --host 0.0.0.0 \
    --port "$((ATTENTION_PORT_BASE + index))" \
    --tcp-port "$((ATTENTION_TCP_PORT_BASE + index))"
}

run_idle_pe() {
  local rank="${INDEX:?idle PE rank is required}"
  set_common_env
  set_nvshmem_env "${rank}"
  export CUDA_VISIBLE_DEVICES="${rank}"
  export PAP_OFFLOAD_EXEC_LOCAL_RANK=0
  export PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH="${RUN_ROOT}/runtime_cuda_context_idle_${rank}.json"
  exec "${PYTHON_BIN}" \
    "${ROOT_DIR}/benchmarks/pap/microbench/nvshmem_idle_pe.py"
}

run_prefill() {
  local index="${INDEX:?Prefill index is required}"
  set_common_env
  load_mps_state "${index}"
  export CUDA_VISIBLE_DEVICES=0
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  export CUDA_MPS_SM_PARTITION="${PREFILL_PARTITION}"
  export VLLM_PORT="$((VLLM_PREFILL_PORT_BASE + index * 20))"
  export PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH="${REMOTE_RUN_ROOT}/runtime_cuda_context_prefill_${index}.json"
  export PAP_RUNTIME_CUDA_CONTEXT_ROLE=prefill
  if [[ "${PAP_ENABLE_DYNAMO_KV_FENCE}" == "1" ]]; then
    export PAP_KV_SNAPSHOT_SOCKET="/tmp/pap-kv-snapshot-${RUN_ID}-${index}.sock"
  fi
  exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "$((PREFILL_PORT_BASE + index))" \
    --compilation-config \
      '{"mode":"VLLM_COMPILE","cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}' \
    --generation-config vllm \
    --dtype float16 \
    --enable-request-id-headers \
    --enable-prompt-tokens-details \
    --enable-prefix-caching \
    --kv-events-config \
      "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$((KV_EVENT_PORT_BASE + index))\",\"replay_endpoint\":\"tcp://*:$((KV_EVENT_REPLAY_PORT_BASE + index))\",\"topic\":\"pap-pa-${index}\",\"hwm\":100000,\"max_queue_size\":0}" \
    --enable-chunked-prefill \
    --block-size 16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${PREFILL_BUDGET}" \
    --no-async-scheduling \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --kv-transfer-config \
      '{"kv_connector":"PAPPrefillConnector","kv_connector_module_path":"vllm.pap.kv_connector","kv_role":"kv_producer"}'
}

run_projection() {
  set_common_env
  set_nvshmem_env 0
  export PAP_NVSHMEM_UID_BIND_HOST=0.0.0.0
  export CUDA_VISIBLE_DEVICES="${LOCAL_GPU}"
  export VLLM_PORT="${VLLM_PROJECTION_PORT}"
  export PAP_TP_SIZE=1
  export PAP_PROJECTION_KV_UNAWARE=1
  export PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH="${RUN_ROOT}/runtime_cuda_context_projection_0.json"
  export PAP_RUNTIME_CUDA_CONTEXT_ROLE=projection
  exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PROJECTION_PORT}" \
    --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}' \
    --generation-config vllm \
    --dtype float16 \
    --enable-request-id-headers \
    --block-size 16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${PROJECTION_BUDGET}" \
    --async-scheduling \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5181
}

build_pap_groups() {
  local index items=()
  for ((index = 0; index < PA_COUNT; index++)); do
    items+=("${REMOTE_IP}:$((PREFILL_PORT_BASE + index)):${REMOTE_IP}:$((ATTENTION_PORT_BASE + index)):$((ATTENTION_TCP_PORT_BASE + index))")
  done
  local IFS=,
  printf '%s\n' "${items[*]}"
}

build_kv_endpoints() {
  local index items=()
  for ((index = 0; index < PA_COUNT; index++)); do
    items+=("tcp://${REMOTE_IP}:$((KV_EVENT_PORT_BASE + index))")
  done
  local IFS=,
  printf '%s\n' "${items[*]}"
}

build_kv_replay_endpoints() {
  local index items=()
  for ((index = 0; index < PA_COUNT; index++)); do
    items+=("tcp://${REMOTE_IP}:$((KV_EVENT_REPLAY_PORT_BASE + index))")
  done
  local IFS=,
  printf '%s\n' "${items[*]}"
}

run_gateway() {
  set_common_env
  local fence_args=()
  if [[ "${PAP_ENABLE_DYNAMO_KV_FENCE}" == "1" ]]; then
    fence_args+=(--enable-dynamo-kv-fence)
  fi
  if [[ "${PAP_CROSS_HOST_ROUTING_TRACE:-0}" == "1" ]]; then
    export PAP_DYNAMO_ROUTING_TRACE_OUTPUT="${RUN_ROOT}/routing_trace.jsonl"
  fi
  exec "${PYTHON_BIN}" -m vllm.pap.gateway.app \
    --host 127.0.0.1 \
    --port "${GATEWAY_PORT}" \
    --model "${MODEL_PATH}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --generation-config vllm \
    --pap-groups "$(build_pap_groups)" \
    --projections "127.0.0.1:${PROJECTION_PORT}" \
    --kv-event-endpoints "$(build_kv_endpoints)" \
    --kv-event-replay-endpoints "$(build_kv_replay_endpoints)" \
    --prefix-block-size 16 \
    --dynamo-site-packages "${ROOT_DIR}/.local/pap-dynamo-router" \
    --dynamo-max-num-batched-tokens "${PREFILL_BUDGET}" \
    --dynamo-prefill-load-scale 2.0 \
    "${fence_args[@]}" \
    --routing-policy dynamo
}

case "${ROLE}" in
  attention) run_attention ;;
  prefill) run_prefill ;;
  idle) run_idle_pe ;;
  projection) run_projection ;;
  gateway) run_gateway ;;
  coordinator) set_common_env ;;
  *)
    echo "Unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/service_logs"
LOCAL_PIDS=()
REMOTE_PIDS=()
MPS_INDICES=()

stop_local_group() {
  local pid="$1"
  kill -TERM -- "-${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

stop_remote_group() {
  local pid="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "kill -TERM -- -${pid} 2>/dev/null || true; for _ in \$(seq 1 30); do kill -0 ${pid} 2>/dev/null || exit 0; sleep 1; done; kill -KILL -- -${pid} 2>/dev/null || true" \
    >/dev/null 2>&1 || true
}

gpu_processes_local() {
  nvidia-smi --id="${LOCAL_NVSHMEM_GPUS}" \
    --query-compute-apps=pid,gpu_bus_id,used_memory \
    --format=csv,noheader,nounits
}

gpu_processes_remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "nvidia-smi --id='${REMOTE_GPUS}' --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits"
}

wait_gpu_cleanup() {
  local deadline=$((SECONDS + 120))
  local local_apps remote_apps
  while true; do
    local_apps="$(gpu_processes_local 2>/dev/null || true)"
    remote_apps="$(gpu_processes_remote 2>/dev/null || true)"
    if [[ -z "${local_apps}" && -z "${remote_apps}" ]]; then
      printf '{"status":"passed","local_processes":0,"remote_processes":0}\n' \
        >"${RUN_ROOT}/cleanup_audit.json"
      return 0
    fi
    if ((SECONDS >= deadline)); then
      jq -n --arg local "${local_apps}" --arg remote "${remote_apps}" \
        '{status:"failed",local:$local,remote:$remote}' \
        >"${RUN_ROOT}/cleanup_audit.json"
      return 1
    fi
    sleep 2
  done
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  local index cleanup_failed=0
  for ((index = ${#LOCAL_PIDS[@]} - 1; index >= 0; index--)); do
    stop_local_group "${LOCAL_PIDS[index]}"
  done
  for ((index = ${#REMOTE_PIDS[@]} - 1; index >= 0; index--)); do
    stop_remote_group "${REMOTE_PIDS[index]}"
  done
  for ((index = ${#MPS_INDICES[@]} - 1; index >= 0; index--)); do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
      "cd '${REMOTE_ROOT}' && PAP_ROOT='${REMOTE_ROOT}' bash '${MPS_HELPER}' cleanup '${MPS_INDICES[index]}' '${PA_GPUS[MPS_INDICES[index]]}' '${REMOTE_RUN_ROOT}'" \
      >"${RUN_ROOT}/service_logs/mps_cleanup_${MPS_INDICES[index]}.log" \
      2>&1 || cleanup_failed=1
  done
  mkdir -p "${RUN_ROOT}/remote_logs"
  rsync -a "${REMOTE_SSH}:${REMOTE_RUN_ROOT}/" \
    "${RUN_ROOT}/remote_logs/" >/dev/null 2>&1 || cleanup_failed=1
  wait_gpu_cleanup || cleanup_failed=1
  if ((code == 0 && cleanup_failed != 0)); then
    code=1
  fi
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

remote_mps_setup() {
  local index="$1"
  local gpu="${PA_GPUS[index]}"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "cd '${REMOTE_ROOT}' && mkdir -p '${REMOTE_RUN_ROOT}' && PAP_ROOT='${REMOTE_ROOT}' bash '${MPS_HELPER}' setup '${index}' '${gpu}' '${REMOTE_RUN_ROOT}'"
  MPS_INDICES+=("${index}")
  rsync -a "${REMOTE_SSH}:${REMOTE_RUN_ROOT}/mps_pa_${index}.env" \
    "${RUN_ROOT}/mps_pa_${index}.env"
}

remote_start() {
  local role="$1"
  local index="$2"
  local command pid
  # shellcheck disable=SC2016  # Expanded by the remote shell after printf.
  printf -v command \
    'cd %q || exit 1; nohup setsid env PAP_ROOT=%q MODEL_PATH=%q RUN_ID=%q PAP_CROSS_HOST_REMOTE_RUN_ROOT=%q PAP_CROSS_HOST_LOCAL_IP=%q PAP_CROSS_HOST_REMOTE_IP=%q PAP_CROSS_HOST_PA_GPUS=%q PAP_CROSS_HOST_PREFILL_PORT_BASE=%q PAP_CROSS_HOST_ATTENTION_PORT_BASE=%q PAP_CROSS_HOST_KV_EVENT_PORT_BASE=%q PAP_CROSS_HOST_KV_EVENT_REPLAY_PORT_BASE=%q PAP_CROSS_HOST_ATTENTION_TCP_PORT_BASE=%q PAP_CROSS_HOST_VLLM_PREFILL_PORT_BASE=%q PAP_CROSS_HOST_UID_PORT=%q PAP_CROSS_HOST_START_TIMEOUT=%q bash %q %q %q > %q 2>&1 < /dev/null & pid=$!; echo "${pid}"' \
    "${REMOTE_ROOT}" "${REMOTE_ROOT}" "${MODEL_PATH}" "${RUN_ID}" \
    "${REMOTE_RUN_ROOT}" "${LOCAL_IP}" "${REMOTE_IP}" "${REMOTE_GPUS}" \
    "${PREFILL_PORT_BASE}" "${ATTENTION_PORT_BASE}" "${KV_EVENT_PORT_BASE}" \
    "${KV_EVENT_REPLAY_PORT_BASE}" \
    "${ATTENTION_TCP_PORT_BASE}" "${VLLM_PREFILL_PORT_BASE}" "${UID_PORT}" \
    "${START_TIMEOUT}" "${SCRIPT_RELATIVE}" "${role}" "${index}" \
    "${REMOTE_RUN_ROOT}/${role}_${index}.log"
  pid="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" "${command}")"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]]
  REMOTE_PIDS+=("${pid}")
  echo "Started remote ${role} ${index}: pid=${pid}"
}

local_start() {
  local role="$1"
  local index="${2:-}"
  local log_name="${role}${index:+_${index}}"
  setsid env \
    PAP_ROOT="${ROOT_DIR}" MODEL_PATH="${MODEL_PATH}" RUN_ID="${RUN_ID}" \
    PAP_CROSS_HOST_RUN_ROOT="${RUN_ROOT}" \
    PAP_CROSS_HOST_REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}" \
    PAP_CROSS_HOST_LOCAL_IP="${LOCAL_IP}" \
    PAP_CROSS_HOST_REMOTE_IP="${REMOTE_IP}" \
    PAP_CROSS_HOST_PROJECTION_GPU="${LOCAL_GPU}" \
    PAP_CROSS_HOST_LOCAL_NVSHMEM_GPUS="${LOCAL_NVSHMEM_GPUS}" \
    PAP_CROSS_HOST_PA_GPUS="${REMOTE_GPUS}" \
    PAP_CROSS_HOST_PREFILL_PORT_BASE="${PREFILL_PORT_BASE}" \
    PAP_CROSS_HOST_PROJECTION_PORT="${PROJECTION_PORT}" \
    PAP_CROSS_HOST_ATTENTION_PORT_BASE="${ATTENTION_PORT_BASE}" \
    PAP_CROSS_HOST_KV_EVENT_PORT_BASE="${KV_EVENT_PORT_BASE}" \
    PAP_CROSS_HOST_KV_EVENT_REPLAY_PORT_BASE="${KV_EVENT_REPLAY_PORT_BASE}" \
    PAP_CROSS_HOST_ATTENTION_TCP_PORT_BASE="${ATTENTION_TCP_PORT_BASE}" \
    PAP_CROSS_HOST_GATEWAY_PORT="${GATEWAY_PORT}" \
    PAP_CROSS_HOST_VLLM_PREFILL_PORT_BASE="${VLLM_PREFILL_PORT_BASE}" \
    PAP_CROSS_HOST_VLLM_PROJECTION_PORT="${VLLM_PROJECTION_PORT}" \
    PAP_CROSS_HOST_UID_PORT="${UID_PORT}" \
    PAP_CROSS_HOST_START_TIMEOUT="${START_TIMEOUT}" \
    PAP_CROSS_HOST_UID_PORT="${UID_PORT}" \
    bash "${ROOT_DIR}/${SCRIPT_RELATIVE}" "${role}" "${index}" \
    >"${RUN_ROOT}/service_logs/${log_name}.log" 2>&1 < /dev/null &
  LOCAL_PIDS+=("$!")
  echo "Started local ${role}: pid=$!"
}

wait_for_health() {
  local endpoint="$1"
  local label="$2"
  local deadline=$((SECONDS + START_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl --noproxy '*' -fsS "${endpoint}" >/dev/null 2>&1; then
      echo "Ready: ${label}"
      return 0
    fi
    check_children_alive
    sleep 1
  done
  echo "Timed out waiting for ${label}: ${endpoint}" >&2
  return 1
}

check_children_alive() {
  local pid
  for pid in "${LOCAL_PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Local PAP process exited during startup: pid=${pid}" >&2
      return 1
    fi
  done
  for pid in "${REMOTE_PIDS[@]}"; do
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
      "kill -0 '${pid}'"; then
      echo "Remote PAP process exited during startup: pid=${pid}" >&2
      return 1
    fi
  done
}

write_effective_config() {
  {
    printf 'export MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'export PAP_AIPERF_INPUT_FILE=%q\n' "${DATASET}"
    printf 'export PAP_AIPERF_SESSIONS=60\n'
    printf 'export PAP_AIPERF_EXPECTED_REQUESTS=180\n'
    printf 'export PAP_AIPERF_CONCURRENCY=30\n'
    printf 'export PAP_CROSS_HOST_REMOTE_SSH=%q\n' "${REMOTE_SSH}"
    printf 'export PAP_CROSS_HOST_REMOTE_ROOT=%q\n' "${REMOTE_ROOT}"
    printf 'export PAP_CROSS_HOST_LOCAL_IP=%q\n' "${LOCAL_IP}"
    printf 'export PAP_CROSS_HOST_REMOTE_IP=%q\n' "${REMOTE_IP}"
    printf 'export PAP_CROSS_HOST_PROJECTION_GPU=%q\n' "${LOCAL_GPU}"
    printf 'export PAP_CROSS_HOST_LOCAL_NVSHMEM_GPUS=%q\n' \
      "${LOCAL_NVSHMEM_GPUS}"
    printf 'export PAP_CROSS_HOST_PA_GPUS=%q\n' "${REMOTE_GPUS}"
    printf 'export PAP_CROSS_HOST_START_TIMEOUT=%q\n' "${START_TIMEOUT}"
    printf 'export PAP_CROSS_HOST_BENCH_TIMEOUT=%q\n' "${BENCH_TIMEOUT}"
    printf 'export PAP_CROSS_HOST_ROUTING_TRACE=%q\n' \
      "${PAP_CROSS_HOST_ROUTING_TRACE:-0}"
    printf 'export PAP_CROSS_HOST_ENABLE_KV_FENCE=%q\n' \
      "${PAP_CROSS_HOST_ENABLE_KV_FENCE:-1}"
    printf 'export PYTHONHASHSEED=%q\n' "${PYTHONHASHSEED}"
    printf 'export PAP_CROSS_HOST_UID_PORT=%q\n' "${UID_PORT}"
    printf 'export PAP_CROSS_HOST_PREFILL_PORT_BASE=%q\n' "${PREFILL_PORT_BASE}"
    printf 'export PAP_CROSS_HOST_PROJECTION_PORT=%q\n' "${PROJECTION_PORT}"
    printf 'export PAP_CROSS_HOST_ATTENTION_PORT_BASE=%q\n' "${ATTENTION_PORT_BASE}"
    printf 'export PAP_CROSS_HOST_KV_EVENT_PORT_BASE=%q\n' "${KV_EVENT_PORT_BASE}"
    printf 'export PAP_CROSS_HOST_KV_EVENT_REPLAY_PORT_BASE=%q\n' \
      "${KV_EVENT_REPLAY_PORT_BASE}"
    printf 'export PAP_CROSS_HOST_ATTENTION_TCP_PORT_BASE=%q\n' \
      "${ATTENTION_TCP_PORT_BASE}"
    printf 'export PAP_CROSS_HOST_GATEWAY_PORT=%q\n' "${GATEWAY_PORT}"
    printf 'export NVSHMEM_SYMMETRIC_SIZE=%q\n' \
      "${NVSHMEM_SYMMETRIC_SIZE:-256M}"
    printf 'export PAP_NVSHMEM_HCA=%q\n' "${PAP_NVSHMEM_HCA:-mlx5_0:1}"
  } >"${RUN_ROOT}/effective_config.env"
}

audit_source() {
  local pap_files=()
  mapfile -t pap_files < <(git ls-files vllm/pap)
  [[ "${#pap_files[@]}" -gt 0 ]]
  local files=(
    "${pap_files[@]}"
    vllm/v1/core/block_pool.py
    vllm/v1/core/kv_cache_utils.py
    benchmarks/pap/scripts/configure_nvshmem.sh
    benchmarks/pap/scripts/build_nvshmem_device_bridge.sh
    benchmarks/pap/microbench/nvshmem_idle_pe.py
    benchmarks/pap/scripts/cross_host_mps.sh
    benchmarks/pap/scripts/run_cross_host_nvshmem_7pa1p_aiperf.sh
    benchmarks/pap/scripts/run_aiperf_profile.sh
    benchmarks/pap/scripts/aiperf_compat_entry.py
  )
  git rev-parse HEAD >"${RUN_ROOT}/local_git_revision.txt"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "cd '${REMOTE_ROOT}' && git rev-parse HEAD" \
    >"${RUN_ROOT}/remote_git_revision.txt"
  sha256sum "${files[@]}" >"${RUN_ROOT}/local_source_checksums.txt"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
    "cd '${REMOTE_ROOT}' && sha256sum ${files[*]}" \
    >"${RUN_ROOT}/remote_source_checksums.txt"
  local revision_match=false checksum_match=false
  if cmp -s "${RUN_ROOT}/local_git_revision.txt" \
    "${RUN_ROOT}/remote_git_revision.txt"; then
    revision_match=true
  fi
  if cmp -s "${RUN_ROOT}/local_source_checksums.txt" \
    "${RUN_ROOT}/remote_source_checksums.txt"; then
    checksum_match=true
  fi
  jq -n \
    --argjson revision_match "${revision_match}" \
    --argjson checksum_match "${checksum_match}" \
    --arg local_revision "$(cat "${RUN_ROOT}/local_git_revision.txt")" \
    --arg remote_revision "$(cat "${RUN_ROOT}/remote_git_revision.txt")" \
    '{
      status: (if $checksum_match then "passed" else "failed" end),
      participating_source_checksums_match: $checksum_match,
      git_revisions_match: $revision_match,
      local_git_revision: $local_revision,
      remote_git_revision: $remote_revision
    }' >"${RUN_ROOT}/source_audit.json"
  [[ "${checksum_match}" == "true" ]]
}

audit_block_hash_contract() {
  local probe remote_command local_hash remote_hash
  probe='from vllm.utils.hashing import get_hash_fn_by_name; from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash; fn = get_hash_fn_by_name("sha256"); init_none_hash(fn); print(hash_block_tokens(fn, None, list(range(16)), None).hex())'
  PYTHONHASHSEED="${PYTHONHASHSEED}" "${PYTHON_BIN}" -c "${probe}" \
    >"${RUN_ROOT}/local_block_hash.txt"
  printf -v remote_command \
    'cd %q && PYTHONHASHSEED=%q %q -c %q' \
    "${REMOTE_ROOT}" "${PYTHONHASHSEED}" "${PYTHON_BIN}" "${probe}"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" "${remote_command}" \
    >"${RUN_ROOT}/remote_block_hash.txt"
  local_hash="$(cat "${RUN_ROOT}/local_block_hash.txt")"
  remote_hash="$(cat "${RUN_ROOT}/remote_block_hash.txt")"
  jq -n \
    --arg local_hash "${local_hash}" \
    --arg remote_hash "${remote_hash}" \
    --arg seed "${PYTHONHASHSEED}" \
    '{
      status: (if $local_hash == $remote_hash then "passed" else "failed" end),
      python_hash_seed: $seed,
      local_hash: $local_hash,
      remote_hash: $remote_hash
    }' >"${RUN_ROOT}/block_hash_audit.json"
  [[ "${local_hash}" =~ ^[0-9a-f]{64}$ && "${local_hash}" == "${remote_hash}" ]]
}

audit_contexts() {
  rsync -a "${REMOTE_SSH}:${REMOTE_RUN_ROOT}/runtime_cuda_context_*.json" \
    "${RUN_ROOT}/"
  gpu_processes_local >"${RUN_ROOT}/gpu_processes_started_local.csv"
  gpu_processes_remote >"${RUN_ROOT}/gpu_processes_started_remote.csv"
  local projection_bus
  projection_bus="$(
    nvidia-smi --id="${LOCAL_GPU}" --query-gpu=pci.bus_id \
      --format=csv,noheader
  )"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${projection_bus}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
projection_bus = sys.argv[2].strip().upper()
attention = [json.loads(path.read_text()) for path in sorted(root.glob("runtime_cuda_context_attention_*.json"))]
prefill = [json.loads(path.read_text()) for path in sorted(root.glob("runtime_cuda_context_prefill_*.json"))]
idle = [json.loads(path.read_text()) for path in sorted(root.glob("runtime_cuda_context_idle_*.json"))]
projection_log = (root / "service_logs/projection.log").read_text(errors="replace")
projection_pids = {
    int(value) for value in re.findall(r"\(EngineCore pid=(\d+)\)", projection_log)
}
gpu_processes = []
for line in (root / "gpu_processes_started_local.csv").read_text().splitlines():
    if not line.strip():
        continue
    pid, bus_id, used_memory = (part.strip() for part in line.split(","))
    gpu_processes.append(
        {"pid": int(pid), "gpu_bus_id": bus_id.upper(), "used_memory_mib": int(used_memory)}
    )
projection = [row for row in gpu_processes if row["gpu_bus_id"] == projection_bus]
errors = []
if len(attention) != 7 or len(prefill) != 7:
    errors.append(f"remote context counts attention={len(attention)} prefill={len(prefill)}")
if any(row["role"] != "attention" or row["multiprocessor_count"] != 12 for row in attention):
    errors.append("Attention MPS context mismatch")
if any(row["role"] != "prefill" or row["multiprocessor_count"] != 80 for row in prefill):
    errors.append("Prefill MPS context mismatch")
if len(gpu_processes) != 7:
    errors.append(f"local NVSHMEM GPU process count={len(gpu_processes)}")
if len(projection_pids) != 1:
    errors.append(f"Projection EngineCore log PID count={len(projection_pids)}")
if len(projection) != 1 or (
    projection_pids and projection[0]["pid"] not in projection_pids
):
    errors.append("Projection PID/GPU-bus mapping mismatch")
if projection and not Path(f"/proc/{projection[0]['pid']}").is_dir():
    errors.append("Projection EngineCore exited before topology audit")
if len(idle) != 6 or any(row["role"] != "nvshmem_idle" or row["multiprocessor_count"] != 92 for row in idle):
    errors.append("local idle PE context mismatch")
if any(not Path(f"/proc/{row['pid']}").is_dir() for row in idle):
    errors.append("local idle PE exited before topology audit")
payload = {
    "status": "failed" if errors else "passed",
    "attention": attention,
    "prefill": prefill,
    "projection": projection,
    "idle": idle,
    "errors": errors,
}
(root / "startup_topology_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
  local context_path pid
  for context_path in "${RUN_ROOT}"/runtime_cuda_context_attention_*.json \
    "${RUN_ROOT}"/runtime_cuda_context_prefill_*.json; do
    pid="$(jq -r '.pid' "${context_path}")"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
      "test -d '/proc/${pid}'"
  done
}

wait_drained() {
  local deadline=$((SECONDS + 180))
  local index healthy
  while ((SECONDS < deadline)); do
    healthy="$(curl --noproxy '*' -fsS "http://127.0.0.1:${GATEWAY_PORT}/health" || true)"
    if jq -e '
      .inflight_requests == 0
      and .lifecycle.active == 0
      and .lifecycle.failed == 0
      and .dynamo_router.healthy == true
      and .dynamo_router.failures == 0
      and .dynamo_router.active_reservations == 0
      and .dynamo_router.pending_cleanup == 0
      and .dynamo_router.pending_selections == 0
      and .kv_fence.healthy == true
      and .kv_fence.failures == 0
      and .kv_fence.active_conversations == 0
      and .kv_fence.pending_releases == 0
    ' <<<"${healthy}" >/dev/null 2>&1; then
      for ((index = 0; index < PA_COUNT; index++)); do
        curl --noproxy '*' -fsS \
          "http://${REMOTE_IP}:$((ATTENTION_PORT_BASE + index))/v1/pap/attention/sessions" \
          | jq -e '.active_sessions == 0' >/dev/null || break
      done
      if ((index == PA_COUNT)); then
        printf 'STATUS=passed\n' >"${RUN_ROOT}/drain_audit.env"
        return 0
      fi
    fi
    sleep 2
  done
  printf 'STATUS=failed\n' >"${RUN_ROOT}/drain_audit.env"
  return 1
}

capture_and_audit_results() {
  local index
  for ((index = 0; index < PA_COUNT; index++)); do
    curl --noproxy '*' -fsS \
      "http://${REMOTE_IP}:$((ATTENTION_PORT_BASE + index))/v1/pap/attention/stats" \
      -o "${RUN_ROOT}/attention_stats_${index}.json"
    if [[ "${PAP_ENABLE_DYNAMO_KV_FENCE}" == "1" ]]; then
      curl --noproxy '*' -fsS \
        "http://${REMOTE_IP}:$((PREFILL_PORT_BASE + index))/v1/pap/prefill/kv-migration/snapshot" \
        -o "${RUN_ROOT}/prefill_kv_snapshot_final_${index}.json"
    fi
  done
  curl --noproxy '*' -fsS \
    "http://127.0.0.1:${GATEWAY_PORT}/v1/pap/topology/stats" \
    -o "${RUN_ROOT}/topology_stats.json"
  rsync -a "${REMOTE_SSH}:${REMOTE_RUN_ROOT}/" \
    "${RUN_ROOT}/remote_logs/"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${PAP_ENABLE_DYNAMO_KV_FENCE}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
fence_enabled = sys.argv[2] == "1"
profile = json.loads((root / "aiperf/profile.json").read_text())
stats = [json.loads((root / f"attention_stats_{index}.json").read_text()) for index in range(7)]
topology = json.loads((root / "topology_stats.json").read_text())
snapshots = (
    [
        json.loads((root / f"prefill_kv_snapshot_final_{index}.json").read_text())
        for index in range(7)
    ]
    if fence_enabled
    else []
)
errors = []
if profile["request_count"]["avg"] != 180 or profile["time_to_first_token"]["count"] != 180:
    errors.append("AIPerf did not complete 180 requests")
if profile["error_summary"] or profile["osl_mismatch_count"]["avg"] != 0 or profile["was_cancelled"]:
    errors.append("AIPerf reported request or output-length errors")
zero_fields = (
    "decode_token_pending_tokens",
    "decode_token_pending_kv",
    "decode_token_dispatching",
    "decode_token_mismatches",
    "decode_token_dispatch_failures",
    "slot_topology_mismatches",
)
for index, row in enumerate(stats):
    if row.get("attention_dispatch_mode") != "nvshmem_graph":
        errors.append(f"attention[{index}] dispatch mode mismatch")
    if row.get("attention_bound_source_ids") != ["projection-r0"]:
        errors.append(f"attention[{index}] peer binding mismatch")
    if int(row.get("offload_exec_peer_batches", 0)) <= 0:
        errors.append(f"attention[{index}] received no batches")
    for field in zero_fields:
        if int(row.get(field, -1)) != 0:
            errors.append(f"attention[{index}] {field}={row.get(field)}")
if topology["total_requests"] != 180 or topology["pending_route_requests"] != 0:
    errors.append("Gateway request accounting mismatch")
if topology["lifecycle"]["completed"] != 180 or topology["lifecycle"]["failed"] != 0:
    errors.append("Gateway lifecycle mismatch")
if topology["dynamo_router"]["failures"] != 0 or topology["dynamo_router"]["active_reservations"] != 0:
    errors.append("Dynamo routing did not drain")
fence = topology.get("kv_fence", {})
if fence.get("enabled") != fence_enabled or not fence.get("healthy"):
    errors.append("Dynamo KV fence is not enabled and healthy")
if fence_enabled and fence.get("checks") != 120:
    errors.append(f"Dynamo KV fence checks={fence.get('checks')} expected=120")
if fence_enabled and fence.get("exports", 0) + fence.get("misses", 0) != fence.get("checks"):
    errors.append("Dynamo KV fence outcome accounting mismatch")
if (
    fence.get("failures") != 0
    or fence.get("active_conversations") != 0
    or fence.get("pending_releases") != 0
):
    errors.append("Dynamo KV fence did not drain")
router = topology["dynamo_router"]
if router.get("kv_fence_checks") != fence.get("exports"):
    errors.append("Dynamo router and KV fence check counts differ")
for index, snapshot in enumerate(snapshots):
    if not snapshot.get("ready") or snapshot.get("failures") != 0:
        errors.append(f"Prefill snapshot[{index}] is unhealthy")
    if snapshot.get("active_exports") != 0:
        errors.append(f"Prefill snapshot[{index}] retained exports")
pattern = re.compile(
    r"CUDA out of memory|EngineDeadError|^Traceback| ERROR .*Traceback|"
    r"Exception in thread|PAP decode commit failed|PAP lease release failed|"
    r"decode-token delivery failed|slot topology mismatch",
    re.MULTILINE,
)
matches = []
for path in [root / "service_logs/projection.log", *sorted((root / "remote_logs").glob("*.log"))]:
    for match in pattern.finditer(path.read_text(errors="replace")):
        matches.append(f"{path}:{match.group(0)}")
if matches:
    errors.append(f"correctness log matches={len(matches)}")
(root / "correctness_audit_matches.log").write_text("\n".join(matches) + ("\n" if matches else ""))
summary = {
    "status": "failed" if errors else "passed",
    "requests": int(profile["request_count"]["avg"]),
    "ttft_mean_ms": profile["time_to_first_token"]["avg"],
    "ttft_p99_ms": profile["time_to_first_token"]["p99"],
    "itl_mean_ms": profile["inter_token_latency"]["avg"],
    "itl_p99_ms": profile["inter_token_latency"]["p99"],
    "e2e_mean_ms": profile["request_latency"]["avg"],
    "e2e_p99_ms": profile["request_latency"]["p99"],
    "request_throughput": profile["request_throughput"]["avg"],
    "output_token_throughput": profile["output_token_throughput"]["avg"],
    "total_output_tokens": int(profile["total_output_tokens"]["avg"]),
    "kv_fence": fence,
    "dynamo_kv_fence_checks": router.get("kv_fence_checks"),
    "dynamo_kv_fence_pins": router.get("kv_fence_pins"),
    "errors": errors,
}
(root / "performance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

command -v ssh >/dev/null
command -v rsync >/dev/null
command -v curl >/dev/null
command -v jq >/dev/null
command -v cmp >/dev/null
command -v sha256sum >/dev/null
command -v nvidia-smi >/dev/null
[[ -x "${PYTHON_BIN}" && -x "${VLLM_BIN}" && -x "${AIPERF_BIN}" ]]
[[ -d "${MODEL_PATH}" && -f "${DATASET}" ]]
[[ "$(sha256sum "${DATASET}" | cut -d' ' -f1)" == "${EXPECTED_DATASET_SHA256}" ]]
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
  "test -x '${REMOTE_ROOT}/.venv/bin/vllm' && test -d '${MODEL_PATH}' && test -f '${REMOTE_ROOT}/${SCRIPT_RELATIVE}' && test -f '${REMOTE_ROOT}/${MPS_HELPER}'"

write_effective_config
audit_source
audit_block_hash_contract
gpu_processes_local >"${RUN_ROOT}/gpu_processes_before_local.csv"
gpu_processes_remote >"${RUN_ROOT}/gpu_processes_before_remote.csv"
[[ ! -s "${RUN_ROOT}/gpu_processes_before_local.csv" ]]
[[ ! -s "${RUN_ROOT}/gpu_processes_before_remote.csv" ]]

for ((INDEX = 0; INDEX < PA_COUNT; INDEX++)); do
  remote_mps_setup "${INDEX}"
done
for ((INDEX = 0; INDEX < PA_COUNT; INDEX++)); do
  remote_start attention "${INDEX}"
done
for ((INDEX = 1; INDEX < ATTENTION_RANK_BASE; INDEX++)); do
  local_start idle "${INDEX}"
done
local_start projection
for ((INDEX = 0; INDEX < PA_COUNT; INDEX++)); do
  wait_for_health \
    "http://${REMOTE_IP}:$((ATTENTION_PORT_BASE + INDEX))/health" \
    "Attention ${INDEX}"
done
for ((INDEX = 0; INDEX < PA_COUNT; INDEX++)); do
  remote_start prefill "${INDEX}"
done
for ((INDEX = 0; INDEX < PA_COUNT; INDEX++)); do
  wait_for_health \
    "http://${REMOTE_IP}:$((PREFILL_PORT_BASE + INDEX))/health" \
    "Prefill ${INDEX}"
done
wait_for_health "http://127.0.0.1:${PROJECTION_PORT}/health" Projection
for ((INDEX = 0; INDEX < PA_COUNT; INDEX++)); do
  curl --noproxy '*' -fsS \
    "http://${REMOTE_IP}:$((PREFILL_PORT_BASE + INDEX))/openapi.json" \
    | jq -e '.paths | has("/v1/pap/prefill/kv-load") and has("/v1/pap/prefill/decode-commit") and has("/v1/pap/prefill/lease-release") and has("/v1/pap/prefill/kv-fence/acquire") and has("/v1/pap/prefill/kv-fence/release")' \
      >/dev/null
  if [[ "${PAP_ENABLE_DYNAMO_KV_FENCE}" == "1" ]]; then
    curl --noproxy '*' -fsS \
      "http://${REMOTE_IP}:$((PREFILL_PORT_BASE + INDEX))/v1/pap/prefill/kv-migration/snapshot" \
      | jq -e '.ready == true and .failures == 0' >/dev/null
  fi
  grep -Fq "Successfully initialized the transport: IBGDA" \
    <(ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
      "cat '${REMOTE_RUN_ROOT}/attention_${INDEX}.log'")
  grep -Fq "NIC handler will be GPU" \
    <(ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_SSH}" \
      "cat '${REMOTE_RUN_ROOT}/attention_${INDEX}.log'")
done
grep -Fq "Successfully initialized the transport: IBGDA" \
  "${RUN_ROOT}/service_logs/projection.log"
grep -Fq "NIC handler will be GPU" \
  "${RUN_ROOT}/service_logs/projection.log"
local_start gateway
wait_for_health "http://127.0.0.1:${GATEWAY_PORT}/health" Gateway
sleep 30
audit_contexts

timeout "${BENCH_TIMEOUT}" env \
  PAP_ROOT="${ROOT_DIR}" \
  AIPERF_BIN="${AIPERF_BIN}" \
  MODEL_PATH="${MODEL_PATH}" \
  AIPERF_INPUT_FILE="${DATASET}" \
  AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
  AIPERF_TARGET_URL="http://127.0.0.1:${GATEWAY_PORT}" \
  AIPERF_OUTPUT_DIR="${RUN_ROOT}/aiperf" \
  AIPERF_SESSIONS=60 \
  AIPERF_CONCURRENCY=30 \
  AIPERF_TIMING_MODE=concurrency \
  AIPERF_REQUEST_TIMEOUT_SECONDS="${BENCH_TIMEOUT}" \
  AIPERF_NUM_PROFILE_RUNS=1 \
  "${AIPERF_RUNNER}" >"${RUN_ROOT}/7PA1P_aiperf.log" 2>&1

wait_drained
capture_and_audit_results
printf 'STATUS=passed\n' >"${RUN_ROOT}/summary.env"
echo "Cross-host NVSHMEM 7PA1P AIPerf C30 passed: ${RUN_ROOT}"
