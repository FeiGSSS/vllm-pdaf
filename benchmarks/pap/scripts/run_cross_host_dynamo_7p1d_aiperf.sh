#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# shellcheck disable=SC2029

set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ROLE="${DYNAMO_CROSS_HOST_ROLE:-controller}"
DYNAMO_PYTHON="${DYNAMO_PYTHON:-${ROOT_DIR}/.venv-dynamo/bin/python}"
PAP_PYTHON="${PAP_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_aiperf_profile.sh"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
REMOTE_HOST="${DYNAMO_REMOTE_HOST:-L20T}"
REMOTE_ROOT="${DYNAMO_REMOTE_ROOT:-${ROOT_DIR}}"
LOCAL_IP="${DYNAMO_LOCAL_IP:-10.128.7.24}"
REMOTE_IP="${DYNAMO_REMOTE_IP:-10.128.7.25}"
LOCAL_UCX_DEVICE="${DYNAMO_LOCAL_UCX_DEVICE:-enx9c69d3247e5c}"
REMOTE_UCX_DEVICE="${DYNAMO_REMOTE_UCX_DEVICE:-enx9c69d341f8cb}"
LOCAL_DECODE_GPU="${DYNAMO_LOCAL_DECODE_GPU:-7}"
PREFILL_COUNT=7

RUN_ID="${DYNAMO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_cross_host_dynamo_7p1d}"
RUN_ROOT="${DYNAMO_RUN_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/e2e/_runs/dynamo/${RUN_ID}}"
REMOTE_RUN_ROOT="${DYNAMO_REMOTE_RUN_ROOT:-${REMOTE_ROOT}/benchmarks/pap/experiments/_staging/${RUN_ID}}"
LOG_ROOT="${RUN_ROOT}/service_logs"
REMOTE_LOG_ROOT="${REMOTE_RUN_ROOT}/service_logs"
NAMESPACE="${DYNAMO_NAMESPACE:-pap-cross-host-${RUN_ID}}"
FRONTEND_PORT="${DYNAMO_FRONTEND_PORT:-19400}"
ETCD_PORT="${DYNAMO_ETCD_PORT:-23379}"
SYSTEM_PORT_BASE="${DYNAMO_SYSTEM_PORT_BASE:-19500}"
NIXL_PORT_BASE="${DYNAMO_NIXL_PORT_BASE:-19600}"
KV_EVENT_PORT_BASE="${DYNAMO_KV_EVENT_PORT_BASE:-19700}"
VLLM_PORT_BASE="${DYNAMO_VLLM_PORT_BASE:-19800}"
ETCD_ENDPOINTS="http://${LOCAL_IP}:${ETCD_PORT}"
ETCD_BIN="${DYNAMO_ETCD_BIN:-${ROOT_DIR}/.local/dynamo-etcd-3.6.1/etcd}"
ETCD_LEASE_TTL="${DYNAMO_ETCD_LEASE_TTL:-3600}"

MAX_MODEL_LEN="${DYNAMO_MAX_MODEL_LEN:-32768}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS:-2048}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DYNAMO_DECODE_MAX_NUM_BATCHED_TOKENS:-256}"
MAX_NUM_SEQS="${DYNAMO_MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${DYNAMO_GPU_MEMORY_UTILIZATION:-0.90}"
BLOCK_SIZE="${DYNAMO_BLOCK_SIZE:-16}"
PREFILL_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
DECODE_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_DECODE_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
ROUTER_PREFILL_LOAD_SCALE="${DYNAMO_ROUTER_PREFILL_LOAD_SCALE:-2.0}"
CLUSTER_READY_WAIT_SECONDS="${DYNAMO_CLUSTER_READY_WAIT_SECONDS:-30}"
MIN_KV_TRANSFER_MB_S="${DYNAMO_MIN_KV_TRANSFER_MB_S:-10}"
EXPECTED_LINK_SPEED_MBIT_S="${DYNAMO_EXPECTED_LINK_SPEED_MBIT_S:-1000}"
KV_LEASE_DURATION="${DYNAMO_KV_LEASE_DURATION:-3600}"
SMOKE_ONLY="${DYNAMO_SMOKE_ONLY:-0}"

AIPERF_INPUT_FILE="${DYNAMO_AIPERF_INPUT_FILE:-}"
AIPERF_INPUT_SHA256="${DYNAMO_AIPERF_INPUT_SHA256:-}"
AIPERF_OUTPUT_DIR="${DYNAMO_AIPERF_OUTPUT_DIR:-${RUN_ROOT}/aiperf}"
AIPERF_SESSIONS="${DYNAMO_AIPERF_SESSIONS:-60}"
AIPERF_CONCURRENCY="${DYNAMO_AIPERF_CONCURRENCY:-30}"
AIPERF_EXPECTED_REQUESTS="${DYNAMO_AIPERF_EXPECTED_REQUESTS:-180}"
AIPERF_REQUEST_TIMEOUT_SECONDS="${DYNAMO_AIPERF_REQUEST_TIMEOUT_SECONDS:-21600}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

configure_worker_env() {
  local host_ip="$1"
  local net_device="$2"
  source "${ROOT_DIR}/benchmarks/pap/scripts/configure_cross_host_nixl.sh"
  pap_configure_cross_host_nixl "${ROOT_DIR}" "${host_ip}" "${net_device}"
  export PYTHONSAFEPATH=1 PYTHONPATH=
  export NO_PROXY="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}"
  export no_proxy="${NO_PROXY}"
  export DYN_NAMESPACE="${NAMESPACE}"
  export DYN_DISCOVERY_BACKEND=etcd
  export DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
  export ETCD_ENDPOINTS ETCD_LEASE_TTL
  export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
}

run_worker() {
  local index="${DYNAMO_CROSS_HOST_INDEX:-0}"
  local host_ip net_device gpu ordinal event_config log_role execution_config
  local -a mode_args scheduling_args event_args
  case "${ROLE}" in
    decode)
      host_ip="${LOCAL_IP}"
      net_device="${LOCAL_UCX_DEVICE}"
      gpu="${LOCAL_DECODE_GPU}"
      ordinal=0
      log_role=decode
      mode_args=(--disaggregation-mode decode)
      scheduling_args=(--async-scheduling)
      event_args=()
      execution_config="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${DECODE_CUDAGRAPH_CAPTURE_SIZES}]}"
      ;;
    prefill)
      (( index >= 0 && index < PREFILL_COUNT )) || die "invalid Prefill index"
      host_ip="${REMOTE_IP}"
      net_device="${REMOTE_UCX_DEVICE}"
      gpu="${index}"
      ordinal=$((index + 1))
      log_role=prefill
      mode_args=(--disaggregation-mode prefill)
      scheduling_args=(--no-async-scheduling)
      event_config="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((KV_EVENT_PORT_BASE + index))\",\"enable_kv_cache_events\":true}"
      event_args=(--kv-events-config "${event_config}")
      execution_config="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
      ;;
    *) die "unsupported worker role: ${ROLE}" ;;
  esac
  configure_worker_env "${host_ip}" "${net_device}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export CUDA_CACHE_PATH="${RUN_ROOT}/cuda_cache/${log_role}_${index}"
  export DYN_SYSTEM_PORT="$((SYSTEM_PORT_BASE + ordinal))"
  export VLLM_NIXL_SIDE_CHANNEL_PORT="$((NIXL_PORT_BASE + ordinal))"
  export VLLM_PORT="$((VLLM_PORT_BASE + ordinal * 20))"
  mkdir -p "${CUDA_CACHE_PATH}"
  local kv_config
  kv_config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"enable_cross_layers_blocks\":\"True\",\"kv_lease_duration\":${KV_LEASE_DURATION}}}"
  exec "${DYNAMO_PYTHON}" -P -m dynamo.vllm \
    --discovery-backend etcd --request-plane tcp --event-plane zmq \
    --model "${MODEL_PATH}" --served-model-name "${MODEL_PATH}" \
    --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
    "${mode_args[@]}" --kv-transfer-config "${kv_config}" \
    "${event_args[@]}" --compilation-config "${execution_config}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "$([[ "${ROLE}" == prefill ]] && printf %s "${PREFILL_MAX_NUM_BATCHED_TOKENS}" || printf %s "${DECODE_MAX_NUM_BATCHED_TOKENS}")" \
    --max-num-seqs "${MAX_NUM_SEQS}" --enable-chunked-prefill \
    --enable-prefix-caching --block-size "${BLOCK_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${scheduling_args[@]}" --disable-log-stats
}

if [[ "${ROLE}" != controller ]]; then
  run_worker
fi

[[ "${SMOKE_ONLY}" =~ ^[01]$ ]] || die "DYNAMO_SMOKE_ONLY must be 0 or 1"
[[ "${LOCAL_IP}" != "${REMOTE_IP}" ]] || die "local and remote IP must differ"
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "invalid Dynamo namespace"
for path in "${DYNAMO_PYTHON}" "${PAP_PYTHON}" "${ETCD_BIN}" "${MODEL_PATH}"; do
  [[ -e "${path}" ]] || die "missing local path: ${path}"
done
if (( SMOKE_ONLY == 0 )); then
  [[ -f "${AIPERF_INPUT_FILE}" ]] || die "missing AIPerf input dataset"
  actual_sha="$(sha256sum "${AIPERF_INPUT_FILE}" | awk '{print $1}')"
  [[ -z "${AIPERF_INPUT_SHA256}" || "${actual_sha}" == "${AIPERF_INPUT_SHA256}" ]] \
    || die "AIPerf input checksum mismatch"
fi

CROSS_HOST_NO_PROXY="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}"
if [[ -n "${NO_PROXY:-}" ]]; then
  CROSS_HOST_NO_PROXY="${NO_PROXY},${CROSS_HOST_NO_PROXY}"
fi
export NO_PROXY="${CROSS_HOST_NO_PROXY}"
export no_proxy="${CROSS_HOST_NO_PROXY}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}/launch" "${RUN_ROOT}/cuda_cache"
LOCAL_PIDS=()
LOCAL_PGIDS=()
REMOTE_PIDS=()
ETCD_PID=""

pull_remote_logs() {
  rsync -a "${REMOTE_HOST}:${REMOTE_LOG_ROOT}/" "${LOG_ROOT}/" >/dev/null 2>&1 || true
  rsync -a "${REMOTE_HOST}:${REMOTE_RUN_ROOT}/remote_gpu_telemetry.csv" \
    "${RUN_ROOT}/" >/dev/null 2>&1 || true
}

cleanup() {
  local code=$?
  set +e
  local pgid pid
  if ((${#REMOTE_PIDS[@]})); then
    for pid in "${REMOTE_PIDS[@]}"; do
      ssh "${REMOTE_HOST}" \
        "kill -TERM -- -${pid} >/dev/null 2>&1 || true" || true
    done
  fi
  for pgid in "${LOCAL_PGIDS[@]:-}"; do
    kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
  done
  sleep 7
  for pgid in "${LOCAL_PGIDS[@]:-}"; do
    kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
  done
  if ((${#REMOTE_PIDS[@]})); then
    for pid in "${REMOTE_PIDS[@]}"; do
      ssh "${REMOTE_HOST}" \
        "kill -KILL -- -${pid} >/dev/null 2>&1 || true" || true
    done
  fi
  if [[ -n "${ETCD_PID}" ]]; then
    kill -TERM -- "-${ETCD_PID}" >/dev/null 2>&1 || true
    sleep 1
    kill -KILL -- "-${ETCD_PID}" >/dev/null 2>&1 || true
  fi
  for pid in "${LOCAL_PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  pull_remote_logs
  local deadline=$((SECONDS + 30))
  local local_gpu_left remote_gpu_left etcd_left
  while true; do
    local_gpu_left="$(nvidia-smi -i "${LOCAL_DECODE_GPU}" \
      --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^$/d' | wc -l)"
    remote_gpu_left="$(ssh "${REMOTE_HOST}" \
      'nvidia-smi -i 0,1,2,3,4,5,6 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed "/^$/d" | wc -l' \
      2>/dev/null || echo unknown)"
    etcd_left="$(ss -ltn 2>/dev/null | grep -c ":${ETCD_PORT} " || true)"
    [[ "${local_gpu_left}" == 0 && "${remote_gpu_left}" == 0 \
      && "${etcd_left}" == 0 ]] && break
    (( SECONDS < deadline )) || break
    sleep 1
  done
  local cleanup_status=passed
  if [[ "${local_gpu_left}" != 0 || "${remote_gpu_left}" != 0 \
    || "${etcd_left}" != 0 ]]; then
    cleanup_status=failed
    (( code != 0 )) || code=2
  fi
  jq -n --arg status "${cleanup_status}" \
    --argjson local_gpu_processes "${local_gpu_left:-null}" \
    --arg remote_gpu_processes "${remote_gpu_left:-unknown}" \
    --argjson etcd_listeners "${etcd_left:-null}" \
    '{status:$status,local_gpu_processes:$local_gpu_processes,remote_gpu_processes:$remote_gpu_processes,etcd_listeners:$etcd_listeners}' \
    > "${RUN_ROOT}/cleanup_audit.json"
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

local_gpu_processes="$(nvidia-smi -i "${LOCAL_DECODE_GPU}" \
  --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${local_gpu_processes//[[:space:]]/}" ]] \
  || die "local Decode GPU is occupied"
remote_gpu_processes="$(ssh "${REMOTE_HOST}" \
  'nvidia-smi -i 0,1,2,3,4,5,6 --query-compute-apps=pid --format=csv,noheader,nounits')"
[[ -z "${remote_gpu_processes//[[:space:]]/}" ]] \
  || die "remote Prefill GPUs are occupied"
local_link_speed="$(cat "/sys/class/net/${LOCAL_UCX_DEVICE}/speed")"
remote_link_speed="$(ssh "${REMOTE_HOST}" \
  "cat '/sys/class/net/${REMOTE_UCX_DEVICE}/speed'")"
[[ "${local_link_speed}" == "${EXPECTED_LINK_SPEED_MBIT_S}" \
  && "${remote_link_speed}" == "${EXPECTED_LINK_SPEED_MBIT_S}" ]] \
  || die "cross-host link speed mismatch"

local_version="$(${DYNAMO_PYTHON} -P -c \
  "import importlib.metadata as m; print(m.version('ai-dynamo'), m.version('vllm'))")"
remote_version="$(ssh "${REMOTE_HOST}" "${REMOTE_ROOT}/.venv-dynamo/bin/python -P -c \"import importlib.metadata as m; print(m.version('ai-dynamo'), m.version('vllm'))\"")"
[[ "${local_version}" == "1.4.1 0.26.0" && "${remote_version}" == "${local_version}" ]] \
  || die "Dynamo/vLLM version mismatch: local=${local_version}, remote=${remote_version}"

rsync -a \
  "${ROOT_DIR}/benchmarks/pap/scripts/run_cross_host_dynamo_7p1d_aiperf.sh" \
  "${ROOT_DIR}/benchmarks/pap/scripts/configure_cross_host_nixl.sh" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/benchmarks/pap/scripts/"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}/launch'"

setsid "${ETCD_BIN}" --name pap-dynamo-cross \
  --data-dir "${RUN_ROOT}/etcd" \
  --listen-client-urls "http://127.0.0.1:${ETCD_PORT},${ETCD_ENDPOINTS}" \
  --advertise-client-urls "${ETCD_ENDPOINTS}" \
  --listen-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
  --initial-advertise-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
  --initial-cluster "pap-dynamo-cross=http://127.0.0.1:$((ETCD_PORT + 1))" \
  --initial-cluster-state new > "${LOG_ROOT}/etcd.log" 2>&1 &
ETCD_PID=$!
LOCAL_PIDS+=("${ETCD_PID}")

deadline=$((SECONDS + 60))
until curl -fsS "${ETCD_ENDPOINTS}/health" >/dev/null 2>&1; do
  kill -0 "${ETCD_PID}" 2>/dev/null || die "etcd exited during startup"
  (( SECONDS < deadline )) || die "timed out waiting for etcd"
  sleep 1
done

frontend_env=(
  PYTHONSAFEPATH=1 PYTHONPATH= DYN_NAMESPACE="${NAMESPACE}"
  DYN_DISCOVERY_BACKEND=etcd DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
  ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" ETCD_LEASE_TTL="${ETCD_LEASE_TTL}"
  NO_PROXY="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}"
  no_proxy="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
)
setsid env "${frontend_env[@]}" "${DYNAMO_PYTHON}" -P -m dynamo.frontend \
  --discovery-backend etcd --request-plane tcp --event-plane zmq \
  --router-mode kv --router-prefill-load-scale "${ROUTER_PREFILL_LOAD_SCALE}" \
  --router-min-initial-workers 1 --kv-cache-block-size "${BLOCK_SIZE}" \
  --http-port "${FRONTEND_PORT}" --model-name "${MODEL_PATH}" \
  --model-path "${MODEL_PATH}" --enforce-disagg \
  > "${LOG_ROOT}/frontend.log" 2>&1 &
LOCAL_PIDS+=("$!")
LOCAL_PGIDS+=("$!")

setsid env \
  DYNAMO_CROSS_HOST_ROLE=decode DYNAMO_CROSS_HOST_INDEX=0 \
  PAP_ROOT="${ROOT_DIR}" DYNAMO_RUN_ID="${RUN_ID}" \
  DYNAMO_RUN_ROOT="${RUN_ROOT}" DYNAMO_NAMESPACE="${NAMESPACE}" \
  DYNAMO_LOCAL_IP="${LOCAL_IP}" DYNAMO_REMOTE_IP="${REMOTE_IP}" \
  DYNAMO_LOCAL_UCX_DEVICE="${LOCAL_UCX_DEVICE}" \
  DYNAMO_REMOTE_UCX_DEVICE="${REMOTE_UCX_DEVICE}" \
  DYNAMO_ETCD_PORT="${ETCD_PORT}" DYNAMO_SYSTEM_PORT_BASE="${SYSTEM_PORT_BASE}" \
  DYNAMO_NIXL_PORT_BASE="${NIXL_PORT_BASE}" \
  DYNAMO_KV_EVENT_PORT_BASE="${KV_EVENT_PORT_BASE}" \
  DYNAMO_VLLM_PORT_BASE="${VLLM_PORT_BASE}" \
  DYNAMO_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
  DYNAMO_DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS}" \
  DYNAMO_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  DYNAMO_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  DYNAMO_BLOCK_SIZE="${BLOCK_SIZE}" \
  DYNAMO_ETCD_LEASE_TTL="${ETCD_LEASE_TTL}" \
  DYNAMO_KV_LEASE_DURATION="${KV_LEASE_DURATION}" \
  DYNAMO_PREFILL_CUDAGRAPH_CAPTURE_SIZES="${PREFILL_CUDAGRAPH_CAPTURE_SIZES}" \
  DYNAMO_DECODE_CUDAGRAPH_CAPTURE_SIZES="${DECODE_CUDAGRAPH_CAPTURE_SIZES}" \
  bash "$0" > "${LOG_ROOT}/decode_0.log" 2>&1 &
LOCAL_PIDS+=("$!")
LOCAL_PGIDS+=("$!")

for ((index = 0; index < PREFILL_COUNT; index++)); do
  launch="${RUN_ROOT}/launch/prefill_${index}.sh"
  cat > "${launch}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export DYNAMO_CROSS_HOST_ROLE=prefill DYNAMO_CROSS_HOST_INDEX=${index}
export PAP_ROOT=$(printf %q "${REMOTE_ROOT}")
export DYNAMO_RUN_ID=$(printf %q "${RUN_ID}")
export DYNAMO_RUN_ROOT=$(printf %q "${REMOTE_RUN_ROOT}")
export DYNAMO_NAMESPACE=$(printf %q "${NAMESPACE}")
export DYNAMO_LOCAL_IP=$(printf %q "${LOCAL_IP}") DYNAMO_REMOTE_IP=$(printf %q "${REMOTE_IP}")
export DYNAMO_LOCAL_UCX_DEVICE=$(printf %q "${LOCAL_UCX_DEVICE}") DYNAMO_REMOTE_UCX_DEVICE=$(printf %q "${REMOTE_UCX_DEVICE}")
export DYNAMO_ETCD_PORT=${ETCD_PORT} DYNAMO_SYSTEM_PORT_BASE=${SYSTEM_PORT_BASE}
export DYNAMO_NIXL_PORT_BASE=${NIXL_PORT_BASE} DYNAMO_KV_EVENT_PORT_BASE=${KV_EVENT_PORT_BASE} DYNAMO_VLLM_PORT_BASE=${VLLM_PORT_BASE}
export DYNAMO_MAX_MODEL_LEN=${MAX_MODEL_LEN} DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_NUM_BATCHED_TOKENS}
export DYNAMO_DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS} DYNAMO_MAX_NUM_SEQS=${MAX_NUM_SEQS}
export DYNAMO_GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} DYNAMO_BLOCK_SIZE=${BLOCK_SIZE}
export DYNAMO_ETCD_LEASE_TTL=${ETCD_LEASE_TTL} DYNAMO_KV_LEASE_DURATION=${KV_LEASE_DURATION}
export DYNAMO_PREFILL_CUDAGRAPH_CAPTURE_SIZES=$(printf %q "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}")
export DYNAMO_DECODE_CUDAGRAPH_CAPTURE_SIZES=$(printf %q "${DECODE_CUDAGRAPH_CAPTURE_SIZES}")
exec bash $(printf %q "${REMOTE_ROOT}/benchmarks/pap/scripts/run_cross_host_dynamo_7p1d_aiperf.sh")
EOF
  chmod +x "${launch}"
  rsync -a "${launch}" "${REMOTE_HOST}:${REMOTE_RUN_ROOT}/launch/"
  remote_pid="$(ssh "${REMOTE_HOST}" \
    "nohup setsid '${REMOTE_RUN_ROOT}/launch/prefill_${index}.sh' > '${REMOTE_LOG_ROOT}/prefill_${index}.log' 2>&1 < /dev/null & echo \$!")"
  [[ "${remote_pid}" =~ ^[0-9]+$ ]] || die "failed to start remote Prefill ${index}"
  REMOTE_PIDS+=("${remote_pid}")
done

deadline=$((SECONDS + 900))
while true; do
  local_ready="$(rg -c 'VllmWorker for .* has been initialized' "${LOG_ROOT}/decode_0.log" || true)"
  remote_ready="$(ssh "${REMOTE_HOST}" \
    "grep -l 'VllmWorker for .* has been initialized' '${REMOTE_LOG_ROOT}'/prefill_*.log 2>/dev/null | wc -l")"
  [[ "${local_ready}" == 1 && "${remote_ready}" == "${PREFILL_COUNT}" ]] && break
  kill -0 "${LOCAL_PIDS[2]}" 2>/dev/null || die "Decode exited during startup"
  (( SECONDS < deadline )) || die "timed out waiting for workers"
  sleep 2
done

deadline=$((SECONDS + 300))
registered=0
until (( registered >= 8 )) && curl -fsS \
  "http://127.0.0.1:${FRONTEND_PORT}/v1/models" > "${RUN_ROOT}/models.json"; do
  registered="$(rg -c 'added model' "${LOG_ROOT}/frontend.log" || true)"
  (( SECONDS < deadline )) || die "timed out waiting for 8 Dynamo workers"
  sleep 1
done
sleep "${CLUSTER_READY_WAIT_SECONDS}"

{
  printf 'TOPOLOGY=7P1D\nPLACEMENT=cross_host\n'
  printf 'LOCAL_HOST=%q\nLOCAL_IP=%q\nLOCAL_DECODE_GPU=%q\n' "$(hostname)" "${LOCAL_IP}" "${LOCAL_DECODE_GPU}"
  printf 'REMOTE_HOST=%q\nREMOTE_IP=%q\nREMOTE_PREFILL_GPUS=0,1,2,3,4,5,6\n' "${REMOTE_HOST}" "${REMOTE_IP}"
  printf 'DYNAMO_VERSION=%q\nVLLM_VERSION=%q\n' "${local_version%% *}" "${local_version##* }"
  printf 'NAMESPACE=%q\nETCD_ENDPOINTS=%q\nROUTER_MODE=kv\n' "${NAMESPACE}" "${ETCD_ENDPOINTS}"
  printf 'ETCD_LEASE_TTL=%q\nKV_LEASE_DURATION=%q\n' \
    "${ETCD_LEASE_TTL}" "${KV_LEASE_DURATION}"
  printf 'UCX_TLS=tcp,cuda_copy\nLOCAL_UCX_DEVICE=%q\nREMOTE_UCX_DEVICE=%q\n' "${LOCAL_UCX_DEVICE}" "${REMOTE_UCX_DEVICE}"
  printf 'LOCAL_LINK_SPEED_MBIT_S=%q\nREMOTE_LINK_SPEED_MBIT_S=%q\n' "${local_link_speed}" "${remote_link_speed}"
  printf 'MODEL_PATH=%q\nMAX_MODEL_LEN=%q\nBLOCK_SIZE=%q\n' "${MODEL_PATH}" "${MAX_MODEL_LEN}" "${BLOCK_SIZE}"
  printf 'PREFILL_MAX_NUM_BATCHED_TOKENS=%q\nDECODE_MAX_NUM_BATCHED_TOKENS=%q\n' "${PREFILL_MAX_NUM_BATCHED_TOKENS}" "${DECODE_MAX_NUM_BATCHED_TOKENS}"
  printf 'MIN_KV_TRANSFER_MB_S=%q\n' "${MIN_KV_TRANSFER_MB_S}"
  printf 'AIPERF_INPUT_FILE=%q\nAIPERF_INPUT_SHA256=%q\n' "${AIPERF_INPUT_FILE}" "${AIPERF_INPUT_SHA256:-${actual_sha:-not_applicable}}"
  printf 'AIPERF_SESSIONS=%q\nAIPERF_CONCURRENCY=%q\nAIPERF_EXPECTED_REQUESTS=%q\n' "${AIPERF_SESSIONS}" "${AIPERF_CONCURRENCY}" "${AIPERF_EXPECTED_REQUESTS}"
  printf 'DYNAMO_RUN_ID=%q\nDYNAMO_RUN_ROOT=%q\nDYNAMO_REMOTE_RUN_ROOT=%q\n' "${RUN_ID}" "${RUN_ROOT}" "${REMOTE_RUN_ROOT}"
  printf 'GIT_COMMIT=%q\n' "$(git rev-parse HEAD)"
} > "${RUN_ROOT}/effective_config.env"

local_compute="$(nvidia-smi -i "${LOCAL_DECODE_GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | sed '/^$/d' | wc -l)"
remote_compute="$(ssh "${REMOTE_HOST}" \
  'for gpu in 0 1 2 3 4 5 6; do nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed "/^$/d" | wc -l; done')"
[[ "${local_compute}" == 1 ]] || die "expected one Decode GPU process"
[[ "${remote_compute}" == $'1\n1\n1\n1\n1\n1\n1' ]] \
  || die "expected one Prefill GPU process per remote GPU"
jq -n --argjson local_compute "${local_compute}" \
  --arg remote_compute "${remote_compute}" --argjson registered "${registered}" \
  --argjson local_pid "${LOCAL_PIDS[2]}" --arg remote_pids "${REMOTE_PIDS[*]}" \
  '{status:"passed",registered_workers:$registered,local_decode:{launcher_pid:$local_pid,gpu_processes:$local_compute},remote_prefills:{launcher_pids:($remote_pids|split(" ")|map(tonumber)),gpu_processes_per_gpu:($remote_compute|split("\n")|map(tonumber))}}' \
  > "${RUN_ROOT}/startup_audit.json"

if (( SMOKE_ONLY == 1 )); then
  jq -n --arg model "${MODEL_PATH}" \
    '{model:$model,messages:[{role:"user",content:"Say hello."}],max_tokens:8,temperature:0}' \
    | curl -fsS -H 'Content-Type: application/json' --data-binary @- \
      "http://127.0.0.1:${FRONTEND_PORT}/v1/chat/completions" \
      > "${RUN_ROOT}/smoke_response.json"
  jq -e '.choices[0].message.content | length > 0' \
    "${RUN_ROOT}/smoke_response.json" >/dev/null || die "smoke returned no content"
else
  telemetry_launch="${RUN_ROOT}/launch/remote_telemetry.sh"
  cat > "${telemetry_launch}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits
  sleep 1
done
EOF
  chmod +x "${telemetry_launch}"
  rsync -a "${telemetry_launch}" "${REMOTE_HOST}:${REMOTE_RUN_ROOT}/launch/"
  ssh "${REMOTE_HOST}" \
    "nohup setsid '${REMOTE_RUN_ROOT}/launch/remote_telemetry.sh' > '${REMOTE_RUN_ROOT}/remote_gpu_telemetry.csv' 2>&1 < /dev/null & echo \$!" \
    > "${RUN_ROOT}/remote_telemetry.pid"
  telemetry_pid="$(cat "${RUN_ROOT}/remote_telemetry.pid")"
  REMOTE_PIDS+=("${telemetry_pid}")
  env PAP_ROOT="${ROOT_DIR}" AIPERF_BIN="${AIPERF_BIN}" \
    MODEL_PATH="${MODEL_PATH}" AIPERF_INPUT_FILE="${AIPERF_INPUT_FILE}" \
    AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
    AIPERF_TARGET_URL="http://127.0.0.1:${FRONTEND_PORT}" \
    AIPERF_OUTPUT_DIR="${AIPERF_OUTPUT_DIR}" \
    AIPERF_SESSIONS="${AIPERF_SESSIONS}" \
    AIPERF_CONCURRENCY="${AIPERF_CONCURRENCY}" \
    AIPERF_TIMING_MODE=concurrency AIPERF_REQUEST_RATE= \
    AIPERF_ARRIVAL_PATTERN=poisson \
    AIPERF_EXPECTED_REQUESTS="${AIPERF_EXPECTED_REQUESTS}" \
    AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_REQUEST_TIMEOUT_SECONDS}" \
    AIPERF_WARMUP_DURATION_SECONDS=0 AIPERF_WARMUP_CONCURRENCY="${AIPERF_CONCURRENCY}" \
    AIPERF_BENCHMARK_DURATION_SECONDS= AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS=0 \
    NO_PROXY="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}" \
    no_proxy="localhost,127.0.0.1,${LOCAL_IP},${REMOTE_IP}" \
    "${AIPERF_RUNNER}" > "${RUN_ROOT}/aiperf.log" 2>&1
  jq -e '(.error_summary // []) | length == 0' \
    "${AIPERF_OUTPUT_DIR}/profile.json" >/dev/null || die "AIPerf reported errors"
  actual_requests="$(jq -r '.request_count.avg' "${AIPERF_OUTPUT_DIR}/profile.json")"
  [[ "${actual_requests}" == "${AIPERF_EXPECTED_REQUESTS}" \
    || "${actual_requests}" == "${AIPERF_EXPECTED_REQUESTS}.0" ]] \
    || die "AIPerf request count mismatch"
fi

pull_remote_logs
if rg --no-ignore -n -i \
  'CUDA out of memory|EngineDeadError|Traceback|NIXL_ERR|NixlConnector.*failed|Keep-alive lease expired|Unable to maintain lease|Releasing expired KV blocks' \
  "${LOG_ROOT}" > "${RUN_ROOT}/correctness_audit_matches.log"; then
  die "Dynamo correctness audit failed"
fi
printf 'STATUS=passed\nMATCH_COUNT=0\n' > "${RUN_ROOT}/correctness_audit.env"

if (( SMOKE_ONLY == 0 )); then
  "${PAP_PYTHON}" "${ROOT_DIR}/benchmarks/pap/tooling/analyze_dynamo_ttft.py" \
    "${RUN_ROOT}" --block-size "${BLOCK_SIZE}" \
    --output "${RUN_ROOT}/dynamo_ttft_analysis.json"
  jq -e --argjson expected "${AIPERF_EXPECTED_REQUESTS}" \
    --argjson sessions "${AIPERF_SESSIONS}" \
    '.joined_requests == $expected and .request_metadata_matches == $expected and .session_routing.complete_conversations == $sessions' \
    "${RUN_ROOT}/dynamo_ttft_analysis.json" >/dev/null \
    || die "Dynamo request/session analysis is incomplete"
  observed="$(jq -r '.kv_transfer.throughput_mb_s.p90 // 0' \
    "${RUN_ROOT}/dynamo_ttft_analysis.json")"
  awk -v observed="${observed}" -v minimum="${MIN_KV_TRANSFER_MB_S}" \
    'BEGIN {exit !(observed >= minimum)}' || die "cross-host KV throughput audit failed"
  printf 'STATUS=passed\nMINIMUM_MB_S=%s\nOBSERVED_MB_S=%s\n' \
    "${MIN_KV_TRANSFER_MB_S}" "${observed}" > "${RUN_ROOT}/kv_transfer_audit.env"
fi

printf 'STATUS=passed\n' > "${RUN_ROOT}/completion_audit.env"
echo "DYNAMO_RUN_ROOT=${RUN_ROOT}"
