#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# Official Dynamo + upstream vLLM Prefill/Decode baseline.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
pap_configure_same_node_nixl "${ROOT_DIR}"
DYNAMO_PYTHON="${DYNAMO_PYTHON:-${ROOT_DIR}/.venv-dynamo/bin/python}"
PAP_PYTHON="${PAP_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
CUDA_GRAPH_AUDITOR="${ROOT_DIR}/benchmarks/pap/scripts/audit_cuda_graph_logs.sh"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"

TOPOLOGY="${DYNAMO_PD_TOPOLOGY:-6p2d}"
ROUTER_MODE="${DYNAMO_PD_ROUTER_MODE:-kv}"
DISCOVERY_BACKEND="${DYNAMO_PD_DISCOVERY_BACKEND:-etcd}"
START_ETCD="${DYNAMO_PD_START_ETCD:-1}"
SMOKE_ONLY="${DYNAMO_PD_SMOKE_ONLY:-0}"
MAX_MODEL_LEN="${DYNAMO_PD_MAX_MODEL_LEN:-32768}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${DYNAMO_PD_PREFILL_MAX_NUM_BATCHED_TOKENS:-2048}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DYNAMO_PD_DECODE_MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${DYNAMO_PD_MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${DYNAMO_PD_GPU_MEMORY_UTILIZATION:-0.90}"
BLOCK_SIZE="${DYNAMO_PD_BLOCK_SIZE:-16}"
MIN_KV_TRANSFER_MB_S="${DYNAMO_PD_MIN_KV_TRANSFER_MB_S:-5000}"
PREFILL_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_PD_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
DECODE_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_PD_DECODE_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,20,24,28,32}"

RUN_ID="${DYNAMO_PD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_dynamo_${TOPOLOGY}_${ROUTER_MODE}}"
RUN_ROOT="${DYNAMO_PD_RUN_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/_staging/dynamo/${RUN_ID}}"
LOG_ROOT="${RUN_ROOT}/service_logs"
FRONTEND_PORT="${DYNAMO_PD_FRONTEND_PORT:-18400}"
SYSTEM_PORT_BASE="${DYNAMO_PD_SYSTEM_PORT_BASE:-18500}"
NIXL_PORT_BASE="${DYNAMO_PD_NIXL_PORT_BASE:-18600}"
KV_EVENT_PORT_BASE="${DYNAMO_PD_KV_EVENT_PORT_BASE:-18700}"
VLLM_PORT_BASE="${DYNAMO_PD_VLLM_PORT_BASE:-18800}"
ETCD_PORT="${DYNAMO_PD_ETCD_PORT:-22379}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:${ETCD_PORT}}"
ETCD_BIN="${DYNAMO_PD_ETCD_BIN:-${ROOT_DIR}/.local/dynamo-etcd-3.6.1/etcd}"
FILE_KV="${DYNAMO_PD_FILE_KV:-${RUN_ROOT}/discovery}"
NAMESPACE="${DYNAMO_PD_NAMESPACE:-pap-dynamo-${RUN_ID}}"

AIPERF_INPUT_FILE="${DYNAMO_PD_AIPERF_INPUT_FILE:-}"
AIPERF_OUTPUT_DIR="${DYNAMO_PD_AIPERF_OUTPUT_DIR:-${RUN_ROOT}/aiperf}"
AIPERF_SESSIONS="${DYNAMO_PD_AIPERF_SESSIONS:-128}"
AIPERF_CONCURRENCY="${DYNAMO_PD_AIPERF_CONCURRENCY:-32}"
AIPERF_EXPECTED_REQUESTS="${DYNAMO_PD_AIPERF_EXPECTED_REQUESTS:-}"
AIPERF_REQUEST_TIMEOUT_SECONDS="${DYNAMO_PD_AIPERF_REQUEST_TIMEOUT_SECONDS:-1200}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${TOPOLOGY}" =~ ^([1-9][0-9]*)p([1-9][0-9]*)d$ ]] \
  || die "invalid Dynamo PD topology: ${TOPOLOGY}"
PREFILL_COUNT="${BASH_REMATCH[1]}"
DECODE_COUNT="${BASH_REMATCH[2]}"
GPU_COUNT=$((PREFILL_COUNT + DECODE_COUNT))
(( GPU_COUNT <= 8 )) || die "Dynamo topology requires more than eight GPUs"
case "${ROUTER_MODE}" in
  round-robin | kv | least-loaded) ;;
  *) die "unsupported Dynamo router mode: ${ROUTER_MODE}" ;;
esac
case "${DISCOVERY_BACKEND}" in
  file)
    [[ "${ROUTER_MODE}" == "round-robin" ]] \
      || die "file discovery is only valid for the round-robin smoke"
    ;;
  etcd) ;;
  *) die "unsupported Dynamo discovery backend: ${DISCOVERY_BACKEND}" ;;
esac
for value in "${START_ETCD}" "${SMOKE_ONLY}"; do
  [[ "${value}" =~ ^[01]$ ]] || die "boolean controls must be 0 or 1"
done
for capture_sizes in \
  "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}" \
  "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"; do
  [[ "${capture_sizes}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
    || die "CUDA Graph capture sizes must be positive integer CSV"
done
for required in "${DYNAMO_PYTHON}" "${PAP_PYTHON}" "${AIPERF_BIN}" \
  "${AIPERF_RUNNER}" "${CUDA_GRAPH_AUDITOR}" "${MODEL_PATH}"; do
  [[ -e "${required}" ]] || die "missing required path: ${required}"
done
if (( SMOKE_ONLY == 0 )); then
  [[ -f "${AIPERF_INPUT_FILE}" ]] \
    || die "set DYNAMO_PD_AIPERF_INPUT_FILE to a valid dataset"
fi

HOST_IP="${DYNAMO_PD_HOST_IP:-$(hostname -I | awk '{print $1}')}"
[[ -n "${HOST_IP}" ]] || die "failed to resolve the local host IP"
LOCAL_NO_PROXY="localhost,127.0.0.1,${HOST_IP}"
if [[ -n "${NO_PROXY:-}" ]]; then
  LOCAL_NO_PROXY="${NO_PROXY},${LOCAL_NO_PROXY}"
fi
export NO_PROXY="${LOCAL_NO_PROXY}"
export no_proxy="${LOCAL_NO_PROXY}"

PIDS=()
PGIDS=()
ETCD_PID=""
ETCD_PGID=""
cleanup() {
  local code=$?
  set +e
  local pgid pid
  for pgid in "${PGIDS[@]:-}"; do
    kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
  done
  sleep 2
  for pgid in "${PGIDS[@]:-}"; do
    kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${ETCD_PGID}" ]]; then
    kill -TERM -- "-${ETCD_PGID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${ETCD_PID}" ]]; then
    wait "${ETCD_PID}" >/dev/null 2>&1 || true
  fi
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ensure_gpus_idle() {
  local gpu processes
  for (( gpu=0; gpu<GPU_COUNT; gpu++ )); do
    processes="$(
      nvidia-smi -i "${gpu}" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    [[ -z "${processes//[[:space:]]/}" ]] \
      || die "GPU ${gpu} is occupied by ${processes//$'\n'/,}"
  done
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local deadline=$((SECONDS + 900))
  until curl -fsS "${url}" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || die "timed out waiting for ${label}: ${url}"
    sleep 2
  done
  echo "${label} is ready at ${url}"
}

wait_for_worker() {
  local pid="$1"
  local log="$2"
  local label="$3"
  local deadline=$((SECONDS + 900))
  until rg -q "VllmWorker for .* has been initialized" "${log}" 2>/dev/null; do
    kill -0 "${pid}" 2>/dev/null \
      || die "${label} exited before initialization; see ${log}"
    (( SECONDS < deadline )) || die "timed out waiting for ${label}"
    sleep 2
  done
  echo "${label} initialized"
}

wait_for_frontend_registration() {
  local url="http://127.0.0.1:${FRONTEND_PORT}/v1/models"
  local output="${RUN_ROOT}/models.json"
  local frontend_log="${LOG_ROOT}/frontend.log"
  local deadline=$((SECONDS + 900))
  local registered=0
  until (( registered >= GPU_COUNT )) \
    && curl -fsS "${url}" > "${output}" \
    && jq -e --arg model "${MODEL_PATH}" \
      '.data | any(.id == $model)' "${output}" >/dev/null; do
    registered="$(rg -c 'added model' "${frontend_log}" || true)"
    (( SECONDS < deadline )) \
      || die "timed out waiting for all Dynamo workers to register"
    sleep 1
  done
  echo "Dynamo frontend registered ${registered}/${GPU_COUNT} workers"
}

start_etcd() {
  (( START_ETCD == 1 )) || {
    wait_for_http "${ETCD_ENDPOINTS}/health" "external etcd"
    return
  }
  [[ -x "${ETCD_BIN}" ]] || die "missing etcd executable: ${ETCD_BIN}"
  local peer_port=$((ETCD_PORT + 1))
  setsid "${ETCD_BIN}" \
    --name pap-dynamo \
    --data-dir "${RUN_ROOT}/etcd" \
    --listen-client-urls "http://127.0.0.1:${ETCD_PORT}" \
    --advertise-client-urls "http://127.0.0.1:${ETCD_PORT}" \
    --listen-peer-urls "http://127.0.0.1:${peer_port}" \
    --initial-advertise-peer-urls "http://127.0.0.1:${peer_port}" \
    --initial-cluster "pap-dynamo=http://127.0.0.1:${peer_port}" \
    --initial-cluster-state new \
    > "${LOG_ROOT}/etcd.log" 2>&1 &
  ETCD_PID="$!"
  ETCD_PGID="${ETCD_PID}"
  wait_for_http "${ETCD_ENDPOINTS}/health" "Dynamo etcd"
}

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}/cuda_cache"
ensure_gpus_idle
if [[ "${DISCOVERY_BACKEND}" == "etcd" ]]; then
  start_etcd
fi
unset PROMETHEUS_MULTIPROC_DIR

"${DYNAMO_PYTHON}" -P -c \
  "import importlib.metadata as m; assert m.version('ai-dynamo') == '1.4.1'; assert m.version('vllm') == '0.26.0'"

git status --short > "${RUN_ROOT}/git_status.txt"
git diff --binary > "${RUN_ROOT}/tracked_worktree.patch"
git diff --cached --binary > "${RUN_ROOT}/tracked_index.patch"
"${DYNAMO_PYTHON}" -P -m pip freeze > "${RUN_ROOT}/python_packages.txt"
{
  printf 'MODE=dynamo_pd\nTOPOLOGY=%q\n' "${TOPOLOGY}"
  printf 'ROUTER_MODE=%q\nDISCOVERY_BACKEND=%q\n' \
    "${ROUTER_MODE}" "${DISCOVERY_BACKEND}"
  printf 'DYNAMO_VERSION=1.4.1\nVLLM_VERSION=0.26.0\n'
  printf 'EXECUTION_MODE=piecewise_cuda_graph\n'
  printf 'PREFILL_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'DECODE_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'MODEL_PATH=%q\nMAX_MODEL_LEN=%q\n' \
    "${MODEL_PATH}" "${MAX_MODEL_LEN}"
  printf 'PREFILL_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${PREFILL_MAX_NUM_BATCHED_TOKENS}"
  printf 'DECODE_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${DECODE_MAX_NUM_BATCHED_TOKENS}"
  printf 'MAX_NUM_SEQS=%q\nBLOCK_SIZE=%q\n' \
    "${MAX_NUM_SEQS}" "${BLOCK_SIZE}"
  printf 'GPU_MEMORY_UTILIZATION=%q\nFRONTEND_PORT=%q\n' \
    "${GPU_MEMORY_UTILIZATION}" "${FRONTEND_PORT}"
  printf 'ETCD_BIN=%q\nETCD_ENDPOINTS=%q\n' \
    "${ETCD_BIN}" "${ETCD_ENDPOINTS}"
  printf 'PAP_NIXL_RUNTIME_MODE=%q\nPAP_NIXL_UCX_VERSION=%q\n' \
    "${PAP_NIXL_RUNTIME_MODE}" "${PAP_NIXL_UCX_VERSION}"
  printf 'NIXL_PLUGIN_DIR=%q\nUCX_PROTO_EMULATION_ENABLE=%q\n' \
    "${NIXL_PLUGIN_DIR}" "${UCX_PROTO_EMULATION_ENABLE}"
  printf 'UCX_CUDA_IPC_ENABLE_GET_ZCOPY=%q\n' \
    "${UCX_CUDA_IPC_ENABLE_GET_ZCOPY}"
  printf 'MIN_KV_TRANSFER_MB_S=%q\n' "${MIN_KV_TRANSFER_MB_S}"
  printf 'HOST_IP=%q\nNAMESPACE=%q\n' "${HOST_IP}" "${NAMESPACE}"
  printf 'AIPERF_BIN=%q\nAIPERF_INPUT_FILE=%q\n' \
    "${AIPERF_BIN}" "${AIPERF_INPUT_FILE}"
  printf 'AIPERF_SESSIONS=%q\nAIPERF_CONCURRENCY=%q\n' \
    "${AIPERF_SESSIONS}" "${AIPERF_CONCURRENCY}"
  printf 'GIT_COMMIT=%q\n' "$(git rev-parse HEAD)"
} > "${RUN_ROOT}/effective_config.env"

COMMON_ENV=(
  PYTHONSAFEPATH=1
  PYTHONPATH=
  NO_PROXY="${LOCAL_NO_PROXY}"
  no_proxy="${LOCAL_NO_PROXY}"
  DYN_NAMESPACE="${NAMESPACE}"
  DYN_DISCOVERY_BACKEND="${DISCOVERY_BACKEND}"
  DYN_FILE_KV="${FILE_KV}"
  ETCD_ENDPOINTS="${ETCD_ENDPOINTS}"
  DYN_REQUEST_PLANE=tcp
  DYN_EVENT_PLANE=zmq
  NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR}"
  UCX_MODULE_DIR="${UCX_MODULE_DIR}"
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
  UCX_PROTO_EMULATION_ENABLE="${UCX_PROTO_EMULATION_ENABLE}"
  UCX_CUDA_IPC_ENABLE_GET_ZCOPY="${UCX_CUDA_IPC_ENABLE_GET_ZCOPY}"
  UCX_TLS="${UCX_TLS}"
  UCX_NET_DEVICES="${UCX_NET_DEVICES}"
  UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED}"
  VLLM_USE_FLASHINFER_SAMPLER=0
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
)
PREFILL_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
DECODE_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${DECODE_CUDAGRAPH_CAPTURE_SIZES}]}"
PREFILL_EXECUTION_ARGS=(
  --compilation-config "${PREFILL_COMPILATION_CONFIG}"
)
DECODE_EXECUTION_ARGS=(
  --compilation-config "${DECODE_COMPILATION_CONFIG}"
)

setsid env "${COMMON_ENV[@]}" \
  DYN_HTTP_PORT="${FRONTEND_PORT}" \
  "${DYNAMO_PYTHON}" -P -m dynamo.frontend \
    --discovery-backend "${DISCOVERY_BACKEND}" \
    --request-plane tcp --event-plane zmq \
    --router-mode "${ROUTER_MODE}" \
    --router-min-initial-workers "${DECODE_COUNT}" \
    --kv-cache-block-size "${BLOCK_SIZE}" \
    --enforce-disagg \
    --http-port "${FRONTEND_PORT}" \
    --model-name "${MODEL_PATH}" --model-path "${MODEL_PATH}" \
    > "${LOG_ROOT}/frontend.log" 2>&1 &
PIDS+=("$!")
PGIDS+=("$!")

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"enable_cross_layers_blocks":"True"}}'
for (( index=0; index<DECODE_COUNT; index++ )); do
  gpu=$((PREFILL_COUNT + index))
  ordinal="${index}"
  log="${LOG_ROOT}/decode_${index}.log"
  setsid env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CUDA_CACHE_PATH="${RUN_ROOT}/cuda_cache/gpu${gpu}" \
    DYN_SYSTEM_PORT="$((SYSTEM_PORT_BASE + ordinal))" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$((NIXL_PORT_BASE + ordinal))" \
    VLLM_PORT="$((VLLM_PORT_BASE + ordinal * 20))" \
    "${DYNAMO_PYTHON}" -P -m dynamo.vllm \
      --discovery-backend "${DISCOVERY_BACKEND}" \
      --request-plane tcp --event-plane zmq \
      --model "${MODEL_PATH}" --served-model-name "${MODEL_PATH}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --disaggregation-mode decode --kv-transfer-config "${KV_CONFIG}" \
      "${DECODE_EXECUTION_ARGS[@]}" --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size "${BLOCK_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --disable-log-stats > "${log}" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
done

for (( index=0; index<PREFILL_COUNT; index++ )); do
  gpu="${index}"
  ordinal=$((DECODE_COUNT + index))
  event_port=$((KV_EVENT_PORT_BASE + index))
  event_config="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${event_port}\",\"enable_kv_cache_events\":true}"
  log="${LOG_ROOT}/prefill_${index}.log"
  setsid env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CUDA_CACHE_PATH="${RUN_ROOT}/cuda_cache/gpu${gpu}" \
    DYN_SYSTEM_PORT="$((SYSTEM_PORT_BASE + ordinal))" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$((NIXL_PORT_BASE + ordinal))" \
    VLLM_PORT="$((VLLM_PORT_BASE + ordinal * 20))" \
    "${DYNAMO_PYTHON}" -P -m dynamo.vllm \
      --discovery-backend "${DISCOVERY_BACKEND}" \
      --request-plane tcp --event-plane zmq \
      --model "${MODEL_PATH}" --served-model-name "${MODEL_PATH}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --disaggregation-mode prefill --kv-transfer-config "${KV_CONFIG}" \
      --kv-events-config "${event_config}" \
      "${PREFILL_EXECUTION_ARGS[@]}" --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size "${BLOCK_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --disable-log-stats > "${log}" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
done

for (( index=0; index<DECODE_COUNT; index++ )); do
  wait_for_worker "${PIDS[index + 1]}" \
    "${LOG_ROOT}/decode_${index}.log" "Dynamo Decode ${index}"
done
for (( index=0; index<PREFILL_COUNT; index++ )); do
  wait_for_worker "${PIDS[DECODE_COUNT + index + 1]}" \
    "${LOG_ROOT}/prefill_${index}.log" "Dynamo Prefill ${index}"
done

wait_for_http "http://127.0.0.1:${FRONTEND_PORT}/v1/models" \
  "Dynamo frontend HTTP service"
wait_for_frontend_registration

DYNAMO_VLLM_GRAPH_LOGS=()
for (( index=0; index<DECODE_COUNT; index++ )); do
  DYNAMO_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/decode_${index}.log")
done
for (( index=0; index<PREFILL_COUNT; index++ )); do
  DYNAMO_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/prefill_${index}.log")
done
"${CUDA_GRAPH_AUDITOR}" "${RUN_ROOT}/vllm_cuda_graph_audit.env" \
  PIECEWISE "${DYNAMO_VLLM_GRAPH_LOGS[@]}"

if (( SMOKE_ONLY == 1 )); then
  jq -n --arg model "${MODEL_PATH}" \
    '{model: $model, messages: [{role: "user", content: "Say hello from Dynamo."}], max_tokens: 8, temperature: 0}' \
    | curl -fsS -H 'Content-Type: application/json' \
      --data-binary @- \
      "http://127.0.0.1:${FRONTEND_PORT}/v1/chat/completions" \
      > "${RUN_ROOT}/smoke_response.json"
  jq -e '.choices[0].message.content | length > 0' \
    "${RUN_ROOT}/smoke_response.json" >/dev/null \
    || die "Dynamo chat smoke returned no content"
else
  env \
    PAP_ROOT="${ROOT_DIR}" \
    AIPERF_ROOT="${AIPERF_ROOT}" \
    AIPERF_BIN="${AIPERF_BIN}" \
    MODEL_PATH="${MODEL_PATH}" \
    AIPERF_INPUT_FILE="${AIPERF_INPUT_FILE}" \
    AIPERF_CUSTOM_DATASET_TYPE=multi-turn \
    AIPERF_TARGET_URL="http://127.0.0.1:${FRONTEND_PORT}" \
    AIPERF_OUTPUT_DIR="${AIPERF_OUTPUT_DIR}" \
    AIPERF_SESSIONS="${AIPERF_SESSIONS}" \
    AIPERF_CONCURRENCY="${AIPERF_CONCURRENCY}" \
    AIPERF_TIMING_MODE=concurrency \
    AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_REQUEST_TIMEOUT_SECONDS}" \
    NO_PROXY="${LOCAL_NO_PROXY}" no_proxy="${LOCAL_NO_PROXY}" \
    "${AIPERF_RUNNER}" 2>&1 | tee "${RUN_ROOT}/aiperf.log"
  [[ -s "${AIPERF_OUTPUT_DIR}/profile.json" ]] \
    || die "AIPerf produced no profile.json"
  jq -e '(.error_summary // []) | length == 0' \
    "${AIPERF_OUTPUT_DIR}/profile.json" >/dev/null \
    || die "AIPerf reported request errors"
  if [[ -n "${AIPERF_EXPECTED_REQUESTS}" ]]; then
    actual_requests="$(jq -r '.request_count.avg' \
      "${AIPERF_OUTPUT_DIR}/profile.json")"
    [[ "${actual_requests}" == "${AIPERF_EXPECTED_REQUESTS}" \
      || "${actual_requests}" == "${AIPERF_EXPECTED_REQUESTS}.0" ]] \
      || die "AIPerf completed ${actual_requests}, expected ${AIPERF_EXPECTED_REQUESTS}"
  fi
fi

if rg -n -i \
  'CUDA out of memory|EngineDeadError|Traceback|NIXL_ERR|NixlConnector.*failed' \
  "${LOG_ROOT}" > "${RUN_ROOT}/correctness_audit_matches.log"; then
  printf 'STATUS=failed\n' > "${RUN_ROOT}/correctness_audit.env"
  die "Dynamo correctness audit failed"
fi
: > "${RUN_ROOT}/correctness_audit_matches.log"
printf 'STATUS=passed\nMATCH_COUNT=0\n' \
  > "${RUN_ROOT}/correctness_audit.env"
if (( SMOKE_ONLY == 0 )); then
  "${PAP_PYTHON}" \
    "${ROOT_DIR}/benchmarks/pap/tooling/analyze_dynamo_ttft.py" \
    "${RUN_ROOT}" --block-size "${BLOCK_SIZE}" \
    --output "${RUN_ROOT}/dynamo_ttft_analysis.json"
  if ! jq -e --argjson minimum "${MIN_KV_TRANSFER_MB_S}" \
    '.kv_transfer.aggregate_throughput_mb_s >= $minimum' \
    "${RUN_ROOT}/dynamo_ttft_analysis.json" >/dev/null; then
    printf 'STATUS=failed\nMINIMUM_MB_S=%q\n' \
      "${MIN_KV_TRANSFER_MB_S}" > "${RUN_ROOT}/kv_transfer_audit.env"
    die "Dynamo KV transfer throughput failed the same-node floor"
  fi
  printf 'STATUS=passed\nMINIMUM_MB_S=%q\n' \
    "${MIN_KV_TRANSFER_MB_S}" > "${RUN_ROOT}/kv_transfer_audit.env"
fi
echo "DYNAMO_PD_RUN_ROOT=${RUN_ROOT}"
