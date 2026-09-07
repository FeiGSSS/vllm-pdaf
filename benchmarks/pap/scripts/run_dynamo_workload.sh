#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# Unified official Dynamo + upstream vLLM DP/PD baselines.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DYNAMO_PYTHON="${DYNAMO_PYTHON:-${ROOT_DIR}/.venv-dynamo/bin/python}"
PAP_PYTHON="${PAP_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_aiperf_profile.sh"
CUDA_GRAPH_AUDITOR="${ROOT_DIR}/benchmarks/pap/scripts/audit_cuda_graph_logs.sh"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"

ARCHITECTURE="${DYNAMO_ARCHITECTURE:-dp8}"
ROUTER_MODE="${DYNAMO_ROUTER_MODE:-kv}"
ROUTER_PREFILL_LOAD_SCALE="${DYNAMO_ROUTER_PREFILL_LOAD_SCALE:-1.0}"
DISCOVERY_BACKEND="${DYNAMO_DISCOVERY_BACKEND:-etcd}"
START_ETCD="${DYNAMO_START_ETCD:-1}"
SMOKE_ONLY="${DYNAMO_SMOKE_ONLY:-0}"
MAX_MODEL_LEN="${DYNAMO_MAX_MODEL_LEN:-131072}"
DEFAULT_HF_OVERRIDES='{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
if [[ -z "${DYNAMO_HF_OVERRIDES+x}" ]]; then
  HF_OVERRIDES="${DEFAULT_HF_OVERRIDES}"
else
  HF_OVERRIDES="${DYNAMO_HF_OVERRIDES}"
fi
AGG_MAX_NUM_BATCHED_TOKENS="${DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS:-32768}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS:-2048}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DYNAMO_DECODE_MAX_NUM_BATCHED_TOKENS:-2048}"
AGG_ASYNC_SCHEDULING="${DYNAMO_AGG_ASYNC_SCHEDULING:-auto}"
PREFILL_ASYNC_SCHEDULING="${DYNAMO_PREFILL_ASYNC_SCHEDULING:-auto}"
DECODE_ASYNC_SCHEDULING="${DYNAMO_DECODE_ASYNC_SCHEDULING:-auto}"
CLUSTER_READY_WAIT_SECONDS="${DYNAMO_CLUSTER_READY_WAIT_SECONDS:-30}"
USE_V2_MODEL_RUNNER="${DYNAMO_USE_V2_MODEL_RUNNER:-auto}"
MAX_NUM_SEQS="${DYNAMO_MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${DYNAMO_GPU_MEMORY_UTILIZATION:-0.90}"
BLOCK_SIZE="${DYNAMO_BLOCK_SIZE:-16}"
MIN_KV_TRANSFER_MB_S="${DYNAMO_MIN_KV_TRANSFER_MB_S:-5000}"
AGG_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_AGG_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,20,24,28,32,64,128}"
PREFILL_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
DECODE_CUDAGRAPH_CAPTURE_SIZES="${DYNAMO_DECODE_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,20,24,28,32}"

RUN_ID="${DYNAMO_RUN_ID:-$(date +%Y%m%d_%H%M%S)_dynamo_${ARCHITECTURE}_${ROUTER_MODE}}"
RUN_ROOT="${DYNAMO_RUN_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/e2e/_runs/dynamo/${RUN_ID}}"
LOG_ROOT="${RUN_ROOT}/service_logs"
FRONTEND_PORT="${DYNAMO_FRONTEND_PORT:-18400}"
SYSTEM_PORT_BASE="${DYNAMO_SYSTEM_PORT_BASE:-18500}"
NIXL_PORT_BASE="${DYNAMO_NIXL_PORT_BASE:-18600}"
KV_EVENT_PORT_BASE="${DYNAMO_KV_EVENT_PORT_BASE:-18700}"
VLLM_PORT_BASE="${DYNAMO_VLLM_PORT_BASE:-18800}"
ETCD_PORT="${DYNAMO_ETCD_PORT:-22379}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:${ETCD_PORT}}"
ETCD_BIN="${DYNAMO_ETCD_BIN:-${ROOT_DIR}/.local/dynamo-etcd-3.6.1/etcd}"
FILE_KV="${DYNAMO_FILE_KV:-${RUN_ROOT}/discovery}"
NAMESPACE="${DYNAMO_NAMESPACE:-pap-dynamo-${RUN_ID}}"

AIPERF_INPUT_FILE="${DYNAMO_AIPERF_INPUT_FILE:-}"
AIPERF_OUTPUT_DIR="${DYNAMO_AIPERF_OUTPUT_DIR:-${RUN_ROOT}/aiperf}"
AIPERF_SESSIONS="${DYNAMO_AIPERF_SESSIONS:-1}"
AIPERF_CONCURRENCY="${DYNAMO_AIPERF_CONCURRENCY:-1}"
AIPERF_TIMING_MODE="${DYNAMO_AIPERF_TIMING_MODE:-concurrency}"
AIPERF_REQUEST_RATE="${DYNAMO_AIPERF_REQUEST_RATE-}"
AIPERF_ARRIVAL_PATTERN="${DYNAMO_AIPERF_ARRIVAL_PATTERN:-poisson}"
AIPERF_CUSTOM_DATASET_TYPE="${DYNAMO_AIPERF_CUSTOM_DATASET_TYPE:-mooncake-trace}"
AIPERF_EXPECTED_REQUESTS="${DYNAMO_AIPERF_EXPECTED_REQUESTS-}"
AIPERF_REQUEST_TIMEOUT_SECONDS="${DYNAMO_AIPERF_REQUEST_TIMEOUT_SECONDS:-21600}"
AIPERF_WARMUP_DURATION_SECONDS="${DYNAMO_AIPERF_WARMUP_DURATION_SECONDS:-0}"
AIPERF_BENCHMARK_DURATION_SECONDS="${DYNAMO_AIPERF_BENCHMARK_DURATION_SECONDS:-}"
AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${DYNAMO_AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

for scheduling in "${AGG_ASYNC_SCHEDULING}" "${PREFILL_ASYNC_SCHEDULING}" \
  "${DECODE_ASYNC_SCHEDULING}"; do
  case "${scheduling}" in
    auto | on | off) ;;
    *) die "Dynamo async scheduling must be auto, on, or off" ;;
  esac
done
[[ "${CLUSTER_READY_WAIT_SECONDS}" =~ ^[0-9]+$ ]] \
  || die "DYNAMO_CLUSTER_READY_WAIT_SECONDS must be a nonnegative integer"
case "${USE_V2_MODEL_RUNNER}" in
  auto | 0 | 1) ;;
  *) die "DYNAMO_USE_V2_MODEL_RUNNER must be auto, 0, or 1" ;;
esac
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] \
  || die "DYNAMO_NAMESPACE must contain only letters, digits, underscores, or hyphens"
[[ "${ROUTER_PREFILL_LOAD_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "DYNAMO_ROUTER_PREFILL_LOAD_SCALE must be a nonnegative number"

case "${ARCHITECTURE}" in
  dp8)
    AGG_COUNT=8
    PREFILL_COUNT=0
    DECODE_COUNT=0
    ;;
  6p2d)
    AGG_COUNT=0
    PREFILL_COUNT=6
    DECODE_COUNT=2
    source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
    pap_configure_same_node_nixl "${ROOT_DIR}"
    ;;
  4p4d)
    AGG_COUNT=0
    PREFILL_COUNT=4
    DECODE_COUNT=4
    source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
    pap_configure_same_node_nixl "${ROOT_DIR}"
    ;;
  2p6d)
    AGG_COUNT=0
    PREFILL_COUNT=2
    DECODE_COUNT=6
    source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
    pap_configure_same_node_nixl "${ROOT_DIR}"
    ;;
  *) die "DYNAMO_ARCHITECTURE must be dp8, 6p2d, 4p4d, or 2p6d" ;;
esac
GPU_COUNT=$((AGG_COUNT + PREFILL_COUNT + DECODE_COUNT))
if (( AGG_COUNT > 0 )); then
  ROUTABLE_COUNT="${AGG_COUNT}"
else
  ROUTABLE_COUNT=$((
    PREFILL_COUNT < DECODE_COUNT ? PREFILL_COUNT : DECODE_COUNT
  ))
fi
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
case "${AIPERF_TIMING_MODE}" in
  concurrency)
    [[ -z "${AIPERF_REQUEST_RATE}" ]] \
      || die "DYNAMO_AIPERF_REQUEST_RATE must be empty in concurrency mode"
    ;;
  request_rate)
    [[ "${AIPERF_REQUEST_RATE}" =~ ^[0-9]+([.][0-9]+)?$ \
      && "${AIPERF_REQUEST_RATE}" =~ [1-9] ]] \
      || die "DYNAMO_AIPERF_REQUEST_RATE must be positive"
    ;;
  *) die "unsupported DYNAMO_AIPERF_TIMING_MODE" ;;
esac
case "${AIPERF_CUSTOM_DATASET_TYPE}" in
  multi-turn | mooncake-trace) ;;
  *) die "unsupported DYNAMO_AIPERF_CUSTOM_DATASET_TYPE" ;;
esac
for capture_sizes in \
  "${AGG_CUDAGRAPH_CAPTURE_SIZES}" \
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
    || die "set DYNAMO_AIPERF_INPUT_FILE to a valid dataset"
fi
AIPERF_INPUT_SHA256=not_applicable
if [[ -f "${AIPERF_INPUT_FILE}" ]]; then
  AIPERF_INPUT_SHA256="$(sha256sum "${AIPERF_INPUT_FILE}" | awk '{print $1}')"
fi
if [[ -n "${DYNAMO_AIPERF_INPUT_SHA256:-}" ]]; then
  [[ "${AIPERF_INPUT_SHA256}" == "${DYNAMO_AIPERF_INPUT_SHA256}" ]] \
    || die "Dynamo replay dataset checksum mismatch"
fi

HOST_IP="${DYNAMO_HOST_IP:-$(hostname -I | awk '{print $1}')}"
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
        --format=csv,noheader,nounits
    )" || die "failed to inspect GPU ${gpu}"
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
uv pip freeze --python "${DYNAMO_PYTHON}" > "${RUN_ROOT}/python_packages.txt"
{
  printf 'MODE=dynamo\nARCHITECTURE=%q\n' "${ARCHITECTURE}"
  printf 'AGG_COUNT=%q\nPREFILL_COUNT=%q\nDECODE_COUNT=%q\n' \
    "${AGG_COUNT}" "${PREFILL_COUNT}" "${DECODE_COUNT}"
  printf 'ROUTER_MODE=%q\nDISCOVERY_BACKEND=%q\n' \
    "${ROUTER_MODE}" "${DISCOVERY_BACKEND}"
  printf 'DYNAMO_VERSION=1.4.1\nVLLM_VERSION=0.26.0\n'
  printf 'EXECUTION_MODE=piecewise_cuda_graph\n'
  printf 'AGG_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${AGG_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'PREFILL_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'DECODE_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'MODEL_PATH=%q\nMAX_MODEL_LEN=%q\n' \
    "${MODEL_PATH}" "${MAX_MODEL_LEN}"
  printf 'HF_OVERRIDES=%q\n' "${HF_OVERRIDES}"
  printf 'PREFILL_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${PREFILL_MAX_NUM_BATCHED_TOKENS}"
  printf 'DECODE_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${DECODE_MAX_NUM_BATCHED_TOKENS}"
  printf 'AGG_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${AGG_MAX_NUM_BATCHED_TOKENS}"
  printf 'MAX_NUM_SEQS=%q\nBLOCK_SIZE=%q\n' \
    "${MAX_NUM_SEQS}" "${BLOCK_SIZE}"
  printf 'GPU_MEMORY_UTILIZATION=%q\nFRONTEND_PORT=%q\n' \
    "${GPU_MEMORY_UTILIZATION}" "${FRONTEND_PORT}"
  printf 'ETCD_BIN=%q\nETCD_ENDPOINTS=%q\n' \
    "${ETCD_BIN}" "${ETCD_ENDPOINTS}"
  if (( PREFILL_COUNT > 0 )); then
    printf 'PAP_NIXL_RUNTIME_MODE=%q\nPAP_NIXL_UCX_VERSION=%q\n' \
      "${PAP_NIXL_RUNTIME_MODE}" "${PAP_NIXL_UCX_VERSION}"
    printf 'NIXL_PLUGIN_DIR=%q\nUCX_PROTO_EMULATION_ENABLE=%q\n' \
      "${NIXL_PLUGIN_DIR}" "${UCX_PROTO_EMULATION_ENABLE}"
    printf 'UCX_CUDA_IPC_ENABLE_GET_ZCOPY=%q\n' \
      "${UCX_CUDA_IPC_ENABLE_GET_ZCOPY}"
  else
    printf 'KV_TRANSFER=not_applicable\n'
  fi
  printf 'MIN_KV_TRANSFER_MB_S=%q\n' "${MIN_KV_TRANSFER_MB_S}"
  printf 'HOST_IP=%q\nNAMESPACE=%q\n' "${HOST_IP}" "${NAMESPACE}"
  printf 'AIPERF_BIN=%q\nAIPERF_INPUT_FILE=%q\n' \
    "${AIPERF_BIN}" "${AIPERF_INPUT_FILE}"
  printf 'AIPERF_SESSIONS=%q\nAIPERF_CONCURRENCY=%q\n' \
    "${AIPERF_SESSIONS}" "${AIPERF_CONCURRENCY}"
  printf 'AIPERF_TIMING_MODE=%q\nAIPERF_REQUEST_RATE=%q\n' \
    "${AIPERF_TIMING_MODE}" "${AIPERF_REQUEST_RATE}"
  printf 'AIPERF_ARRIVAL_PATTERN=%q\nAIPERF_CUSTOM_DATASET_TYPE=%q\n' \
    "${AIPERF_ARRIVAL_PATTERN}" "${AIPERF_CUSTOM_DATASET_TYPE}"
  printf 'AIPERF_EXPECTED_REQUESTS=%q\n' "${AIPERF_EXPECTED_REQUESTS}"
  printf 'AIPERF_WARMUP_DURATION_SECONDS=%q\n' \
    "${AIPERF_WARMUP_DURATION_SECONDS}"
  printf 'AIPERF_BENCHMARK_DURATION_SECONDS=%q\n' \
    "${AIPERF_BENCHMARK_DURATION_SECONDS}"
  printf 'AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS=%q\n' \
    "${AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS}"
  printf 'AIPERF_INPUT_SHA256=%q\n' "${AIPERF_INPUT_SHA256}"
  printf 'GIT_COMMIT=%q\n' "$(git rev-parse HEAD)"
  # These are the names consumed by the launcher when replaying this file.
  for setting in \
    ARCHITECTURE ROUTER_MODE ROUTER_PREFILL_LOAD_SCALE \
    DISCOVERY_BACKEND START_ETCD SMOKE_ONLY \
    MAX_MODEL_LEN HF_OVERRIDES AGG_MAX_NUM_BATCHED_TOKENS \
    PREFILL_MAX_NUM_BATCHED_TOKENS DECODE_MAX_NUM_BATCHED_TOKENS \
    AGG_ASYNC_SCHEDULING PREFILL_ASYNC_SCHEDULING DECODE_ASYNC_SCHEDULING \
    CLUSTER_READY_WAIT_SECONDS USE_V2_MODEL_RUNNER \
    MAX_NUM_SEQS GPU_MEMORY_UTILIZATION BLOCK_SIZE \
    MIN_KV_TRANSFER_MB_S AGG_CUDAGRAPH_CAPTURE_SIZES \
    PREFILL_CUDAGRAPH_CAPTURE_SIZES DECODE_CUDAGRAPH_CAPTURE_SIZES \
    RUN_ID RUN_ROOT FRONTEND_PORT SYSTEM_PORT_BASE NIXL_PORT_BASE \
    KV_EVENT_PORT_BASE VLLM_PORT_BASE ETCD_PORT ETCD_BIN FILE_KV NAMESPACE \
    HOST_IP AIPERF_INPUT_FILE AIPERF_OUTPUT_DIR AIPERF_SESSIONS \
    AIPERF_CONCURRENCY AIPERF_TIMING_MODE AIPERF_REQUEST_RATE \
    AIPERF_ARRIVAL_PATTERN AIPERF_CUSTOM_DATASET_TYPE AIPERF_EXPECTED_REQUESTS \
    AIPERF_REQUEST_TIMEOUT_SECONDS AIPERF_WARMUP_DURATION_SECONDS \
    AIPERF_BENCHMARK_DURATION_SECONDS AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS \
    AIPERF_INPUT_SHA256; do
    printf 'DYNAMO_%s=%q\n' "${setting}" "${!setting}"
  done
  printf 'DYNAMO_PYTHON=%q\nPAP_PYTHON=%q\n' \
    "${DYNAMO_PYTHON}" "${PAP_PYTHON}"
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
  VLLM_USE_FLASHINFER_SAMPLER=0
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
)
if [[ "${USE_V2_MODEL_RUNNER}" != auto ]]; then
  COMMON_ENV+=(VLLM_USE_V2_MODEL_RUNNER="${USE_V2_MODEL_RUNNER}")
fi
if (( PREFILL_COUNT > 0 )); then
  COMMON_ENV+=(
    NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR}"
    UCX_MODULE_DIR="${UCX_MODULE_DIR}"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
    UCX_PROTO_EMULATION_ENABLE="${UCX_PROTO_EMULATION_ENABLE}"
    UCX_CUDA_IPC_ENABLE_GET_ZCOPY="${UCX_CUDA_IPC_ENABLE_GET_ZCOPY}"
    UCX_TLS="${UCX_TLS}"
    UCX_NET_DEVICES="${UCX_NET_DEVICES}"
    UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED}"
  )
fi
AGG_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${AGG_CUDAGRAPH_CAPTURE_SIZES}]}"
PREFILL_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
DECODE_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${DECODE_CUDAGRAPH_CAPTURE_SIZES}]}"
AGG_EXECUTION_ARGS=(
  --compilation-config "${AGG_COMPILATION_CONFIG}"
)
PREFILL_EXECUTION_ARGS=(
  --compilation-config "${PREFILL_COMPILATION_CONFIG}"
)
DECODE_EXECUTION_ARGS=(
  --compilation-config "${DECODE_COMPILATION_CONFIG}"
)
case "${AGG_ASYNC_SCHEDULING}" in
  on) AGG_EXECUTION_ARGS+=(--async-scheduling) ;;
  off) AGG_EXECUTION_ARGS+=(--no-async-scheduling) ;;
esac
case "${PREFILL_ASYNC_SCHEDULING}" in
  on) PREFILL_EXECUTION_ARGS+=(--async-scheduling) ;;
  off) PREFILL_EXECUTION_ARGS+=(--no-async-scheduling) ;;
esac
case "${DECODE_ASYNC_SCHEDULING}" in
  on) DECODE_EXECUTION_ARGS+=(--async-scheduling) ;;
  off) DECODE_EXECUTION_ARGS+=(--no-async-scheduling) ;;
esac
if [[ -n "${HF_OVERRIDES}" ]]; then
  AGG_EXECUTION_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
  PREFILL_EXECUTION_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
  DECODE_EXECUTION_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi

FRONTEND_ARGS=(
  --discovery-backend "${DISCOVERY_BACKEND}"
  --request-plane tcp
  --event-plane zmq
  --router-mode "${ROUTER_MODE}"
  --router-prefill-load-scale "${ROUTER_PREFILL_LOAD_SCALE}"
  --router-min-initial-workers "${ROUTABLE_COUNT}"
  --kv-cache-block-size "${BLOCK_SIZE}"
  --http-port "${FRONTEND_PORT}"
  --model-name "${MODEL_PATH}"
  --model-path "${MODEL_PATH}"
)
if (( PREFILL_COUNT > 0 )); then
  FRONTEND_ARGS+=(--enforce-disagg)
fi

setsid env "${COMMON_ENV[@]}" \
  DYN_HTTP_PORT="${FRONTEND_PORT}" \
  "${DYNAMO_PYTHON}" -P -m dynamo.frontend \
    "${FRONTEND_ARGS[@]}" \
    > "${LOG_ROOT}/frontend.log" 2>&1 &
PIDS+=("$!")
PGIDS+=("$!")

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"enable_cross_layers_blocks":"True"}}'
for (( index=0; index<AGG_COUNT; index++ )); do
  gpu="${index}"
  ordinal="${index}"
  event_port=$((KV_EVENT_PORT_BASE + index))
  event_config="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${event_port}\",\"enable_kv_cache_events\":true}"
  log="${LOG_ROOT}/aggregated_${index}.log"
  setsid env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CUDA_CACHE_PATH="${RUN_ROOT}/cuda_cache/gpu${gpu}" \
    DYN_SYSTEM_PORT="$((SYSTEM_PORT_BASE + ordinal))" \
    VLLM_PORT="$((VLLM_PORT_BASE + ordinal * 20))" \
    "${DYNAMO_PYTHON}" -P -m dynamo.vllm \
      --discovery-backend "${DISCOVERY_BACKEND}" \
      --request-plane tcp --event-plane zmq \
      --model "${MODEL_PATH}" --served-model-name "${MODEL_PATH}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --kv-events-config "${event_config}" \
      "${AGG_EXECUTION_ARGS[@]}" --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${AGG_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size "${BLOCK_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --disable-log-stats > "${log}" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
done

for (( index=0; index<AGG_COUNT; index++ )); do
  wait_for_worker "${PIDS[index + 1]}" \
    "${LOG_ROOT}/aggregated_${index}.log" "Dynamo Aggregated ${index}"
done
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
for (( index=0; index<AGG_COUNT; index++ )); do
  DYNAMO_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/aggregated_${index}.log")
done
for (( index=0; index<DECODE_COUNT; index++ )); do
  DYNAMO_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/decode_${index}.log")
done
for (( index=0; index<PREFILL_COUNT; index++ )); do
  DYNAMO_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/prefill_${index}.log")
done
"${CUDA_GRAPH_AUDITOR}" "${RUN_ROOT}/vllm_cuda_graph_audit.env" \
  PIECEWISE "${DYNAMO_VLLM_GRAPH_LOGS[@]}"
sleep "${CLUSTER_READY_WAIT_SECONDS}"

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
    AIPERF_BIN="${AIPERF_BIN}" \
    MODEL_PATH="${MODEL_PATH}" \
    AIPERF_INPUT_FILE="${AIPERF_INPUT_FILE}" \
    AIPERF_CUSTOM_DATASET_TYPE="${AIPERF_CUSTOM_DATASET_TYPE}" \
    AIPERF_TARGET_URL="http://127.0.0.1:${FRONTEND_PORT}" \
    AIPERF_OUTPUT_DIR="${AIPERF_OUTPUT_DIR}" \
    AIPERF_SESSIONS="${AIPERF_SESSIONS}" \
    AIPERF_CONCURRENCY="${AIPERF_CONCURRENCY}" \
    AIPERF_TIMING_MODE="${AIPERF_TIMING_MODE}" \
    AIPERF_REQUEST_RATE="${AIPERF_REQUEST_RATE}" \
    AIPERF_ARRIVAL_PATTERN="${AIPERF_ARRIVAL_PATTERN}" \
    AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_REQUEST_TIMEOUT_SECONDS}" \
    AIPERF_WARMUP_DURATION_SECONDS="${AIPERF_WARMUP_DURATION_SECONDS}" \
    AIPERF_WARMUP_CONCURRENCY="${AIPERF_CONCURRENCY}" \
    AIPERF_WARMUP_REQUEST_RATE="${AIPERF_REQUEST_RATE}" \
    AIPERF_WARMUP_ARRIVAL_PATTERN="${AIPERF_ARRIVAL_PATTERN}" \
    AIPERF_BENCHMARK_DURATION_SECONDS="${AIPERF_BENCHMARK_DURATION_SECONDS}" \
    AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS}" \
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

if rg --no-ignore -n -i \
  'CUDA out of memory|EngineDeadError|Traceback|NIXL_ERR|NixlConnector.*failed' \
  "${LOG_ROOT}" > "${RUN_ROOT}/correctness_audit_matches.log"; then
  printf 'STATUS=failed\n' > "${RUN_ROOT}/correctness_audit.env"
  die "Dynamo correctness audit failed"
else
  scan_status=$?
  if (( scan_status != 1 )); then
    printf 'STATUS=failed\nREASON=log_scan_failed\n' \
      > "${RUN_ROOT}/correctness_audit.env"
    die "Dynamo correctness log scan failed"
  fi
fi
: > "${RUN_ROOT}/correctness_audit_matches.log"
printf 'STATUS=passed\nMATCH_COUNT=0\n' \
  > "${RUN_ROOT}/correctness_audit.env"
if (( SMOKE_ONLY == 0 && PREFILL_COUNT > 0 )); then
  "${PAP_PYTHON}" \
    "${ROOT_DIR}/benchmarks/pap/tooling/analyze_dynamo_ttft.py" \
    "${RUN_ROOT}" --block-size "${BLOCK_SIZE}" \
    --output "${RUN_ROOT}/dynamo_ttft_analysis.json"
  if ! jq -e --argjson minimum "${MIN_KV_TRANSFER_MB_S}" \
    '.kv_transfer.throughput_mb_s.p90 >= $minimum' \
    "${RUN_ROOT}/dynamo_ttft_analysis.json" >/dev/null; then
    observed="$(jq -r \
      '.kv_transfer.throughput_mb_s.p90 // "missing"' \
      "${RUN_ROOT}/dynamo_ttft_analysis.json")"
    printf 'STATUS=failed\nMETRIC=%q\nMINIMUM_MB_S=%q\nOBSERVED_MB_S=%q\n' \
      least_contended_window_p90 "${MIN_KV_TRANSFER_MB_S}" "${observed}" \
      > "${RUN_ROOT}/kv_transfer_audit.env"
    die "Dynamo KV transfer throughput failed the same-node floor"
  fi
  observed="$(jq -r \
    '.kv_transfer.throughput_mb_s.p90' \
    "${RUN_ROOT}/dynamo_ttft_analysis.json")"
  aggregate="$(jq -r '.kv_transfer.aggregate_throughput_mb_s' \
    "${RUN_ROOT}/dynamo_ttft_analysis.json")"
  printf '%s\n' \
    'STATUS=passed' \
    'METRIC=least_contended_window_p90' \
    "MINIMUM_MB_S=${MIN_KV_TRANSFER_MB_S}" \
    "OBSERVED_MB_S=${observed}" \
    "CONTENTION_WEIGHTED_MB_S=${aggregate}" \
    > "${RUN_ROOT}/kv_transfer_audit.env"
elif [[ "${ARCHITECTURE}" == "dp8" ]]; then
  printf 'STATUS=not_applicable\nREASON=aggregated_workers_have_no_kv_transfer\n' \
    > "${RUN_ROOT}/kv_transfer_audit.env"
fi
echo "DYNAMO_RUN_ROOT=${RUN_ROOT}"
