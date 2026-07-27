#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
pap_configure_same_node_nixl "${ROOT_DIR}"
if [[ -v PD_LOAD_CLIENT_MODE ]]; then
  echo "PD_LOAD_CLIENT_MODE was removed; the PD runner is AIPerf-only" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
PROXY="${ROOT_DIR}/examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
AIPERF_DATASET_GENERATOR="${ROOT_DIR}/benchmarks/pap/aiperf/generate_multiturn_dataset.py"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${AIPERF_ROOT}/.venv/bin/aiperf}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"

TOPOLOGY="${PD_LOAD_TOPOLOGY:-3p1d}"
TRANSFER_MODE="${1:-${PD_LOAD_TRANSFER_MODE:-oneway}}"
ROUNDS="${PD_LOAD_ROUNDS:-10}"
CONVERSATIONS="${PD_LOAD_CONVERSATIONS:-32}"
REQUEST_RATE="${PD_LOAD_REQUEST_RATE:-12}"
DOCUMENT_TOKENS="${PD_LOAD_DOCUMENT_TOKENS:-8192}"
APPEND_TOKENS="${PD_LOAD_APPEND_TOKENS:-512}"
OUTPUT_TOKENS="${PD_LOAD_OUTPUT_TOKENS:-32}"
AIPERF_DOCUMENT_TOKENS_MEDIAN="${AIPERF_DOCUMENT_TOKENS_MEDIAN:-8000}"
AIPERF_DOCUMENT_TOKENS_MIN="${AIPERF_DOCUMENT_TOKENS_MIN:-4096}"
AIPERF_DOCUMENT_TOKENS_MAX="${AIPERF_DOCUMENT_TOKENS_MAX:-11264}"
AIPERF_APPEND_TOKENS_MEDIAN="${AIPERF_APPEND_TOKENS_MEDIAN:-500}"
AIPERF_APPEND_TOKENS_MIN="${AIPERF_APPEND_TOKENS_MIN:-256}"
AIPERF_APPEND_TOKENS_MAX="${AIPERF_APPEND_TOKENS_MAX:-768}"
AIPERF_OUTPUT_TOKENS_MEDIAN="${AIPERF_OUTPUT_TOKENS_MEDIAN:-30}"
AIPERF_OUTPUT_TOKENS_MIN="${AIPERF_OUTPUT_TOKENS_MIN:-16}"
AIPERF_OUTPUT_TOKENS_MAX="${AIPERF_OUTPUT_TOKENS_MAX:-64}"
AIPERF_RANDOM_SEED="${AIPERF_RANDOM_SEED:-42}"
AIPERF_THINK_TIME_MS="${AIPERF_THINK_TIME_MS:-3000}"
AIPERF_TOOL_TIME_MS="${AIPERF_TOOL_TIME_MS:-1000}"
AIPERF_TOOL_EVERY="${AIPERF_TOOL_EVERY:-3}"
MAX_MODEL_LEN="${PD_LOAD_MAX_MODEL_LEN:-20000}"
MAX_NUM_BATCHED_TOKENS="${PD_LOAD_MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${PD_LOAD_MAX_NUM_SEQS:-64}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PD_LOAD_PREFILL_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
PREFILL_MAX_NUM_SEQS="${PD_LOAD_PREFILL_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
DECODE_MAX_NUM_BATCHED_TOKENS="${PD_LOAD_DECODE_MAX_NUM_BATCHED_TOKENS:-64}"
DECODE_MAX_NUM_SEQS="${PD_LOAD_DECODE_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
EXECUTION_MODE="${PD_LOAD_EXECUTION_MODE:-eager}"
PREFILL_CUDAGRAPH_CAPTURE_SIZES="${PD_LOAD_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
DECODE_CUDAGRAPH_CAPTURE_SIZES="${PD_LOAD_DECODE_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,20,24,28,32}"
GPU_MEMORY_UTILIZATION="${PD_LOAD_GPU_MEMORY_UTILIZATION:-0.90}"
REQUEST_TIMEOUT_SECONDS="${PD_LOAD_REQUEST_TIMEOUT_SECONDS:-180}"
case "${EXECUTION_MODE}" in
  eager | piecewise) ;;
  *)
    echo "PD_LOAD_EXECUTION_MODE must be eager or piecewise" >&2
    exit 2
    ;;
esac
for capture_sizes in \
  "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}" \
  "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"; do
  if ! [[ "${capture_sizes}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "CUDA Graph capture sizes must be comma-separated positive integers" >&2
    exit 2
  fi
done

PREFILL_EXECUTION_ARGS=(--enforce-eager)
DECODE_EXECUTION_ARGS=(--enforce-eager)
PREFILL_COMPILATION_CONFIG=""
DECODE_COMPILATION_CONFIG=""
if [[ "${EXECUTION_MODE}" == "piecewise" ]]; then
  PREFILL_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
  DECODE_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${DECODE_CUDAGRAPH_CAPTURE_SIZES}]}"
  PREFILL_EXECUTION_ARGS=(
    --compilation-config "${PREFILL_COMPILATION_CONFIG}"
  )
  DECODE_EXECUTION_ARGS=(
    --compilation-config "${DECODE_COMPILATION_CONFIG}"
  )
fi
EXPERIMENTS_ROOT="${PAP_EXPERIMENTS_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXPERIMENTS_ROOT}/_staging}"
RUN_ID="${PD_LOAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_pd_${TOPOLOGY}_${TRANSFER_MODE}}"
RUN_ROOT="${PD_LOAD_RUN_ROOT:-${RESULTS_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${RUN_ROOT}/service_logs"
AIPERF_INPUT_FILE_PROVIDED=0
if [[ -n "${PD_AIPERF_INPUT_FILE:-${AIPERF_INPUT_FILE:-}}" ]]; then
  AIPERF_INPUT_FILE_PROVIDED=1
fi
PD_AIPERF_INPUT_FILE="${PD_AIPERF_INPUT_FILE:-${AIPERF_INPUT_FILE:-${RUN_ROOT}/aiperf_multiturn.jsonl}}"
PD_AIPERF_OUTPUT_DIR="${PD_AIPERF_OUTPUT_DIR:-${RUN_ROOT}/aiperf}"
PD_AIPERF_CONCURRENCY="${PD_AIPERF_CONCURRENCY:-${CONVERSATIONS}}"
PD_AIPERF_TIMING_MODE="${PD_AIPERF_TIMING_MODE:-concurrency}"
PD_AIPERF_REQUEST_RATE="${PD_AIPERF_REQUEST_RATE-}"
if [[ ! "${PD_AIPERF_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PD_AIPERF_CONCURRENCY must be positive" >&2
  exit 2
fi
ACTIVE_CONVERSATIONS="${PD_AIPERF_CONCURRENCY}"
if (( ACTIVE_CONVERSATIONS > CONVERSATIONS )); then
  echo "AIPerf concurrency exceeds total conversations" >&2
  exit 2
fi
if [[ "${PD_AIPERF_TIMING_MODE}" == "request_rate" \
  && -z "${PD_AIPERF_REQUEST_RATE}" ]]; then
  PD_AIPERF_REQUEST_RATE="${REQUEST_RATE}"
fi

if [[ ! "${TOPOLOGY}" =~ ^([1-9][0-9]*)p([1-9][0-9]*)d$ ]]; then
  echo "invalid PD topology: ${TOPOLOGY}" >&2
  exit 2
fi
PREFILL_COUNT="${BASH_REMATCH[1]}"
DECODE_COUNT="${BASH_REMATCH[2]}"
GPU_COUNT=$((PREFILL_COUNT + DECODE_COUNT))
case "${TRANSFER_MODE}" in
  oneway)
    EXTRA_CONFIG='"bidirectional_kv_xfer":false,"enable_cross_layers_blocks":"True"'
    ;;
  twoway)
    EXTRA_CONFIG='"bidirectional_kv_xfer":true,"kv_recompute_threshold":0,"decoder_kv_blocks_ttl":480,"enable_cross_layers_blocks":"True"'
    ;;
  *)
    echo "usage: $0 [oneway|twoway]" >&2
    exit 2
    ;;
esac

default_gpu_csv() {
  local start="$1"
  local count_value="$2"
  local values=()
  local index
  for (( index=0; index<count_value; index++ )); do
    values+=("$((start + index))")
  done
  local IFS=,
  printf '%s' "${values[*]}"
}

PREFILL_GPUS_CSV="${PD_PREFILL_GPUS:-$(default_gpu_csv 0 "${PREFILL_COUNT}")}"
DECODE_GPUS_CSV="${PD_DECODE_GPUS:-$(
  default_gpu_csv "${PREFILL_COUNT}" "${DECODE_COUNT}"
)}"
IFS=, read -r -a PREFILL_GPUS <<< "${PREFILL_GPUS_CSV}"
IFS=, read -r -a DECODE_GPUS <<< "${DECODE_GPUS_CSV}"
if (( ${#PREFILL_GPUS[@]} != PREFILL_COUNT \
      || ${#DECODE_GPUS[@]} != DECODE_COUNT )); then
  echo "GPU list does not match topology ${TOPOLOGY}" >&2
  exit 2
fi

for required in "${PYTHON_BIN}" "${VLLM_BIN}" "${PROXY}" \
  "${DATASET_PATH}"; do
  [[ -e "${required}" ]] || {
    echo "required path is missing: ${required}" >&2
    exit 1
  }
done
[[ -x "${AIPERF_BIN}" ]] || {
  echo "AIPerf is not installed at ${AIPERF_BIN}" >&2
  exit 1
}
for required in "${AIPERF_RUNNER}" "${AIPERF_DATASET_GENERATOR}"; do
  [[ -e "${required}" ]] || {
    echo "required AIPerf path is missing: ${required}" >&2
    exit 1
  }
done

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

PIDS=()
PGIDS=()

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
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ensure_gpus_idle() {
  local gpu processes
  for gpu in "${PREFILL_GPUS[@]}" "${DECODE_GPUS[@]}"; do
    processes="$(
      nvidia-smi -i "${gpu}" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    if [[ -n "${processes//[[:space:]]/}" ]]; then
      echo "GPU ${gpu} is occupied by ${processes}" >&2
      return 1
    fi
  done
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local deadline=$((SECONDS + 900))
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for ${label}: ${url}" >&2
      return 1
    fi
    sleep 2
  done
  echo "${label} is ready at ${url}"
}

kv_config() {
  local engine_id="$1"
  local role="$2"
  printf '{"kv_connector":"NixlConnector","engine_id":"%s","kv_role":"%s","kv_load_failure_policy":"fail","kv_connector_extra_config":{%s}}' \
    "${engine_id}" "${role}" "${EXTRA_CONFIG}"
}

ensure_gpus_idle
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
git status --short > "${RUN_ROOT}/git_status.txt"
git diff --binary > "${RUN_ROOT}/tracked_worktree.patch"
git diff --cached --binary > "${RUN_ROOT}/tracked_index.patch"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_TRACKED_WORKTREE_DIRTY=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  GIT_TRACKED_WORKTREE_DIRTY=1
fi
if [[ "${AIPERF_INPUT_FILE_PROVIDED}" == "1" ]]; then
  [[ -f "${PD_AIPERF_INPUT_FILE}" ]] || {
    echo "AIPerf input file is missing: ${PD_AIPERF_INPUT_FILE}" >&2
    exit 1
  }
else
  "${PYTHON_BIN}" "${AIPERF_DATASET_GENERATOR}" \
    --model "${MODEL_PATH}" --corpus "${DATASET_PATH}" \
    --output "${PD_AIPERF_INPUT_FILE}" \
    --sessions "${CONVERSATIONS}" --turns "${ROUNDS}" \
    --document-tokens "${DOCUMENT_TOKENS}" \
    --document-tokens-median "${AIPERF_DOCUMENT_TOKENS_MEDIAN}" \
    --document-tokens-min "${AIPERF_DOCUMENT_TOKENS_MIN}" \
    --document-tokens-max "${AIPERF_DOCUMENT_TOKENS_MAX}" \
    --append-tokens "${APPEND_TOKENS}" \
    --append-tokens-median "${AIPERF_APPEND_TOKENS_MEDIAN}" \
    --append-tokens-min "${AIPERF_APPEND_TOKENS_MIN}" \
    --append-tokens-max "${AIPERF_APPEND_TOKENS_MAX}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --output-tokens-median "${AIPERF_OUTPUT_TOKENS_MEDIAN}" \
    --output-tokens-min "${AIPERF_OUTPUT_TOKENS_MIN}" \
    --output-tokens-max "${AIPERF_OUTPUT_TOKENS_MAX}" \
    --random-seed "${AIPERF_RANDOM_SEED}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --think-time-ms "${AIPERF_THINK_TIME_MS}" \
    --tool-time-ms "${AIPERF_TOOL_TIME_MS}" \
    --tool-every "${AIPERF_TOOL_EVERY}" \
    --session-prefix "${RUN_ID}-aiperf"
fi

PREFILL_PORT_BASE="${PD_PREFILL_PORT_BASE:-24100}"
DECODE_PORT_BASE="${PD_DECODE_PORT_BASE:-24200}"
PROXY_PORT="${PD_PROXY_PORT:-24300}"
SIDE_PORT_BASE="${PD_SIDE_PORT_BASE:-5860}"
VLLM_PORT_BASE="${PD_VLLM_PORT_BASE:-51600}"
PREFILL_PORTS=()
DECODE_PORTS=()
HOSTS_PREFILL=()
HOSTS_DECODE=()

{
  printf 'MODE=pd\nTOPOLOGY=%q\nPD_TRANSFER_MODE=%q\n' \
    "${TOPOLOGY}" "${TRANSFER_MODE}"
  printf 'GPU_COUNT=%q\n' "${GPU_COUNT}"
  printf 'MODEL_PATH=%q\nROUNDS=%q\nTOTAL_CONVERSATIONS=%q\n' \
    "${MODEL_PATH}" "${ROUNDS}" "${CONVERSATIONS}"
  printf 'ACTIVE_CONVERSATIONS=%q\n' "${ACTIVE_CONVERSATIONS}"
  printf 'REQUEST_RATE=%q\nDOCUMENT_TOKENS=%q\nAPPEND_TOKENS=%q\n' \
    "${REQUEST_RATE}" "${DOCUMENT_TOKENS}" "${APPEND_TOKENS}"
  printf 'OUTPUT_TOKENS=%q\nMAX_MODEL_LEN=%q\n' \
    "${OUTPUT_TOKENS}" "${MAX_MODEL_LEN}"
  printf 'MAX_NUM_BATCHED_TOKENS=%q\nMAX_NUM_SEQS=%q\n' \
    "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}"
  printf 'PREFILL_MAX_NUM_BATCHED_TOKENS=%q\nPREFILL_MAX_NUM_SEQS=%q\n' \
    "${PREFILL_MAX_NUM_BATCHED_TOKENS}" "${PREFILL_MAX_NUM_SEQS}"
  printf 'DECODE_MAX_NUM_BATCHED_TOKENS=%q\nDECODE_MAX_NUM_SEQS=%q\n' \
    "${DECODE_MAX_NUM_BATCHED_TOKENS}" "${DECODE_MAX_NUM_SEQS}"
  printf 'EXECUTION_MODE=%q\n' "${EXECUTION_MODE}"
  printf 'PREFILL_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'DECODE_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'PREFILL_COMPILATION_CONFIG=%q\n' \
    "${PREFILL_COMPILATION_CONFIG}"
  printf 'DECODE_COMPILATION_CONFIG=%q\n' \
    "${DECODE_COMPILATION_CONFIG}"
  printf 'GPU_MEMORY_UTILIZATION=%q\nPREFILL_GPUS=%q\nDECODE_GPUS=%q\n' \
    "${GPU_MEMORY_UTILIZATION}" "${PREFILL_GPUS_CSV}" \
    "${DECODE_GPUS_CSV}"
  printf 'REQUEST_TIMEOUT_SECONDS=%q\n' "${REQUEST_TIMEOUT_SECONDS}"
  printf 'CLIENT=%q\n' "aiperf"
  printf 'AIPERF_ROOT=%q\nAIPERF_BIN=%q\n' \
    "${AIPERF_ROOT}" "${AIPERF_BIN}"
  printf 'PD_AIPERF_INPUT_FILE=%q\nPD_AIPERF_OUTPUT_DIR=%q\n' \
    "${PD_AIPERF_INPUT_FILE}" "${PD_AIPERF_OUTPUT_DIR}"
  printf 'PD_AIPERF_CONCURRENCY=%q\nPD_AIPERF_TIMING_MODE=%q\n' \
    "${PD_AIPERF_CONCURRENCY}" "${PD_AIPERF_TIMING_MODE}"
  printf 'PD_AIPERF_REQUEST_RATE=%q\n' "${PD_AIPERF_REQUEST_RATE}"
  printf 'PAP_NIXL_RUNTIME_MODE=%q\nPAP_NIXL_UCX_VERSION=%q\n' \
    "${PAP_NIXL_RUNTIME_MODE}" "${PAP_NIXL_UCX_VERSION}"
  printf 'NIXL_PLUGIN_DIR=%q\nUCX_PROTO_EMULATION_ENABLE=%q\n' \
    "${NIXL_PLUGIN_DIR}" "${UCX_PROTO_EMULATION_ENABLE}"
  printf 'GIT_COMMIT=%q\nGIT_TRACKED_WORKTREE_DIRTY=%q\n' \
    "${GIT_COMMIT}" "${GIT_TRACKED_WORKTREE_DIRTY}"
} > "${RUN_ROOT}/effective_config.env"

for (( index=0; index<PREFILL_COUNT; index++ )); do
  port=$((PREFILL_PORT_BASE + index))
  side_port=$((SIDE_PORT_BASE + index))
  config="$(kv_config "${RUN_ID}-prefill-${index}" kv_producer)"
  PREFILL_PORTS+=("${port}")
  HOSTS_PREFILL+=(127.0.0.1)
  echo "Starting PD Prefill ${index} on GPU ${PREFILL_GPUS[index]}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${PREFILL_GPUS[index]}" \
    VLLM_PORT="$((VLLM_PORT_BASE + index * 20))" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    PAP_MODEL_HOOKS=0 \
    PAP_CUDAGRAPH_COMPATIBLE=0 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${port}" \
      "${PREFILL_EXECUTION_ARGS[@]}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${PREFILL_MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size 16 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --kv-transfer-config "${config}" \
      > "${LOG_ROOT}/prefill_${index}.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
done

for (( index=0; index<DECODE_COUNT; index++ )); do
  port=$((DECODE_PORT_BASE + index))
  side_port=$((SIDE_PORT_BASE + PREFILL_COUNT + index))
  config="$(kv_config "${RUN_ID}-decode-${index}" kv_consumer)"
  DECODE_PORTS+=("${port}")
  HOSTS_DECODE+=(127.0.0.1)
  echo "Starting PD Decode ${index} on GPU ${DECODE_GPUS[index]}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${DECODE_GPUS[index]}" \
    VLLM_PORT="$((VLLM_PORT_BASE + (PREFILL_COUNT + index) * 20))" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    PAP_MODEL_HOOKS=0 \
    PAP_CUDAGRAPH_COMPATIBLE=0 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${port}" \
      "${DECODE_EXECUTION_ARGS[@]}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${DECODE_MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size 16 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --kv-transfer-config "${config}" \
      > "${LOG_ROOT}/decode_${index}.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
done

for (( index=0; index<PREFILL_COUNT; index++ )); do
  wait_for_http \
    "http://127.0.0.1:${PREFILL_PORTS[index]}/health" \
    "PD Prefill ${index}"
done
for (( index=0; index<DECODE_COUNT; index++ )); do
  wait_for_http \
    "http://127.0.0.1:${DECODE_PORTS[index]}/health" \
    "PD Decode ${index}"
done

echo "Starting conversation-affine PD proxy"
setsid "${PYTHON_BIN}" "${PROXY}" \
  --host 127.0.0.1 --port "${PROXY_PORT}" \
  --prefiller-hosts "${HOSTS_PREFILL[@]}" \
  --prefiller-ports "${PREFILL_PORTS[@]}" \
  --decoder-hosts "${HOSTS_DECODE[@]}" \
  --decoder-ports "${DECODE_PORTS[@]}" \
  > "${LOG_ROOT}/proxy.log" 2>&1 &
PIDS+=("$!")
PGIDS+=("$!")
wait_for_http "http://127.0.0.1:${PROXY_PORT}/health" "PD proxy"

env \
  PAP_ROOT="${ROOT_DIR}" \
  AIPERF_ROOT="${AIPERF_ROOT}" \
  AIPERF_BIN="${AIPERF_BIN}" \
  MODEL_PATH="${MODEL_PATH}" \
  AIPERF_INPUT_FILE="${PD_AIPERF_INPUT_FILE}" \
  AIPERF_TARGET_URL="http://127.0.0.1:${PROXY_PORT}" \
  AIPERF_OUTPUT_DIR="${PD_AIPERF_OUTPUT_DIR}" \
  AIPERF_SESSIONS="${CONVERSATIONS}" \
  AIPERF_CONCURRENCY="${PD_AIPERF_CONCURRENCY}" \
  AIPERF_TIMING_MODE="${PD_AIPERF_TIMING_MODE}" \
  AIPERF_REQUEST_RATE="${PD_AIPERF_REQUEST_RATE}" \
  AIPERF_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
  "${AIPERF_RUNNER}" \
  2>&1 | tee "${RUN_ROOT}/client.log"
if [[ -z "$(find "${PD_AIPERF_OUTPUT_DIR}" -type f \
  -name 'profile*.json' -size +0c -print -quit)" ]]; then
  echo "AIPerf produced no profile JSON under ${PD_AIPERF_OUTPUT_DIR}" >&2
  exit 1
fi

curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" \
  -o "${RUN_ROOT}/proxy_health.json"
if rg -n -i \
  'CUDA out of memory|EngineDeadError|Traceback|NIXL.*failed|NIXL_ERR' \
  "${LOG_ROOT}" > "${RUN_ROOT}/correctness_audit_matches.log"; then
  printf 'STATUS=failed\n' > "${RUN_ROOT}/correctness_audit.env"
  echo "PD correctness audit failed" >&2
  exit 1
fi
: > "${RUN_ROOT}/correctness_audit_matches.log"
printf 'STATUS=passed\nMATCH_COUNT=0\n' \
  > "${RUN_ROOT}/correctness_audit.env"
echo "PD_LOAD_RUN_ROOT=${RUN_ROOT}"
