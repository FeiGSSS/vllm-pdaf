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
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv-dynamo/bin/vllm}"
VLLM_PYTHON="${PD_VLLM_PYTHON:-$(dirname "${VLLM_BIN}")/python}"
EXPECTED_VLLM_VERSION="${PD_EXPECTED_VLLM_VERSION:-0.26.0}"
CUDA_GRAPH_AUDITOR="${ROOT_DIR}/benchmarks/pap/scripts/audit_cuda_graph_logs.sh"
KV_TRANSFER_ANALYZER="${ROOT_DIR}/benchmarks/pap/tooling/analyze_dynamo_ttft.py"
PROXY="${ROOT_DIR}/examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
AIPERF_DATASET_GENERATOR="${ROOT_DIR}/benchmarks/pap/aiperf/generate_multiturn_dataset.py"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
ENABLE_AUTO_TOOL_CHOICE="${PAP_BENCH_ENABLE_AUTO_TOOL_CHOICE:-0}"
TOOL_CALL_PARSER="${PAP_BENCH_TOOL_CALL_PARSER:-}"

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
MAX_NUM_BATCHED_TOKENS="${PD_LOAD_MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${PD_LOAD_MAX_NUM_SEQS:-256}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PD_LOAD_PREFILL_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
PREFILL_MAX_NUM_SEQS="${PD_LOAD_PREFILL_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
DECODE_MAX_NUM_BATCHED_TOKENS="${PD_LOAD_DECODE_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
DECODE_MAX_NUM_SEQS="${PD_LOAD_DECODE_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
GPU_MEMORY_UTILIZATION="${PD_LOAD_GPU_MEMORY_UTILIZATION:-0.90}"
REQUEST_TIMEOUT_SECONDS="${PD_LOAD_REQUEST_TIMEOUT_SECONDS:-180}"
MIN_KV_TRANSFER_MB_S="${PD_LOAD_MIN_KV_TRANSFER_MB_S:-5000}"

PREFILL_CUDAGRAPH_CAPTURE_SIZES="${PD_LOAD_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
DECODE_CUDAGRAPH_CAPTURE_SIZES="${PD_LOAD_DECODE_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,20,24,28,32}"
for capture_sizes in \
  "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}" \
  "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"; do
  if ! [[ "${capture_sizes}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "CUDA Graph capture sizes must be positive integer CSV" >&2
    exit 2
  fi
done
PREFILL_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
DECODE_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${DECODE_CUDAGRAPH_CAPTURE_SIZES}]}"
PREFILL_EXECUTION_ARGS=(
  --compilation-config "${PREFILL_COMPILATION_CONFIG}"
)
DECODE_EXECUTION_ARGS=(
  --compilation-config "${DECODE_COMPILATION_CONFIG}"
)
TOOL_ARGS=()
case "${ENABLE_AUTO_TOOL_CHOICE}" in
  0)
    [[ -z "${TOOL_CALL_PARSER}" ]] || {
      echo "tool parser requires PAP_BENCH_ENABLE_AUTO_TOOL_CHOICE=1" >&2
      exit 2
    }
    ;;
  1)
    [[ -n "${TOOL_CALL_PARSER}" ]] || {
      echo "PAP_BENCH_TOOL_CALL_PARSER is required for automatic tools" >&2
      exit 2
    }
    TOOL_ARGS=(
      --enable-auto-tool-choice
      --tool-call-parser "${TOOL_CALL_PARSER}"
    )
    ;;
  *)
    echo "PAP_BENCH_ENABLE_AUTO_TOOL_CHOICE must be 0 or 1" >&2
    exit 2
    ;;
esac
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
PD_AIPERF_EXPECTED_REQUESTS="${PD_AIPERF_EXPECTED_REQUESTS:-}"
if [[ ! "${PD_AIPERF_CONCURRENCY}" \
  =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "PD_AIPERF_CONCURRENCY must be a positive integer or CSV list" >&2
  exit 2
fi
ACTIVE_CONVERSATIONS=0
IFS=, read -r -a PD_AIPERF_CONCURRENCY_POINTS \
  <<< "${PD_AIPERF_CONCURRENCY}"
for concurrency in "${PD_AIPERF_CONCURRENCY_POINTS[@]}"; do
  if (( concurrency > CONVERSATIONS )); then
    echo "AIPerf concurrency exceeds total conversations" >&2
    exit 2
  fi
  if (( concurrency > ACTIVE_CONVERSATIONS )); then
    ACTIVE_CONVERSATIONS="${concurrency}"
  fi
done
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

for required in "${PYTHON_BIN}" "${VLLM_BIN}" "${VLLM_PYTHON}" "${PROXY}" \
  "${DATASET_PATH}" "${CUDA_GRAPH_AUDITOR}" "${KV_TRANSFER_ANALYZER}"; do
  [[ -e "${required}" ]] || {
    echo "required path is missing: ${required}" >&2
    exit 1
  }
done

VLLM_VERSION="$(
  "${VLLM_PYTHON}" -P -c \
    'import importlib.metadata as m; print(m.version("vllm"))'
)"
if [[ "${VLLM_VERSION}" != "${EXPECTED_VLLM_VERSION}" ]]; then
  echo "PD worker vLLM version ${VLLM_VERSION} does not match " \
    "the required baseline ${EXPECTED_VLLM_VERSION}" >&2
  exit 2
fi
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
  printf 'EXECUTION_MODE=piecewise_cuda_graph\n'
  printf 'PREFILL_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${PREFILL_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'DECODE_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
    "${DECODE_CUDAGRAPH_CAPTURE_SIZES}"
  printf 'GPU_MEMORY_UTILIZATION=%q\nPREFILL_GPUS=%q\nDECODE_GPUS=%q\n' \
    "${GPU_MEMORY_UTILIZATION}" "${PREFILL_GPUS_CSV}" \
    "${DECODE_GPUS_CSV}"
  printf 'REQUEST_TIMEOUT_SECONDS=%q\n' "${REQUEST_TIMEOUT_SECONDS}"
  printf 'ENABLE_AUTO_TOOL_CHOICE=%q\nTOOL_CALL_PARSER=%q\n' \
    "${ENABLE_AUTO_TOOL_CHOICE}" "${TOOL_CALL_PARSER}"
  printf 'CLIENT=%q\n' "aiperf"
  printf 'VLLM_BIN=%q\nVLLM_PYTHON=%q\n' \
    "${VLLM_BIN}" "${VLLM_PYTHON}"
  printf 'VLLM_VERSION=%q\nEXPECTED_VLLM_VERSION=%q\n' \
    "${VLLM_VERSION}" "${EXPECTED_VLLM_VERSION}"
  printf 'AIPERF_ROOT=%q\nAIPERF_BIN=%q\n' \
    "${AIPERF_ROOT}" "${AIPERF_BIN}"
  printf 'PD_AIPERF_INPUT_FILE=%q\nPD_AIPERF_OUTPUT_DIR=%q\n' \
    "${PD_AIPERF_INPUT_FILE}" "${PD_AIPERF_OUTPUT_DIR}"
  printf 'PD_AIPERF_CONCURRENCY=%q\nPD_AIPERF_TIMING_MODE=%q\n' \
    "${PD_AIPERF_CONCURRENCY}" "${PD_AIPERF_TIMING_MODE}"
  printf 'PD_AIPERF_REQUEST_RATE=%q\n' "${PD_AIPERF_REQUEST_RATE}"
  printf 'PD_AIPERF_EXPECTED_REQUESTS=%q\n' \
    "${PD_AIPERF_EXPECTED_REQUESTS}"
  printf 'PAP_NIXL_RUNTIME_MODE=%q\nPAP_NIXL_UCX_VERSION=%q\n' \
    "${PAP_NIXL_RUNTIME_MODE}" "${PAP_NIXL_UCX_VERSION}"
  printf 'NIXL_PLUGIN_DIR=%q\nUCX_PROTO_EMULATION_ENABLE=%q\n' \
    "${NIXL_PLUGIN_DIR}" "${UCX_PROTO_EMULATION_ENABLE}"
  printf 'UCX_CUDA_IPC_ENABLE_GET_ZCOPY=%q\n' \
    "${UCX_CUDA_IPC_ENABLE_GET_ZCOPY}"
  printf 'MIN_KV_TRANSFER_MB_S=%q\n' "${MIN_KV_TRANSFER_MB_S}"
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
    PYTHONSAFEPATH=1 \
    PYTHONPATH= \
    CUDA_VISIBLE_DEVICES="${PREFILL_GPUS[index]}" \
    VLLM_PORT="$((VLLM_PORT_BASE + index * 20))" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    PAP_MODEL_HOOKS=0 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${port}" \
      "${PREFILL_EXECUTION_ARGS[@]}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${PREFILL_MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size 16 \
      "${TOOL_ARGS[@]}" \
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
    PYTHONSAFEPATH=1 \
    PYTHONPATH= \
    CUDA_VISIBLE_DEVICES="${DECODE_GPUS[index]}" \
    VLLM_PORT="$((VLLM_PORT_BASE + (PREFILL_COUNT + index) * 20))" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    PAP_MODEL_HOOKS=0 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${port}" \
      "${DECODE_EXECUTION_ARGS[@]}" \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${DECODE_MAX_NUM_SEQS}" --enable-chunked-prefill \
      --enable-prefix-caching --block-size 16 \
      "${TOOL_ARGS[@]}" \
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

PD_VLLM_GRAPH_LOGS=()
for (( index=0; index<PREFILL_COUNT; index++ )); do
  PD_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/prefill_${index}.log")
done
for (( index=0; index<DECODE_COUNT; index++ )); do
  PD_VLLM_GRAPH_LOGS+=("${LOG_ROOT}/decode_${index}.log")
done
"${CUDA_GRAPH_AUDITOR}" "${RUN_ROOT}/vllm_cuda_graph_audit.env" \
  PIECEWISE "${PD_VLLM_GRAPH_LOGS[@]}"

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
  AIPERF_CUSTOM_DATASET_TYPE="${AIPERF_CUSTOM_DATASET_TYPE:-multi-turn}" \
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
if [[ -n "${PD_AIPERF_EXPECTED_REQUESTS}" ]]; then
  actual_requests="$(jq -r '.request_count.avg' \
    "${PD_AIPERF_OUTPUT_DIR}/profile.json")"
  if [[ "${actual_requests}" != "${PD_AIPERF_EXPECTED_REQUESTS}" \
    && "${actual_requests}" != "${PD_AIPERF_EXPECTED_REQUESTS}.0" ]]; then
    echo "AIPerf completed ${actual_requests}, expected " \
      "${PD_AIPERF_EXPECTED_REQUESTS}" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" "${KV_TRANSFER_ANALYZER}" "${RUN_ROOT}" \
  --block-size 16 --output "${RUN_ROOT}/pd_ttft_analysis.json"
if ! jq -e --argjson minimum "${MIN_KV_TRANSFER_MB_S}" \
  '.kv_transfer.aggregate_throughput_mb_s >= $minimum' \
  "${RUN_ROOT}/pd_ttft_analysis.json" >/dev/null; then
  printf 'STATUS=failed\nMINIMUM_MB_S=%q\n' \
    "${MIN_KV_TRANSFER_MB_S}" > "${RUN_ROOT}/kv_transfer_audit.env"
  echo "PD KV transfer throughput failed the same-node floor" >&2
  exit 1
fi
printf 'STATUS=passed\nMINIMUM_MB_S=%q\n' \
  "${MIN_KV_TRANSFER_MB_S}" > "${RUN_ROOT}/kv_transfer_audit.env"

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
