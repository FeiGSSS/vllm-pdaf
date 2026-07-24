#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${AIPERF_ROOT}/.venv/bin/aiperf}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"

DP_SIZE="${DP_LOAD_SIZE:-8}"
GPU_CSV="${DP_LOAD_GPUS:-$(seq -s, 0 $((DP_SIZE - 1)))}"
RUN_ID="${DP_LOAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_dp${DP_SIZE}}"
EXPERIMENTS_ROOT="${PAP_EXPERIMENTS_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXPERIMENTS_ROOT}/_staging}"
RUN_ROOT="${DP_LOAD_RUN_ROOT:-${RESULTS_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${RUN_ROOT}/service_logs"
PORT="${DP_LOAD_PORT:-24400}"
VLLM_PORT_BASE="${DP_LOAD_VLLM_PORT_BASE:-53600}"

ROUNDS="${DP_LOAD_ROUNDS:-5}"
CONVERSATIONS="${DP_LOAD_CONVERSATIONS:-128}"
CONCURRENCY="${DP_AIPERF_CONCURRENCY:-${CONVERSATIONS}}"
INPUT_FILE="${DP_AIPERF_INPUT_FILE:?DP_AIPERF_INPUT_FILE is required}"
OUTPUT_DIR="${DP_AIPERF_OUTPUT_DIR:-${RUN_ROOT}/aiperf}"
MAX_MODEL_LEN="${DP_LOAD_MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${DP_LOAD_MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${DP_LOAD_MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${DP_LOAD_GPU_MEMORY_UTILIZATION:-0.90}"
EXECUTION_MODE="${DP_LOAD_EXECUTION_MODE:-eager}"
REQUEST_TIMEOUT_SECONDS="${DP_LOAD_REQUEST_TIMEOUT_SECONDS:-600}"
TARGET_URLS=""
for (( rank=0; rank<DP_SIZE; rank++ )); do
  [[ -z "${TARGET_URLS}" ]] || TARGET_URLS+=","
  TARGET_URLS+="http://127.0.0.1:$((PORT + rank))"
done

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${DP_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "DP_LOAD_SIZE must be positive"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] \
  || die "DP_AIPERF_CONCURRENCY must be positive"
(( CONCURRENCY <= CONVERSATIONS )) \
  || die "AIPerf concurrency exceeds total conversations"
IFS=, read -r -a GPUS <<< "${GPU_CSV}"
(( ${#GPUS[@]} == DP_SIZE )) || die "DP_LOAD_GPUS does not match DP_LOAD_SIZE"
for required in "${PYTHON_BIN}" "${VLLM_BIN}" "${AIPERF_BIN}" \
  "${AIPERF_RUNNER}" "${MODEL_PATH}" "${INPUT_FILE}"; do
  [[ -e "${required}" ]] || die "required path is missing: ${required}"
done

EXECUTION_ARGS=(--enforce-eager)
COMPILATION_CONFIG=""
if [[ "${EXECUTION_MODE}" == "piecewise" ]]; then
  COMPILATION_CONFIG='{"mode":"VLLM_COMPILE","cudagraph_mode":"PIECEWISE"}'
  EXECUTION_ARGS=(--compilation-config "${COMPILATION_CONFIG}")
elif [[ "${EXECUTION_MODE}" != "eager" ]]; then
  die "DP_LOAD_EXECUTION_MODE must be eager or piecewise"
fi

PIDS=()
cleanup() {
  local code=$?
  set +e
  for pid in "${PIDS[@]:-}"; do
    kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  done
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for gpu in "${GPUS[@]}"; do
  processes="$(
    nvidia-smi -i "${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  [[ -z "${processes//[[:space:]]/}" ]] \
    || die "GPU ${gpu} is occupied by ${processes}"
done

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
git status --short > "${RUN_ROOT}/git_status.txt"
git diff --binary > "${RUN_ROOT}/tracked_worktree.patch"
git diff --cached --binary > "${RUN_ROOT}/tracked_index.patch"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_TRACKED_WORKTREE_DIRTY=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  GIT_TRACKED_WORKTREE_DIRTY=1
fi

{
  printf 'MODE=dp\nTOPOLOGY=%q\nGPU_COUNT=%q\n' \
    "${DP_SIZE}dp" "${DP_SIZE}"
  printf 'MODEL_PATH=%q\nGPUS=%q\nPORT_BASE=%q\nVLLM_PORT_BASE=%q\n' \
    "${MODEL_PATH}" "${GPU_CSV}" "${PORT}" "${VLLM_PORT_BASE}"
  printf 'ROUTING_POLICY=%q\nTARGET_URLS=%q\n' \
    "aiperf_sticky_user_sessions" "${TARGET_URLS}"
  printf 'ROUNDS=%q\nTOTAL_CONVERSATIONS=%q\nACTIVE_CONVERSATIONS=%q\n' \
    "${ROUNDS}" "${CONVERSATIONS}" "${CONCURRENCY}"
  printf 'MAX_MODEL_LEN=%q\nMAX_NUM_BATCHED_TOKENS=%q\nMAX_NUM_SEQS=%q\n' \
    "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}"
  printf 'GPU_MEMORY_UTILIZATION=%q\nEXECUTION_MODE=%q\n' \
    "${GPU_MEMORY_UTILIZATION}" "${EXECUTION_MODE}"
  printf 'COMPILATION_CONFIG=%q\nAIPERF_INPUT_FILE=%q\n' \
    "${COMPILATION_CONFIG}" "${INPUT_FILE}"
  printf 'GIT_COMMIT=%q\nGIT_TRACKED_WORKTREE_DIRTY=%q\n' \
    "${GIT_COMMIT}" "${GIT_TRACKED_WORKTREE_DIRTY}"
} > "${RUN_ROOT}/effective_config.env"

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

for (( rank=0; rank<DP_SIZE; rank++ )); do
  setsid env \
    CUDA_VISIBLE_DEVICES="${GPUS[rank]}" \
    VLLM_PORT="$((VLLM_PORT_BASE + rank * 20))" \
    PAP_MODEL_HOOKS=0 \
    PAP_CUDAGRAPH_COMPATIBLE=0 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "$((PORT + rank))" \
      "${EXECUTION_ARGS[@]}" \
      --generation-config vllm --dtype float16 \
      --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --enable-chunked-prefill --enable-prefix-caching --block-size 16 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      > "${LOG_ROOT}/dp_${rank}.log" 2>&1 &
  PIDS+=("$!")
done

deadline=$((SECONDS + 900))
for (( rank=0; rank<DP_SIZE; rank++ )); do
  until curl -fsS "http://127.0.0.1:$((PORT + rank))/health" \
    >/dev/null 2>&1; do
    kill -0 "${PIDS[rank]}" >/dev/null 2>&1 \
      || die "DP instance ${rank} exited during startup"
    (( SECONDS < deadline )) || die "timed out waiting for DP instances"
    sleep 2
  done
done

env \
  PAP_ROOT="${ROOT_DIR}" \
  AIPERF_ROOT="${AIPERF_ROOT}" \
  AIPERF_BIN="${AIPERF_BIN}" \
  MODEL_PATH="${MODEL_PATH}" \
  AIPERF_INPUT_FILE="${INPUT_FILE}" \
  AIPERF_TARGET_URLS="${TARGET_URLS}" \
  AIPERF_CONNECTION_REUSE_STRATEGY=sticky-user-sessions \
  AIPERF_OUTPUT_DIR="${OUTPUT_DIR}" \
  AIPERF_SESSIONS="${CONVERSATIONS}" \
  AIPERF_CONCURRENCY="${CONCURRENCY}" \
  AIPERF_TIMING_MODE=concurrency \
  AIPERF_REQUEST_RATE= \
  AIPERF_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
  AIPERF_EXPORT_LEVEL="${AIPERF_EXPORT_LEVEL:-records}" \
  "${AIPERF_RUNNER}" 2>&1 | tee "${RUN_ROOT}/client.log"

if [[ -z "$(find "${OUTPUT_DIR}" -type f \
  -name 'profile*.json' -size +0c -print -quit)" ]]; then
  die "AIPerf produced no profile JSON under ${OUTPUT_DIR}"
fi
if rg -n -i \
  'CUDA out of memory|EngineDeadError|Traceback' \
  "${LOG_ROOT}" > "${RUN_ROOT}/correctness_audit_matches.log"; then
  printf 'STATUS=failed\n' > "${RUN_ROOT}/correctness_audit.env"
  die "DP correctness audit failed"
fi
: > "${RUN_ROOT}/correctness_audit_matches.log"
printf 'STATUS=passed\nMATCH_COUNT=0\n' \
  > "${RUN_ROOT}/correctness_audit.env"
echo "DP_LOAD_RUN_ROOT=${RUN_ROOT}"
