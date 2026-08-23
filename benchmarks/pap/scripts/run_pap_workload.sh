#!/usr/bin/env bash
set -euo pipefail

# PAP service/workload runner used by project-owned benchmark entrypoints.

for removed_flag in \
  PAP_ASYNC_DECODE_TOKEN \
  PAP_PREFILL_KV_ASYNC \
  PAP_KV_HANDOFF_MODE \
  PAP_UNIFIED_KV \
  PAP_BATCHED_ROUTE_COPY \
  PAP_UNIFIED_MD_FAST_KEY \
  PAP_ATTENTION_DISPATCH_MODE \
  PAP_ATTENTION_COMBINE_WAIT_US \
  PAP_ATTENTION_ACTIVE_PEER_TRACKING \
  PAP_ATTENTION_MAILBOX_PREFETCH \
  PAP_RUNNER_MICROBATCH_COUNT \
  PAP_MPS_MODE \
  PAP_BENCH_MPS_PROFILE \
  PAP_ASYNC_DECODE_TOKEN_SYNC_ONLY_BARRIER \
  PAP_PROJECTION_SYNC_ONLY_BARRIER \
  PAP_PREFILL_SYNC_ONLY_BARRIER \
  PAP_DIAG_R1_PROJECTION_GATE_COUNT \
  PAP_DIAG_R1_COMMIT_GATE_COUNT \
  PAP_DIAG_DECODE_COMMIT_GATE_FILE \
  PAP_DIAG_DECODE_COMMIT_GATE_TIMEOUT \
  PAP_BENCH_CLIENT_MODE \
  PAP_MULTITURN_LOAD_ROUNDS \
  PAP_MULTITURN_LOAD_CONVERSATIONS \
  PAP_MULTITURN_LOAD_REQUEST_RATE \
  PAP_MULTITURN_APPEND_TOKENS \
  PAP_MULTITURN_FIRST_OUTPUT_TOKENS \
  PAP_MULTITURN_BLOCK_SIZE \
  PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS; do
  if [[ -v "${removed_flag}" ]]; then
    case "${removed_flag}" in
      PAP_ASYNC_DECODE_TOKEN)
        replacement="unconditional asynchronous sampled-token delivery"
        experiment_id="PAP-20260713-ASYNC-DECODE-TOKEN-D2H"
        ;;
      PAP_PREFILL_KV_ASYNC)
        replacement="unconditional safe asynchronous Prefill KV import"
        experiment_id="PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC"
        ;;
      PAP_KV_HANDOFF_MODE)
        replacement="sealed catalog and request manifest handoff"
        experiment_id="PAP-20260714-SEAL-HANDOFF-KV"
        ;;
      PAP_UNIFIED_KV | PAP_BATCHED_ROUTE_COPY | PAP_UNIFIED_MD_FAST_KEY)
        replacement="the current PAP data path"
        experiment_id="PAP-20260703-UNIFIED-KV"
        ;;
      PAP_ATTENTION_MAILBOX_PREFETCH)
        replacement="the serial Attention receive loop"
        experiment_id="PAP-20260701-ATTENTION-MAILBOX-PREFETCH"
        ;;
      PAP_ATTENTION_*)
        replacement="topology-derived Attention execution"
        experiment_id="PAP-20260711-ATTENTION-COMBINE"
        ;;
      PAP_RUNNER_MICROBATCH_COUNT)
        replacement="one unsplit vLLM Projection scheduler batch"
        experiment_id="PAP-20260724-SINGLE-PROJECTION-BATCH"
        ;;
      PAP_MPS_MODE | PAP_BENCH_MPS_PROFILE)
        replacement="the AIPerf static 80/12 MPS partition"
        experiment_id="PAP-20260714-ASYNC-STATIC-BASELINE"
        ;;
      PAP_*_SYNC_ONLY_BARRIER)
        replacement="no Projection or Prefill timing barrier"
        experiment_id="PAP-20260714-ASYNC-TTFT-ROOTCAUSE"
        ;;
      PAP_BENCH_CLIENT_MODE | PAP_MULTITURN_*)
        replacement="the AIPerf-only PAP benchmark interface"
        experiment_id="PAP-20260722-AIPERF-CANONICAL-CUTOVER"
        ;;
      *)
        replacement="unconditional decode-commit delivery"
        experiment_id="PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION"
        ;;
    esac
    echo "ERROR: ${removed_flag} was removed; use ${replacement}. Historical evidence: ${experiment_id}." >&2
    exit 2
  fi
done

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
CUDA_GRAPH_AUDITOR="${ROOT_DIR}/benchmarks/pap/scripts/audit_cuda_graph_logs.sh"
DEFERRED_TRACE_VALIDATOR="${ROOT_DIR}/benchmarks/pap/tooling/validate_deferred_trace.py"
PROJECTION_MEMORY_PLANNER="${ROOT_DIR}/vllm/pap/model/memory.py"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
AIPERF_DATASET_GENERATOR="${ROOT_DIR}/benchmarks/pap/aiperf/generate_multiturn_dataset.py"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}"
PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE:-0}"
PAP_BENCH_STRICT_CORRECTNESS_AUDIT="${PAP_BENCH_STRICT_CORRECTNESS_AUDIT:-1}"
PAP_BENCH_CLIENT="aiperf"

GIT_COMMIT=""
GIT_COMMIT_SHORT=""
GIT_TRACKED_WORKTREE_DIRTY=0

MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
BENCH_DIR="${BENCH_DIR:-/home/fei/research/PD/refer_codes/vllm/benchmarks}"
DATASET_PATH="${DATASET_PATH:-${BENCH_DIR}/sonnet_4x.txt}"
ENABLE_AUTO_TOOL_CHOICE="${PAP_BENCH_ENABLE_AUTO_TOOL_CHOICE:-0}"
TOOL_CALL_PARSER="${PAP_BENCH_TOOL_CALL_PARSER:-}"

INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PAP_AIPERF_TURNS="${PAP_AIPERF_TURNS:-10}"
PAP_AIPERF_SESSIONS="${PAP_AIPERF_SESSIONS:-32}"
PAP_AIPERF_VARIABLE_TURNS="${PAP_AIPERF_VARIABLE_TURNS:-0}"
PAP_AIPERF_APPEND_TOKENS="${PAP_AIPERF_APPEND_TOKENS:-512}"
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
if ! [[ "${PAP_AIPERF_TURNS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_AIPERF_SESSIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: AIPerf turns/sessions must be positive" >&2
  exit 2
fi
REQUESTS_PER_AIPERF_VARIATION=$((
  PAP_AIPERF_TURNS * PAP_AIPERF_SESSIONS
))
if [[ -n "${PAP_AIPERF_EXPECTED_REQUESTS:-}" ]]; then
  [[ "${PAP_AIPERF_EXPECTED_REQUESTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: PAP_AIPERF_EXPECTED_REQUESTS must be positive" >&2
    exit 2
  }
  REQUESTS_PER_AIPERF_VARIATION="${PAP_AIPERF_EXPECTED_REQUESTS}"
fi
[[ "${PAP_AIPERF_VARIABLE_TURNS}" =~ ^[01]$ ]] || {
  echo "ERROR: PAP_AIPERF_VARIABLE_TURNS must be 0 or 1" >&2
  exit 2
}
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
CLUSTER_READY_WAIT_SECONDS="${CLUSTER_READY_WAIT_SECONDS:-30}"
PAP_BENCH_SESSION_DRAIN_TIMEOUT="${PAP_BENCH_SESSION_DRAIN_TIMEOUT:-15}"
PAP_BENCH_GATEWAY_DRAIN_TIMEOUT="${PAP_BENCH_GATEWAY_DRAIN_TIMEOUT:-120}"
PAP_DEFERRED_TRACE_FLUSH_TIMEOUT="${PAP_DEFERRED_TRACE_FLUSH_TIMEOUT:-30}"

TOPOLOGY="${PAP_TOPOLOGY:-3pa1p}"
if [[ ! "${TOPOLOGY}" =~ ^([0-9]+)pa([0-9]+)p$ ]]; then
  echo "ERROR: unsupported PAP topology: ${TOPOLOGY}" >&2
  exit 2
fi
PA_COUNT="${PAP_PA_COUNT:-${BASH_REMATCH[1]}}"
PROJECTION_COUNT="${PAP_PROJECTION_COUNT:-${BASH_REMATCH[2]}}"
if [[ ! "${PA_COUNT}" =~ ^[1-9][0-9]*$ \
  || ! "${PROJECTION_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAP topology counts must be positive: ${TOPOLOGY}" >&2
  exit 2
fi
TOPOLOGY_TAG="$(printf '%s' "${TOPOLOGY}" | tr '[:lower:]' '[:upper:]')"
EXPERIMENTS_ROOT="${PAP_EXPERIMENTS_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXPERIMENTS_ROOT}/_staging}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_ROOT}/service_logs}"
AIPERF_INPUT_FILE_PROVIDED=0
if [[ -n "${PAP_AIPERF_INPUT_FILE:-${AIPERF_INPUT_FILE:-}}" ]]; then
  AIPERF_INPUT_FILE_PROVIDED=1
fi
PAP_AIPERF_INPUT_FILE="${PAP_AIPERF_INPUT_FILE:-${AIPERF_INPUT_FILE:-${RUN_ROOT}/aiperf_multiturn.jsonl}}"
PAP_AIPERF_OUTPUT_DIR="${PAP_AIPERF_OUTPUT_DIR:-${RUN_ROOT}/aiperf}"
if [[ "${AIPERF_INPUT_FILE_PROVIDED}" == "1" \
  && "${PAP_AIPERF_VARIABLE_TURNS}" == "1" \
  && -z "${PAP_AIPERF_EXPECTED_REQUESTS:-}" ]]; then
  [[ -f "${PAP_AIPERF_INPUT_FILE}" ]] || {
    echo "ERROR: AIPerf input file is missing: ${PAP_AIPERF_INPUT_FILE}" >&2
    exit 2
  }
  REQUESTS_PER_AIPERF_VARIATION="$(
    jq -er '[.[].turns | length] | add | select(. > 0)' \
      "${PAP_AIPERF_INPUT_FILE}"
  )" || {
    echo "ERROR: cannot count turns in ${PAP_AIPERF_INPUT_FILE}" >&2
    exit 2
  }
fi
PAP_AIPERF_CONCURRENCY="${PAP_AIPERF_CONCURRENCY:-12}"
PAP_AIPERF_TIMING_MODE="${PAP_AIPERF_TIMING_MODE:-concurrency}"
PAP_AIPERF_REQUEST_RATE="${PAP_AIPERF_REQUEST_RATE-}"
if [[ ! "${PAP_AIPERF_CONCURRENCY}" \
  =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "ERROR: PAP_AIPERF_CONCURRENCY must be a positive integer or CSV list" >&2
  exit 2
fi
IFS=, read -r -a PAP_AIPERF_CONCURRENCY_POINTS \
  <<< "${PAP_AIPERF_CONCURRENCY}"
for concurrency in "${PAP_AIPERF_CONCURRENCY_POINTS[@]}"; do
  if (( concurrency > PAP_AIPERF_SESSIONS )); then
    echo "ERROR: AIPerf concurrency exceeds total sessions" >&2
    exit 2
  fi
done
AIPERF_NUM_PROFILE_RUNS="${AIPERF_NUM_PROFILE_RUNS:-1}"
if [[ ! "${AIPERF_NUM_PROFILE_RUNS}" =~ ^[1-9][0-9]*$ ]] \
  || (( AIPERF_NUM_PROFILE_RUNS > 10 )); then
  echo "ERROR: AIPERF_NUM_PROFILE_RUNS must be between 1 and 10" >&2
  exit 2
fi
AIPERF_VARIATION_COUNT="${#PAP_AIPERF_CONCURRENCY_POINTS[@]}"
NUM_PROMPTS=$((
  REQUESTS_PER_AIPERF_VARIATION
  * AIPERF_VARIATION_COUNT
  * AIPERF_NUM_PROFILE_RUNS
))
if [[ "${PAP_AIPERF_TIMING_MODE}" == "request_rate" \
  && -z "${PAP_AIPERF_REQUEST_RATE}" ]]; then
  echo "ERROR: request_rate timing requires PAP_AIPERF_REQUEST_RATE" >&2
  exit 2
fi
PAP_PROXY_PORT="${PAP_PROXY_PORT:-9460}"
PREFILL_PORT_BASE="${PAP_PREFILL_PORT_BASE:-${PAP_PREFILL_PORT:-8100}}"
PROJECTION_PORT_BASE="${PAP_PROJECTION_PORT_BASE:-${PAP_PROJECTION_PORT:-8200}}"
ATTENTION_PORT_BASE="${PAP_ATTENTION_PORT_BASE:-${PAP_ATTENTION_PORT:-8300}}"
ATTENTION_TCP_PORT_BASE="${PAP_ATTENTION_TCP_PORT_BASE:-${PAP_ATTENTION_TCP_PORT:-9300}}"
VLLM_PREFILL_PORT_BASE="${PAP_VLLM_PREFILL_PORT_BASE:-${PAP_VLLM_PREFILL_PORT:-50000}}"
VLLM_PROJECTION_PORT_BASE="${PAP_VLLM_PROJECTION_PORT_BASE:-${PAP_VLLM_PROJECTION_PORT:-$((VLLM_PREFILL_PORT_BASE + PA_COUNT * 20))}}"

if (( PA_COUNT == 1 && PROJECTION_COUNT == 1 )); then
  DEFAULT_PREFILL_GPUS=1
  DEFAULT_PROJECTION_GPUS=2
else
  DEFAULT_PREFILL_GPUS="$(seq -s, 0 $((PA_COUNT - 1)))"
  DEFAULT_PROJECTION_GPUS="$(seq -s, "${PA_COUNT}" $((PA_COUNT + PROJECTION_COUNT - 1)))"
fi
PAP_PREFILL_GPUS="${PAP_PREFILL_GPUS:-${DEFAULT_PREFILL_GPUS}}"
PAP_PROJECTION_GPUS="${PAP_PROJECTION_GPUS:-${DEFAULT_PROJECTION_GPUS}}"
PAP_TP_SIZE="${PAP_TP_SIZE:-1}"
PAP_VLLM_DTYPE="${PAP_VLLM_DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS:-2048}"
PAP_PREFILL_MAX_NUM_SEQS="${PAP_PREFILL_MAX_NUM_SEQS:-256}"
PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS="${PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS:-256}"
PAP_PROJECTION_MAX_NUM_SEQS="${PAP_PROJECTION_MAX_NUM_SEQS:-256}"
PAP_PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.90}"
PAP_STATIC_PREFILL_CHUNKS="${PAP_STATIC_PREFILL_CHUNKS:-20}"
PAP_STATIC_ATTENTION_CHUNKS="${PAP_STATIC_ATTENTION_CHUNKS:-3}"
PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_STATIC_PREFILL_EXPECTED_SMS:-80}"
PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_STATIC_ATTENTION_EXPECTED_SMS:-12}"
PAP_ENABLE_MPS=1
if ! [[ "${PAP_STATIC_PREFILL_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_STATIC_ATTENTION_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_STATIC_PREFILL_EXPECTED_SMS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_STATIC_ATTENTION_EXPECTED_SMS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: static MPS chunk and SM counts must be positive integers" >&2
  exit 2
fi
if (( PAP_STATIC_PREFILL_CHUNKS + PAP_STATIC_ATTENTION_CHUNKS != 23 )); then
  echo "ERROR: PAP AIPerf requires all 23 L20 MPS chunks" >&2
  exit 2
fi
if (( PAP_STATIC_PREFILL_EXPECTED_SMS != PAP_STATIC_PREFILL_CHUNKS * 4 \
  || PAP_STATIC_ATTENTION_EXPECTED_SMS != PAP_STATIC_ATTENTION_CHUNKS * 4 )); then
  echo "ERROR: PAP AIPerf expects four visible SMs per L20 MPS chunk" >&2
  exit 2
fi
PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"
PAP_NVSHMEM_WORLD_SIZE=$((PA_COUNT + PROJECTION_COUNT))
PAP_NVSHMEM_UID_FILE="${PAP_NVSHMEM_UID_FILE:-${RUN_ROOT}/nvshmem.uid}"
PAP_NVSHMEM_INIT_TIMEOUT="${PAP_NVSHMEM_INIT_TIMEOUT:-${SERVER_START_TIMEOUT}}"
if (( PROJECTION_COUNT != 1 )); then
  echo "ERROR: PAP NVSHMEM whole-step Graph requires one Projection" >&2
  exit 2
fi
if (( PAP_TP_SIZE != 1 )); then
  echo "ERROR: PAP NVSHMEM whole-step Graph requires PAP_TP_SIZE=1" >&2
  exit 2
fi
source "${ROOT_DIR}/benchmarks/pap/scripts/configure_nvshmem.sh"
pap_configure_nvshmem "${ROOT_DIR}"
export PAP_NVSHMEM_INIT_TIMEOUT
PAP_OFFLOAD_EXEC_TRACE="${PAP_OFFLOAD_EXEC_TRACE:-0}"
PAP_OFFLOAD_EXEC_TRACE_LAYER="${PAP_OFFLOAD_EXEC_TRACE_LAYER:-}"
PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE:-0}"
PAP_DEFERRED_CUDA_TRACE_MAX_PENDING="${PAP_DEFERRED_CUDA_TRACE_MAX_PENDING:-1024}"
PAP_UNIFIED_MD_CACHE_LIMIT="${PAP_UNIFIED_MD_CACHE_LIMIT:-256}"
PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT:-256}"
PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY:-conversation_affinity}"
PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS:-64}"
PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS:-1}"
PAP_PREFIX_CACHE_AUDIT="${PAP_PREFIX_CACHE_AUDIT:-0}"
PAP_BLOCK_SIZE="${PAP_BLOCK_SIZE:-16}"
PAP_DECODE_COMMIT_ENDPOINT="${PAP_DECODE_COMMIT_ENDPOINT:-}"
PAP_LEASE_RELEASE_ENDPOINT="${PAP_LEASE_RELEASE_ENDPOINT:-}"
PAP_DECODE_COMMIT_FAIL_CLOSED="${PAP_DECODE_COMMIT_FAIL_CLOSED:-1}"
PAP_DECODE_COMMIT_TIMEOUT="${PAP_DECODE_COMMIT_TIMEOUT:-5.0}"
PAP_DECODE_COMMIT_QUEUE_SIZE="${PAP_DECODE_COMMIT_QUEUE_SIZE:-1024}"
PAP_DECODE_COMMIT_MAX_ATTEMPTS="${PAP_DECODE_COMMIT_MAX_ATTEMPTS:-8}"
PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS="${PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS:-0.05}"
PAP_DECODE_COMMIT_RETRY_MAX_SECONDS="${PAP_DECODE_COMMIT_RETRY_MAX_SECONDS:-0.5}"
PAP_DECODE_COMMIT_FLUSH_TIMEOUT="${PAP_DECODE_COMMIT_FLUSH_TIMEOUT:-15.0}"
PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE:-0}"
PAP_RUNTIME_CUDA_CONTEXT_AUDIT="${PAP_RUNTIME_CUDA_CONTEXT_AUDIT:-0}"
PAP_PREFILL_TORCH_PROFILE="${PAP_PREFILL_TORCH_PROFILE:-0}"
PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS="${PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS:-32}"
PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT="${PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT:-120}"
PAP_DECODE_TOKEN_TIMEOUT="${PAP_DECODE_TOKEN_TIMEOUT:-0.2}"
PAP_DECODE_TOKEN_QUEUE_SIZE="${PAP_DECODE_TOKEN_QUEUE_SIZE:-1024}"
PAP_DECODE_TOKEN_MAX_ATTEMPTS="${PAP_DECODE_TOKEN_MAX_ATTEMPTS:-8}"
PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS="${PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS:-0.05}"
PAP_DECODE_TOKEN_RETRY_MAX_SECONDS="${PAP_DECODE_TOKEN_RETRY_MAX_SECONDS:-0.5}"
PAP_DECODE_TOKEN_FLUSH_TIMEOUT="${PAP_DECODE_TOKEN_FLUSH_TIMEOUT:-5.0}"
PAP_LEASE_RELEASE_TIMEOUT="${PAP_LEASE_RELEASE_TIMEOUT:-5.0}"
PAP_LEASE_RELEASE_MAX_ATTEMPTS="${PAP_LEASE_RELEASE_MAX_ATTEMPTS:-5}"
PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS="${PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS:-0.05}"
PAP_LEASE_RELEASE_RETRY_MAX_SECONDS="${PAP_LEASE_RELEASE_RETRY_MAX_SECONDS:-0.5}"
PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS:-300}"

case "${PAP_DEFERRED_CUDA_TRACE,,}" in
  1|true|yes|on)
    case "${PAP_OFFLOAD_EXEC_TRACE,,}" in
      0|false|no|off) ;;
      *)
        echo "PAP_DEFERRED_CUDA_TRACE requires PAP_OFFLOAD_EXEC_TRACE=0" >&2
        exit 2
        ;;
    esac
    ;;
esac
if ! [[ "${PAP_DEFERRED_CUDA_TRACE_MAX_PENDING}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PAP_DEFERRED_CUDA_TRACE_MAX_PENDING must be a positive integer" >&2
  exit 2
fi
if [[ -n "${PAP_OFFLOAD_EXEC_TRACE_LAYER}" ]] \
  && ! [[ "${PAP_OFFLOAD_EXEC_TRACE_LAYER}" =~ ^[0-9]+$ ]]; then
  echo "PAP_OFFLOAD_EXEC_TRACE_LAYER must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${PAP_PREFILL_TORCH_PROFILE}" =~ ^[01]$ ]]; then
  echo "PAP_PREFILL_TORCH_PROFILE must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${PAP_PREFILL_IPC_PROFILE}" =~ ^[01]$ ]]; then
  echo "PAP_PREFILL_IPC_PROFILE must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PAP Prefill torch-profile limits must be positive integers" >&2
  exit 2
fi
if ! [[ "${PAP_DEFERRED_TRACE_FLUSH_TIMEOUT}" =~ ^[1-9][0-9]*$ \
  && "${PAP_BENCH_GATEWAY_DRAIN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PAP drain and trace timeouts must be positive integers" >&2
  exit 2
fi
PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES="${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
for capture_sizes in "${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES}"; do
  if ! [[ "${capture_sizes}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "ERROR: CUDA Graph capture sizes must be positive integer CSV" >&2
    exit 2
  fi
done
for incompatible_flag in \
  PAP_OFFLOAD_EXEC_TRACE \
  PAP_DEFERRED_CUDA_TRACE \
  PAP_PREFILL_TORCH_PROFILE; do
  case "${!incompatible_flag:-0}" in
    1 | true | True | TRUE | yes | Yes | YES | on | On | ON)
      echo "ERROR: ${incompatible_flag} is incompatible with the audited CUDA Graph lane" >&2
      exit 2
      ;;
  esac
done
PAP_PREFILL_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
PAP_PROJECTION_COMPILATION_CONFIG='{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}'
PREFILL_EXECUTION_ARGS=(
  --compilation-config "${PAP_PREFILL_COMPILATION_CONFIG}"
)
PROJECTION_EXECUTION_ARGS=(
  --compilation-config "${PAP_PROJECTION_COMPILATION_CONFIG}"
)

export PAP_OFFLOAD_EXEC_TRACE
export PAP_OFFLOAD_EXEC_TRACE_LAYER
export PAP_DEFERRED_CUDA_TRACE
export PAP_DEFERRED_CUDA_TRACE_MAX_PENDING
export PAP_UNIFIED_MD_CACHE_LIMIT
export PAP_DECODE_SLOT_PLAN_CACHE_LIMIT
export PAP_BLOCK_SIZE
export PAP_DECODE_COMMIT_FAIL_CLOSED
export PAP_DECODE_COMMIT_TIMEOUT
export PAP_DECODE_COMMIT_QUEUE_SIZE
export PAP_DECODE_COMMIT_MAX_ATTEMPTS
export PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS
export PAP_DECODE_COMMIT_RETRY_MAX_SECONDS
export PAP_DECODE_COMMIT_FLUSH_TIMEOUT
export PAP_PREFILL_IPC_PROFILE
export PAP_RUNTIME_CUDA_CONTEXT_AUDIT
export PAP_PREFILL_TORCH_PROFILE
export PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS
export PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT
export PAP_DECODE_TOKEN_TIMEOUT
export PAP_DECODE_TOKEN_QUEUE_SIZE
export PAP_DECODE_TOKEN_MAX_ATTEMPTS
export PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS
export PAP_DECODE_TOKEN_RETRY_MAX_SECONDS
export PAP_DECODE_TOKEN_FLUSH_TIMEOUT
export PAP_LEASE_RELEASE_TIMEOUT
export PAP_LEASE_RELEASE_MAX_ATTEMPTS
export PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS
export PAP_LEASE_RELEASE_RETRY_MAX_SECONDS
export PAP_KV_LEASE_TTL_SECONDS

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

append_no_proxy() {
  local var_name="$1"
  local current="${!var_name:-}"
  local host
  for host in 127.0.0.1 localhost; do
    case ",${current}," in
      *",${host},"*) ;;
      *) current="${current:+${current},}${host}" ;;
    esac
  done
  printf -v "${var_name}" '%s' "${current}"
  export "${var_name}"
}

append_no_proxy NO_PROXY
append_no_proxy no_proxy

PREFILL_OBSERVABILITY_ARGS=()
case "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" in
  1 | true | True | TRUE | yes | Yes | YES)
    PREFILL_OBSERVABILITY_ARGS+=("--enable-prompt-tokens-details")
    ;;
  0 | false | False | FALSE | no | No | NO)
    ;;
  *)
    echo "ERROR: PAP_ENABLE_PROMPT_TOKENS_DETAILS must be 0 or 1" >&2
    exit 2
    ;;
esac

PIDS=()
PREFILL_GPUS=()
PROJECTION_GPUS=()
MPS_PIPE_DIRS=()
MPS_LOG_DIRS=()
MPS_STARTED_DIRS=()
MPS_GPU_UUIDS=()
MPS_PREFILL_PARTITIONS=()
MPS_ATTENTION_PARTITIONS=()
MPS_PREFILL_VISIBLE_SMS=()
MPS_ATTENTION_VISIBLE_SMS=()
MPS_PIPE_BASE_DIR="${PAP_MPS_PIPE_BASE_DIR:-/tmp/pap-mps-${USER:-user}-${TOPOLOGY}-$$}"
MPS_LOG_BASE_DIR="${PAP_MPS_LOG_BASE_DIR:-${RUN_LOG_DIR}/mps-log}"

split_csv() {
  local csv="$1"
  local -n output="$2"
  IFS=',' read -r -a output <<< "${csv}"
}

require_count() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  (( actual >= expected )) \
    || die "${name} has ${actual} entries but needs at least ${expected}"
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

build_pap_groups_spec() {
  local items=()
  local idx
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    items+=(
      "127.0.0.1:$((PREFILL_PORT_BASE + idx)):127.0.0.1:$((ATTENTION_PORT_BASE + idx)):$((ATTENTION_TCP_PORT_BASE + idx))"
    )
  done
  join_by_comma "${items[@]}"
}

build_projections_spec() {
  local items=()
  local idx
  for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    items+=("127.0.0.1:$((PROJECTION_PORT_BASE + idx))")
  done
  join_by_comma "${items[@]}"
}

mps_control() {
  local pipe_dir="$1"
  local command="$2"
  timeout 10 env \
    CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" \
    nvidia-cuda-mps-control <<< "${command}"
}

parse_created_partition_id() {
  local response="$1"
  local line partition_id
  while IFS= read -r line; do
    line="${line%$'\r'}"
    if [[ "${line}" == Partition\ *\ created ]]; then
      partition_id="${line#Partition }"
      partition_id="${partition_id% created}"
      [[ -n "${partition_id}" ]] || return 1
      printf '%s\n' "${partition_id}"
      return 0
    fi
  done <<< "${response}"
  return 1
}

create_static_partition() {
  local pipe_dir="$1"
  local chunks="$2"
  local gpu_uuid="$3"
  local response partition_id
  if ! response="$(
    mps_control \
      "${pipe_dir}" "sm_partition add ${gpu_uuid} ${chunks}" 2>&1
  )"; then
    echo "static MPS partition creation failed: ${response}" >&2
    return 1
  fi
  if ! partition_id="$(parse_created_partition_id "${response}")"; then
    echo "unexpected static MPS create response: ${response}" >&2
    return 1
  fi
  printf '%s\n' "${partition_id}"
}

validate_static_partition_visible_sms() {
  local idx="$1"
  local role="$2"
  local partition_id="$3"
  local expected_sms="$4"
  local probe_log="${RUN_LOG_DIR}/mps_static_${idx}_${role}_probe.log"
  local visible_sms
  if ! visible_sms="$(
    env \
      CUDA_VISIBLE_DEVICES=0 \
      CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIRS[idx]}" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIRS[idx]}" \
      CUDA_MPS_SM_PARTITION="${partition_id}" \
      "${PYTHON_BIN}" -c \
        'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' \
      2> "${probe_log}"
  )"; then
    sed -n '1,120p' "${probe_log}" >&2 || true
    die "static MPS ${role} visibility probe failed for PA ${idx}"
  fi
  if [[ "${visible_sms}" != "${expected_sms}" ]]; then
    die "static MPS ${role} on PA ${idx} exposed ${visible_sms} SMs; expected ${expected_sms}"
  fi
  case "${role}" in
    prefill) MPS_PREFILL_VISIBLE_SMS[idx]="${visible_sms}" ;;
    attention) MPS_ATTENTION_VISIBLE_SMS[idx]="${visible_sms}" ;;
    *) die "unknown static MPS role: ${role}" ;;
  esac
}

write_static_mps_audit() {
  local idx="$1"
  local lspart gpu_uuid prefill_partition_id attention_partition_id
  if ! lspart="$(mps_control "${MPS_PIPE_DIRS[idx]}" lspart 2>&1)"; then
    die "static MPS lspart failed for PA ${idx}: ${lspart}"
  fi
  gpu_uuid="${MPS_GPU_UUIDS[idx]}"
  [[ "${MPS_PREFILL_PARTITIONS[idx]}" == "${gpu_uuid}/"* ]] \
    || die "invalid PA ${idx} Prefill partition ID"
  [[ "${MPS_ATTENTION_PARTITIONS[idx]}" == "${gpu_uuid}/"* ]] \
    || die "invalid PA ${idx} Attention partition ID"
  prefill_partition_id="${MPS_PREFILL_PARTITIONS[idx]#"${gpu_uuid}/"}"
  attention_partition_id="${MPS_ATTENTION_PARTITIONS[idx]#"${gpu_uuid}/"}"
  [[ "${lspart}" == *"${prefill_partition_id}"* ]] \
    || die "static MPS lspart omitted PA ${idx} Prefill partition"
  [[ "${lspart}" == *"${attention_partition_id}"* ]] \
    || die "static MPS lspart omitted PA ${idx} Attention partition"
  {
    printf 'MPS_MODE=static\n'
    printf 'PHYSICAL_GPU_INDEX=%q\n' "${PREFILL_GPUS[idx]}"
    printf 'GPU_UUID=%q\n' "${MPS_GPU_UUIDS[idx]}"
    printf 'PREFILL_PARTITION_ID=%q\n' "${MPS_PREFILL_PARTITIONS[idx]}"
    printf 'ATTENTION_PARTITION_ID=%q\n' \
      "${MPS_ATTENTION_PARTITIONS[idx]}"
    printf 'PREFILL_CHUNKS=%q\n' "${PAP_STATIC_PREFILL_CHUNKS}"
    printf 'ATTENTION_CHUNKS=%q\n' "${PAP_STATIC_ATTENTION_CHUNKS}"
    printf 'PREFILL_VISIBLE_SMS=%q\n' "${MPS_PREFILL_VISIBLE_SMS[idx]}"
    printf 'ATTENTION_VISIBLE_SMS=%q\n' \
      "${MPS_ATTENTION_VISIBLE_SMS[idx]}"
    printf 'LSPART_OUTPUT=%q\n' "${lspart}"
  } > "${RUN_ROOT}/mps_static_audit_pa_${idx}.env"
}

audit_runtime_cuda_contexts() {
  [[ "${PAP_RUNTIME_CUDA_CONTEXT_AUDIT}" == "1" ]] || return 0
  local idx role path expected_sms expected_partition actual_sms
  local actual_partition attempts lspart
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    for role in prefill attention; do
      path="${RUN_ROOT}/runtime_cuda_context_${role}_${idx}.json"
      attempts=0
      while [[ ! -s "${path}" ]] && (( attempts < 100 )); do
        sleep 0.1
        attempts=$((attempts + 1))
      done
      [[ -s "${path}" ]] \
        || die "missing live ${role} CUDA-context audit for PA ${idx}"
      if [[ "${role}" == "prefill" ]]; then
        expected_sms="${PAP_STATIC_PREFILL_EXPECTED_SMS}"
        expected_partition="${MPS_PREFILL_PARTITIONS[idx]}"
      else
        expected_sms="${PAP_STATIC_ATTENTION_EXPECTED_SMS}"
        expected_partition="${MPS_ATTENTION_PARTITIONS[idx]}"
      fi
      actual_sms="$(jq -r '.multiprocessor_count' "${path}")"
      actual_partition="$(jq -r '.cuda_mps_sm_partition' "${path}")"
      [[ "${actual_sms}" == "${expected_sms}" ]] \
        || die "live ${role} context exposed ${actual_sms} SMs; expected ${expected_sms}"
      [[ "${actual_partition}" == "${expected_partition}" ]] \
        || die "live ${role} context used unexpected static MPS partition"
    done
    lspart="$(mps_control "${MPS_PIPE_DIRS[idx]}" lspart)" \
      || die "live static MPS lspart failed for PA ${idx}"
    {
      printf 'STATUS=passed\n'
      printf 'PA_INDEX=%q\n' "${idx}"
      printf 'PREFILL_VISIBLE_SMS=%q\n' \
        "${PAP_STATIC_PREFILL_EXPECTED_SMS}"
      printf 'ATTENTION_VISIBLE_SMS=%q\n' \
        "${PAP_STATIC_ATTENTION_EXPECTED_SMS}"
      printf 'LSPART_OUTPUT=%q\n' "${lspart}"
    } > "${RUN_ROOT}/runtime_cuda_context_audit_pa_${idx}.env"
  done
}

start_mps_for_pa() {
  local idx="$1"
  local gpu="$2"
  local pipe_dir="${MPS_PIPE_BASE_DIR}/pa-${idx}"
  local log_dir="${MPS_LOG_BASE_DIR}/pa-${idx}"
  local gpu_uuid prefill_partition attention_partition
  mkdir -p "${pipe_dir}" "${log_dir}"
  MPS_PIPE_DIRS[idx]="${pipe_dir}"
  MPS_LOG_DIRS[idx]="${log_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" \
  CUDA_MPS_LOG_DIRECTORY="${log_dir}" \
  nvidia-cuda-mps-control -d -S
  MPS_STARTED_DIRS+=("${pipe_dir}")

  if ! gpu_uuid="$(
    nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader
  )"; then
    die "failed to resolve UUID for PA ${idx} GPU ${gpu}"
  fi
  gpu_uuid="${gpu_uuid//[[:space:]]/}"
  [[ "${gpu_uuid}" == GPU-* ]] \
    || die "invalid UUID for PA ${idx} GPU ${gpu}: ${gpu_uuid}"
  MPS_GPU_UUIDS[idx]="${gpu_uuid}"

  prefill_partition="$(
    create_static_partition \
      "${pipe_dir}" "${PAP_STATIC_PREFILL_CHUNKS}" "${gpu_uuid}"
  )" || die "failed to create PA ${idx} Prefill static MPS partition"
  MPS_PREFILL_PARTITIONS[idx]="${prefill_partition}"
  attention_partition="$(
    create_static_partition \
      "${pipe_dir}" "${PAP_STATIC_ATTENTION_CHUNKS}" "${gpu_uuid}"
  )" || die "failed to create PA ${idx} Attention static MPS partition"
  MPS_ATTENTION_PARTITIONS[idx]="${attention_partition}"

  validate_static_partition_visible_sms \
    "${idx}" prefill "${prefill_partition}" \
    "${PAP_STATIC_PREFILL_EXPECTED_SMS}"
  validate_static_partition_visible_sms \
    "${idx}" attention "${attention_partition}" \
    "${PAP_STATIC_ATTENTION_EXPECTED_SMS}"
  write_static_mps_audit "${idx}"
}

cleanup() {
  local code=$?
  set +e
  local pid idx pipe_dir partition_id full_partition_id gpu_uuid cleanup_log
  local cleanup_failed=0
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    kill -0 "${pid}" >/dev/null 2>&1 && kill -KILL "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  for (( idx=${#MPS_STARTED_DIRS[@]} - 1; idx>=0; idx-- )); do
      pipe_dir="${MPS_STARTED_DIRS[idx]}"
      cleanup_log="${RUN_LOG_DIR}/mps_static_cleanup_pa_${idx}.log"
      : > "${cleanup_log}"
      gpu_uuid="${MPS_GPU_UUIDS[idx]:-}"
      full_partition_id="${MPS_ATTENTION_PARTITIONS[idx]:-}"
      if [[ -n "${full_partition_id}" ]]; then
        if [[ "${full_partition_id}" != "${gpu_uuid}/"* ]]; then
          echo "WARNING: invalid Attention partition ID ${full_partition_id}" >&2
          cleanup_failed=1
        else
          partition_id="${full_partition_id#"${gpu_uuid}/"}"
          if ! mps_control \
            "${pipe_dir}" "sm_partition rm ${gpu_uuid} ${partition_id}" \
            >> "${cleanup_log}" 2>&1; then
            echo "WARNING: failed to remove Attention partition ${full_partition_id}" >&2
            cleanup_failed=1
          fi
        fi
      fi
      full_partition_id="${MPS_PREFILL_PARTITIONS[idx]:-}"
      if [[ -n "${full_partition_id}" ]]; then
        if [[ "${full_partition_id}" != "${gpu_uuid}/"* ]]; then
          echo "WARNING: invalid Prefill partition ID ${full_partition_id}" >&2
          cleanup_failed=1
        else
          partition_id="${full_partition_id#"${gpu_uuid}/"}"
          if ! mps_control \
            "${pipe_dir}" "sm_partition rm ${gpu_uuid} ${partition_id}" \
            >> "${cleanup_log}" 2>&1; then
            echo "WARNING: failed to remove Prefill partition ${full_partition_id}" >&2
            cleanup_failed=1
          fi
        fi
      fi
  done
  for pipe_dir in "${MPS_STARTED_DIRS[@]}"; do
    if ! mps_control "${pipe_dir}" quit >/dev/null 2>&1; then
      echo "WARNING: failed to stop MPS daemon at ${pipe_dir}" >&2
      cleanup_failed=1
    fi
  done
  if (( code == 0 && cleanup_failed != 0 )); then
    code=1
  fi
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

die() {
  echo "ERROR: $*" >&2
  exit 1
}

TOOL_ARGS=()
case "${ENABLE_AUTO_TOOL_CHOICE}" in
  0)
    [[ -z "${TOOL_CALL_PARSER}" ]] \
      || die "tool parser requires PAP_BENCH_ENABLE_AUTO_TOOL_CHOICE=1"
    ;;
  1)
    [[ -n "${TOOL_CALL_PARSER}" ]] \
      || die "PAP_BENCH_TOOL_CALL_PARSER is required for automatic tools"
    TOOL_ARGS=(
      --enable-auto-tool-choice
      --tool-call-parser "${TOOL_CALL_PARSER}"
    )
    ;;
  *)
    die "PAP_BENCH_ENABLE_AUTO_TOOL_CHOICE must be 0 or 1"
    ;;
esac

wait_for_http() {
  local url="$1"
  local name="$2"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} is ready at ${url}"
      return 0
    fi
    if (( "$(date +%s)" - start > SERVER_START_TIMEOUT )); then
      die "Timed out waiting for ${name} at ${url}"
    fi
    sleep 2
  done
}

check_children_alive() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      die "managed child exited unexpectedly: pid=${pid}"
    fi
  done
}

wait_cluster_stable() {
  local remaining="${CLUSTER_READY_WAIT_SECONDS}"
  while (( remaining > 0 )); do
    check_children_alive
    sleep 2
    remaining=$((remaining - 2))
  done
}

audit_projection_scheduling() {
  local idx log_path
  for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    log_path="${RUN_LOG_DIR}/projection_${idx}.log"
    if ! rg -q "Asynchronous scheduling is enabled" "${log_path}"; then
      die "Projection ${idx} did not confirm asynchronous scheduling"
    fi
  done
  {
    printf 'STATUS=passed\n'
    printf 'ASYNC_SCHEDULING=1\n'
    printf 'SCHEDULER_QUEUE_DEPTH=2\n'
    printf 'PAP_RUNNER_MICROBATCH_PIPELINE=0\n'
  } > "${RUN_ROOT}/projection_scheduling_audit.env"
}

wait_attention_sessions_drained() {
  local start response idx active_sessions last_responses
  start="$(date +%s)"
  while true; do
    active_sessions=0
    last_responses=""
    for (( idx=0; idx<PA_COUNT; idx++ )); do
      response="$(
        curl -fsS \
          "http://127.0.0.1:$((ATTENTION_PORT_BASE + idx))/v1/pap/attention/sessions" \
          2>/dev/null || true
      )"
      last_responses+="pa${idx}=${response} "
      if [[ "${response}" != *'"active_sessions":0'* ]]; then
        active_sessions=$((active_sessions + 1))
      fi
    done
    if (( active_sessions == 0 )); then
      {
        printf 'STATUS=passed\n'
        printf 'ACTIVE_SESSIONS=0\n'
        printf 'ATTENTION_INSTANCE_COUNT=%q\n' "${PA_COUNT}"
        printf 'TIMEOUT_SECONDS=%q\n' "${PAP_BENCH_SESSION_DRAIN_TIMEOUT}"
      } > "${RUN_ROOT}/session_drain.env"
      echo "All ${PA_COUNT} PAP Attention sessions drained"
      return 0
    fi
    check_children_alive
    if (( "$(date +%s)" - start > PAP_BENCH_SESSION_DRAIN_TIMEOUT )); then
      {
        printf 'STATUS=failed\n'
        printf 'ACTIVE_ATTENTION_INSTANCES=%q\n' "${active_sessions}"
        printf 'LAST_RESPONSES=%q\n' "${last_responses}"
        printf 'TIMEOUT_SECONDS=%q\n' "${PAP_BENCH_SESSION_DRAIN_TIMEOUT}"
      } > "${RUN_ROOT}/session_drain.env"
      die "Timed out waiting for PAP Attention sessions to drain"
    fi
    sleep 1
  done
}

wait_gateway_requests_drained() {
  local start response
  start="$(date +%s)"
  while true; do
    response="$(
      curl -fsS "http://127.0.0.1:${PAP_PROXY_PORT}/health" 2>/dev/null \
        || true
    )"
    if [[ "${response}" == *'"inflight_requests":0'* ]]; then
      {
        printf 'STATUS=passed\n'
        printf 'INFLIGHT_REQUESTS=0\n'
        printf 'TIMEOUT_SECONDS=%q\n' "${PAP_BENCH_GATEWAY_DRAIN_TIMEOUT}"
      } > "${RUN_ROOT}/gateway_drain.env"
      echo "PAP Gateway requests drained"
      return 0
    fi
    check_children_alive
    if (( "$(date +%s)" - start > PAP_BENCH_GATEWAY_DRAIN_TIMEOUT )); then
      {
        printf 'STATUS=failed\n'
        printf 'LAST_RESPONSE=%q\n' "${response}"
        printf 'TIMEOUT_SECONDS=%q\n' "${PAP_BENCH_GATEWAY_DRAIN_TIMEOUT}"
      } > "${RUN_ROOT}/gateway_drain.env"
      die "Timed out waiting for PAP Gateway requests to drain"
    fi
    sleep 1
  done
}

capture_attention_fast_path_stats() {
  local idx
  local stats_paths=()
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    local output_path="${RUN_ROOT}/attention_fast_path_stats_${idx}.json"
    curl -fsS \
      "http://127.0.0.1:$((ATTENTION_PORT_BASE + idx))/v1/pap/attention/stats" \
      -o "${output_path}" \
      || die "Failed to capture PAP Attention ${idx} fast-path stats"
    stats_paths+=("${output_path}")
  done
  if (( PA_COUNT == 1 )); then
    cp "${stats_paths[0]}" "${RUN_ROOT}/attention_fast_path_stats.json"
    return
  fi
  "${PYTHON_BIN}" - "${RUN_ROOT}/attention_fast_path_stats.json" \
    "${stats_paths[@]}" <<'PY'
import json
import sys

output_path = sys.argv[1]
instances = []
for index, path in enumerate(sys.argv[2:]):
    with open(path, encoding="utf-8") as source:
        instances.append({"attention_index": index, "stats": json.load(source)})
with open(output_path, "w", encoding="utf-8") as output:
    json.dump({"instances": instances}, output, indent=2)
    output.write("\n")
PY
}

audit_pap_whole_step_graph() {
  local capture_count=0
  local errors=()
  local idx log
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    log="${RUN_LOG_DIR}/attention_${idx}_0.log"
    if rg -q 'PAP Attention whole-step CUDA Graph capture complete' "${log}"; then
      (( capture_count += 1 ))
    else
      errors+=("Attention ${idx} did not capture the PAP whole-step Graph")
    fi
  done
  for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    log="${RUN_LOG_DIR}/projection_${idx}.log"
    if rg -q 'PAP Projection whole-step CUDA Graph capture complete' "${log}"; then
      (( capture_count += 1 ))
    else
      errors+=("Projection ${idx} did not capture the PAP whole-step Graph")
    fi
  done
  {
    if (( ${#errors[@]} == 0 )); then
      printf 'STATUS=passed\n'
    else
      printf 'STATUS=failed\n'
    fi
    printf 'EXPECTED_PROCESS_COUNT=%q\n' "$((PA_COUNT + PROJECTION_COUNT))"
    printf 'CAPTURED_PROCESS_COUNT=%q\n' "${capture_count}"
    printf 'ERROR_COUNT=%q\n' "${#errors[@]}"
  } > "${RUN_ROOT}/pap_whole_step_graph_audit.env"
  if (( ${#errors[@]} > 0 )); then
    printf 'ERROR: %s\n' "${errors[@]}" >&2
    die "PAP whole-step CUDA Graph audit failed"
  fi
}

audit_projection_outer_graph_configuration() {
  local errors=()
  local idx log
  for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    log="${RUN_LOG_DIR}/projection_${idx}.log"
    rg -q 'enforce_eager=False' "${log}" \
      || errors+=("Projection ${idx} did not retain torch.compile")
    rg -q 'cudagraph_mode.*NONE' "${log}" \
      || errors+=("Projection ${idx} enabled a nested vLLM CUDA Graph")
    if rg -q 'Graph capturing finished' "${log}"; then
      errors+=("Projection ${idx} captured a forbidden nested vLLM Graph")
    fi
  done
  {
    if (( ${#errors[@]} == 0 )); then
      printf 'STATUS=passed\n'
    else
      printf 'STATUS=failed\n'
    fi
    printf 'PROJECTION_VLLM_CUDAGRAPH_MODE=NONE\n'
    printf 'PROJECTION_OUTER_GRAPH_MODE=PAP_WHOLE_STEP\n'
    printf 'ERROR_COUNT=%q\n' "${#errors[@]}"
  } > "${RUN_ROOT}/projection_outer_graph_config_audit.env"
  if (( ${#errors[@]} > 0 )); then
    printf 'ERROR: %s\n' "${errors[@]}" >&2
    die "PAP Projection outer-Graph configuration audit failed"
  fi
}

audit_decode_token_join() {
  local stats_path="${RUN_ROOT}/attention_fast_path_stats.json"
  local summary_path="${RUN_ROOT}/decode_token_join_audit.env"
  "${PYTHON_BIN}" - "${stats_path}" "${summary_path}" <<'PY'
import json
import sys

stats_path, summary_path = sys.argv[1:]
with open(stats_path, encoding="utf-8") as source:
    payload = json.load(source)

if "instances" in payload:
    instances = [entry["stats"] for entry in payload["instances"]]
else:
    instances = [payload]

zero_fields = (
    "decode_token_pending_tokens",
    "decode_token_pending_kv",
    "decode_token_dispatching",
    "decode_token_mismatches",
    "decode_token_dispatch_failures",
)
errors = []
for index, stats in enumerate(instances):
    missing = [
        field
        for field in (*zero_fields, "decode_token_received", "decode_token_matched")
        if field not in stats
    ]
    if missing:
        errors.append(f"attention[{index}] missing fields: {','.join(missing)}")
        continue
    for field in zero_fields:
        if int(stats[field]) != 0:
            errors.append(f"attention[{index}] {field}={stats[field]}")
    if int(stats["decode_token_received"]) <= 0:
        errors.append(f"attention[{index}] received no async decode tokens")
    if int(stats["decode_token_matched"]) <= 0:
        errors.append(f"attention[{index}] matched no async decode tokens")

with open(summary_path, "w", encoding="utf-8") as output:
    output.write(f"STATUS={'failed' if errors else 'passed'}\n")
    output.write("DECODE_TOKEN_DELIVERY=async\n")
    output.write(f"ATTENTION_INSTANCE_COUNT={len(instances)}\n")
    output.write(f"ERROR_COUNT={len(errors)}\n")

if errors:
    for error in errors:
        print(f"PAP decode-token join audit: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

deferred_trace_enabled() {
  case "${PAP_DEFERRED_CUDA_TRACE,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

capture_projection_deferred_traces() {
  deferred_trace_enabled || return 0
  local idx output_path deadline
  local -a trace_paths=()
  local -a validator_args=()
  for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    output_path="${RUN_ROOT}/projection_deferred_trace_${idx}.json"
    [[ ! -e "${output_path}" ]] \
      || die "Projection deferred trace already exists: ${output_path}"
    [[ ! -e "${output_path}.flush" ]] \
      || die "Projection deferred trace trigger exists: ${output_path}.flush"
    : > "${output_path}.flush"
    trace_paths+=("${output_path}")
  done
  for output_path in "${trace_paths[@]}"; do
    deadline=$((SECONDS + PAP_DEFERRED_TRACE_FLUSH_TIMEOUT))
    until [[ -s "${output_path}" ]]; do
      check_children_alive
      if (( SECONDS >= deadline )); then
        die "Timed out waiting for Projection deferred trace: ${output_path}"
      fi
      sleep 0.1
    done
    validator_args=(
      --trace "${output_path}"
      --scope projection_process_critical_chain
      --num-layers 36
    )
    if (( PROJECTION_COUNT == 1 )); then
      validator_args+=(
        --attention-stats "${RUN_ROOT}/attention_fast_path_stats.json"
      )
    fi
    "${PYTHON_BIN}" "${DEFERRED_TRACE_VALIDATOR}" \
      "${validator_args[@]}"
  done
  if (( PROJECTION_COUNT == 1 )); then
    cp "${trace_paths[0]}" "${RUN_ROOT}/projection_deferred_trace.json"
    return
  fi
  "${PYTHON_BIN}" - "${RUN_ROOT}/projection_deferred_trace.json" \
    "${trace_paths[@]}" <<'PY'
import json
import sys

output_path = sys.argv[1]
instances = []
for index, path in enumerate(sys.argv[2:]):
    with open(path, encoding="utf-8") as source:
        instances.append({"projection_index": index, "trace": json.load(source)})
with open(output_path, "w", encoding="utf-8") as output:
    json.dump({"instances": instances}, output, indent=2)
    output.write("\n")
PY
}

start_prefill_torch_profiles() {
  [[ "${PAP_PREFILL_TORCH_PROFILE}" == "1" ]] || return 0
  local idx
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    curl -fsS -X POST \
      "http://127.0.0.1:$((PREFILL_PORT_BASE + idx))/start_profile" \
      >/dev/null \
      || die "Failed to start Prefill ${idx} torch profiler"
  done
}

wait_prefill_torch_profiles() {
  [[ "${PAP_PREFILL_TORCH_PROFILE}" == "1" ]] || return 0
  local idx profile_dir deadline
  local -a trace_files=()
  : > "${RUN_ROOT}/prefill_torch_profile_audit.env"
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    profile_dir="${RUN_ROOT}/prefill_torch_profile_${idx}"
    deadline=$((SECONDS + PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT))
    while true; do
      shopt -s nullglob
      trace_files=(
        "${profile_dir}"/*.trace.json
        "${profile_dir}"/*.trace.json.gz
      )
      shopt -u nullglob
      if (( ${#trace_files[@]} > 0 )) \
        && [[ -s "${trace_files[0]}" ]]; then
        break
      fi
      check_children_alive
      if (( SECONDS >= deadline )); then
        die "Timed out waiting for Prefill ${idx} torch profile"
      fi
      sleep 0.2
    done
    {
      printf 'PREFILL_%d_PROFILE_DIR=%q\n' "${idx}" "${profile_dir}"
      printf 'PREFILL_%d_TRACE=%q\n' "${idx}" "${trace_files[0]}"
    } >> "${RUN_ROOT}/prefill_torch_profile_audit.env"
  done
  {
    printf 'STATUS=passed\n'
    printf 'PREFILL_INSTANCE_COUNT=%q\n' "${PA_COUNT}"
    printf 'MAX_ITERATIONS=%q\n' \
      "${PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS}"
  } >> "${RUN_ROOT}/prefill_torch_profile_audit.env"
}

capture_proxy_topology_stats() {
  curl -fsS \
    "http://127.0.0.1:${PAP_PROXY_PORT}/v1/pap/topology/stats" \
    -o "${RUN_ROOT}/topology_runtime_stats.json" \
    || die "Failed to capture PAP proxy topology stats"
}

ensure_ports_free() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import socket
import sys

busy = []
for raw_port in sys.argv[1:]:
    port = int(raw_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            busy.append(str(port))
    finally:
        sock.close()

if busy:
    print("Ports already in use: " + ", ".join(busy), file=sys.stderr)
    raise SystemExit(1)
PY
}

ensure_dataset() {
  if [[ -f "${DATASET_PATH}" ]]; then
    return
  fi
  [[ -f "${BENCH_DIR}/sonnet.txt" ]] || die "Missing ${BENCH_DIR}/sonnet.txt"
  mkdir -p "$(dirname "${DATASET_PATH}")"
  : > "${DATASET_PATH}"
  for _ in 1 2 3 4; do
    cat "${BENCH_DIR}/sonnet.txt" >> "${DATASET_PATH}"
  done
}

prepare_aiperf_dataset() {
  [[ -x "${AIPERF_BIN}" ]] \
    || die "AIPerf is not installed at ${AIPERF_BIN}"
  [[ -x "${AIPERF_RUNNER}" ]] \
    || die "Missing AIPerf runner: ${AIPERF_RUNNER}"
  [[ -f "${AIPERF_DATASET_GENERATOR}" ]] \
    || die "Missing AIPerf dataset generator: ${AIPERF_DATASET_GENERATOR}"
  if [[ "${AIPERF_INPUT_FILE_PROVIDED}" == "1" ]]; then
    [[ -f "${PAP_AIPERF_INPUT_FILE}" ]] \
      || die "Missing AIPerf input file: ${PAP_AIPERF_INPUT_FILE}"
    return
  fi
  "${PYTHON_BIN}" "${AIPERF_DATASET_GENERATOR}" \
    --model "${MODEL_PATH}" \
    --corpus "${DATASET_PATH}" \
    --output "${PAP_AIPERF_INPUT_FILE}" \
    --sessions "${PAP_AIPERF_SESSIONS}" \
    --turns "${PAP_AIPERF_TURNS}" \
    --document-tokens "${INPUT_LEN}" \
    --document-tokens-median "${AIPERF_DOCUMENT_TOKENS_MEDIAN}" \
    --document-tokens-min "${AIPERF_DOCUMENT_TOKENS_MIN}" \
    --document-tokens-max "${AIPERF_DOCUMENT_TOKENS_MAX}" \
    --append-tokens "${PAP_AIPERF_APPEND_TOKENS}" \
    --append-tokens-median "${AIPERF_APPEND_TOKENS_MEDIAN}" \
    --append-tokens-min "${AIPERF_APPEND_TOKENS_MIN}" \
    --append-tokens-max "${AIPERF_APPEND_TOKENS_MAX}" \
    --output-tokens "${OUTPUT_LEN}" \
    --output-tokens-median "${AIPERF_OUTPUT_TOKENS_MEDIAN}" \
    --output-tokens-min "${AIPERF_OUTPUT_TOKENS_MIN}" \
    --output-tokens-max "${AIPERF_OUTPUT_TOKENS_MAX}" \
    --random-seed "${AIPERF_RANDOM_SEED}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --think-time-ms "${AIPERF_THINK_TIME_MS}" \
    --tool-time-ms "${AIPERF_TOOL_TIME_MS}" \
    --tool-every "${AIPERF_TOOL_EVERY}" \
    --session-prefix "${RUN_ID}-aiperf"
}

capture_git_state() {
  GIT_COMMIT="$(git rev-parse HEAD)"
  GIT_COMMIT_SHORT="$(git rev-parse --short HEAD)"
  git status --short > "${RUN_ROOT}/git_status.txt"
  git diff --binary > "${RUN_ROOT}/tracked_worktree.patch"
  git diff --cached --binary > "${RUN_ROOT}/tracked_index.patch"
  if [[ -s "${RUN_ROOT}/tracked_worktree.patch" \
    || -s "${RUN_ROOT}/tracked_index.patch" ]]; then
    GIT_TRACKED_WORKTREE_DIRTY=1
  fi
  if [[ "${PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE}" == "1" \
    && "${GIT_TRACKED_WORKTREE_DIRTY}" == "1" ]]; then
    die "Tracked worktree is dirty; see ${RUN_ROOT}/git_status.txt"
  fi
}

write_effective_config() {
  {
    printf 'MODE=%q\n' "pap"
    printf 'PAP_BENCH_CLIENT=%q\n' "${PAP_BENCH_CLIENT}"
    printf 'TOPOLOGY=%q\n' "${TOPOLOGY}"
    printf 'PA_COUNT=%q\n' "${PA_COUNT}"
    printf 'PROJECTION_COUNT=%q\n' "${PROJECTION_COUNT}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
    printf 'BENCH_DIR=%q\n' "${BENCH_DIR}"
    printf 'NUM_PROMPTS=%q\n' "${NUM_PROMPTS}"
    printf 'PAP_ENABLE_PROMPT_TOKENS_DETAILS=%q\n' "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}"
    printf 'PAP_PREFIX_CACHE_AUDIT=%q\n' "${PAP_PREFIX_CACHE_AUDIT}"
    printf 'PAP_BLOCK_SIZE=%q\n' "${PAP_BLOCK_SIZE}"
    printf 'PAP_AIPERF_TURNS=%q\n' "${PAP_AIPERF_TURNS}"
    printf 'PAP_AIPERF_SESSIONS=%q\n' "${PAP_AIPERF_SESSIONS}"
    printf 'PAP_AIPERF_VARIABLE_TURNS=%q\n' \
      "${PAP_AIPERF_VARIABLE_TURNS}"
    printf 'PAP_AIPERF_EXPECTED_REQUESTS=%q\n' \
      "${REQUESTS_PER_AIPERF_VARIATION}"
    printf 'PAP_AIPERF_APPEND_TOKENS=%q\n' \
      "${PAP_AIPERF_APPEND_TOKENS}"
    printf 'AIPERF_ROOT=%q\n' "${AIPERF_ROOT}"
    printf 'AIPERF_BIN=%q\n' "${AIPERF_BIN}"
    printf 'PAP_AIPERF_INPUT_FILE=%q\n' "${PAP_AIPERF_INPUT_FILE}"
    printf 'PAP_AIPERF_OUTPUT_DIR=%q\n' "${PAP_AIPERF_OUTPUT_DIR}"
    printf 'PAP_AIPERF_CONCURRENCY=%q\n' "${PAP_AIPERF_CONCURRENCY}"
    printf 'AIPERF_VARIATION_COUNT=%q\n' "${AIPERF_VARIATION_COUNT}"
    printf 'AIPERF_NUM_PROFILE_RUNS=%q\n' "${AIPERF_NUM_PROFILE_RUNS}"
    printf 'PAP_AIPERF_TIMING_MODE=%q\n' "${PAP_AIPERF_TIMING_MODE}"
    printf 'PAP_AIPERF_REQUEST_RATE=%q\n' "${PAP_AIPERF_REQUEST_RATE}"
    printf 'INPUT_LENS_CSV=%q\n' "${INPUT_LEN}"
    printf 'OUTPUT_LENS_CSV=%q\n' "${OUTPUT_LEN}"
    printf 'BENCH_TIMEOUT=%q\n' "${BENCH_TIMEOUT}"
    printf 'SERVER_START_TIMEOUT=%q\n' "${SERVER_START_TIMEOUT}"
    printf 'RESULTS_ROOT=%q\n' "${RESULTS_ROOT}"
    printf 'RUN_ROOT=%q\n' "${RUN_ROOT}"
    printf 'RUN_LOG_DIR=%q\n' "${RUN_LOG_DIR}"
    printf 'PROXY_PORT=%q\n' "${PAP_PROXY_PORT}"
    printf 'VLLM_BIN=%q\n' "${VLLM_BIN}"
    printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
    printf 'GIT_COMMIT=%q\n' "${GIT_COMMIT}"
    printf 'GIT_COMMIT_SHORT=%q\n' "${GIT_COMMIT_SHORT}"
    printf 'GIT_TRACKED_WORKTREE_DIRTY=%q\n' "${GIT_TRACKED_WORKTREE_DIRTY}"
    printf 'PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=%q\n' "${PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE}"
    printf 'PAP_BENCH_STRICT_CORRECTNESS_AUDIT=%q\n' "${PAP_BENCH_STRICT_CORRECTNESS_AUDIT}"
    printf 'VLLM_USE_FLASHINFER_SAMPLER=%q\n' "${VLLM_USE_FLASHINFER_SAMPLER}"
    printf 'NO_PROXY=%q\n' "${NO_PROXY}"
    printf 'no_proxy=%q\n' "${no_proxy}"
    printf 'PAP_PROXY_PORT=%q\n' "${PAP_PROXY_PORT}"
    printf 'PAP_PREFILL_GPU_MEMORY_UTILIZATION=%q\n' "${PAP_PREFILL_GPU_MEMORY_UTILIZATION}"
    printf 'PROJECTION_MEMORY_POLICY=%q\n' \
      "model_weights_x1.20_plus_kv_validation"
    printf 'PROJECTION_GPU_MEMORY_UTILIZATION=%q\n' \
      "${PROJECTION_GPU_MEMORY_UTILIZATION}"
    printf 'PROJECTION_MODEL_WEIGHT_BYTES=%q\n' \
      "${PROJECTION_MODEL_WEIGHT_BYTES}"
    printf 'PROJECTION_PER_RANK_WEIGHT_BYTES=%q\n' \
      "${PROJECTION_PER_RANK_WEIGHT_BYTES}"
    printf 'PROJECTION_VALIDATION_KV_BYTES=%q\n' \
      "${PROJECTION_VALIDATION_KV_BYTES}"
    printf 'PROJECTION_MEMORY_TARGET_BYTES=%q\n' \
      "${PROJECTION_MEMORY_TARGET_BYTES}"
    printf 'PROJECTION_GPU_TOTAL_BYTES=%q\n' \
      "${PROJECTION_GPU_TOTAL_BYTES}"
    printf 'PAP_STATIC_PREFILL_CHUNKS=%q\n' \
      "${PAP_STATIC_PREFILL_CHUNKS}"
    printf 'PAP_STATIC_ATTENTION_CHUNKS=%q\n' \
      "${PAP_STATIC_ATTENTION_CHUNKS}"
    printf 'PAP_STATIC_PREFILL_EXPECTED_SMS=%q\n' \
      "${PAP_STATIC_PREFILL_EXPECTED_SMS}"
    printf 'PAP_STATIC_ATTENTION_EXPECTED_SMS=%q\n' \
      "${PAP_STATIC_ATTENTION_EXPECTED_SMS}"
    printf 'PAP_ENABLE_MPS=%q\n' "${PAP_ENABLE_MPS}"
    printf 'PAP_OFFLOAD_KV_TRANSPORT=%q\n' "${PAP_OFFLOAD_KV_TRANSPORT}"
    printf 'PAP_NVSHMEM_INIT_TIMEOUT=%q\n' "${PAP_NVSHMEM_INIT_TIMEOUT}"
    printf 'PAP_NVSHMEM_BUFFER_BYTES=%q\n' "${PAP_NVSHMEM_BUFFER_BYTES}"
    printf 'PAP_NVSHMEM_CONTROL_BYTES=%q\n' "${PAP_NVSHMEM_CONTROL_BYTES}"
    printf 'PAP_OFFLOAD_EXEC_TRACE=%q\n' "${PAP_OFFLOAD_EXEC_TRACE}"
    printf 'PAP_OFFLOAD_EXEC_TRACE_LAYER=%q\n' \
      "${PAP_OFFLOAD_EXEC_TRACE_LAYER}"
    printf 'PAP_DEFERRED_CUDA_TRACE=%q\n' "${PAP_DEFERRED_CUDA_TRACE}"
    printf 'PAP_DEFERRED_CUDA_TRACE_MAX_PENDING=%q\n' \
      "${PAP_DEFERRED_CUDA_TRACE_MAX_PENDING}"
    printf 'PAP_UNIFIED_MD_CACHE_LIMIT=%q\n' "${PAP_UNIFIED_MD_CACHE_LIMIT}"
    printf 'PAP_DECODE_SLOT_PLAN_CACHE_LIMIT=%q\n' "${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}"
    printf 'PAP_PREFILL_GPUS=%q\n' "${PAP_PREFILL_GPUS}"
    printf 'PAP_PROJECTION_GPUS=%q\n' "${PAP_PROJECTION_GPUS}"
    printf 'PAP_VLLM_DTYPE=%q\n' "${PAP_VLLM_DTYPE}"
    printf 'ENABLE_AUTO_TOOL_CHOICE=%q\nTOOL_CALL_PARSER=%q\n' \
      "${ENABLE_AUTO_TOOL_CHOICE}" "${TOOL_CALL_PARSER}"
    printf 'PAP_ROUTING_POLICY=%q\n' "${PAP_ROUTING_POLICY}"
    printf 'PREFILL_PORT_BASE=%q\n' "${PREFILL_PORT_BASE}"
    printf 'PROJECTION_PORT_BASE=%q\n' "${PROJECTION_PORT_BASE}"
    printf 'ATTENTION_PORT_BASE=%q\n' "${ATTENTION_PORT_BASE}"
    printf 'PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=%q\n' "${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS}"
    printf 'PAP_DECODE_CAPACITY_SOURCE=%q\n' \
      "request_max_tokens_with_environment_fallback"
    printf 'PAP_DECODE_COMMIT_ENDPOINT=%q\n' "${PAP_DECODE_COMMIT_ENDPOINT}"
    printf 'PAP_DECODE_COMMIT_FAIL_CLOSED=%q\n' "${PAP_DECODE_COMMIT_FAIL_CLOSED}"
    printf 'PAP_DECODE_COMMIT_TIMEOUT=%q\n' "${PAP_DECODE_COMMIT_TIMEOUT}"
    printf 'PAP_DECODE_COMMIT_QUEUE_SIZE=%q\n' "${PAP_DECODE_COMMIT_QUEUE_SIZE}"
    printf 'PAP_DECODE_COMMIT_MAX_ATTEMPTS=%q\n' "${PAP_DECODE_COMMIT_MAX_ATTEMPTS}"
    printf 'PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS=%q\n' "${PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS}"
    printf 'PAP_DECODE_COMMIT_RETRY_MAX_SECONDS=%q\n' "${PAP_DECODE_COMMIT_RETRY_MAX_SECONDS}"
    printf 'PAP_DECODE_COMMIT_FLUSH_TIMEOUT=%q\n' "${PAP_DECODE_COMMIT_FLUSH_TIMEOUT}"
    printf 'VLLM_USE_V2_MODEL_RUNNER=%q\n' "${VLLM_USE_V2_MODEL_RUNNER}"
    printf 'PAP_PREFILL_IPC_PROFILE=%q\n' "${PAP_PREFILL_IPC_PROFILE}"
    printf 'PAP_RUNTIME_CUDA_CONTEXT_AUDIT=%q\n' \
      "${PAP_RUNTIME_CUDA_CONTEXT_AUDIT}"
    printf 'PAP_PREFILL_TORCH_PROFILE=%q\n' \
      "${PAP_PREFILL_TORCH_PROFILE}"
    printf 'PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS=%q\n' \
      "${PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS}"
    printf 'PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT=%q\n' \
      "${PAP_PREFILL_TORCH_PROFILE_FLUSH_TIMEOUT}"
    printf 'PAP_DECODE_TOKEN_TIMEOUT=%q\n' "${PAP_DECODE_TOKEN_TIMEOUT}"
    printf 'PAP_DECODE_TOKEN_QUEUE_SIZE=%q\n' "${PAP_DECODE_TOKEN_QUEUE_SIZE}"
    printf 'PAP_DECODE_TOKEN_MAX_ATTEMPTS=%q\n' "${PAP_DECODE_TOKEN_MAX_ATTEMPTS}"
    printf 'PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS=%q\n' \
      "${PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS}"
    printf 'PAP_DECODE_TOKEN_RETRY_MAX_SECONDS=%q\n' \
      "${PAP_DECODE_TOKEN_RETRY_MAX_SECONDS}"
    printf 'PAP_DECODE_TOKEN_FLUSH_TIMEOUT=%q\n' \
      "${PAP_DECODE_TOKEN_FLUSH_TIMEOUT}"
    printf 'PAP_LEASE_RELEASE_ENDPOINT=%q\n' "${PAP_LEASE_RELEASE_ENDPOINT}"
    printf 'PAP_LEASE_RELEASE_TIMEOUT=%q\n' "${PAP_LEASE_RELEASE_TIMEOUT}"
    printf 'PAP_LEASE_RELEASE_MAX_ATTEMPTS=%q\n' "${PAP_LEASE_RELEASE_MAX_ATTEMPTS}"
    printf 'PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS=%q\n' "${PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS}"
    printf 'PAP_LEASE_RELEASE_RETRY_MAX_SECONDS=%q\n' "${PAP_LEASE_RELEASE_RETRY_MAX_SECONDS}"
    printf 'PAP_KV_LEASE_TTL_SECONDS=%q\n' "${PAP_KV_LEASE_TTL_SECONDS}"
    printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN}"
    printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "${MAX_NUM_BATCHED_TOKENS}"
    printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS}"
    printf 'PAP_PREFILL_MAX_NUM_BATCHED_TOKENS=%q\n' \
      "${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS}"
    printf 'PAP_PREFILL_MAX_NUM_SEQS=%q\n' \
      "${PAP_PREFILL_MAX_NUM_SEQS}"
    printf 'PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS=%q\n' \
      "${PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS}"
    printf 'PAP_PROJECTION_MAX_NUM_SEQS=%q\n' \
      "${PAP_PROJECTION_MAX_NUM_SEQS}"
    printf 'PAP_PROJECTION_ASYNC_SCHEDULING=%q\n' "1"
    printf 'PAP_PREFILL_EXECUTION_MODE=piecewise_cuda_graph\n'
    printf 'PAP_PROJECTION_EXECUTION_MODE=pap_whole_step_cuda_graph\n'
    printf 'PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
      "${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES}"
    printf 'PAP_ATTENTION_PROJECTION_GRAPH_MODE=whole_step\n'
    printf 'CLUSTER_READY_WAIT_SECONDS=%q\n' "${CLUSTER_READY_WAIT_SECONDS}"
    printf 'PAP_BENCH_GATEWAY_DRAIN_TIMEOUT=%q\n' \
      "${PAP_BENCH_GATEWAY_DRAIN_TIMEOUT}"
    printf 'PAP_BENCH_SESSION_DRAIN_TIMEOUT=%q\n' "${PAP_BENCH_SESSION_DRAIN_TIMEOUT}"
    printf 'PAP_DEFERRED_TRACE_FLUSH_TIMEOUT=%q\n' \
      "${PAP_DEFERRED_TRACE_FLUSH_TIMEOUT}"
  } > "${RUN_ROOT}/effective_config.env"
}

write_topology_manifest() {
  RUN_ROOT="${RUN_ROOT}" \
  TOPOLOGY="${TOPOLOGY}" \
  PA_COUNT="${PA_COUNT}" \
  PROJECTION_COUNT="${PROJECTION_COUNT}" \
  PREFILL_GPUS="${PAP_PREFILL_GPUS}" \
  PROJECTION_GPUS="${PAP_PROJECTION_GPUS}" \
  PREFILL_PORT_BASE="${PREFILL_PORT_BASE}" \
  ATTENTION_PORT_BASE="${ATTENTION_PORT_BASE}" \
  ATTENTION_TCP_PORT_BASE="${ATTENTION_TCP_PORT_BASE}" \
  PROJECTION_PORT_BASE="${PROJECTION_PORT_BASE}" \
  PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os

pa_count = int(os.environ["PA_COUNT"])
projection_count = int(os.environ["PROJECTION_COUNT"])
prefill_gpus = os.environ["PREFILL_GPUS"].split(",")
projection_gpus = os.environ["PROJECTION_GPUS"].split(",")

def base(name: str) -> int:
    return int(os.environ[name])

manifest = {
    "topology": os.environ["TOPOLOGY"],
    "routing_policy": os.environ["PAP_ROUTING_POLICY"],
    "pa_groups": [
        {
            "id": index,
            "gpu": prefill_gpus[index],
            "prefill": f"http://127.0.0.1:{base('PREFILL_PORT_BASE') + index}",
            "attention": (
                f"http://127.0.0.1:{base('ATTENTION_PORT_BASE') + index}"
            ),
            "attention_tcp_port": base("ATTENTION_TCP_PORT_BASE") + index,
        }
        for index in range(pa_count)
    ],
    "projections": [
        {
            "id": index,
            "gpu": projection_gpus[index],
            "endpoint": (
                f"http://127.0.0.1:{base('PROJECTION_PORT_BASE') + index}"
            ),
        }
        for index in range(projection_count)
    ],
}
with open(
    os.path.join(os.environ["RUN_ROOT"], "topology_manifest.json"),
    "w",
    encoding="utf-8",
) as output:
    json.dump(manifest, output, indent=2)
    output.write("\n")
PY
}

write_run_metadata() {
  RUN_ROOT="${RUN_ROOT}" \
  PAP_BENCH_CLIENT="${PAP_BENCH_CLIENT}" \
  INPUT_LEN="${INPUT_LEN}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  MODEL_PATH="${MODEL_PATH}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PAP_NVSHMEM_BUFFER_BYTES="${PAP_NVSHMEM_BUFFER_BYTES:-}" \
  PAP_NVSHMEM_CONTROL_BYTES="${PAP_NVSHMEM_CONTROL_BYTES:-}" \
  PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
  PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE}" \
  PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}" \
  PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" \
  PAP_AIPERF_TURNS="${PAP_AIPERF_TURNS}" \
  PAP_AIPERF_SESSIONS="${PAP_AIPERF_SESSIONS}" \
  PAP_AIPERF_VARIABLE_TURNS="${PAP_AIPERF_VARIABLE_TURNS}" \
  PAP_AIPERF_APPEND_TOKENS="${PAP_AIPERF_APPEND_TOKENS}" \
  PAP_AIPERF_CONCURRENCY="${PAP_AIPERF_CONCURRENCY}" \
  AIPERF_NUM_PROFILE_RUNS="${AIPERF_NUM_PROFILE_RUNS}" \
  PAP_AIPERF_TIMING_MODE="${PAP_AIPERF_TIMING_MODE}" \
  PAP_AIPERF_REQUEST_RATE="${PAP_AIPERF_REQUEST_RATE}" \
  PAP_VLLM_DTYPE="${PAP_VLLM_DTYPE}" \
  GIT_COMMIT="${GIT_COMMIT}" \
  GIT_COMMIT_SHORT="${GIT_COMMIT_SHORT}" \
  GIT_TRACKED_WORKTREE_DIRTY="${GIT_TRACKED_WORKTREE_DIRTY}" \
  PAP_PROXY_PORT="${PAP_PROXY_PORT}" \
  TOPOLOGY="${TOPOLOGY}" \
  PA_COUNT="${PA_COUNT}" \
  PROJECTION_COUNT="${PROJECTION_COUNT}" \
  PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY}" \
  RUN_ID="${RUN_ID}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime

pa_count = int(os.environ["PA_COUNT"])
projection_count = int(os.environ["PROJECTION_COUNT"])
metadata = {
    "mode": "pap",
    "client": os.environ["PAP_BENCH_CLIENT"],
    "topology": os.environ["TOPOLOGY"],
    "pa_count": pa_count,
    "projection_count": projection_count,
    "routing_policy": os.environ["PAP_ROUTING_POLICY"],
    "run_id": os.environ["RUN_ID"],
    "result_root": os.environ["RUN_ROOT"],
    "input_lens": [os.environ["INPUT_LEN"]],
    "output_lens": [os.environ["OUTPUT_LEN"]],
    "expected_requests": int(os.environ["NUM_PROMPTS"]),
    "model_path": os.environ["MODEL_PATH"],
    "max_model_len": os.environ["MAX_MODEL_LEN"],
    "max_num_seqs": os.environ["MAX_NUM_SEQS"],
    "offload_exec_transport": "nvshmem_graph",
    "nvshmem_buffer_bytes": (
        int(os.environ["PAP_NVSHMEM_BUFFER_BYTES"])
        if os.environ["PAP_NVSHMEM_BUFFER_BYTES"]
        else None
    ),
    "nvshmem_control_bytes": (
        int(os.environ["PAP_NVSHMEM_CONTROL_BYTES"])
        if os.environ["PAP_NVSHMEM_CONTROL_BYTES"]
        else None
    ),
    "offload_kv_transport": os.environ["PAP_OFFLOAD_KV_TRANSPORT"],
    "batched_route_copy": True,
    "unified_md_fast_key": True,
    "projection_async_scheduling": True,
    "projection_scheduler_queue_depth": 2,
    "projection_runner_microbatch_pipeline": False,
    "prefill_kv_async": True,
    "prefill_ipc_profile": os.environ["PAP_PREFILL_IPC_PROFILE"] == "1",
    "kv_handoff_mode": "sealed_manifest",
    "decode_slot_plan_cache_limit": int(
        os.environ["PAP_DECODE_SLOT_PLAN_CACHE_LIMIT"]
    ),
    "attention_dispatch_mode": "nvshmem_graph",
    "prompt_tokens_details": (
        os.environ["PAP_ENABLE_PROMPT_TOKENS_DETAILS"] == "1"
    ),
    "aiperf_turns": int(os.environ["PAP_AIPERF_TURNS"]),
    "aiperf_variable_turns": (
        os.environ["PAP_AIPERF_VARIABLE_TURNS"] == "1"
    ),
    "aiperf_sessions": int(os.environ["PAP_AIPERF_SESSIONS"]),
    "aiperf_append_tokens": int(os.environ["PAP_AIPERF_APPEND_TOKENS"]),
    "aiperf_concurrency_points": [
        int(value)
        for value in os.environ["PAP_AIPERF_CONCURRENCY"].split(",")
    ],
    "aiperf_num_profile_runs": int(
        os.environ["AIPERF_NUM_PROFILE_RUNS"]
    ),
    "aiperf_timing_mode": os.environ["PAP_AIPERF_TIMING_MODE"],
    "aiperf_request_rate": (
        float(os.environ["PAP_AIPERF_REQUEST_RATE"])
        if os.environ["PAP_AIPERF_REQUEST_RATE"]
        else None
    ),
    "dtype": os.environ["PAP_VLLM_DTYPE"],
    "git_commit": os.environ["GIT_COMMIT"],
    "git_commit_short": os.environ["GIT_COMMIT_SHORT"],
    "git_tracked_worktree_dirty": (
        os.environ["GIT_TRACKED_WORKTREE_DIRTY"] == "1"
    ),
    "proxy_port": os.environ["PAP_PROXY_PORT"],
    "config_dir": "project-owned PAP runner",
    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}
with open(os.path.join(os.environ["RUN_ROOT"], "run_metadata.json"), "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")
PY
}

audit_correctness_logs() {
  local matches_path="${RUN_ROOT}/correctness_audit_matches.log"
  local summary_path="${RUN_ROOT}/correctness_audit.env"
  local pattern
  pattern='CUDA out of memory|EngineDeadError|^Traceback| ERROR .*Traceback|Exception in thread|PAP decode commit failed|non-contiguous PAP decode commit|conflicting duplicate PAP decode commit|PAP lease release raced|PAP Prefill request .* without a KV lease|PAP control is not initialized|deferred PAP EngineCore control failed|new_token_ids length must match new_seq_len delta|PAP decode-token delivery failed|PAP decode-token queue is full|PAP decode-token join flush timed out|PAP lease release failed|PAP unified KV append out of range|PAP unified KV state missing|PAP unified KV state changed during decode append|PAP unified KV seq_len changed during decode append|prefill KV must reach the registered prefix before unified decode attention|PAP unified paged FlashAttention failed'

  if rg -n --no-heading "${pattern}" "${RUN_LOG_DIR}" > "${matches_path}"; then
    {
      printf 'STATUS=failed\n'
      printf 'MATCH_COUNT=%s\n' "$(wc -l < "${matches_path}")"
      printf 'STRICT=%q\n' "${PAP_BENCH_STRICT_CORRECTNESS_AUDIT}"
    } > "${summary_path}"
    if [[ "${PAP_BENCH_STRICT_CORRECTNESS_AUDIT}" == "1" ]]; then
      cat "${matches_path}" >&2
      die "PAP correctness audit failed; see ${matches_path}"
    fi
    return
  fi

  : > "${matches_path}"
  {
    printf 'STATUS=passed\n'
    printf 'MATCH_COUNT=0\n'
    printf 'STRICT=%q\n' "${PAP_BENCH_STRICT_CORRECTNESS_AUDIT}"
  } > "${summary_path}"
}

audit_xy_routes() {
  local summary_path="${RUN_ROOT}/routing_audit.env"
  if RUN_ROOT="${RUN_ROOT}" \
    RUN_LOG_DIR="${RUN_LOG_DIR}" \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    PA_COUNT="${PA_COUNT}" \
    PROJECTION_COUNT="${PROJECTION_COUNT}" \
    PREFILL_PORT_BASE="${PREFILL_PORT_BASE}" \
    PROJECTION_PORT_BASE="${PROJECTION_PORT_BASE}" \
    PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY}" \
    PAP_AIPERF_TURNS="${PAP_AIPERF_TURNS}" \
    PAP_AIPERF_SESSIONS="${PAP_AIPERF_SESSIONS}" \
    PAP_AIPERF_VARIABLE_TURNS="${PAP_AIPERF_VARIABLE_TURNS}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import regex as re
from collections import Counter
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
log_dir = Path(os.environ["RUN_LOG_DIR"])
expected_requests = int(os.environ["NUM_PROMPTS"])
pa_count = int(os.environ["PA_COUNT"])
projection_count = int(os.environ["PROJECTION_COUNT"])
prefill_base = int(os.environ["PREFILL_PORT_BASE"])
projection_base = int(os.environ["PROJECTION_PORT_BASE"])
routing_policy = os.environ["PAP_ROUTING_POLICY"]
load_rounds = int(os.environ["PAP_AIPERF_TURNS"])
load_conversations = int(os.environ["PAP_AIPERF_SESSIONS"])
variable_turns = os.environ["PAP_AIPERF_VARIABLE_TURNS"] == "1"
load_repetitions = expected_requests // (load_rounds * load_conversations)
route_pattern = re.compile(
    r"request_id=\S+ pa=[^:\s]+:(\d+).* projection=[^:\s]+:(\d+)"
)

proxy_text = (log_dir / "proxy.log").read_text(errors="replace")
routes = [
    (int(match.group(1)), int(match.group(2)))
    for match in route_pattern.finditer(proxy_text)
]
pa_routes = Counter(pa_port for pa_port, _ in routes)
projection_routes = Counter(projection_port for _, projection_port in routes)
pair_routes = Counter(
    f"pa{pa_port - prefill_base}:p{projection_port - projection_base}"
    for pa_port, projection_port in routes
)
expected_pa_routes = Counter()
expected_projection_routes = Counter()
expected_pair_routes = Counter()
errors = []
affinity_variable = (
    routing_policy == "conversation_affinity" and variable_turns
)
if routing_policy == "conversation_affinity" and not variable_turns:
    expected_group_indices = [
        (repetition * load_conversations + conversation) % pa_count
        for repetition in range(load_repetitions)
        for conversation in range(load_conversations)
        for _ in range(load_rounds)
    ]
elif affinity_variable:
    # Conversation assignment depends on the first-seen order. Validate the
    # endpoint range and request count instead of reconstructing interleaving.
    expected_group_indices = [0 for _ in range(expected_requests)]
else:
    expected_group_indices = [
        request_number % pa_count for request_number in range(expected_requests)
    ]

for request_number, group_index in enumerate(expected_group_indices):
    projection_index = request_number % projection_count
    if routing_policy == "crossbar_round_robin":
        projection_index = (
            request_number // pa_count + group_index
        ) % projection_count
    elif routing_policy == "projection_affinity":
        groups_per_projection = (
            pa_count + projection_count - 1
        ) // projection_count
        projection_index = min(
            group_index // groups_per_projection,
            projection_count - 1,
        )
    elif routing_policy == "projection_sticky":
        group_index = projection_index % pa_count
    elif routing_policy not in (
        "round_robin",
        "conversation_affinity",
    ):
        errors.append(f"unsupported routing policy {routing_policy!r}")
        group_index = 0
        projection_index = 0
    expected_pa_routes[prefill_base + group_index] += 1
    expected_projection_routes[projection_base + projection_index] += 1
    expected_pair_routes[f"pa{group_index}:p{projection_index}"] += 1

if len(routes) != expected_requests:
    errors.append(
        f"routed request count {len(routes)} != expected {expected_requests}"
    )
if affinity_variable:
    if any(
        port < prefill_base or port >= prefill_base + pa_count
        for port in pa_routes
    ):
        errors.append(f"PA routes contain an unknown endpoint: {dict(pa_routes)}")
elif pa_routes != expected_pa_routes:
    errors.append(
        f"PA route counts {dict(pa_routes)} != expected "
        f"{dict(expected_pa_routes)}"
    )
if projection_routes != expected_projection_routes:
    errors.append(
        f"Projection route counts {dict(projection_routes)} != expected "
        f"{dict(expected_projection_routes)}"
    )
if routing_policy != "conversation_affinity" and (
    pair_routes != expected_pair_routes
):
    errors.append(
        f"PA/Projection pair counts {dict(pair_routes)} != expected "
        f"{dict(expected_pair_routes)}"
    )

control_counts = {}
total_releases = 0
release_marker = '"POST /v1/pap/prefill/lease-release HTTP/1.1" 200 OK'
commit_marker = '"POST /v1/pap/prefill/decode-commit HTTP/1.1" 200 OK'
for index in range(pa_count):
    port = prefill_base + index
    prefill_text = (log_dir / f"prefill_{index}.log").read_text(
        errors="replace"
    )
    releases = prefill_text.count(release_marker)
    commits = prefill_text.count(commit_marker)
    routed = pa_routes[port]
    total_releases += releases
    control_counts[port] = {
        "routed_requests": routed,
        "decode_commit_200": commits,
        "lease_release_200": releases,
    }
    if releases != routed:
        errors.append(
            f"PA port {port} release count {releases} != routed {routed}"
        )
    if commits < routed:
        errors.append(
            f"PA port {port} commit count {commits} < routed {routed}"
        )

audit = {
    "status": "failed" if errors else "passed",
    "route_count": len(routes),
    "pa_routes": dict(sorted(pa_routes.items())),
    "projection_routes": dict(sorted(projection_routes.items())),
    "pair_routes": dict(sorted(pair_routes.items())),
    "expected_pair_routes": dict(sorted(expected_pair_routes.items())),
    "prefill_control_counts": control_counts,
    "total_lease_release_200": total_releases,
    "errors": errors,
}
with open(run_root / "routing_audit.json", "w", encoding="utf-8") as output:
    json.dump(audit, output, indent=2)
    output.write("\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
  then
    {
      printf 'STATUS=passed\n'
      printf 'EXPECTED_REQUESTS=%q\n' "${NUM_PROMPTS}"
      printf 'PA_COUNT=%q\n' "${PA_COUNT}"
      printf 'PROJECTION_COUNT=%q\n' "${PROJECTION_COUNT}"
      printf 'ROUTING_POLICY=%q\n' "${PAP_ROUTING_POLICY}"
    } > "${summary_path}"
  else
    printf 'STATUS=failed\n' > "${summary_path}"
    die "PAP x:y routing audit failed; see ${RUN_ROOT}/routing_audit.json"
  fi
}

cd "${ROOT_DIR}"

(( PA_COUNT >= 1 && PROJECTION_COUNT >= 1 )) \
  || die "PAP topology must contain at least one PA and one Projection"
[[ "${PAP_TP_SIZE}" == "1" ]] || die "This runner is intentionally fixed to PAP_TP_SIZE=1"
[[ "${PAP_ENABLE_MPS}" == "1" ]] \
  || die "the PAP benchmark runner requires static MPS"
[[ "${PAP_VLLM_DTYPE}" == "float16" ]] \
  || die "AIPerf requires PAP_VLLM_DTYPE=float16"
[[ "${PAP_PREFIX_CACHE_AUDIT}" == "0" \
  && "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" == "1" ]] \
  || die "AIPerf requires prompt details and forbids cache audit"
if (( PA_COUNT > 1 )); then
  [[ "${PAP_ROUTING_POLICY}" == "conversation_affinity" ]] \
    || die "multi-PA AIPerf requires conversation_affinity"
else
  [[ "${PAP_ROUTING_POLICY}" == "round_robin" \
    || "${PAP_ROUTING_POLICY}" == "conversation_affinity" ]] \
    || die "single-PA AIPerf requires a supported PAP routing policy"
fi
(( PAP_AIPERF_TURNS >= 2 || PAP_AIPERF_VARIABLE_TURNS == 1 )) \
  || die "PAP multi-turn benchmarks require at least two turns"
(( INPUT_LEN > 0 && PAP_AIPERF_APPEND_TOKENS > 0 && OUTPUT_LEN > 1 )) \
  || die "multi-turn token counts must be positive"
(( PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS >= OUTPUT_LEN )) \
  || die "PAP unified KV decode capacity is too small for load output"
[[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ -x "${VLLM_BIN}" ]] || die "VLLM_BIN is not executable: ${VLLM_BIN}"
[[ -f "${DEFERRED_TRACE_VALIDATOR}" ]] \
  || die "Missing deferred trace validator: ${DEFERRED_TRACE_VALIDATOR}"
[[ -f "${PROJECTION_MEMORY_PLANNER}" ]] \
  || die "Missing Projection memory planner: ${PROJECTION_MEMORY_PLANNER}"
[[ -x "${CUDA_GRAPH_AUDITOR}" ]] \
  || die "Missing CUDA Graph auditor: ${CUDA_GRAPH_AUDITOR}"
[[ -d "${MODEL_PATH}" ]] || die "Model path does not exist: ${MODEL_PATH}"

"${PYTHON_BIN}" - <<'PY' || die "PAP v0.26 plugin preflight failed"
import importlib.metadata as metadata
import vllm

version = vllm.__version__.split("+", 1)[0]
parts = version.split(".")
if len(parts) < 2 or tuple(map(int, parts[:2])) != (0, 26):
    raise SystemExit(f"expected vLLM 0.26, got {vllm.__version__}")

expected = {
    ("vllm.general_plugins", "pap"): "vllm.pap.plugin:register_pap_plugin",
    ("vllm.endpoint_plugins", "pap"): (
        "vllm.pap.endpoint_plugin:PAPEndpointPlugin"
    ),
}
for (group, name), value in expected.items():
    entries = [entry for entry in metadata.entry_points(group=group) if entry.name == name]
    if len(entries) != 1 or entries[0].value != value:
        raise SystemExit(f"invalid {group}/{name} entry point: {entries}")

from vllm.pap.kv_connector import PAPPrefillConnector  # noqa: F401

flashinfer = metadata.version("flashinfer-python")
cubin = metadata.version("flashinfer-cubin")
if flashinfer != cubin:
    raise SystemExit(
        f"FlashInfer package mismatch: flashinfer-python={flashinfer}, "
        f"flashinfer-cubin={cubin}"
    )
PY

ensure_dataset
mkdir -p "${RUN_ROOT}" "${RUN_LOG_DIR}"
capture_git_state
prepare_aiperf_dataset
split_csv "${PAP_PREFILL_GPUS}" PREFILL_GPUS
split_csv "${PAP_PROJECTION_GPUS}" PROJECTION_GPUS
require_count "PAP_PREFILL_GPUS" "${#PREFILL_GPUS[@]}" "${PA_COUNT}"
require_count \
  "PAP_PROJECTION_GPUS" "${#PROJECTION_GPUS[@]}" "${PROJECTION_COUNT}"

read -r MODEL_NUM_LAYERS MODEL_NUM_HEADS MODEL_NUM_KV_HEADS \
  MODEL_HEAD_DIM < <(
  "${PYTHON_BIN}" - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
num_layers = int(config["num_hidden_layers"])
num_heads = int(config["num_attention_heads"])
num_kv_heads = int(config["num_key_value_heads"])
head_dim = int(config.get("head_dim") or config["hidden_size"] // num_heads)
print(num_layers, num_heads, num_kv_heads, head_dim)
PY
)

projection_memory_args=(
  "${PROJECTION_MEMORY_PLANNER}"
  --model-path "${MODEL_PATH}"
  --tensor-parallel-size "${PAP_TP_SIZE}"
  --validation-kv-bytes "$((
    MODEL_NUM_LAYERS * MAX_MODEL_LEN * 2
    * MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM * 2 / PAP_TP_SIZE
  ))"
)
for gpu in "${PROJECTION_GPUS[@]}"; do
  projection_memory_args+=(--gpu-id "${gpu}")
done
read -r PROJECTION_GPU_MEMORY_UTILIZATION PROJECTION_MODEL_WEIGHT_BYTES \
  PROJECTION_PER_RANK_WEIGHT_BYTES PROJECTION_VALIDATION_KV_BYTES \
  PROJECTION_MEMORY_TARGET_BYTES PROJECTION_GPU_TOTAL_BYTES < <(
    "${PYTHON_BIN}" "${projection_memory_args[@]}"
  )
echo "Projection memory budget: utilization=${PROJECTION_GPU_MEMORY_UTILIZATION}, target_bytes=${PROJECTION_MEMORY_TARGET_BYTES}"

ports=("${PAP_PROXY_PORT}")
for (( idx=0; idx<PA_COUNT; idx++ )); do
  ports+=(
    "$((PREFILL_PORT_BASE + idx))"
    "$((ATTENTION_PORT_BASE + idx))"
    "$((ATTENTION_TCP_PORT_BASE + idx))"
    "$((VLLM_PREFILL_PORT_BASE + idx * 20))"
  )
done
for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
  ports+=(
    "$((PROJECTION_PORT_BASE + idx))"
    "$((VLLM_PROJECTION_PORT_BASE + idx * 20))"
  )
done
ensure_ports_free "${ports[@]}"

PAP_OFFLOAD_EXEC_NUM_HEADS="${PAP_OFFLOAD_EXEC_NUM_HEADS:-${MODEL_NUM_HEADS}}"
PAP_OFFLOAD_EXEC_NUM_KV_HEADS="${PAP_OFFLOAD_EXEC_NUM_KV_HEADS:-${MODEL_NUM_KV_HEADS}}"
PAP_OFFLOAD_EXEC_HEAD_DIM="${PAP_OFFLOAD_EXEC_HEAD_DIM:-${MODEL_HEAD_DIM}}"
PAP_OFFLOAD_EXEC_Q_SIZE="${PAP_OFFLOAD_EXEC_Q_SIZE:-$((PAP_OFFLOAD_EXEC_NUM_HEADS * PAP_OFFLOAD_EXEC_HEAD_DIM))}"
PAP_OFFLOAD_EXEC_KV_SIZE="${PAP_OFFLOAD_EXEC_KV_SIZE:-$((PAP_OFFLOAD_EXEC_NUM_KV_HEADS * PAP_OFFLOAD_EXEC_HEAD_DIM))}"
case "${PAP_VLLM_DTYPE}" in
  float16 | half | bfloat16)
    pap_nvshmem_dtype_bytes=2
    ;;
  float32 | float)
    pap_nvshmem_dtype_bytes=4
    ;;
  *)
    die "cannot size NVSHMEM buffers for dtype ${PAP_VLLM_DTYPE}"
    ;;
esac
pap_nvshmem_qkv_bytes=$((
  PAP_PROJECTION_MAX_NUM_SEQS
  * (PAP_OFFLOAD_EXEC_Q_SIZE + 2 * PAP_OFFLOAD_EXEC_KV_SIZE)
  * pap_nvshmem_dtype_bytes
))
pap_nvshmem_output_bytes=$((
  PAP_PROJECTION_MAX_NUM_SEQS
  * PAP_OFFLOAD_EXEC_Q_SIZE
  * pap_nvshmem_dtype_bytes
))
pap_nvshmem_required_bytes="${pap_nvshmem_qkv_bytes}"
if (( pap_nvshmem_output_bytes > pap_nvshmem_required_bytes )); then
  pap_nvshmem_required_bytes="${pap_nvshmem_output_bytes}"
fi
pap_nvshmem_alignment=$((2 * 1024 * 1024))
PAP_NVSHMEM_BUFFER_BYTES="${PAP_NVSHMEM_BUFFER_BYTES:-$((
  (pap_nvshmem_required_bytes + pap_nvshmem_alignment - 1)
  / pap_nvshmem_alignment
  * pap_nvshmem_alignment
))}"
PAP_NVSHMEM_CONTROL_BYTES="${PAP_NVSHMEM_CONTROL_BYTES:-32768}"

if [[ "${PAP_ENABLE_MPS}" == "1" ]]; then
  command -v nvidia-cuda-mps-control >/dev/null 2>&1 \
    || die "PAP_ENABLE_MPS=1 but nvidia-cuda-mps-control was not found"
  for (( idx=0; idx<PA_COUNT; idx++ )); do
    start_mps_for_pa "${idx}" "${PREFILL_GPUS[idx]}"
  done
fi

for (( idx=0; idx<PA_COUNT; idx++ )); do
  attention_port=$((ATTENTION_PORT_BASE + idx))
  attention_tcp_port=$((ATTENTION_TCP_PORT_BASE + idx))
  prefill_control_port=$((PREFILL_PORT_BASE + idx))
  decode_commit_endpoint="${PAP_DECODE_COMMIT_ENDPOINT:-http://127.0.0.1:${prefill_control_port}/v1/pap/prefill/decode-commit}"
  lease_release_endpoint="${PAP_LEASE_RELEASE_ENDPOINT:-http://127.0.0.1:${prefill_control_port}/v1/pap/prefill/lease-release}"
  attention_env=("CUDA_VISIBLE_DEVICES=${PREFILL_GPUS[idx]}")
  if [[ "${PAP_ENABLE_MPS}" == "1" ]]; then
    attention_env=(
      "CUDA_VISIBLE_DEVICES=0"
      "CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIRS[idx]}"
      "CUDA_MPS_LOG_DIRECTORY=${MPS_LOG_DIRS[idx]}"
    )
    attention_env+=(
      "CUDA_MPS_SM_PARTITION=${MPS_ATTENTION_PARTITIONS[idx]}"
    )
  fi
  echo "Starting PAP Attention ${idx} on GPU ${PREFILL_GPUS[idx]}"
  env \
    "${attention_env[@]}" \
    PAP_TOPOLOGY="${TOPOLOGY}" \
    PAP_PA_COUNT="${PA_COUNT}" \
    PAP_PROJECTION_COUNT="${PROJECTION_COUNT}" \
    PAP_NVSHMEM_RANK="$((PROJECTION_COUNT + idx))" \
    PAP_NVSHMEM_WORLD_SIZE="${PAP_NVSHMEM_WORLD_SIZE}" \
    PAP_NVSHMEM_UID_FILE="${PAP_NVSHMEM_UID_FILE}" \
    PAP_NVSHMEM_BUFFER_BYTES="${PAP_NVSHMEM_BUFFER_BYTES:-}" \
    PAP_NVSHMEM_CONTROL_BYTES="${PAP_NVSHMEM_CONTROL_BYTES:-}" \
    PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
    PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE}" \
    PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}" \
    PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE}" \
    PAP_DEFERRED_TRACE_ROLE=attention \
    PAP_DEFERRED_TRACE_OUTPUT= \
    PAP_OFFLOAD_EXEC_LOCAL_RANK=0 \
    PAP_OFFLOAD_EXEC_Q_SIZE="${PAP_OFFLOAD_EXEC_Q_SIZE}" \
    PAP_OFFLOAD_EXEC_KV_SIZE="${PAP_OFFLOAD_EXEC_KV_SIZE}" \
    PAP_OFFLOAD_EXEC_NUM_HEADS="${PAP_OFFLOAD_EXEC_NUM_HEADS}" \
    PAP_OFFLOAD_EXEC_NUM_KV_HEADS="${PAP_OFFLOAD_EXEC_NUM_KV_HEADS}" \
    PAP_OFFLOAD_EXEC_HEAD_DIM="${PAP_OFFLOAD_EXEC_HEAD_DIM}" \
    PAP_DECODE_COMMIT_ENDPOINT="${decode_commit_endpoint}" \
    PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH="$(
      if [[ "${PAP_RUNTIME_CUDA_CONTEXT_AUDIT}" == "1" ]]; then
        printf '%s' "${RUN_ROOT}/runtime_cuda_context_attention_${idx}.json"
      fi
    )" \
    PAP_LEASE_RELEASE_ENDPOINT="${lease_release_endpoint}" \
    "${PYTHON_BIN}" -m vllm.pap.service \
      --host 127.0.0.1 \
      --port "${attention_port}" \
      --tcp-port "${attention_tcp_port}" \
      > "${RUN_LOG_DIR}/attention_${idx}_0.log" 2>&1 &
  PIDS+=("$!")
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
  wait_for_http \
    "http://127.0.0.1:$((ATTENTION_PORT_BASE + idx))/health" \
    "PAP Attention ${idx}"
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
  prefill_port=$((PREFILL_PORT_BASE + idx))
  prefill_env=("CUDA_VISIBLE_DEVICES=${PREFILL_GPUS[idx]}")
  if [[ "${PAP_ENABLE_MPS}" == "1" ]]; then
    prefill_env=(
      "CUDA_VISIBLE_DEVICES=0"
      "CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIRS[idx]}"
      "CUDA_MPS_LOG_DIRECTORY=${MPS_LOG_DIRS[idx]}"
    )
    prefill_env+=(
      "CUDA_MPS_SM_PARTITION=${MPS_PREFILL_PARTITIONS[idx]}"
    )
  fi
  echo "Starting PAP Prefill ${idx} on GPU ${PREFILL_GPUS[idx]}"
  prefill_profiler_args=()
  if [[ "${PAP_PREFILL_TORCH_PROFILE}" == "1" ]]; then
    prefill_profile_dir="${RUN_ROOT}/prefill_torch_profile_${idx}"
    mkdir -p "${prefill_profile_dir}"
    prefill_profiler_args=(
      --profiler-config
      "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${prefill_profile_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"max_iterations\":${PAP_PREFILL_TORCH_PROFILE_MAX_ITERATIONS}}"
    )
  fi
  env \
    "${prefill_env[@]}" \
    VLLM_PORT="$((VLLM_PREFILL_PORT_BASE + idx * 20))" \
    PAP_TOPOLOGY="${TOPOLOGY}" \
    PAP_PA_COUNT="${PA_COUNT}" \
    PAP_PROJECTION_COUNT="${PROJECTION_COUNT}" \
    PAP_DEFERRED_CUDA_TRACE=0 \
    PAP_DEFERRED_TRACE_ROLE= \
    PAP_DEFERRED_TRACE_OUTPUT= \
    PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
    PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE}" \
    PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS}" \
    PAP_PREFIX_CACHE_AUDIT="${PAP_PREFIX_CACHE_AUDIT}" \
    PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH="$(
      if [[ "${PAP_RUNTIME_CUDA_CONTEXT_AUDIT}" == "1" ]]; then
        printf '%s' "${RUN_ROOT}/runtime_cuda_context_prefill_${idx}.json"
      fi
    )" \
    PAP_RUNTIME_CUDA_CONTEXT_ROLE=prefill \
    VLLM_PLUGINS=pap \
    PAP_MODEL_HOOKS=1 \
    PAP_CUDAGRAPH_COMPATIBLE=1 \
    PAP_CUDAGRAPH_ROLE=prefill \
    PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --port "${prefill_port}" \
      --host 127.0.0.1 \
      "${PREFILL_EXECUTION_ARGS[@]}" \
      --generation-config vllm \
      --dtype "${PAP_VLLM_DTYPE}" \
      --enable-request-id-headers \
      "${PREFILL_OBSERVABILITY_ARGS[@]}" \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --block-size "${PAP_BLOCK_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${PAP_PREFILL_MAX_NUM_SEQS}" \
      --max-num-batched-tokens \
        "${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      "${TOOL_ARGS[@]}" \
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --gpu-memory-utilization "${PAP_PREFILL_GPU_MEMORY_UTILIZATION}" \
      "${prefill_profiler_args[@]}" \
      --kv-transfer-config \
        '{"kv_connector":"PAPPrefillConnector","kv_connector_module_path":"vllm.pap.kv_connector","kv_role":"kv_producer"}' \
      > "${RUN_LOG_DIR}/prefill_${idx}.log" 2>&1 &
  PIDS+=("$!")
done

for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
  projection_port=$((PROJECTION_PORT_BASE + idx))
  echo "Starting PAP Projection ${idx} on GPU ${PROJECTION_GPUS[idx]}"
  env \
    CUDA_VISIBLE_DEVICES="${PROJECTION_GPUS[idx]}" \
    VLLM_PORT="$((VLLM_PROJECTION_PORT_BASE + idx * 20))" \
    PAP_TOPOLOGY="${TOPOLOGY}" \
    PAP_PA_COUNT="${PA_COUNT}" \
    PAP_PROJECTION_COUNT="${PROJECTION_COUNT}" \
    PAP_NVSHMEM_RANK="${idx}" \
    PAP_NVSHMEM_WORLD_SIZE="${PAP_NVSHMEM_WORLD_SIZE}" \
    PAP_NVSHMEM_UID_FILE="${PAP_NVSHMEM_UID_FILE}" \
    PAP_NVSHMEM_BUFFER_BYTES="${PAP_NVSHMEM_BUFFER_BYTES:-}" \
    PAP_NVSHMEM_CONTROL_BYTES="${PAP_NVSHMEM_CONTROL_BYTES:-}" \
    PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE}" \
    PAP_DEFERRED_TRACE_ROLE=projection \
    PAP_DEFERRED_TRACE_OUTPUT="${RUN_ROOT}/projection_deferred_trace_${idx}.json" \
    PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
    PAP_ATTENTION_PORT_BASE="${ATTENTION_PORT_BASE}" \
    PAP_TP_SIZE="${PAP_TP_SIZE}" \
    PAP_PROJECTION_KV_UNAWARE=1 \
    VLLM_PLUGINS=pap \
    PAP_MODEL_HOOKS=1 \
    PAP_CUDAGRAPH_COMPATIBLE=1 \
    PAP_CUDAGRAPH_ROLE=projection \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --port "${projection_port}" \
      --host 127.0.0.1 \
      "${PROJECTION_EXECUTION_ARGS[@]}" \
      --generation-config vllm \
      --dtype "${PAP_VLLM_DTYPE}" \
      --enable-request-id-headers \
      --block-size "${PAP_BLOCK_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${PAP_PROJECTION_MAX_NUM_SEQS}" \
      --max-num-batched-tokens \
        "${PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS}" \
      "${TOOL_ARGS[@]}" \
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --gpu-memory-utilization "${PROJECTION_GPU_MEMORY_UTILIZATION}" \
      > "${RUN_LOG_DIR}/projection_${idx}.log" 2>&1 &
  PIDS+=("$!")
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
  wait_for_http \
    "http://127.0.0.1:$((PREFILL_PORT_BASE + idx))/health" \
    "PAP Prefill ${idx}"
  curl -fsS "http://127.0.0.1:$((PREFILL_PORT_BASE + idx))/openapi.json" \
    | jq -e '
        .paths
        | has("/v1/pap/prefill/decode-commit")
          and has("/v1/pap/prefill/lease-release")
      ' >/dev/null \
    || die "PAP Prefill ${idx} control plugin routes are missing"
done
for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
  wait_for_http \
    "http://127.0.0.1:$((PROJECTION_PORT_BASE + idx))/health" \
    "PAP Projection ${idx}"
done

PAP_VLLM_GRAPH_LOGS=()
for (( idx=0; idx<PA_COUNT; idx++ )); do
  PAP_VLLM_GRAPH_LOGS+=("${RUN_LOG_DIR}/prefill_${idx}.log")
done
"${CUDA_GRAPH_AUDITOR}" "${RUN_ROOT}/prefill_cuda_graph_audit.env" \
  PIECEWISE "${PAP_VLLM_GRAPH_LOGS[@]}"
audit_projection_outer_graph_configuration

audit_runtime_cuda_contexts

PAP_GROUPS_SPEC="$(build_pap_groups_spec)"
PROJECTIONS_SPEC="$(build_projections_spec)"

echo "Starting PAP Gateway on port ${PAP_PROXY_PORT}"
env \
  "${PYTHON_BIN}" -m vllm.pap.gateway.app \
  --host 127.0.0.1 \
  --port "${PAP_PROXY_PORT}" \
  --pap-groups "${PAP_GROUPS_SPEC}" \
  --projections "${PROJECTIONS_SPEC}" \
  --routing-policy "${PAP_ROUTING_POLICY}" \
  > "${RUN_LOG_DIR}/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:${PAP_PROXY_PORT}/health" "PAP Gateway"
wait_cluster_stable
audit_projection_scheduling

write_effective_config
write_topology_manifest
write_run_metadata
start_prefill_torch_profiles

TAG="${TOPOLOGY_TAG}_aiperf"
echo "=== Running ${TAG} on port ${PAP_PROXY_PORT} ==="
timeout "${BENCH_TIMEOUT}" env \
  PAP_ROOT="${ROOT_DIR}" \
  AIPERF_ROOT="${AIPERF_ROOT}" \
  AIPERF_BIN="${AIPERF_BIN}" \
  MODEL_PATH="${MODEL_PATH}" \
  AIPERF_INPUT_FILE="${PAP_AIPERF_INPUT_FILE}" \
  AIPERF_CUSTOM_DATASET_TYPE="${AIPERF_CUSTOM_DATASET_TYPE:-multi-turn}" \
  AIPERF_TARGET_URL="http://127.0.0.1:${PAP_PROXY_PORT}" \
  AIPERF_OUTPUT_DIR="${PAP_AIPERF_OUTPUT_DIR}" \
  AIPERF_SESSIONS="${PAP_AIPERF_SESSIONS}" \
  AIPERF_CONCURRENCY="${PAP_AIPERF_CONCURRENCY}" \
  AIPERF_TIMING_MODE="${PAP_AIPERF_TIMING_MODE}" \
  AIPERF_REQUEST_RATE="${PAP_AIPERF_REQUEST_RATE}" \
  AIPERF_REQUEST_TIMEOUT_SECONDS="${BENCH_TIMEOUT}" \
  "${AIPERF_RUNNER}" \
  2>&1 | tee "${RUN_ROOT}/${TAG}.log"
if [[ -z "$(find "${PAP_AIPERF_OUTPUT_DIR}" -type f \
  -name 'profile*.json' -size +0c -print -quit)" ]]; then
  die "AIPerf produced no profile JSON under ${PAP_AIPERF_OUTPUT_DIR}"
fi

wait_prefill_torch_profiles
wait_gateway_requests_drained
wait_attention_sessions_drained
capture_proxy_topology_stats
capture_attention_fast_path_stats
audit_pap_whole_step_graph
audit_decode_token_join
capture_projection_deferred_traces
audit_xy_routes
audit_correctness_logs

echo "RUN_ROOT=${RUN_ROOT}"
