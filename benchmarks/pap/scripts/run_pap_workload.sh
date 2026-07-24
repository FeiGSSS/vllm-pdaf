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
        replacement="the AIPerf static 72/20 MPS partition"
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
DEFERRED_TRACE_VALIDATOR="${ROOT_DIR}/benchmarks/pap/tooling/validate_deferred_trace.py"
PROJECTION_MEMORY_PLANNER="${ROOT_DIR}/vllm/pap/model/memory.py"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
AIPERF_DATASET_GENERATOR="${ROOT_DIR}/benchmarks/pap/aiperf/generate_multiturn_dataset.py"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${AIPERF_ROOT}/.venv/bin/aiperf}"
PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE:-0}"
PAP_BENCH_STRICT_CORRECTNESS_AUDIT="${PAP_BENCH_STRICT_CORRECTNESS_AUDIT:-1}"
PAP_BENCH_CLIENT="aiperf"

GIT_COMMIT=""
GIT_COMMIT_SHORT=""
GIT_TRACKED_WORKTREE_DIRTY=0

MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
BENCH_DIR="${BENCH_DIR:-/home/fei/research/PD/refer_codes/vllm/benchmarks}"
DATASET_PATH="${DATASET_PATH:-${BENCH_DIR}/sonnet_4x.txt}"

INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PAP_AIPERF_TURNS="${PAP_AIPERF_TURNS:-10}"
PAP_AIPERF_SESSIONS="${PAP_AIPERF_SESSIONS:-32}"
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
NUM_PROMPTS=$((PAP_AIPERF_TURNS * PAP_AIPERF_SESSIONS))
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
CLUSTER_READY_WAIT_SECONDS="${CLUSTER_READY_WAIT_SECONDS:-30}"
PAP_BENCH_SESSION_DRAIN_TIMEOUT="${PAP_BENCH_SESSION_DRAIN_TIMEOUT:-15}"
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
PAP_AIPERF_CONCURRENCY="${PAP_AIPERF_CONCURRENCY:-12}"
PAP_AIPERF_TIMING_MODE="${PAP_AIPERF_TIMING_MODE:-concurrency}"
PAP_AIPERF_REQUEST_RATE="${PAP_AIPERF_REQUEST_RATE-}"
if [[ ! "${PAP_AIPERF_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAP_AIPERF_CONCURRENCY must be positive" >&2
  exit 2
fi
if (( PAP_AIPERF_CONCURRENCY > PAP_AIPERF_SESSIONS )); then
  echo "ERROR: AIPerf concurrency exceeds total sessions" >&2
  exit 2
fi
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
ATTENTION_ZMQ_PORT_BASE="${PAP_ATTENTION_ZMQ_PORT_BASE:-${PAP_ATTENTION_ZMQ_PORT:-10300}}"
PROJECTION_ZMQ_PORT_BASE="${PAP_PROJECTION_ZMQ_PORT_BASE:-${PAP_PROJECTION_ZMQ_PORT:-11300}}"
PREFILL_NIXL_PORT_BASE="${PAP_PREFILL_NIXL_PORT_BASE:-${PAP_PREFILL_NIXL_PORT:-5559}}"
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
PAP_PREFILL_MAX_NUM_SEQS="${PAP_PREFILL_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS="${PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS:-64}"
PAP_PROJECTION_MAX_NUM_SEQS="${PAP_PROJECTION_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
PAP_EXECUTION_MODE="${PAP_EXECUTION_MODE:-eager}"
PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES="${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,32,64,128}"
PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES="${PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,20,24,28,32}"
PAP_PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.90}"
PAP_PREFILL_MPS_PERCENT="${PAP_PREFILL_MPS_PERCENT:-80}"
PAP_ATTENTION_MPS_PERCENT="${PAP_ATTENTION_MPS_PERCENT:-20}"
PAP_STATIC_PREFILL_CHUNKS="${PAP_STATIC_PREFILL_CHUNKS:-18}"
PAP_STATIC_ATTENTION_CHUNKS="${PAP_STATIC_ATTENTION_CHUNKS:-5}"
PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_STATIC_PREFILL_EXPECTED_SMS:-72}"
PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_STATIC_ATTENTION_EXPECTED_SMS:-20}"
PAP_ENABLE_MPS=1
if ! [[ "${PAP_STATIC_PREFILL_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_STATIC_ATTENTION_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: static MPS chunk counts must be positive integers" >&2
  exit 2
fi
PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-local_fast}"
PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"
PAP_OFFLOAD_EXEC_TRACE="${PAP_OFFLOAD_EXEC_TRACE:-0}"
PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE:-0}"
PAP_DEFERRED_CUDA_TRACE_MAX_PENDING="${PAP_DEFERRED_CUDA_TRACE_MAX_PENDING:-1024}"
PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND="${PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND:-1}"
PAP_NIXL_MAILBOX_INLINE_PUBLISH="${PAP_NIXL_MAILBOX_INLINE_PUBLISH:-1}"
PAP_UNIFIED_MD_CACHE_LIMIT="${PAP_UNIFIED_MD_CACHE_LIMIT:-256}"
PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT:-256}"
PAP_ATTENTION_DISPATCH_QUEUE_SIZE="${PAP_ATTENTION_DISPATCH_QUEUE_SIZE:-0}"
DEFAULT_ATTENTION_COMBINE_WAIT_US=0
if (( PROJECTION_COUNT > 1 )); then
  DEFAULT_ATTENTION_COMBINE_WAIT_US=200
  if (( PA_COUNT > 1 )); then
    DEFAULT_ATTENTION_COMBINE_WAIT_US=1000
  fi
fi
PAP_DIRECT_MAILBOX_OUTPUT="${PAP_DIRECT_MAILBOX_OUTPUT:-}"
if [[ -z "${PAP_DIRECT_MAILBOX_OUTPUT}" ]]; then
  if [[ "${PAP_OFFLOAD_EXEC_TRANSPORT}" == "local_fast" ]]; then
    PAP_DIRECT_MAILBOX_OUTPUT=1
  else
    PAP_DIRECT_MAILBOX_OUTPUT=0
  fi
fi
PAP_LOCAL_FAST_SPIN_ITERS="${PAP_LOCAL_FAST_SPIN_ITERS:-2048}"
PAP_LOCAL_FAST_YIELD_ITERS="${PAP_LOCAL_FAST_YIELD_ITERS:-64}"
PAP_LOCAL_FAST_SLEEP_US="${PAP_LOCAL_FAST_SLEEP_US:-20}"
PAP_LOCAL_FAST_SLEEP_AFTER_US="${PAP_LOCAL_FAST_SLEEP_AFTER_US:-50}"
PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM:-16}"
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
if ! [[ "${PAP_DEFERRED_TRACE_FLUSH_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PAP_DEFERRED_TRACE_FLUSH_TIMEOUT must be a positive integer" >&2
  exit 2
fi

case "${PAP_EXECUTION_MODE}" in
  eager | piecewise) ;;
  *)
    echo "ERROR: PAP_EXECUTION_MODE must be eager or piecewise" >&2
    exit 2
    ;;
esac
for capture_sizes in \
  "${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES}" \
  "${PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES}"; do
  if ! [[ "${capture_sizes}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "ERROR: CUDA Graph capture sizes must be comma-separated positive integers" >&2
    exit 2
  fi
done

PAP_CUDAGRAPH_COMPATIBLE=0
PREFILL_EXECUTION_ARGS=(--enforce-eager)
PROJECTION_EXECUTION_ARGS=(--enforce-eager)
PAP_PREFILL_COMPILATION_CONFIG=""
PAP_PROJECTION_COMPILATION_CONFIG=""
if [[ "${PAP_EXECUTION_MODE}" == "piecewise" ]]; then
  for incompatible_flag in \
    PAP_OFFLOAD_EXEC_TRACE \
    PAP_DEFERRED_CUDA_TRACE \
    PAP_PREFILL_TORCH_PROFILE \
    PAP_PROJECTION_CRITICAL_TRACE \
    VLLM_QWEN3_LAYER_PROFILE; do
    case "${!incompatible_flag:-0}" in
      1 | true | True | TRUE | yes | Yes | YES | on | On | ON)
        echo "ERROR: ${incompatible_flag} is incompatible with PAP piecewise CUDA Graph" >&2
        exit 2
        ;;
    esac
  done
  PAP_CUDAGRAPH_COMPATIBLE=1
  PAP_PREFILL_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES}]}"
  PAP_PROJECTION_COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES}]}"
  PREFILL_EXECUTION_ARGS=(
    --compilation-config "${PAP_PREFILL_COMPILATION_CONFIG}"
  )
  PROJECTION_EXECUTION_ARGS=(
    --compilation-config "${PAP_PROJECTION_COMPILATION_CONFIG}"
  )
fi

export PAP_OFFLOAD_EXEC_TRACE
export PAP_DEFERRED_CUDA_TRACE
export PAP_DEFERRED_CUDA_TRACE_MAX_PENDING
export PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND
export PAP_NIXL_MAILBOX_INLINE_PUBLISH
export PAP_UNIFIED_MD_CACHE_LIMIT
export PAP_DECODE_SLOT_PLAN_CACHE_LIMIT
export PAP_ATTENTION_DISPATCH_QUEUE_SIZE
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
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
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
      "127.0.0.1:$((PREFILL_PORT_BASE + idx)):$((PREFILL_NIXL_PORT_BASE + idx)):127.0.0.1:$((ATTENTION_PORT_BASE + idx)):$((ATTENTION_TCP_PORT_BASE + idx)):$((ATTENTION_ZMQ_PORT_BASE + idx))"
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
    printf 'PAP_AIPERF_APPEND_TOKENS=%q\n' \
      "${PAP_AIPERF_APPEND_TOKENS}"
    printf 'AIPERF_ROOT=%q\n' "${AIPERF_ROOT}"
    printf 'AIPERF_BIN=%q\n' "${AIPERF_BIN}"
    printf 'PAP_AIPERF_INPUT_FILE=%q\n' "${PAP_AIPERF_INPUT_FILE}"
    printf 'PAP_AIPERF_OUTPUT_DIR=%q\n' "${PAP_AIPERF_OUTPUT_DIR}"
    printf 'PAP_AIPERF_CONCURRENCY=%q\n' "${PAP_AIPERF_CONCURRENCY}"
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
    printf 'PROJECTION_MEMORY_POLICY=%q\n' "model_weights_x1.20"
    printf 'PROJECTION_GPU_MEMORY_UTILIZATION=%q\n' \
      "${PROJECTION_GPU_MEMORY_UTILIZATION}"
    printf 'PROJECTION_MODEL_WEIGHT_BYTES=%q\n' \
      "${PROJECTION_MODEL_WEIGHT_BYTES}"
    printf 'PROJECTION_PER_RANK_WEIGHT_BYTES=%q\n' \
      "${PROJECTION_PER_RANK_WEIGHT_BYTES}"
    printf 'PROJECTION_MEMORY_TARGET_BYTES=%q\n' \
      "${PROJECTION_MEMORY_TARGET_BYTES}"
    printf 'PROJECTION_GPU_TOTAL_BYTES=%q\n' \
      "${PROJECTION_GPU_TOTAL_BYTES}"
    printf 'PAP_PREFILL_MPS_PERCENT=%q\n' "${PAP_PREFILL_MPS_PERCENT}"
    printf 'PAP_ATTENTION_MPS_PERCENT=%q\n' "${PAP_ATTENTION_MPS_PERCENT}"
    printf 'PAP_STATIC_PREFILL_CHUNKS=%q\n' \
      "${PAP_STATIC_PREFILL_CHUNKS}"
    printf 'PAP_STATIC_ATTENTION_CHUNKS=%q\n' \
      "${PAP_STATIC_ATTENTION_CHUNKS}"
    printf 'PAP_STATIC_PREFILL_EXPECTED_SMS=%q\n' \
      "${PAP_STATIC_PREFILL_EXPECTED_SMS}"
    printf 'PAP_STATIC_ATTENTION_EXPECTED_SMS=%q\n' \
      "${PAP_STATIC_ATTENTION_EXPECTED_SMS}"
    printf 'PAP_ENABLE_MPS=%q\n' "${PAP_ENABLE_MPS}"
    printf 'PAP_OFFLOAD_EXEC_TRANSPORT=%q\n' "${PAP_OFFLOAD_EXEC_TRANSPORT}"
    printf 'PAP_OFFLOAD_KV_TRANSPORT=%q\n' "${PAP_OFFLOAD_KV_TRANSPORT}"
    printf 'PAP_OFFLOAD_EXEC_TRACE=%q\n' "${PAP_OFFLOAD_EXEC_TRACE}"
    printf 'PAP_DEFERRED_CUDA_TRACE=%q\n' "${PAP_DEFERRED_CUDA_TRACE}"
    printf 'PAP_DEFERRED_CUDA_TRACE_MAX_PENDING=%q\n' \
      "${PAP_DEFERRED_CUDA_TRACE_MAX_PENDING}"
    printf 'PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND=%q\n' "${PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND}"
    printf 'PAP_NIXL_MAILBOX_INLINE_PUBLISH=%q\n' "${PAP_NIXL_MAILBOX_INLINE_PUBLISH}"
    printf 'PAP_UNIFIED_MD_CACHE_LIMIT=%q\n' "${PAP_UNIFIED_MD_CACHE_LIMIT}"
    printf 'PAP_DECODE_SLOT_PLAN_CACHE_LIMIT=%q\n' "${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}"
    printf 'PAP_ATTENTION_DISPATCH_QUEUE_SIZE=%q\n' \
      "${PAP_ATTENTION_DISPATCH_QUEUE_SIZE}"
    printf 'PAP_DIRECT_MAILBOX_OUTPUT=%q\n' "${PAP_DIRECT_MAILBOX_OUTPUT}"
    printf 'PAP_LOCAL_FAST_SPIN_ITERS=%q\n' "${PAP_LOCAL_FAST_SPIN_ITERS}"
    printf 'PAP_LOCAL_FAST_YIELD_ITERS=%q\n' "${PAP_LOCAL_FAST_YIELD_ITERS}"
    printf 'PAP_LOCAL_FAST_SLEEP_US=%q\n' "${PAP_LOCAL_FAST_SLEEP_US}"
    printf 'PAP_LOCAL_FAST_SLEEP_AFTER_US=%q\n' "${PAP_LOCAL_FAST_SLEEP_AFTER_US}"
    printf 'PAP_PREFILL_GPUS=%q\n' "${PAP_PREFILL_GPUS}"
    printf 'PAP_PROJECTION_GPUS=%q\n' "${PAP_PROJECTION_GPUS}"
    printf 'PAP_VLLM_DTYPE=%q\n' "${PAP_VLLM_DTYPE}"
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
    printf 'EXECUTION_MODE=%q\n' "${PAP_EXECUTION_MODE}"
    printf 'PAP_CUDAGRAPH_COMPATIBLE=%q\n' \
      "${PAP_CUDAGRAPH_COMPATIBLE}"
    printf 'PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
      "${PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES}"
    printf 'PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES=%q\n' \
      "${PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES}"
    printf 'PAP_PREFILL_COMPILATION_CONFIG=%q\n' \
      "${PAP_PREFILL_COMPILATION_CONFIG}"
    printf 'PAP_PROJECTION_COMPILATION_CONFIG=%q\n' \
      "${PAP_PROJECTION_COMPILATION_CONFIG}"
    printf 'CLUSTER_READY_WAIT_SECONDS=%q\n' "${CLUSTER_READY_WAIT_SECONDS}"
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
  PREFILL_NIXL_PORT_BASE="${PREFILL_NIXL_PORT_BASE}" \
  ATTENTION_PORT_BASE="${ATTENTION_PORT_BASE}" \
  ATTENTION_TCP_PORT_BASE="${ATTENTION_TCP_PORT_BASE}" \
  ATTENTION_ZMQ_PORT_BASE="${ATTENTION_ZMQ_PORT_BASE}" \
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
            "prefill_nixl_port": base("PREFILL_NIXL_PORT_BASE") + index,
            "attention": (
                f"http://127.0.0.1:{base('ATTENTION_PORT_BASE') + index}"
            ),
            "attention_tcp_port": base("ATTENTION_TCP_PORT_BASE") + index,
            "attention_zmq_port": base("ATTENTION_ZMQ_PORT_BASE") + index,
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
  PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT}" \
  PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
  PAP_DIRECT_MAILBOX_OUTPUT="${PAP_DIRECT_MAILBOX_OUTPUT}" \
  PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE}" \
  PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}" \
  PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" \
  PAP_AIPERF_TURNS="${PAP_AIPERF_TURNS}" \
  PAP_AIPERF_SESSIONS="${PAP_AIPERF_SESSIONS}" \
  PAP_AIPERF_APPEND_TOKENS="${PAP_AIPERF_APPEND_TOKENS}" \
  PAP_AIPERF_CONCURRENCY="${PAP_AIPERF_CONCURRENCY}" \
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
attention_dispatch_mode = (
    "direct" if projection_count == 1 else "central_combine"
)
attention_combine_wait_us = (
    0.0 if projection_count == 1 else (200.0 if pa_count == 1 else 1000.0)
)
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
    "offload_exec_transport": os.environ["PAP_OFFLOAD_EXEC_TRANSPORT"],
    "offload_kv_transport": os.environ["PAP_OFFLOAD_KV_TRANSPORT"],
    "batched_route_copy": True,
    "direct_mailbox_output": os.environ["PAP_DIRECT_MAILBOX_OUTPUT"] == "1",
    "unified_md_fast_key": True,
    "local_fast_stream_ordered": True,
    "projection_async_scheduling": True,
    "projection_scheduler_queue_depth": 2,
    "projection_runner_microbatch_pipeline": False,
    "prefill_kv_async": True,
    "prefill_ipc_profile": os.environ["PAP_PREFILL_IPC_PROFILE"] == "1",
    "kv_handoff_mode": "sealed_manifest",
    "decode_slot_plan_cache_limit": int(
        os.environ["PAP_DECODE_SLOT_PLAN_CACHE_LIMIT"]
    ),
    "attention_dispatch_mode": attention_dispatch_mode,
    "attention_combine_wait_us": attention_combine_wait_us,
    "attention_active_peer_tracking": projection_count > 1,
    "prompt_tokens_details": (
        os.environ["PAP_ENABLE_PROMPT_TOKENS_DETAILS"] == "1"
    ),
    "aiperf_turns": int(os.environ["PAP_AIPERF_TURNS"]),
    "aiperf_sessions": int(os.environ["PAP_AIPERF_SESSIONS"]),
    "aiperf_append_tokens": int(os.environ["PAP_AIPERF_APPEND_TOKENS"]),
    "aiperf_concurrency": int(os.environ["PAP_AIPERF_CONCURRENCY"]),
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
  pattern='CUDA out of memory|EngineDeadError|Traceback|NIXL.*failed|PAP local fast.*failed|PAP decode commit failed|new_token_ids length must match new_seq_len delta|PAP decode commit flush timed out|PAP decode commit queue full|PAP decode-token delivery failed|PAP decode-token queue is full|PAP decode-token join flush timed out|PAP lease release failed|PAP unified KV append out of range|PAP unified KV state missing|PAP unified KV state changed during decode append|PAP unified KV seq_len changed during decode append|prefill KV must reach the registered prefix before unified decode attention|PAP unified paged FlashAttention failed'

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
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import re
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
expected_group_indices = []
if routing_policy == "conversation_affinity":
    if projection_count != 1:
        errors.append("conversation-affinity load audit requires one Projection")
    expected_group_indices = [
        conversation % pa_count
        for _ in range(load_rounds)
        for conversation in range(load_conversations)
    ]
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
    elif routing_policy not in ("round_robin", "conversation_affinity"):
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
if pa_routes != expected_pa_routes:
    errors.append(
        f"PA route counts {dict(pa_routes)} != expected "
        f"{dict(expected_pa_routes)}"
    )
if projection_routes != expected_projection_routes:
    errors.append(
        f"Projection route counts {dict(projection_routes)} != expected "
        f"{dict(expected_projection_routes)}"
    )
if pair_routes != expected_pair_routes:
    errors.append(
        f"PA/Projection pair counts {dict(pair_routes)} != expected "
        f"{dict(expected_pair_routes)}"
    )

control_counts = {}
release_marker = '"POST /v1/pap/prefill/lease-release HTTP/1.1" 200 OK'
commit_marker = '"POST /v1/pap/prefill/decode-commit HTTP/1.1" 200 OK'
for index in range(pa_count):
    port = prefill_base + index
    prefill_text = (log_dir / f"prefill_{index}.log").read_text(errors="replace")
    releases = prefill_text.count(release_marker)
    commits = prefill_text.count(commit_marker)
    routed = pa_routes[port]
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
[[ "${PAP_STATIC_PREFILL_CHUNKS}" == "18" \
  && "${PAP_STATIC_ATTENTION_CHUNKS}" == "5" ]] \
  || die "the AIPerf PAP path requires 18/5 static-MPS chunks"
if (( PA_COUNT > 1 )); then
  [[ "${PAP_ROUTING_POLICY}" == "conversation_affinity" ]] \
    || die "multi-PA AIPerf requires conversation_affinity routing"
else
  [[ "${PAP_ROUTING_POLICY}" == "round_robin" \
    || "${PAP_ROUTING_POLICY}" == "conversation_affinity" ]] \
    || die "single-PA AIPerf requires round_robin or conversation_affinity"
fi
(( PROJECTION_COUNT == 1 )) \
  || die "AIPerf currently requires one Projection"
(( PAP_AIPERF_TURNS >= 4 )) \
  || die "AIPerf requires at least four turns"
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
[[ -d "${MODEL_PATH}" ]] || die "Model path does not exist: ${MODEL_PATH}"

"${PYTHON_BIN}" -c 'import nixl' >/dev/null 2>&1 \
  || die "Python package 'nixl' is not installed in .venv"

ensure_dataset
mkdir -p "${RUN_ROOT}" "${RUN_LOG_DIR}"
capture_git_state
prepare_aiperf_dataset
split_csv "${PAP_PREFILL_GPUS}" PREFILL_GPUS
split_csv "${PAP_PROJECTION_GPUS}" PROJECTION_GPUS
require_count "PAP_PREFILL_GPUS" "${#PREFILL_GPUS[@]}" "${PA_COUNT}"
require_count \
  "PAP_PROJECTION_GPUS" "${#PROJECTION_GPUS[@]}" "${PROJECTION_COUNT}"

projection_memory_args=(
  "${PROJECTION_MEMORY_PLANNER}"
  --model-path "${MODEL_PATH}"
  --tensor-parallel-size "${PAP_TP_SIZE}"
)
for gpu in "${PROJECTION_GPUS[@]}"; do
  projection_memory_args+=(--gpu-id "${gpu}")
done
read -r PROJECTION_GPU_MEMORY_UTILIZATION PROJECTION_MODEL_WEIGHT_BYTES \
  PROJECTION_PER_RANK_WEIGHT_BYTES PROJECTION_MEMORY_TARGET_BYTES \
  PROJECTION_GPU_TOTAL_BYTES < <(
    "${PYTHON_BIN}" "${projection_memory_args[@]}"
  )
echo "Projection memory budget: utilization=${PROJECTION_GPU_MEMORY_UTILIZATION}, target_bytes=${PROJECTION_MEMORY_TARGET_BYTES}"

ports=("${PAP_PROXY_PORT}")
for (( idx=0; idx<PA_COUNT; idx++ )); do
  ports+=(
    "$((PREFILL_PORT_BASE + idx))"
    "$((ATTENTION_PORT_BASE + idx))"
    "$((ATTENTION_TCP_PORT_BASE + idx))"
    "$((ATTENTION_ZMQ_PORT_BASE + idx))"
    "$((PREFILL_NIXL_PORT_BASE + idx))"
    "$((VLLM_PREFILL_PORT_BASE + idx * 20))"
  )
done
for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
  ports+=(
    "$((PROJECTION_PORT_BASE + idx))"
    "$((PROJECTION_ZMQ_PORT_BASE + idx))"
    "$((VLLM_PROJECTION_PORT_BASE + idx * 20))"
  )
done
ensure_ports_free "${ports[@]}"

read -r MODEL_NUM_HEADS MODEL_NUM_KV_HEADS MODEL_HEAD_DIM < <(
  "${PYTHON_BIN}" - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
num_heads = int(config["num_attention_heads"])
num_kv_heads = int(config["num_key_value_heads"])
head_dim = int(config.get("head_dim") or config["hidden_size"] // num_heads)
print(num_heads, num_kv_heads, head_dim)
PY
)

PAP_OFFLOAD_EXEC_NUM_HEADS="${PAP_OFFLOAD_EXEC_NUM_HEADS:-${MODEL_NUM_HEADS}}"
PAP_OFFLOAD_EXEC_NUM_KV_HEADS="${PAP_OFFLOAD_EXEC_NUM_KV_HEADS:-${MODEL_NUM_KV_HEADS}}"
PAP_OFFLOAD_EXEC_HEAD_DIM="${PAP_OFFLOAD_EXEC_HEAD_DIM:-${MODEL_HEAD_DIM}}"
PAP_OFFLOAD_EXEC_Q_SIZE="${PAP_OFFLOAD_EXEC_Q_SIZE:-$((PAP_OFFLOAD_EXEC_NUM_HEADS * PAP_OFFLOAD_EXEC_HEAD_DIM))}"
PAP_OFFLOAD_EXEC_KV_SIZE="${PAP_OFFLOAD_EXEC_KV_SIZE:-$((PAP_OFFLOAD_EXEC_NUM_KV_HEADS * PAP_OFFLOAD_EXEC_HEAD_DIM))}"

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
  attention_zmq_port=$((ATTENTION_ZMQ_PORT_BASE + idx))
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
    PAP_NIXL_MAILBOX_ACTOR_ID="attention-${idx}-0" \
    PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT}" \
    PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
    PAP_DIRECT_MAILBOX_OUTPUT="${PAP_DIRECT_MAILBOX_OUTPUT}" \
    PAP_LOCAL_FAST_SPIN_ITERS="${PAP_LOCAL_FAST_SPIN_ITERS}" \
    PAP_LOCAL_FAST_YIELD_ITERS="${PAP_LOCAL_FAST_YIELD_ITERS}" \
    PAP_LOCAL_FAST_SLEEP_US="${PAP_LOCAL_FAST_SLEEP_US}" \
    PAP_LOCAL_FAST_SLEEP_AFTER_US="${PAP_LOCAL_FAST_SLEEP_AFTER_US}" \
    PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE}" \
    PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}" \
    PAP_ATTENTION_DISPATCH_QUEUE_SIZE="${PAP_ATTENTION_DISPATCH_QUEUE_SIZE}" \
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
      --offload-exec-zmq-port "${attention_zmq_port}" \
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
  prefill_nixl_port=$((PREFILL_NIXL_PORT_BASE + idx))
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
    PAP_MODEL_HOOKS=1 \
    PAP_CUDAGRAPH_COMPATIBLE="${PAP_CUDAGRAPH_COMPATIBLE}" \
    PAP_CUDAGRAPH_ROLE=prefill \
    PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${prefill_nixl_port}" \
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
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --gpu-memory-utilization "${PAP_PREFILL_GPU_MEMORY_UTILIZATION}" \
      "${prefill_profiler_args[@]}" \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
      > "${RUN_LOG_DIR}/prefill_${idx}.log" 2>&1 &
  PIDS+=("$!")
done

for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
  projection_port=$((PROJECTION_PORT_BASE + idx))
  projection_zmq_port=$((PROJECTION_ZMQ_PORT_BASE + idx))
  echo "Starting PAP Projection ${idx} on GPU ${PROJECTION_GPUS[idx]}"
  env \
    CUDA_VISIBLE_DEVICES="${PROJECTION_GPUS[idx]}" \
    VLLM_PORT="$((VLLM_PROJECTION_PORT_BASE + idx * 20))" \
    PAP_TOPOLOGY="${TOPOLOGY}" \
    PAP_PA_COUNT="${PA_COUNT}" \
    PAP_PROJECTION_COUNT="${PROJECTION_COUNT}" \
    PAP_NIXL_MAILBOX_ACTOR_ID="projection-${idx}" \
    PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE}" \
    PAP_DEFERRED_TRACE_ROLE=projection \
    PAP_DEFERRED_TRACE_OUTPUT="${RUN_ROOT}/projection_deferred_trace_${idx}.json" \
    PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT}" \
    PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
    PAP_DIRECT_MAILBOX_OUTPUT="${PAP_DIRECT_MAILBOX_OUTPUT}" \
    PAP_LOCAL_FAST_SPIN_ITERS="${PAP_LOCAL_FAST_SPIN_ITERS}" \
    PAP_LOCAL_FAST_YIELD_ITERS="${PAP_LOCAL_FAST_YIELD_ITERS}" \
    PAP_LOCAL_FAST_SLEEP_US="${PAP_LOCAL_FAST_SLEEP_US}" \
    PAP_LOCAL_FAST_SLEEP_AFTER_US="${PAP_LOCAL_FAST_SLEEP_AFTER_US}" \
    PAP_OFFLOAD_EXEC_HOST=127.0.0.1 \
    PAP_OFFLOAD_EXEC_ZMQ_PORT="${projection_zmq_port}" \
    PAP_ATTENTION_ZMQ_PORT_BASE="${ATTENTION_ZMQ_PORT_BASE}" \
    PAP_ATTENTION_PORT_BASE="${ATTENTION_PORT_BASE}" \
    PAP_TP_SIZE="${PAP_TP_SIZE}" \
    PAP_PROJECTION_KV_UNAWARE=1 \
    PAP_MODEL_HOOKS=1 \
    PAP_CUDAGRAPH_COMPATIBLE="${PAP_CUDAGRAPH_COMPATIBLE}" \
    PAP_CUDAGRAPH_ROLE=projection \
    PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM}" \
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
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --gpu-memory-utilization "${PROJECTION_GPU_MEMORY_UTILIZATION}" \
      > "${RUN_LOG_DIR}/projection_${idx}.log" 2>&1 &
  PIDS+=("$!")
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
  wait_for_http \
    "http://127.0.0.1:$((PREFILL_PORT_BASE + idx))/health" \
    "PAP Prefill ${idx}"
done
for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
  wait_for_http \
    "http://127.0.0.1:$((PROJECTION_PORT_BASE + idx))/health" \
    "PAP Projection ${idx}"
done

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
wait_attention_sessions_drained
capture_proxy_topology_stats
capture_attention_fast_path_stats
audit_decode_token_join
capture_projection_deferred_traces
audit_xy_routes
audit_correctness_logs

echo "RUN_ROOT=${RUN_ROOT}"
