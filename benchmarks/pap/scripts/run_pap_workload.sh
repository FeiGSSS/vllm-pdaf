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
  PAP_MPS_MODE \
  PAP_BENCH_MPS_PROFILE \
  PAP_ASYNC_DECODE_TOKEN_SYNC_ONLY_BARRIER \
  PAP_PROJECTION_SYNC_ONLY_BARRIER \
  PAP_PREFILL_SYNC_ONLY_BARRIER \
  PAP_DIAG_R1_PROJECTION_GATE_COUNT \
  PAP_DIAG_R1_COMMIT_GATE_COUNT \
  PAP_DIAG_DECODE_COMMIT_GATE_FILE \
  PAP_DIAG_DECODE_COMMIT_GATE_TIMEOUT; do
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
        replacement="the frozen P17 data path"
        experiment_id="PAP-20260703-UNIFIED-KV"
        ;;
      PAP_ATTENTION_*)
        replacement="topology-derived Attention execution"
        experiment_id="PAP-20260711-ATTENTION-COMBINE"
        ;;
      PAP_MPS_MODE | PAP_BENCH_MPS_PROFILE)
        replacement="the P17 static 64/28 MPS partition"
        experiment_id="PAP-20260714-ASYNC-STATIC-BASELINE"
        ;;
      PAP_*_SYNC_ONLY_BARRIER)
        replacement="no Projection or Prefill timing barrier"
        experiment_id="PAP-20260714-ASYNC-TTFT-ROOTCAUSE"
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
NORTH_STAR_FINALIZER="${ROOT_DIR}/benchmarks/multi_turn/finalize_pap_pd_multiturn.py"
DEFERRED_TRACE_VALIDATOR="${ROOT_DIR}/benchmarks/multi_turn/validate_deferred_trace.py"
PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE:-0}"
PAP_BENCH_STRICT_CORRECTNESS_AUDIT="${PAP_BENCH_STRICT_CORRECTNESS_AUDIT:-1}"
PAP_BENCH_CLIENT_MODE="${PAP_BENCH_CLIENT_MODE:-canonical}"
case "${PAP_BENCH_CLIENT_MODE}" in
  canonical | multiturn_prefix_cache | multiturn_chat_prefix_cache \
    | multiturn_north_star | multiturn_load) ;;
  *)
    echo "ERROR: unsupported PAP_BENCH_CLIENT_MODE=${PAP_BENCH_CLIENT_MODE}" >&2
    exit 2
    ;;
esac

GIT_COMMIT=""
GIT_COMMIT_SHORT=""
GIT_TRACKED_WORKTREE_DIRTY=0

MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_NAME="${DATASET_NAME:-sonnet}"
BENCH_DIR="${BENCH_DIR:-/home/fei/research/PD/refer_codes/vllm/benchmarks}"
DATASET_PATH="${DATASET_PATH:-${BENCH_DIR}/sonnet_4x.txt}"

INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PREFIX_LEN="${PREFIX_LEN:-50}"
QPS="${QPS:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
PAP_MULTITURN_LOAD_ROUNDS="${PAP_MULTITURN_LOAD_ROUNDS:-5}"
PAP_MULTITURN_LOAD_CONVERSATIONS="${PAP_MULTITURN_LOAD_CONVERSATIONS:-4}"
PAP_MULTITURN_LOAD_REQUEST_RATE="${PAP_MULTITURN_LOAD_REQUEST_RATE:-2}"
PAP_MULTITURN_APPEND_TOKENS="${PAP_MULTITURN_APPEND_TOKENS:-120}"
if [[ "${PAP_BENCH_CLIENT_MODE}" == "multiturn_load" ]]; then
  if ! [[ "${PAP_MULTITURN_LOAD_ROUNDS}" =~ ^[1-9][0-9]*$ \
    && "${PAP_MULTITURN_LOAD_CONVERSATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: multi-turn load rounds/conversations must be positive" >&2
    exit 2
  fi
  NUM_PROMPTS=$((
    PAP_MULTITURN_LOAD_ROUNDS * PAP_MULTITURN_LOAD_CONVERSATIONS
  ))
elif [[ "${PAP_BENCH_CLIENT_MODE}" == "multiturn_north_star" ]]; then
  NUM_PROMPTS=2
elif [[ "${PAP_BENCH_CLIENT_MODE}" != "canonical" ]]; then
  NUM_PROMPTS=3
fi
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
CLUSTER_READY_WAIT_SECONDS="${CLUSTER_READY_WAIT_SECONDS:-30}"
PAP_BENCH_SESSION_DRAIN_TIMEOUT="${PAP_BENCH_SESSION_DRAIN_TIMEOUT:-15}"
PAP_DEFERRED_TRACE_FLUSH_TIMEOUT="${PAP_DEFERRED_TRACE_FLUSH_TIMEOUT:-30}"

TOPOLOGY="${PAP_TOPOLOGY:-1pa1p}"
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
RESULTS_ROOT="${RESULTS_ROOT:-/home/fei/research/PD/test/baseline/pap/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_ROOT}/service_logs}"
PAP_NORTH_STAR_HARDWARE_SIGNATURE="${PAP_NORTH_STAR_HARDWARE_SIGNATURE:-NVIDIA-L20x2}"
PAP_NORTH_STAR_CONVERSATION_ID="${PAP_NORTH_STAR_CONVERSATION_ID:-${RUN_ID}-conversation-0}"
PAP_NORTH_STAR_CACHE_SALT="${PAP_NORTH_STAR_CACHE_SALT:-${RUN_ID}-cache-salt}"

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
PAP_VLLM_DTYPE="${PAP_VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PAP_PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.76}"
PAP_PROJECTION_GPU_MEMORY_UTILIZATION="${PAP_PROJECTION_GPU_MEMORY_UTILIZATION:-0.76}"
PAP_PREFILL_MPS_PERCENT="${PAP_PREFILL_MPS_PERCENT:-70}"
PAP_ATTENTION_MPS_PERCENT="${PAP_ATTENTION_MPS_PERCENT:-30}"
PAP_STATIC_PREFILL_CHUNKS="${PAP_STATIC_PREFILL_CHUNKS:-16}"
PAP_STATIC_ATTENTION_CHUNKS="${PAP_STATIC_ATTENTION_CHUNKS:-7}"
PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_STATIC_PREFILL_EXPECTED_SMS:-64}"
PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_STATIC_ATTENTION_EXPECTED_SMS:-28}"
PAP_ENABLE_MPS=1
if ! [[ "${PAP_STATIC_PREFILL_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PAP_STATIC_ATTENTION_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: static MPS chunk counts must be positive integers" >&2
  exit 2
fi
PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-nixl_mailbox}"
PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"
PAP_OFFLOAD_EXEC_TRACE="${PAP_OFFLOAD_EXEC_TRACE:-0}"
PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE:-0}"
PAP_DEFERRED_CUDA_TRACE_MAX_PENDING="${PAP_DEFERRED_CUDA_TRACE_MAX_PENDING:-1024}"
PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND="${PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND:-1}"
PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT="${PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT:-0}"
PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT_OUTPUT="${PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT_OUTPUT:-0}"
PAP_NIXL_MAILBOX_INLINE_PUBLISH="${PAP_NIXL_MAILBOX_INLINE_PUBLISH:-1}"
PAP_NIXL_MAILBOX_BATCH_PLAN="${PAP_NIXL_MAILBOX_BATCH_PLAN:-1}"
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
PAP_LOCAL_FAST_ASYNC_DOORBELL="${PAP_LOCAL_FAST_ASYNC_DOORBELL:-0}"
PAP_LOCAL_FAST_STREAM_ORDERED="${PAP_LOCAL_FAST_STREAM_ORDERED:-1}"
PAP_LOCAL_FAST_SLOT_COUNT="${PAP_LOCAL_FAST_SLOT_COUNT:-2}"
PAP_LOCAL_FAST_BATCH_PLAN="${PAP_LOCAL_FAST_BATCH_PLAN:-1}"
PAP_LOCAL_FAST_SPIN_ITERS="${PAP_LOCAL_FAST_SPIN_ITERS:-2048}"
PAP_LOCAL_FAST_YIELD_ITERS="${PAP_LOCAL_FAST_YIELD_ITERS:-64}"
PAP_LOCAL_FAST_SLEEP_US="${PAP_LOCAL_FAST_SLEEP_US:-20}"
PAP_LOCAL_FAST_SLEEP_AFTER_US="${PAP_LOCAL_FAST_SLEEP_AFTER_US:-50}"
PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM:-16}"
PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY:-round_robin}"
DEFAULT_DECODE_CAPACITY_TOKENS=32
DEFAULT_PROMPT_TOKENS_DETAILS=0
if [[ "${PAP_BENCH_CLIENT_MODE}" != "canonical" ]]; then
  DEFAULT_DECODE_CAPACITY_TOKENS=64
  DEFAULT_PROMPT_TOKENS_DETAILS=1
fi
PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS:-${DEFAULT_DECODE_CAPACITY_TOKENS}}"
PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS:-${DEFAULT_PROMPT_TOKENS_DETAILS}}"
PAP_PREFIX_CACHE_AUDIT="${PAP_PREFIX_CACHE_AUDIT:-0}"
PAP_MULTITURN_FIRST_OUTPUT_TOKENS="${PAP_MULTITURN_FIRST_OUTPUT_TOKENS:-48}"
PAP_MULTITURN_BLOCK_SIZE="${PAP_MULTITURN_BLOCK_SIZE:-16}"
PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS="${PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS:-1}"
PAP_DECODE_COMMIT_ENDPOINT="${PAP_DECODE_COMMIT_ENDPOINT:-}"
PAP_LEASE_RELEASE_ENDPOINT="${PAP_LEASE_RELEASE_ENDPOINT:-}"
PAP_DECODE_COMMIT_FAIL_CLOSED="${PAP_DECODE_COMMIT_FAIL_CLOSED:-1}"
PAP_DECODE_COMMIT_TIMEOUT="${PAP_DECODE_COMMIT_TIMEOUT:-0.2}"
PAP_DECODE_COMMIT_QUEUE_SIZE="${PAP_DECODE_COMMIT_QUEUE_SIZE:-1024}"
PAP_DECODE_COMMIT_MAX_ATTEMPTS="${PAP_DECODE_COMMIT_MAX_ATTEMPTS:-8}"
PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS="${PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS:-0.05}"
PAP_DECODE_COMMIT_RETRY_MAX_SECONDS="${PAP_DECODE_COMMIT_RETRY_MAX_SECONDS:-0.5}"
PAP_DECODE_COMMIT_FLUSH_TIMEOUT="${PAP_DECODE_COMMIT_FLUSH_TIMEOUT:-5.0}"
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

export PAP_OFFLOAD_EXEC_TRACE
export PAP_DEFERRED_CUDA_TRACE
export PAP_DEFERRED_CUDA_TRACE_MAX_PENDING
export PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND
export PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT
export PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT_OUTPUT
export PAP_NIXL_MAILBOX_INLINE_PUBLISH
export PAP_NIXL_MAILBOX_BATCH_PLAN
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
  for pipe_dir in "${MPS_STARTED_DIRS[@]:-}"; do
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
    printf 'PAP_BENCH_CLIENT_MODE=%q\n' "${PAP_BENCH_CLIENT_MODE}"
    printf 'TOPOLOGY=%q\n' "${TOPOLOGY}"
    printf 'PA_COUNT=%q\n' "${PA_COUNT}"
    printf 'PROJECTION_COUNT=%q\n' "${PROJECTION_COUNT}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'DATASET_NAME=%q\n' "${DATASET_NAME}"
    printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
    printf 'BENCH_DIR=%q\n' "${BENCH_DIR}"
    printf 'PREFIX_LEN=%q\n' "${PREFIX_LEN}"
    printf 'NUM_PROMPTS=%q\n' "${NUM_PROMPTS}"
    printf 'PAP_ENABLE_PROMPT_TOKENS_DETAILS=%q\n' "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}"
    printf 'PAP_PREFIX_CACHE_AUDIT=%q\n' "${PAP_PREFIX_CACHE_AUDIT}"
    printf 'PAP_MULTITURN_FIRST_OUTPUT_TOKENS=%q\n' "${PAP_MULTITURN_FIRST_OUTPUT_TOKENS}"
    printf 'PAP_MULTITURN_BLOCK_SIZE=%q\n' "${PAP_MULTITURN_BLOCK_SIZE}"
    printf 'PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS=%q\n' "${PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS}"
    printf 'PAP_MULTITURN_LOAD_ROUNDS=%q\n' \
      "${PAP_MULTITURN_LOAD_ROUNDS}"
    printf 'PAP_MULTITURN_LOAD_CONVERSATIONS=%q\n' \
      "${PAP_MULTITURN_LOAD_CONVERSATIONS}"
    printf 'PAP_MULTITURN_LOAD_REQUEST_RATE=%q\n' \
      "${PAP_MULTITURN_LOAD_REQUEST_RATE}"
    printf 'PAP_MULTITURN_APPEND_TOKENS=%q\n' \
      "${PAP_MULTITURN_APPEND_TOKENS}"
    printf 'INPUT_LENS_CSV=%q\n' "${INPUT_LEN}"
    printf 'OUTPUT_LENS_CSV=%q\n' "${OUTPUT_LEN}"
    printf 'QPS_CSV=%q\n' "${QPS}"
    printf 'BENCH_NUM_WARMUPS=%q\n' "${BENCH_NUM_WARMUPS}"
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
    printf 'PAP_PROJECTION_GPU_MEMORY_UTILIZATION=%q\n' "${PAP_PROJECTION_GPU_MEMORY_UTILIZATION}"
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
    printf 'PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT=%q\n' "${PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT}"
    printf 'PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT_OUTPUT=%q\n' "${PAP_OFFLOAD_EXEC_DIRECT_QKV_SENDER_SLOT_OUTPUT}"
    printf 'PAP_NIXL_MAILBOX_INLINE_PUBLISH=%q\n' "${PAP_NIXL_MAILBOX_INLINE_PUBLISH}"
    printf 'PAP_NIXL_MAILBOX_BATCH_PLAN=%q\n' "${PAP_NIXL_MAILBOX_BATCH_PLAN}"
    printf 'PAP_UNIFIED_MD_CACHE_LIMIT=%q\n' "${PAP_UNIFIED_MD_CACHE_LIMIT}"
    printf 'PAP_DECODE_SLOT_PLAN_CACHE_LIMIT=%q\n' "${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}"
    printf 'PAP_ATTENTION_DISPATCH_QUEUE_SIZE=%q\n' \
      "${PAP_ATTENTION_DISPATCH_QUEUE_SIZE}"
    printf 'PAP_DIRECT_MAILBOX_OUTPUT=%q\n' "${PAP_DIRECT_MAILBOX_OUTPUT}"
    printf 'PAP_LOCAL_FAST_ASYNC_DOORBELL=%q\n' "${PAP_LOCAL_FAST_ASYNC_DOORBELL}"
    printf 'PAP_LOCAL_FAST_STREAM_ORDERED=%q\n' "${PAP_LOCAL_FAST_STREAM_ORDERED}"
    printf 'PAP_LOCAL_FAST_SLOT_COUNT=%q\n' "${PAP_LOCAL_FAST_SLOT_COUNT}"
    printf 'PAP_LOCAL_FAST_BATCH_PLAN=%q\n' "${PAP_LOCAL_FAST_BATCH_PLAN}"
    printf 'PAP_LOCAL_FAST_SPIN_ITERS=%q\n' "${PAP_LOCAL_FAST_SPIN_ITERS}"
    printf 'PAP_LOCAL_FAST_YIELD_ITERS=%q\n' "${PAP_LOCAL_FAST_YIELD_ITERS}"
    printf 'PAP_LOCAL_FAST_SLEEP_US=%q\n' "${PAP_LOCAL_FAST_SLEEP_US}"
    printf 'PAP_LOCAL_FAST_SLEEP_AFTER_US=%q\n' "${PAP_LOCAL_FAST_SLEEP_AFTER_US}"
    printf 'PAP_PREFILL_GPUS=%q\n' "${PAP_PREFILL_GPUS}"
    printf 'PAP_PROJECTION_GPUS=%q\n' "${PAP_PROJECTION_GPUS}"
    printf 'PAP_VLLM_DTYPE=%q\n' "${PAP_VLLM_DTYPE}"
    printf 'PAP_NORTH_STAR_HARDWARE_SIGNATURE=%q\n' \
      "${PAP_NORTH_STAR_HARDWARE_SIGNATURE}"
    printf 'PAP_NORTH_STAR_CONVERSATION_ID=%q\n' \
      "${PAP_NORTH_STAR_CONVERSATION_ID}"
    printf 'PAP_NORTH_STAR_CACHE_SALT=%q\n' \
      "${PAP_NORTH_STAR_CACHE_SALT}"
    printf 'PAP_ROUTING_POLICY=%q\n' "${PAP_ROUTING_POLICY}"
    printf 'PREFILL_PORT_BASE=%q\n' "${PREFILL_PORT_BASE}"
    printf 'PROJECTION_PORT_BASE=%q\n' "${PROJECTION_PORT_BASE}"
    printf 'ATTENTION_PORT_BASE=%q\n' "${ATTENTION_PORT_BASE}"
    printf 'PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=%q\n' "${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS}"
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
  PAP_BENCH_CLIENT_MODE="${PAP_BENCH_CLIENT_MODE}" \
  INPUT_LEN="${INPUT_LEN}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  QPS="${QPS}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  MODEL_PATH="${MODEL_PATH}" \
  PREFIX_LEN="${PREFIX_LEN}" \
  BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT}" \
  PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
  PAP_DIRECT_MAILBOX_OUTPUT="${PAP_DIRECT_MAILBOX_OUTPUT}" \
  PAP_LOCAL_FAST_STREAM_ORDERED="${PAP_LOCAL_FAST_STREAM_ORDERED}" \
  PAP_LOCAL_FAST_SLOT_COUNT="${PAP_LOCAL_FAST_SLOT_COUNT}" \
  PAP_PREFILL_IPC_PROFILE="${PAP_PREFILL_IPC_PROFILE}" \
  PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}" \
  PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" \
  PAP_MULTITURN_LOAD_ROUNDS="${PAP_MULTITURN_LOAD_ROUNDS}" \
  PAP_MULTITURN_LOAD_CONVERSATIONS="${PAP_MULTITURN_LOAD_CONVERSATIONS}" \
  PAP_MULTITURN_LOAD_REQUEST_RATE="${PAP_MULTITURN_LOAD_REQUEST_RATE}" \
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
    "client_mode": os.environ["PAP_BENCH_CLIENT_MODE"],
    "topology": os.environ["TOPOLOGY"],
    "pa_count": pa_count,
    "projection_count": projection_count,
    "routing_policy": os.environ["PAP_ROUTING_POLICY"],
    "run_id": os.environ["RUN_ID"],
    "result_root": os.environ["RUN_ROOT"],
    "input_lens": [os.environ["INPUT_LEN"]],
    "output_lens": [os.environ["OUTPUT_LEN"]],
    "qps": [os.environ["QPS"]],
    "num_prompts": os.environ["NUM_PROMPTS"],
    "model_path": os.environ["MODEL_PATH"],
    "prefix_len": os.environ["PREFIX_LEN"],
    "num_warmups": os.environ["BENCH_NUM_WARMUPS"],
    "max_model_len": os.environ["MAX_MODEL_LEN"],
    "max_num_seqs": os.environ["MAX_NUM_SEQS"],
    "offload_exec_transport": os.environ["PAP_OFFLOAD_EXEC_TRANSPORT"],
    "offload_kv_transport": os.environ["PAP_OFFLOAD_KV_TRANSPORT"],
    "batched_route_copy": True,
    "direct_mailbox_output": os.environ["PAP_DIRECT_MAILBOX_OUTPUT"] == "1",
    "unified_md_fast_key": True,
    "local_fast_stream_ordered": (
        os.environ["PAP_LOCAL_FAST_STREAM_ORDERED"] == "1"
    ),
    "local_fast_slot_count": int(os.environ["PAP_LOCAL_FAST_SLOT_COUNT"]),
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
    "multiturn_load_rounds": int(os.environ["PAP_MULTITURN_LOAD_ROUNDS"]),
    "multiturn_load_conversations": int(
        os.environ["PAP_MULTITURN_LOAD_CONVERSATIONS"]
    ),
    "multiturn_load_request_rate": float(
        os.environ["PAP_MULTITURN_LOAD_REQUEST_RATE"]
    ),
    "dtype": os.environ["PAP_VLLM_DTYPE"],
    "git_commit": os.environ["GIT_COMMIT"],
    "git_commit_short": os.environ["GIT_COMMIT_SHORT"],
    "git_tracked_worktree_dirty": (
        os.environ["GIT_TRACKED_WORKTREE_DIRTY"] == "1"
    ),
    "proxy_port": os.environ["PAP_PROXY_PORT"],
    "config_dir": "self-contained skill runner",
    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}
with open(os.path.join(os.environ["RUN_ROOT"], "run_metadata.json"), "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")
PY
}

validate_benchmark_result() {
  local result_path="$1"
  NUM_PROMPTS="${NUM_PROMPTS}" "${PYTHON_BIN}" - "${result_path}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    result = json.load(f)

expected = int(os.environ["NUM_PROMPTS"])
completed = int(result.get("completed", 0))
failed = int(result.get("failed", 0))
if completed != expected or failed != 0:
    raise SystemExit(
        f"benchmark result is incomplete: completed={completed}, "
        f"failed={failed}, expected={expected}"
    )
PY
}

validate_north_star_result() {
  local result_path="$1"
  "${PYTHON_BIN}" - "${result_path}" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as file_obj:
    result = json.load(file_obj)

if result.get("schema_version") != 2:
    raise SystemExit(f"north-star schema mismatch: {result.get('schema_version')}")
if result.get("metric_definition") != "last_output_token_v2":
    raise SystemExit(
        f"north-star metric definition mismatch: {result.get('metric_definition')}"
    )
if result.get("validity") != {"status": "passed", "cache_gate": "passed"}:
    raise SystemExit(f"north-star validity failed: {result.get('validity')}")
if result.get("architecture") != "pap":
    raise SystemExit(f"north-star architecture is not pap: {result.get('architecture')}")
if (result.get("topology") or {}).get("name") != "1pa1p":
    raise SystemExit(f"north-star topology is not 1pa1p: {result.get('topology')}")
rounds = result.get("rounds") or []
if len(rounds) != 2:
    raise SystemExit(f"north-star expected two rounds, got {len(rounds)}")
for index, round_result in enumerate(rounds, start=1):
    if round_result.get("completion_tokens") != 256:
        raise SystemExit(f"round {index} did not return 256 tokens")
    if round_result.get("finish_reason") != "length":
        raise SystemExit(f"round {index} did not finish by length")
    for metric in ("ttft_ms", "tpot_ms", "latency_ms", "eof_latency_ms"):
        value = round_result.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"round {index} has invalid {metric}: {value}")
cache = result.get("cache_validation") or {}
if cache.get("status") != "passed":
    raise SystemExit(f"north-star cache validation failed: {cache}")
if int(cache.get("decode_derived_hit_tokens", 0)) < 16:
    raise SystemExit("north-star second turn has no Decode-derived cache block")
if not result.get("profile_fingerprint"):
    raise SystemExit("north-star profile fingerprint is missing")
if not result.get("implementation_fingerprint"):
    raise SystemExit("north-star implementation fingerprint is missing")
PY
}

validate_multiturn_load_result() {
  local result_path="$1"
  PAP_MULTITURN_LOAD_ROUNDS="${PAP_MULTITURN_LOAD_ROUNDS}" \
  PAP_MULTITURN_LOAD_CONVERSATIONS="${PAP_MULTITURN_LOAD_CONVERSATIONS}" \
  "${PYTHON_BIN}" - "${result_path}" <<'PY'
import json
import math
import os
import sys

with open(sys.argv[1], encoding="utf-8") as file_obj:
    result = json.load(file_obj)

rounds = int(os.environ["PAP_MULTITURN_LOAD_ROUNDS"])
conversations = int(os.environ["PAP_MULTITURN_LOAD_CONVERSATIONS"])
if result.get("architecture") != "pap":
    raise SystemExit("multi-turn load architecture is not pap")
validity = result.get("validity") or {}
if validity.get("status") != "passed":
    raise SystemExit(f"multi-turn load validity failed: {validity}")
cache = result.get("cache_validation") or {}
if cache.get("status") != "passed":
    raise SystemExit(f"multi-turn load cache validation failed: {cache}")
requests = result.get("requests") or []
expected = rounds * conversations
if len(requests) != expected:
    raise SystemExit(
        f"multi-turn load request count mismatch: {len(requests)} != {expected}"
    )
for request in requests:
    if request.get("completion_tokens") != 256:
        raise SystemExit("multi-turn load request did not return 256 tokens")
    if request.get("finish_reason") != "length":
        raise SystemExit("multi-turn load request did not finish by length")
    for metric in ("ttft_ms", "tpot_ms", "latency_ms", "eof_latency_ms"):
        value = request.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value <= 0:
            raise SystemExit(f"multi-turn load has invalid {metric}: {value}")
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
for request_number in range(expected_requests):
    group_index = request_number % pa_count
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
    elif routing_policy != "round_robin":
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
  || die "P17 requires static MPS"
if [[ "${PAP_BENCH_CLIENT_MODE}" == "multiturn_north_star" ]]; then
  [[ "${TOPOLOGY}" == "1pa1p" ]] \
    || die "multiturn_north_star requires PAP_TOPOLOGY=1pa1p"
  [[ "${INPUT_LEN}" == "16000" ]] \
    || die "multiturn_north_star requires INPUT_LEN=16000"
  [[ "${OUTPUT_LEN}" == "256" ]] \
    || die "multiturn_north_star requires OUTPUT_LEN=256"
  [[ "${MAX_MODEL_LEN}" == "20000" ]] \
    || die "multiturn_north_star requires MAX_MODEL_LEN=20000"
  [[ "${MAX_NUM_BATCHED_TOKENS}" == "4096" ]] \
    || die "multiturn_north_star requires MAX_NUM_BATCHED_TOKENS=4096"
  [[ "${MAX_NUM_SEQS}" == "2" ]] \
    || die "multiturn_north_star requires MAX_NUM_SEQS=2"
  [[ "${PAP_VLLM_DTYPE}" == "float16" ]] \
    || die "multiturn_north_star requires PAP_VLLM_DTYPE=float16"
  [[ "${PAP_PREFIX_CACHE_AUDIT}" == "0" ]] \
    || die "multiturn_north_star forbids PAP_PREFIX_CACHE_AUDIT"
  [[ "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" == "1" ]] \
    || die "multiturn_north_star requires prompt token details"
  [[ "${PAP_ENABLE_MPS}" == "1" ]] \
    || die "multiturn_north_star requires PAP_ENABLE_MPS=1"
  [[ "${PAP_PREFILL_MPS_PERCENT}" == "70" \
    && "${PAP_ATTENTION_MPS_PERCENT}" == "30" ]] \
    || die "multiturn_north_star requires PAP MPS 70/30"
  (( PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS >= OUTPUT_LEN )) \
    || die "PAP unified KV decode capacity is too small for north-star output"
elif [[ "${PAP_BENCH_CLIENT_MODE}" == "multiturn_load" ]]; then
  [[ "${TOPOLOGY}" == "1pa1p" ]] \
    || die "multiturn_load requires PAP_TOPOLOGY=1pa1p"
  [[ "${INPUT_LEN}" == "16000" && "${OUTPUT_LEN}" == "256" ]] \
    || die "multiturn_load requires INPUT_LEN=16000 and OUTPUT_LEN=256"
  [[ "${MAX_MODEL_LEN}" == "20000" ]] \
    || die "multiturn_load requires MAX_MODEL_LEN=20000"
  [[ "${MAX_NUM_BATCHED_TOKENS}" == "4096" ]] \
    || die "multiturn_load requires MAX_NUM_BATCHED_TOKENS=4096"
  [[ "${MAX_NUM_SEQS}" == "4" ]] \
    || die "multiturn_load requires MAX_NUM_SEQS=4"
  [[ "${PAP_VLLM_DTYPE}" == "float16" ]] \
    || die "multiturn_load requires PAP_VLLM_DTYPE=float16"
  [[ "${PAP_PREFIX_CACHE_AUDIT}" == "0" \
    && "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" == "1" ]] \
    || die "multiturn_load requires prompt details and forbids cache audit"
  [[ "${PAP_ENABLE_MPS}" == "1" ]] \
    || die "multiturn_load requires PAP MPS"
  [[ "${PAP_STATIC_PREFILL_CHUNKS}" == "16" \
    && "${PAP_STATIC_ATTENTION_CHUNKS}" == "7" ]] \
    || die "P17 static MPS requires 16/7 chunks"
  (( PAP_MULTITURN_LOAD_ROUNDS >= 4 )) \
    || die "multiturn_load requires at least four rounds"
  (( PAP_MULTITURN_LOAD_CONVERSATIONS <= MAX_NUM_SEQS )) \
    || die "active conversations exceed MAX_NUM_SEQS"
  (( PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS >= OUTPUT_LEN )) \
    || die "PAP unified KV decode capacity is too small for load output"
elif [[ "${PAP_BENCH_CLIENT_MODE}" != "canonical" ]]; then
  [[ "${PA_COUNT}" == "1" && "${PROJECTION_COUNT}" == "1" ]] \
    || die "multi-turn prefix-cache modes require PAP_TOPOLOGY=1pa1p"
  [[ "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" == "1" ]] \
    || die "multi-turn prefix-cache modes require prompt token details"
  [[ "${PAP_MULTITURN_FIRST_OUTPUT_TOKENS}" =~ ^[1-9][0-9]*$ ]] \
    || die "PAP_MULTITURN_FIRST_OUTPUT_TOKENS must be positive"
  (( PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS >= PAP_MULTITURN_FIRST_OUTPUT_TOKENS )) \
    || die "PAP unified KV decode capacity is too small for first turn"
fi
[[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ -x "${VLLM_BIN}" ]] || die "VLLM_BIN is not executable: ${VLLM_BIN}"
[[ -f "${NORTH_STAR_FINALIZER}" ]] \
  || die "Missing north-star finalizer: ${NORTH_STAR_FINALIZER}"
[[ -f "${DEFERRED_TRACE_VALIDATOR}" ]] \
  || die "Missing deferred trace validator: ${DEFERRED_TRACE_VALIDATOR}"
[[ -d "${MODEL_PATH}" ]] || die "Model path does not exist: ${MODEL_PATH}"

"${PYTHON_BIN}" -c 'import nixl' >/dev/null 2>&1 \
  || die "Python package 'nixl' is not installed in .venv"

ensure_dataset
mkdir -p "${RUN_ROOT}" "${RUN_LOG_DIR}"
capture_git_state
split_csv "${PAP_PREFILL_GPUS}" PREFILL_GPUS
split_csv "${PAP_PROJECTION_GPUS}" PROJECTION_GPUS
require_count "PAP_PREFILL_GPUS" "${#PREFILL_GPUS[@]}" "${PA_COUNT}"
require_count \
  "PAP_PROJECTION_GPUS" "${#PROJECTION_GPUS[@]}" "${PROJECTION_COUNT}"

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
    PAP_LOCAL_FAST_ASYNC_DOORBELL="${PAP_LOCAL_FAST_ASYNC_DOORBELL}" \
    PAP_LOCAL_FAST_STREAM_ORDERED="${PAP_LOCAL_FAST_STREAM_ORDERED}" \
    PAP_LOCAL_FAST_SLOT_COUNT="${PAP_LOCAL_FAST_SLOT_COUNT}" \
    PAP_LOCAL_FAST_BATCH_PLAN="${PAP_LOCAL_FAST_BATCH_PLAN}" \
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
    PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${prefill_nixl_port}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --port "${prefill_port}" \
      --host 127.0.0.1 \
      --enforce-eager \
      --generation-config vllm \
      --dtype "${PAP_VLLM_DTYPE}" \
      --enable-request-id-headers \
      "${PREFILL_OBSERVABILITY_ARGS[@]}" \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
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
    PAP_LOCAL_FAST_ASYNC_DOORBELL="${PAP_LOCAL_FAST_ASYNC_DOORBELL}" \
    PAP_LOCAL_FAST_STREAM_ORDERED="${PAP_LOCAL_FAST_STREAM_ORDERED}" \
    PAP_LOCAL_FAST_SLOT_COUNT="${PAP_LOCAL_FAST_SLOT_COUNT}" \
    PAP_LOCAL_FAST_BATCH_PLAN="${PAP_LOCAL_FAST_BATCH_PLAN}" \
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
    PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --port "${projection_port}" \
      --host 127.0.0.1 \
      --enforce-eager \
      --generation-config vllm \
      --dtype "${PAP_VLLM_DTYPE}" \
      --enable-request-id-headers \
      --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --gpu-memory-utilization "${PAP_PROJECTION_GPU_MEMORY_UTILIZATION}" \
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

write_effective_config
write_topology_manifest
write_run_metadata
start_prefill_torch_profiles

case "${PAP_BENCH_CLIENT_MODE}" in
  canonical)
    TAG="${TOPOLOGY_TAG}_i${INPUT_LEN}_o${OUTPUT_LEN}_q${QPS}"
    echo "=== Running ${TAG} on port ${PAP_PROXY_PORT} ==="
    timeout "${BENCH_TIMEOUT}" "${VLLM_BIN}" bench serve \
      --backend vllm \
      --model "${MODEL_PATH}" \
      --dataset-name "${DATASET_NAME}" \
      --dataset-path "${DATASET_PATH}" \
      --sonnet-input-len "${INPUT_LEN}" \
      --sonnet-output-len "${OUTPUT_LEN}" \
      --sonnet-prefix-len "${PREFIX_LEN}" \
      --num-prompts "${NUM_PROMPTS}" \
      --port "${PAP_PROXY_PORT}" \
      --save-result \
      --result-dir "${RUN_ROOT}" \
      --result-filename "${TAG}.json" \
      --request-rate "${QPS}" \
      --num-warmups "${BENCH_NUM_WARMUPS}" \
      2>&1 | tee "${RUN_ROOT}/${TAG}.log"

    validate_benchmark_result "${RUN_ROOT}/${TAG}.json"
    ;;
  multiturn_prefix_cache)
    TAG="${TOPOLOGY_TAG}_multiturn_prefix_cache"
    echo "=== Running ${TAG} on port ${PAP_PROXY_PORT} ==="
    timeout "${BENCH_TIMEOUT}" "${PYTHON_BIN}" \
      examples/pap/pap_multiturn_prefix_cache.py \
      --base-url "http://127.0.0.1:${PAP_PROXY_PORT}" \
      --model "${MODEL_PATH}" \
      --result-path "${RUN_ROOT}/multiturn_prefix_cache.json" \
      --prompt-tokens "${INPUT_LEN}" \
      --first-output-tokens "${PAP_MULTITURN_FIRST_OUTPUT_TOKENS}" \
      --second-output-tokens "${OUTPUT_LEN}" \
      --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
      --min-decode-hit-blocks "${PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS}" \
      2>&1 | tee "${RUN_ROOT}/${TAG}.log"
    ;;
  multiturn_chat_prefix_cache)
    TAG="${TOPOLOGY_TAG}_multiturn_chat_prefix_cache"
    echo "=== Running ${TAG} on port ${PAP_PROXY_PORT} ==="
    timeout "${BENCH_TIMEOUT}" "${PYTHON_BIN}" \
      examples/pap/pap_multiturn_chat_prefix_cache.py \
      --base-url "http://127.0.0.1:${PAP_PROXY_PORT}" \
      --model "${MODEL_PATH}" \
      --result-path "${RUN_ROOT}/multiturn_chat_prefix_cache.json" \
      --min-first-prompt-tokens "${INPUT_LEN}" \
      --first-output-tokens "${PAP_MULTITURN_FIRST_OUTPUT_TOKENS}" \
      --second-output-tokens "${OUTPUT_LEN}" \
      --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
      --min-decode-hit-blocks "${PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS}" \
      2>&1 | tee "${RUN_ROOT}/${TAG}.log"
    ;;
  multiturn_north_star)
    TAG="${TOPOLOGY_TAG}_multiturn_north_star"
    echo "=== Running ${TAG} on port ${PAP_PROXY_PORT} ==="
    timeout "${BENCH_TIMEOUT}" "${PYTHON_BIN}" \
      benchmarks/multi_turn/pap_pd_multiturn_client.py \
      --base-url "http://127.0.0.1:${PAP_PROXY_PORT}" \
      --model "${MODEL_PATH}" \
      --corpus "${DATASET_PATH}" \
      --result "${RUN_ROOT}/result.json" \
      --architecture "pap" \
      --topology "${TOPOLOGY}" \
      --conversation-id "${PAP_NORTH_STAR_CONVERSATION_ID}" \
      --cache-salt "${PAP_NORTH_STAR_CACHE_SALT}" \
      --hardware-signature "${PAP_NORTH_STAR_HARDWARE_SIGNATURE}" \
      --git-commit "${GIT_COMMIT}" \
      --git-tracked-worktree-dirty "${GIT_TRACKED_WORKTREE_DIRTY}" \
      --offload-exec-transport "${PAP_OFFLOAD_EXEC_TRANSPORT}" \
      --direct-mailbox-output "${PAP_DIRECT_MAILBOX_OUTPUT}" \
      --prefill-ipc-profile "${PAP_PREFILL_IPC_PROFILE}" \
      --document-tokens "${INPUT_LEN}" \
      --append-tokens "${PAP_MULTITURN_APPEND_TOKENS}" \
      --output-tokens "${OUTPUT_LEN}" \
      --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
      --dtype "${PAP_VLLM_DTYPE}" \
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      2>&1 | tee "${RUN_ROOT}/${TAG}.log"
    validate_north_star_result "${RUN_ROOT}/result.json"
    ;;
  multiturn_load)
    TAG="${TOPOLOGY_TAG}_multiturn_load"
    echo "=== Running ${TAG} on port ${PAP_PROXY_PORT} ==="
    timeout "${BENCH_TIMEOUT}" "${PYTHON_BIN}" \
      benchmarks/multi_turn/pap_pd_multiturn_load_client.py \
      --base-url "http://127.0.0.1:${PAP_PROXY_PORT}" \
      --model "${MODEL_PATH}" \
      --corpus "${DATASET_PATH}" \
      --result "${RUN_ROOT}/result.json" \
      --architecture pap \
      --topology "${TOPOLOGY}" \
      --conversation-id-prefix "${PAP_NORTH_STAR_CONVERSATION_ID}" \
      --cache-salt-prefix "${PAP_NORTH_STAR_CACHE_SALT}" \
      --hardware-signature "${PAP_NORTH_STAR_HARDWARE_SIGNATURE}" \
      --git-commit "${GIT_COMMIT}" \
      --git-tracked-worktree-dirty "${GIT_TRACKED_WORKTREE_DIRTY}" \
      --offload-exec-transport "${PAP_OFFLOAD_EXEC_TRANSPORT}" \
      --direct-mailbox-output "${PAP_DIRECT_MAILBOX_OUTPUT}" \
      --prefill-ipc-profile "${PAP_PREFILL_IPC_PROFILE}" \
      --document-tokens "${INPUT_LEN}" \
      --append-tokens "${PAP_MULTITURN_APPEND_TOKENS}" \
      --output-tokens "${OUTPUT_LEN}" \
      --rounds "${PAP_MULTITURN_LOAD_ROUNDS}" \
      --active-conversations "${PAP_MULTITURN_LOAD_CONVERSATIONS}" \
      --request-rate "${PAP_MULTITURN_LOAD_REQUEST_RATE}" \
      --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
      --dtype "${PAP_VLLM_DTYPE}" \
      --tensor-parallel-size "${PAP_TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      2>&1 | tee "${RUN_ROOT}/${TAG}.log"
    validate_multiturn_load_result "${RUN_ROOT}/result.json"
    ;;
esac

wait_prefill_torch_profiles
wait_attention_sessions_drained
capture_proxy_topology_stats
capture_attention_fast_path_stats
audit_decode_token_join
capture_projection_deferred_traces
audit_xy_routes
audit_correctness_logs

if [[ "${PAP_BENCH_CLIENT_MODE}" == "multiturn_north_star" \
  || "${PAP_BENCH_CLIENT_MODE}" == "multiturn_load" ]]; then
  deferred_trace_artifact_args=()
  if deferred_trace_enabled; then
    deferred_trace_artifact_args+=(
      --artifact \
      "projection_deferred_trace=${RUN_ROOT}/projection_deferred_trace.json"
    )
  fi
  "${PYTHON_BIN}" "${NORTH_STAR_FINALIZER}" \
    --result "${RUN_ROOT}/result.json" \
    --architecture pap \
    --passed-gate session_drain \
    --passed-gate routing \
    --passed-gate correctness_logs \
    --passed-gate attention_stats_capture \
    --passed-gate decode_token_join \
    --artifact "session_drain=${RUN_ROOT}/session_drain.env" \
    --artifact "routing=${RUN_ROOT}/routing_audit.json" \
    --artifact "correctness_logs=${RUN_ROOT}/correctness_audit.env" \
    --artifact "run_metadata=${RUN_ROOT}/run_metadata.json" \
    --artifact "effective_config=${RUN_ROOT}/effective_config.env" \
    --artifact \
      "tracked_worktree_patch=${RUN_ROOT}/tracked_worktree.patch" \
    --artifact "tracked_index_patch=${RUN_ROOT}/tracked_index.patch" \
    --artifact \
      "attention_stats=${RUN_ROOT}/attention_fast_path_stats.json" \
    --artifact \
      "decode_token_join=${RUN_ROOT}/decode_token_join_audit.env" \
    "${deferred_trace_artifact_args[@]}"
fi

echo "RUN_ROOT=${RUN_ROOT}"
