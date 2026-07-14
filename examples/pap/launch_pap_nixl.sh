#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -v PAP_ASYNC_DECODE_TOKEN ]]; then
    echo "PAP_ASYNC_DECODE_TOKEN was removed; sampled-token delivery is "\
"unconditionally asynchronous. Historical evidence: "\
"PAP-20260713-ASYNC-DECODE-TOKEN-D2H." >&2
    exit 2
fi

usage() {
    cat <<'EOF'
Usage: launch_pap_nixl.sh [--model MODEL_PATH_OR_NAME]

Environment overrides:
  PAP_MODEL_PATH          Model path or Hugging Face model id.
  PAP_TOPOLOGY           Topology such as 1pa1p or 6pa2p.
EOF
}

MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-0.6B}"
while (($#)); do
    case "$1" in
        -m | --model)
            if (($# < 2)); then
                echo "$1 requires a model path or name" >&2
                exit 2
            fi
            MODEL_PATH="$2"
            shift 2
            ;;
        --model=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
export PAP_MODEL_PATH="$MODEL_PATH"
TOPOLOGY="${PAP_TOPOLOGY:-6pa2p}"
if [[ ! "$TOPOLOGY" =~ ^([0-9]+)pa([0-9]+)p$ ]]; then
    echo "Unsupported PAP topology: $TOPOLOGY" >&2
    exit 1
fi
TOPOLOGY_TAG="$(printf '%s' "$TOPOLOGY" | tr '[:lower:]' '[:upper:]')"
PA_COUNT="${PAP_PA_COUNT:-${BASH_REMATCH[1]}}"
PROJECTION_COUNT="${PAP_PROJECTION_COUNT:-${BASH_REMATCH[2]}}"
PAP_TP_SIZE="${PAP_TP_SIZE:-1}"
PAP_DISABLE_CUSTOM_ALL_REDUCE="${PAP_DISABLE_CUSTOM_ALL_REDUCE:-auto}"
if (( PA_COUNT < 1 || PROJECTION_COUNT < 1 )); then
    echo "PAP topology must include at least one PA and one Projection: $TOPOLOGY" >&2
    exit 1
fi
if (( PAP_TP_SIZE < 1 )); then
    echo "PAP_TP_SIZE must be positive: $PAP_TP_SIZE" >&2
    exit 1
fi
PREFILL_GPU_COUNT=$((PA_COUNT * PAP_TP_SIZE))
PROJECTION_GPU_COUNT=$((PROJECTION_COUNT * PAP_TP_SIZE))
TOTAL_GPU_COUNT=$(((PA_COUNT + PROJECTION_COUNT) * PAP_TP_SIZE))
if (( TOTAL_GPU_COUNT > 8 )); then
    echo "PAP topology $TOPOLOGY requires $TOTAL_GPU_COUNT GPUs; max is 8" >&2
    exit 1
fi
PREFILL_PORT_BASE="${PAP_PREFILL_PORT_BASE:-8100}"
PROJECTION_PORT_BASE="${PAP_PROJECTION_PORT_BASE:-8200}"
ATTENTION_PORT_BASE="${PAP_ATTENTION_PORT_BASE:-8300}"
ATTENTION_TCP_PORT_BASE="${PAP_ATTENTION_TCP_PORT_BASE:-9300}"
ATTENTION_ZMQ_PORT_BASE="${PAP_ATTENTION_ZMQ_PORT_BASE:-10300}"
PROJECTION_ZMQ_PORT_BASE="${PAP_PROJECTION_ZMQ_PORT_BASE:-11300}"
PROXY_PORT="${PAP_PROXY_PORT:-9000}"
SERVICE_ONLY="${PAP_SERVICE_ONLY:-0}"
STATUS_FILE="${PAP_STATUS_FILE:-}"
SKIP_SMOKE_REQUEST="${PAP_SKIP_SMOKE_REQUEST:-0}"
PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-nixl_mailbox}"
case "$PAP_OFFLOAD_EXEC_TRANSPORT" in
    nixl|nixl_mailbox) ;;
    local_fast|local-fast|cuda_ipc_fast) ;;
    *)
        echo "PAP_OFFLOAD_EXEC_TRANSPORT=$PAP_OFFLOAD_EXEC_TRANSPORT is not supported; use nixl_mailbox or local_fast" >&2
        exit 1
        ;;
esac
PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"
PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM:-16}"
PAP_OFFLOAD_EXEC_NUM_HEADS="${PAP_OFFLOAD_EXEC_NUM_HEADS:-}"
PAP_OFFLOAD_EXEC_NUM_KV_HEADS="${PAP_OFFLOAD_EXEC_NUM_KV_HEADS:-}"
PAP_OFFLOAD_EXEC_HEAD_DIM="${PAP_OFFLOAD_EXEC_HEAD_DIM:-}"
PAP_OFFLOAD_EXEC_Q_SIZE="${PAP_OFFLOAD_EXEC_Q_SIZE:-}"
PAP_OFFLOAD_EXEC_KV_SIZE="${PAP_OFFLOAD_EXEC_KV_SIZE:-}"
PAP_OFFLOAD_EXEC_TRACE="${PAP_OFFLOAD_EXEC_TRACE:-0}"
PAP_ATTENTION_KV_DEBUG="${PAP_ATTENTION_KV_DEBUG:-0}"
PAP_KV_LOCALITY_PROFILE="${PAP_KV_LOCALITY_PROFILE:-0}"
PAP_KV_LOCALITY_PROFILE_MIN_BATCH="${PAP_KV_LOCALITY_PROFILE_MIN_BATCH:-1}"
PAP_UNIFIED_KV="${PAP_UNIFIED_KV:-1}"
PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS:-32}"
PAP_DECODE_COMMIT_ENDPOINT="${PAP_DECODE_COMMIT_ENDPOINT:-}"
PAP_DECODE_COMMIT_TIMEOUT="${PAP_DECODE_COMMIT_TIMEOUT:-0.2}"
PAP_DECODE_COMMIT_QUEUE_SIZE="${PAP_DECODE_COMMIT_QUEUE_SIZE:-1024}"
PAP_DECODE_COMMIT_MAX_ATTEMPTS="${PAP_DECODE_COMMIT_MAX_ATTEMPTS:-8}"
PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS="${PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS:-0.05}"
PAP_DECODE_COMMIT_RETRY_MAX_SECONDS="${PAP_DECODE_COMMIT_RETRY_MAX_SECONDS:-0.5}"
PAP_DECODE_COMMIT_FLUSH_TIMEOUT="${PAP_DECODE_COMMIT_FLUSH_TIMEOUT:-5.0}"
PAP_DECODE_TOKEN_TIMEOUT="${PAP_DECODE_TOKEN_TIMEOUT:-0.2}"
PAP_DECODE_TOKEN_QUEUE_SIZE="${PAP_DECODE_TOKEN_QUEUE_SIZE:-1024}"
PAP_DECODE_TOKEN_MAX_ATTEMPTS="${PAP_DECODE_TOKEN_MAX_ATTEMPTS:-8}"
PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS="${PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS:-0.05}"
PAP_DECODE_TOKEN_RETRY_MAX_SECONDS="${PAP_DECODE_TOKEN_RETRY_MAX_SECONDS:-0.5}"
PAP_DECODE_TOKEN_FLUSH_TIMEOUT="${PAP_DECODE_TOKEN_FLUSH_TIMEOUT:-5.0}"
PAP_LEASE_RELEASE_ENDPOINT="${PAP_LEASE_RELEASE_ENDPOINT:-}"
PAP_LEASE_RELEASE_TIMEOUT="${PAP_LEASE_RELEASE_TIMEOUT:-5.0}"
PAP_LEASE_RELEASE_MAX_ATTEMPTS="${PAP_LEASE_RELEASE_MAX_ATTEMPTS:-5}"
PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS="${PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS:-0.05}"
PAP_LEASE_RELEASE_RETRY_MAX_SECONDS="${PAP_LEASE_RELEASE_RETRY_MAX_SECONDS:-0.5}"
PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS:-300}"
PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS:-0}"
export PAP_DECODE_COMMIT_TIMEOUT PAP_DECODE_COMMIT_QUEUE_SIZE
export PAP_DECODE_COMMIT_MAX_ATTEMPTS PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS
export PAP_DECODE_COMMIT_RETRY_MAX_SECONDS PAP_DECODE_COMMIT_FLUSH_TIMEOUT
export PAP_DECODE_TOKEN_TIMEOUT PAP_DECODE_TOKEN_QUEUE_SIZE
export PAP_DECODE_TOKEN_MAX_ATTEMPTS PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS
export PAP_DECODE_TOKEN_RETRY_MAX_SECONDS PAP_DECODE_TOKEN_FLUSH_TIMEOUT
export PAP_LEASE_RELEASE_TIMEOUT PAP_LEASE_RELEASE_MAX_ATTEMPTS
export PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS PAP_LEASE_RELEASE_RETRY_MAX_SECONDS
export PAP_KV_LEASE_TTL_SECONDS
PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY:-round_robin}"
unset PAP_ATTENTION_COPY_PREFIX_KV
PREFILL_NIXL_PORT_BASE="${PAP_PREFILL_NIXL_PORT_BASE:-5559}"
PROJECTION_NIXL_PORT_BASE="${PAP_PROJECTION_NIXL_PORT_BASE:-6000}"
VLLM_PORT_BASE="${PAP_VLLM_PORT_BASE:-50000}"
MAX_MODEL_LEN="${PAP_MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${PAP_MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${PAP_MAX_NUM_BATCHED_TOKENS:-8192}"
PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.80}"
PROJECTION_GPU_MEMORY_UTILIZATION="${PAP_PROJECTION_GPU_MEMORY_UTILIZATION:-0.80}"
PREFILL_MPS_PERCENT="${PAP_PREFILL_MPS_PERCENT:-70}"
ATTENTION_MPS_PERCENT="${PAP_ATTENTION_MPS_PERCENT:-30}"
ENABLE_MPS="${PAP_ENABLE_MPS:-1}"
LOG_DIR="${PAP_LOG_DIR:-$ROOT_DIR/examples/pap/logs/$TOPOLOGY}"
MPS_PIPE_BASE_DIR="${PAP_MPS_PIPE_BASE_DIR:-/tmp/pap-mps-${USER:-user}-${TOPOLOGY}-$$}"
MPS_LOG_BASE_DIR="${PAP_MPS_LOG_BASE_DIR:-$LOG_DIR/mps-log}"
RESULT_PATH="${PAP_RESULT_PATH:-$LOG_DIR/result.json}"
PROMPT="${PAP_PROMPT:-Briefly explain what PAP does.}"
DEFAULT_PREFILL_GPUS="$(seq -s, 0 $((PREFILL_GPU_COUNT - 1)))"
DEFAULT_PROJECTION_GPUS="$(
    seq -s, "$PREFILL_GPU_COUNT" $((TOTAL_GPU_COUNT - 1))
)"
PREFILL_GPUS_CSV="${PAP_PREFILL_GPUS:-$DEFAULT_PREFILL_GPUS}"
PROJECTION_GPUS_CSV="${PAP_PROJECTION_GPUS:-$DEFAULT_PROJECTION_GPUS}"

mkdir -p "$LOG_DIR"

vllm_scheduler_args=()
vllm_tp_args=()
vllm_prefill_observability_args=()
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    vllm_scheduler_args+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
fi
case "$PAP_ENABLE_PROMPT_TOKENS_DETAILS" in
    1 | true | True | TRUE | yes | Yes | YES)
        vllm_prefill_observability_args+=("--enable-prompt-tokens-details")
        ;;
    0 | false | False | FALSE | no | No | NO)
        ;;
    *)
        echo "PAP_ENABLE_PROMPT_TOKENS_DETAILS must be 0 or 1" >&2
        exit 1
        ;;
esac
case "$PAP_DISABLE_CUSTOM_ALL_REDUCE" in
    auto)
        if (( PAP_TP_SIZE > 1 )); then
            vllm_tp_args+=("--disable-custom-all-reduce")
        fi
        ;;
    1 | true | True | TRUE | yes | Yes | YES)
        vllm_tp_args+=("--disable-custom-all-reduce")
        ;;
    0 | false | False | FALSE | no | No | NO)
        ;;
    *)
        echo "PAP_DISABLE_CUSTOM_ALL_REDUCE must be auto, 0, or 1: $PAP_DISABLE_CUSTOM_ALL_REDUCE" >&2
        exit 1
        ;;
esac

PIDS=()
MPS_PIPE_DIRS=()
MPS_LOG_DIRS=()
MPS_STARTED_DIRS=()
PREFILL_GPUS=()
PROJECTION_GPUS=()

split_csv() {
    local csv=$1
    local -n out=$2
    IFS=',' read -r -a out <<<"$csv"
}

join_by_comma() {
    local IFS=','
    echo "$*"
}

join_by_pipe() {
    local IFS='|'
    echo "$*"
}

gpu_group_csv() {
    local -n gpus=$1
    local group_idx=$2
    local start=$((group_idx * PAP_TP_SIZE))
    local selected=()
    local rank
    for (( rank=0; rank<PAP_TP_SIZE; rank++ )); do
        selected+=("${gpus[$((start + rank))]}")
    done
    join_by_comma "${selected[@]}"
}

gpu_group_rank() {
    local -n gpus=$1
    local group_idx=$2
    local rank=$3
    echo "${gpus[$((group_idx * PAP_TP_SIZE + rank))]}"
}

build_rank_ports() {
    local base=$1
    local group_idx=$2
    local ports=()
    local rank
    for (( rank=0; rank<PAP_TP_SIZE; rank++ )); do
        ports+=("$((base + group_idx * PAP_TP_SIZE + rank))")
    done
    join_by_pipe "${ports[@]}"
}

build_rank_ports_csv() {
    local base=$1
    local group_idx=$2
    local ports=()
    local rank
    for (( rank=0; rank<PAP_TP_SIZE; rank++ )); do
        ports+=("$((base + group_idx * PAP_TP_SIZE + rank))")
    done
    join_by_comma "${ports[@]}"
}

cleanup() {
    local code=$?
    set +e
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    sleep 2
    for pid in "${PIDS[@]:-}"; do
        kill -0 "$pid" >/dev/null 2>&1 && kill -KILL "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${PIDS[@]:-}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
    for pipe_dir in "${MPS_STARTED_DIRS[@]:-}"; do
        timeout 5 bash -c 'echo quit | CUDA_MPS_PIPE_DIRECTORY="$0" nvidia-cuda-mps-control' "$pipe_dir" >/dev/null 2>&1 || true
    done
    exit "$code"
}
trap cleanup INT TERM EXIT

require_count() {
    local name=$1
    local actual=$2
    local expected=$3
    if (( actual < expected )); then
        echo "$name has $actual entries but needs at least $expected" >&2
        exit 1
    fi
}

wait_for_http() {
    local url=$1
    local name=$2
    local timeout_seconds=${3:-900}
    local start
    start=$(date +%s)
    while true; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "$name is ready at $url"
            return 0
        fi
        if (( $(date +%s) - start > timeout_seconds )); then
            echo "Timed out waiting for $name at $url" >&2
            return 1
        fi
        sleep 2
    done
}

start_mps_for_pa() {
    local idx=$1
    local gpu=$2
    if [[ "$ENABLE_MPS" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "PAP_ENABLE_MPS=1 but nvidia-cuda-mps-control was not found" >&2
        return 1
    fi
    local pipe_dir="$MPS_PIPE_BASE_DIR/pa-$idx"
    local log_dir="$MPS_LOG_BASE_DIR/pa-$idx"
    mkdir -p "$pipe_dir" "$log_dir"
    MPS_PIPE_DIRS[$idx]="$pipe_dir"
    MPS_LOG_DIRS[$idx]="$log_dir"
    local attempt
    for attempt in 1 2 3; do
        echo "Starting CUDA MPS daemon for PA group $idx on GPU $gpu (attempt $attempt)"
        if CUDA_VISIBLE_DEVICES="$gpu" \
            CUDA_MPS_PIPE_DIRECTORY="$pipe_dir" \
            CUDA_MPS_LOG_DIRECTORY="$log_dir" \
            nvidia-cuda-mps-control -d; then
            MPS_STARTED_DIRS+=("$pipe_dir")
            return 0
        fi
        timeout 5 bash -c 'echo quit | CUDA_MPS_PIPE_DIRECTORY="$0" nvidia-cuda-mps-control' "$pipe_dir" >/dev/null 2>&1 || true
        sleep 2
    done
    echo "Failed to start CUDA MPS daemon for PA group $idx after 3 attempts" >&2
    return 1
}

with_pa_mps_env() {
    local idx=$1
    shift
    if [[ "$ENABLE_MPS" == "1" ]]; then
        exec env CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIRS[$idx]}" CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIRS[$idx]}" "$@"
    else
        exec env "$@"
    fi
}

build_pap_groups_spec() {
    local spec=""
    local idx
    for (( idx=0; idx<PA_COUNT; idx++ )); do
        local prefill_port=$((PREFILL_PORT_BASE + idx))
        local prefill_nixl_port=$((PREFILL_NIXL_PORT_BASE + idx))
        local attention_port
        local attention_tcp_port
        local attention_zmq_port
        attention_port="$(build_rank_ports "$ATTENTION_PORT_BASE" "$idx")"
        attention_tcp_port="$(build_rank_ports "$ATTENTION_TCP_PORT_BASE" "$idx")"
        attention_zmq_port="$(build_rank_ports "$ATTENTION_ZMQ_PORT_BASE" "$idx")"
        local item="127.0.0.1:${prefill_port}:${prefill_nixl_port}:127.0.0.1:${attention_port}:${attention_tcp_port}:${attention_zmq_port}"
        if [[ -z "$spec" ]]; then
            spec="$item"
        else
            spec="$spec,$item"
        fi
    done
    echo "$spec"
}

build_projections_spec() {
    local spec=""
    local idx
    for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
        local projection_port=$((PROJECTION_PORT_BASE + idx))
        local item="127.0.0.1:${projection_port}"
        if [[ -z "$spec" ]]; then
            spec="$item"
        else
            spec="$spec,$item"
        fi
    done
    echo "$spec"
}

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

if ! "$ROOT_DIR/.venv/bin/python" -c 'import nixl' >/dev/null 2>&1; then
    echo "Python package 'nixl' is not installed in .venv; install it before running PAP NIXL." >&2
    exit 1
fi

read -r MODEL_NUM_HEADS MODEL_NUM_KV_HEADS MODEL_HEAD_DIM < <(
    "$ROOT_DIR/.venv/bin/python" - "$MODEL_PATH/config.json" <<'PY'
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
if (( MODEL_NUM_HEADS % PAP_TP_SIZE != 0 )); then
    echo "Model num_attention_heads=$MODEL_NUM_HEADS is not divisible by PAP_TP_SIZE=$PAP_TP_SIZE" >&2
    exit 1
fi
if (( MODEL_NUM_KV_HEADS >= PAP_TP_SIZE && MODEL_NUM_KV_HEADS % PAP_TP_SIZE != 0 )); then
    echo "Model num_key_value_heads=$MODEL_NUM_KV_HEADS is not divisible by PAP_TP_SIZE=$PAP_TP_SIZE" >&2
    exit 1
fi
PAP_OFFLOAD_EXEC_NUM_HEADS="${PAP_OFFLOAD_EXEC_NUM_HEADS:-$((MODEL_NUM_HEADS / PAP_TP_SIZE))}"
PAP_OFFLOAD_EXEC_NUM_KV_HEADS="${PAP_OFFLOAD_EXEC_NUM_KV_HEADS:-$((MODEL_NUM_KV_HEADS >= PAP_TP_SIZE ? MODEL_NUM_KV_HEADS / PAP_TP_SIZE : 1))}"
PAP_OFFLOAD_EXEC_HEAD_DIM="${PAP_OFFLOAD_EXEC_HEAD_DIM:-$MODEL_HEAD_DIM}"
PAP_OFFLOAD_EXEC_Q_SIZE="${PAP_OFFLOAD_EXEC_Q_SIZE:-$((PAP_OFFLOAD_EXEC_NUM_HEADS * PAP_OFFLOAD_EXEC_HEAD_DIM))}"
PAP_OFFLOAD_EXEC_KV_SIZE="${PAP_OFFLOAD_EXEC_KV_SIZE:-$((PAP_OFFLOAD_EXEC_NUM_KV_HEADS * PAP_OFFLOAD_EXEC_HEAD_DIM))}"

split_csv "$PREFILL_GPUS_CSV" PREFILL_GPUS
split_csv "$PROJECTION_GPUS_CSV" PROJECTION_GPUS
require_count "PAP_PREFILL_GPUS" "${#PREFILL_GPUS[@]}" "$PREFILL_GPU_COUNT"
require_count "PAP_PROJECTION_GPUS" "${#PROJECTION_GPUS[@]}" "$PROJECTION_GPU_COUNT"
if (( PAP_TP_SIZE > 1 )) && [[ "$ENABLE_MPS" == "1" ]]; then
    echo "PAP_ENABLE_MPS=1 is not supported with PAP_TP_SIZE > 1; set PAP_ENABLE_MPS=0" >&2
    exit 1
fi

cd "$ROOT_DIR"

export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

if [[ "$ENABLE_MPS" == "1" ]]; then
    for (( idx=0; idx<PA_COUNT; idx++ )); do
        start_mps_for_pa "$idx" "${PREFILL_GPUS[$idx]}"
    done
fi

for (( idx=0; idx<PA_COUNT; idx++ )); do
    prefill_control_port=$((PREFILL_PORT_BASE + idx))
    decode_commit_endpoint="${PAP_DECODE_COMMIT_ENDPOINT:-http://127.0.0.1:${prefill_control_port}/v1/pap/prefill/decode-commit}"
    lease_release_endpoint="${PAP_LEASE_RELEASE_ENDPOINT:-http://127.0.0.1:${prefill_control_port}/v1/pap/prefill/lease-release}"
    for (( rank=0; rank<PAP_TP_SIZE; rank++ )); do
        gpu="$(gpu_group_rank PREFILL_GPUS "$idx" "$rank")"
        attention_port=$((ATTENTION_PORT_BASE + idx * PAP_TP_SIZE + rank))
        attention_tcp_port=$((ATTENTION_TCP_PORT_BASE + idx * PAP_TP_SIZE + rank))
        attention_zmq_port=$((ATTENTION_ZMQ_PORT_BASE + idx * PAP_TP_SIZE + rank))
        echo "Starting PAP Attention internal executor $idx rank $rank on GPU $gpu"
        if [[ "$ENABLE_MPS" == "1" ]]; then
            with_pa_mps_env "$idx" \
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$ATTENTION_MPS_PERCENT" \
                PAP_NIXL_MAILBOX_ACTOR_ID="attention-${idx}-${rank}" \
                PAP_OFFLOAD_EXEC_TRANSPORT="$PAP_OFFLOAD_EXEC_TRANSPORT" \
                PAP_OFFLOAD_KV_TRANSPORT="$PAP_OFFLOAD_KV_TRANSPORT" \
                PAP_OFFLOAD_EXEC_LOCAL_RANK=0 \
                PAP_OFFLOAD_EXEC_Q_SIZE="$PAP_OFFLOAD_EXEC_Q_SIZE" \
                PAP_OFFLOAD_EXEC_KV_SIZE="$PAP_OFFLOAD_EXEC_KV_SIZE" \
                PAP_OFFLOAD_EXEC_NUM_HEADS="$PAP_OFFLOAD_EXEC_NUM_HEADS" \
                PAP_OFFLOAD_EXEC_NUM_KV_HEADS="$PAP_OFFLOAD_EXEC_NUM_KV_HEADS" \
                PAP_OFFLOAD_EXEC_HEAD_DIM="$PAP_OFFLOAD_EXEC_HEAD_DIM" \
                PAP_OFFLOAD_EXEC_TRACE="$PAP_OFFLOAD_EXEC_TRACE" \
                PAP_ATTENTION_KV_DEBUG="$PAP_ATTENTION_KV_DEBUG" \
                PAP_KV_LOCALITY_PROFILE="$PAP_KV_LOCALITY_PROFILE" \
                PAP_KV_LOCALITY_PROFILE_MIN_BATCH="$PAP_KV_LOCALITY_PROFILE_MIN_BATCH" \
                PAP_DECODE_COMMIT_ENDPOINT="$decode_commit_endpoint" \
                PAP_LEASE_RELEASE_ENDPOINT="$lease_release_endpoint" \
                .venv/bin/python -m vllm.pap.attention_executor \
                --host 127.0.0.1 \
                --port "$attention_port" \
                --tcp-port "$attention_tcp_port" \
                --offload-exec-zmq-port "$attention_zmq_port" \
                >"$LOG_DIR/attention_${idx}_${rank}.log" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="$gpu" \
            PAP_NIXL_MAILBOX_ACTOR_ID="attention-${idx}-${rank}" \
            PAP_OFFLOAD_EXEC_TRANSPORT="$PAP_OFFLOAD_EXEC_TRANSPORT" \
            PAP_OFFLOAD_KV_TRANSPORT="$PAP_OFFLOAD_KV_TRANSPORT" \
            PAP_OFFLOAD_EXEC_LOCAL_RANK=0 \
            PAP_OFFLOAD_EXEC_Q_SIZE="$PAP_OFFLOAD_EXEC_Q_SIZE" \
            PAP_OFFLOAD_EXEC_KV_SIZE="$PAP_OFFLOAD_EXEC_KV_SIZE" \
            PAP_OFFLOAD_EXEC_NUM_HEADS="$PAP_OFFLOAD_EXEC_NUM_HEADS" \
            PAP_OFFLOAD_EXEC_NUM_KV_HEADS="$PAP_OFFLOAD_EXEC_NUM_KV_HEADS" \
            PAP_OFFLOAD_EXEC_HEAD_DIM="$PAP_OFFLOAD_EXEC_HEAD_DIM" \
            PAP_OFFLOAD_EXEC_TRACE="$PAP_OFFLOAD_EXEC_TRACE" \
            PAP_ATTENTION_KV_DEBUG="$PAP_ATTENTION_KV_DEBUG" \
            PAP_KV_LOCALITY_PROFILE="$PAP_KV_LOCALITY_PROFILE" \
            PAP_KV_LOCALITY_PROFILE_MIN_BATCH="$PAP_KV_LOCALITY_PROFILE_MIN_BATCH" \
            PAP_DECODE_COMMIT_ENDPOINT="$decode_commit_endpoint" \
            PAP_LEASE_RELEASE_ENDPOINT="$lease_release_endpoint" \
            .venv/bin/python -m vllm.pap.attention_executor \
            --host 127.0.0.1 \
            --port "$attention_port" \
            --tcp-port "$attention_tcp_port" \
            --offload-exec-zmq-port "$attention_zmq_port" \
            >"$LOG_DIR/attention_${idx}_${rank}.log" 2>&1 &
        fi
        PIDS+=("$!")
    done
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
    for (( rank=0; rank<PAP_TP_SIZE; rank++ )); do
        wait_for_http "http://127.0.0.1:$((ATTENTION_PORT_BASE + idx * PAP_TP_SIZE + rank))/health" "PAP Attention $idx rank $rank"
    done
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
    gpu_csv="$(gpu_group_csv PREFILL_GPUS "$idx")"
    prefill_port=$((PREFILL_PORT_BASE + idx))
    prefill_nixl_port=$((PREFILL_NIXL_PORT_BASE + idx))
    echo "Starting PAP Prefill vLLM NIXL producer $idx on GPU(s) $gpu_csv"
    if [[ "$ENABLE_MPS" == "1" ]]; then
        with_pa_mps_env "$idx" \
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PREFILL_MPS_PERCENT" \
            VLLM_PORT="$((VLLM_PORT_BASE + idx * 20))" \
            PAP_OFFLOAD_KV_TRANSPORT="$PAP_OFFLOAD_KV_TRANSPORT" \
            PAP_UNIFIED_KV="$PAP_UNIFIED_KV" \
            PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="$PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS" \
            PAP_KV_LEASE_TTL_SECONDS="$PAP_KV_LEASE_TTL_SECONDS" \
            VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
            VLLM_NIXL_SIDE_CHANNEL_PORT="$prefill_nixl_port" \
            .venv/bin/vllm serve "$MODEL_PATH" \
            --port "$prefill_port" \
            --host 127.0.0.1 \
            --enforce-eager \
            --generation-config vllm \
            --enable-request-id-headers \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            "${vllm_scheduler_args[@]}" \
            "${vllm_prefill_observability_args[@]}" \
            --tensor-parallel-size "$PAP_TP_SIZE" \
            --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
            "${vllm_tp_args[@]}" \
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
            >"$LOG_DIR/prefill_${idx}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu_csv" \
        VLLM_PORT="$((VLLM_PORT_BASE + idx * 20))" \
        PAP_OFFLOAD_KV_TRANSPORT="$PAP_OFFLOAD_KV_TRANSPORT" \
        PAP_UNIFIED_KV="$PAP_UNIFIED_KV" \
        PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="$PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS" \
        PAP_KV_LEASE_TTL_SECONDS="$PAP_KV_LEASE_TTL_SECONDS" \
        VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
        VLLM_NIXL_SIDE_CHANNEL_PORT="$prefill_nixl_port" \
        .venv/bin/vllm serve "$MODEL_PATH" \
            --port "$prefill_port" \
            --host 127.0.0.1 \
            --enforce-eager \
            --generation-config vllm \
            --enable-request-id-headers \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            "${vllm_scheduler_args[@]}" \
            "${vllm_prefill_observability_args[@]}" \
            --tensor-parallel-size "$PAP_TP_SIZE" \
            --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
            "${vllm_tp_args[@]}" \
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
            >"$LOG_DIR/prefill_${idx}.log" 2>&1 &
    fi
    PIDS+=("$!")
done

for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    gpu_csv="$(gpu_group_csv PROJECTION_GPUS "$idx")"
    projection_port=$((PROJECTION_PORT_BASE + idx))
    projection_zmq_port="$(build_rank_ports_csv "$PROJECTION_ZMQ_PORT_BASE" "$idx")"
    echo "Starting PAP Projection vLLM metadata-only $idx on GPU(s) $gpu_csv"
    CUDA_VISIBLE_DEVICES="$gpu_csv" \
    VLLM_PORT="$((VLLM_PORT_BASE + PA_COUNT * 20 + idx * 20))" \
    PAP_NIXL_MAILBOX_ACTOR_ID="projection-${idx}" \
    PAP_OFFLOAD_EXEC_TRANSPORT="$PAP_OFFLOAD_EXEC_TRANSPORT" \
    PAP_OFFLOAD_KV_TRANSPORT="$PAP_OFFLOAD_KV_TRANSPORT" \
    PAP_OFFLOAD_EXEC_HOST=127.0.0.1 \
    PAP_OFFLOAD_EXEC_ZMQ_PORT="$projection_zmq_port" \
    PAP_ATTENTION_ZMQ_PORT_BASE="$ATTENTION_ZMQ_PORT_BASE" \
    PAP_ATTENTION_PORT_BASE="$ATTENTION_PORT_BASE" \
    PAP_TP_SIZE="$PAP_TP_SIZE" \
    PAP_OFFLOAD_EXEC_TRACE="$PAP_OFFLOAD_EXEC_TRACE" \
    PAP_PROJECTION_KV_UNAWARE=1 \
    PAP_REMOTE_ATTENTION_PARALLELISM="$PAP_REMOTE_ATTENTION_PARALLELISM" \
    .venv/bin/vllm serve "$MODEL_PATH" \
        --port "$projection_port" \
        --host 127.0.0.1 \
        --enforce-eager \
        --generation-config vllm \
        --enable-request-id-headers \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        "${vllm_scheduler_args[@]}" \
        --tensor-parallel-size "$PAP_TP_SIZE" \
        --gpu-memory-utilization "$PROJECTION_GPU_MEMORY_UTILIZATION" \
        "${vllm_tp_args[@]}" \
        >"$LOG_DIR/projection_${idx}.log" 2>&1 &
    PIDS+=("$!")
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
    wait_for_http "http://127.0.0.1:$((PREFILL_PORT_BASE + idx))/health" "PAP Prefill $idx"
done
for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    wait_for_http "http://127.0.0.1:$((PROJECTION_PORT_BASE + idx))/health" "PAP Projection $idx"
done

pap_groups_spec="$(build_pap_groups_spec)"
projections_spec="$(build_projections_spec)"

echo "Starting multi PAP proxy on port $PROXY_PORT"
.venv/bin/python examples/pap/multi_pap_proxy_server.py \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --pap-groups "$pap_groups_spec" \
    --projections "$projections_spec" \
    --routing-policy "$PAP_ROUTING_POLICY" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$PROXY_PORT/health" "multi PAP proxy"

if [[ -n "$STATUS_FILE" ]]; then
    echo "$PROXY_PORT" >"$STATUS_FILE"
fi

if [[ "${SERVICE_ONLY}" == "1" ]]; then
    echo "PAP_SERVICE_ONLY=1; services remain running. Logs are in $LOG_DIR"
    wait
fi

if [[ "${SKIP_SMOKE_REQUEST}" != "1" ]]; then
    echo "Running one PAP $TOPOLOGY_TAG request through proxy"
    .venv/bin/python examples/pap/run_one_request.py \
        --host 127.0.0.1 \
        --port "$PROXY_PORT" \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-tokens "${PAP_MAX_TOKENS:-8}" \
        --conversation-id "pap-${TOPOLOGY}-one-turn" \
        --output "$RESULT_PATH" \
        | tee "$LOG_DIR/one_request.log"
fi

if [[ "${PAP_KEEP_SERVERS:-0}" == "1" ]]; then
    echo "PAP_KEEP_SERVERS=1; services remain running. Press Ctrl-C to stop. Logs are in $LOG_DIR"
    wait
fi

echo "PAP $TOPOLOGY_TAG experiment finished. Logs are in $LOG_DIR"
