#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
PA_COUNT="${PAP_PA_COUNT:-6}"
PROJECTION_COUNT="${PAP_PROJECTION_COUNT:-2}"
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
PAP_MODE="${PAP_MODE:-debug_remote_attention}"
PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-prototype_tcp}"
PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM:-16}"
PAP_OFFLOAD_EXEC_NUM_HEADS="${PAP_OFFLOAD_EXEC_NUM_HEADS:-32}"
PAP_OFFLOAD_EXEC_NUM_KV_HEADS="${PAP_OFFLOAD_EXEC_NUM_KV_HEADS:-8}"
PAP_OFFLOAD_EXEC_HEAD_DIM="${PAP_OFFLOAD_EXEC_HEAD_DIM:-128}"
PAP_OFFLOAD_EXEC_Q_SIZE="${PAP_OFFLOAD_EXEC_Q_SIZE:-4096}"
PAP_OFFLOAD_EXEC_KV_SIZE="${PAP_OFFLOAD_EXEC_KV_SIZE:-1024}"
PAP_OFFLOAD_EXEC_TRACE="${PAP_OFFLOAD_EXEC_TRACE:-0}"
PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY:-round_robin}"
PREFILL_NIXL_PORT_BASE="${PAP_PREFILL_NIXL_PORT_BASE:-5559}"
PROJECTION_NIXL_PORT_BASE="${PAP_PROJECTION_NIXL_PORT_BASE:-6000}"
VLLM_PORT_BASE="${PAP_VLLM_PORT_BASE:-50000}"
MAX_MODEL_LEN="${PAP_MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${PAP_MAX_NUM_SEQS:-2}"
PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.45}"
PROJECTION_GPU_MEMORY_UTILIZATION="${PAP_PROJECTION_GPU_MEMORY_UTILIZATION:-0.80}"
PREFILL_MPS_PERCENT="${PAP_PREFILL_MPS_PERCENT:-70}"
ATTENTION_MPS_PERCENT="${PAP_ATTENTION_MPS_PERCENT:-30}"
ENABLE_MPS="${PAP_ENABLE_MPS:-1}"
LOG_DIR="${PAP_LOG_DIR:-$ROOT_DIR/examples/pap/logs/6pa2p}"
MPS_PIPE_BASE_DIR="${PAP_MPS_PIPE_BASE_DIR:-/tmp/pap-mps-${USER:-user}-6pa2p-$$}"
MPS_LOG_BASE_DIR="${PAP_MPS_LOG_BASE_DIR:-$LOG_DIR/mps-log}"
RESULT_PATH="${PAP_RESULT_PATH:-$LOG_DIR/result.json}"
PROMPT="${PAP_PROMPT:-Briefly explain what PAP does.}"
PREFILL_GPUS_CSV="${PAP_PREFILL_GPUS:-0,1,2,3,4,5}"
PROJECTION_GPUS_CSV="${PAP_PROJECTION_GPUS:-6,7}"

mkdir -p "$LOG_DIR"

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
        local attention_port=$((ATTENTION_PORT_BASE + idx))
        local attention_tcp_port=$((ATTENTION_TCP_PORT_BASE + idx))
        local attention_zmq_port=$((ATTENTION_ZMQ_PORT_BASE + idx))
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

if ! .venv/bin/python -c 'import nixl' >/dev/null 2>&1; then
    echo "Python package 'nixl' is not installed in .venv; install it before running PAP NIXL." >&2
    exit 1
fi

split_csv "$PREFILL_GPUS_CSV" PREFILL_GPUS
split_csv "$PROJECTION_GPUS_CSV" PROJECTION_GPUS
require_count "PAP_PREFILL_GPUS" "${#PREFILL_GPUS[@]}" "$PA_COUNT"
require_count "PAP_PROJECTION_GPUS" "${#PROJECTION_GPUS[@]}" "$PROJECTION_COUNT"

cd "$ROOT_DIR"

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

for (( idx=0; idx<PA_COUNT; idx++ )); do
    start_mps_for_pa "$idx" "${PREFILL_GPUS[$idx]}"
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
    gpu="${PREFILL_GPUS[$idx]}"
    attention_port=$((ATTENTION_PORT_BASE + idx))
    attention_tcp_port=$((ATTENTION_TCP_PORT_BASE + idx))
    attention_zmq_port=$((ATTENTION_ZMQ_PORT_BASE + idx))
    echo "Starting PAP Attention internal executor $idx on GPU $gpu"
    with_pa_mps_env "$idx" \
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$ATTENTION_MPS_PERCENT" \
        PAP_OFFLOAD_EXEC_TRANSPORT="$PAP_OFFLOAD_EXEC_TRANSPORT" \
        PAP_OFFLOAD_EXEC_LOCAL_RANK=0 \
        PAP_OFFLOAD_EXEC_Q_SIZE="$PAP_OFFLOAD_EXEC_Q_SIZE" \
        PAP_OFFLOAD_EXEC_KV_SIZE="$PAP_OFFLOAD_EXEC_KV_SIZE" \
        PAP_OFFLOAD_EXEC_NUM_HEADS="$PAP_OFFLOAD_EXEC_NUM_HEADS" \
        PAP_OFFLOAD_EXEC_NUM_KV_HEADS="$PAP_OFFLOAD_EXEC_NUM_KV_HEADS" \
        PAP_OFFLOAD_EXEC_HEAD_DIM="$PAP_OFFLOAD_EXEC_HEAD_DIM" \
        PAP_OFFLOAD_EXEC_TRACE="$PAP_OFFLOAD_EXEC_TRACE" \
        .venv/bin/python examples/pap/pap_attention_executor.py \
        --host 127.0.0.1 \
        --port "$attention_port" \
        --tcp-port "$attention_tcp_port" \
        --offload-exec-zmq-port "$attention_zmq_port" \
        >"$LOG_DIR/attention_${idx}.log" 2>&1 &
    PIDS+=("$!")
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
    wait_for_http "http://127.0.0.1:$((ATTENTION_PORT_BASE + idx))/health" "PAP Attention $idx"
done

for (( idx=0; idx<PA_COUNT; idx++ )); do
    gpu="${PREFILL_GPUS[$idx]}"
    prefill_port=$((PREFILL_PORT_BASE + idx))
    prefill_nixl_port=$((PREFILL_NIXL_PORT_BASE + idx))
    echo "Starting PAP Prefill vLLM NIXL producer $idx on GPU $gpu"
    with_pa_mps_env "$idx" \
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PREFILL_MPS_PERCENT" \
        VLLM_PORT="$((VLLM_PORT_BASE + idx * 20))" \
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
        --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
        >"$LOG_DIR/prefill_${idx}.log" 2>&1 &
    PIDS+=("$!")
done

for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do
    gpu="${PROJECTION_GPUS[$idx]}"
    projection_port=$((PROJECTION_PORT_BASE + idx))
    projection_nixl_port=$((PROJECTION_NIXL_PORT_BASE + idx))
    projection_zmq_port=$((PROJECTION_ZMQ_PORT_BASE + idx))
    projection_kv_transfer_config=$(printf '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"pap_shadow_attention":true,"pap_remote_attention":true,"pap_attention_endpoint":"http://127.0.0.1:%s","pap_attention_tcp_endpoint":"tcp://127.0.0.1:%s","pap_mode":"%s"}}' "$ATTENTION_PORT_BASE" "$ATTENTION_TCP_PORT_BASE" "$PAP_MODE")
    echo "Starting PAP Projection vLLM NIXL consumer $idx on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    VLLM_PORT="$((VLLM_PORT_BASE + PA_COUNT * 20 + idx * 20))" \
    PAP_SHADOW_ATTENTION="${PAP_SHADOW_ATTENTION:-1}" \
    PAP_REMOTE_ATTENTION="${PAP_REMOTE_ATTENTION:-1}" \
    PAP_ATTENTION_ENDPOINT="http://127.0.0.1:$ATTENTION_PORT_BASE" \
    PAP_REMOTE_ATTENTION_TRANSPORT="${PAP_REMOTE_ATTENTION_TRANSPORT:-tcp}" \
    PAP_OFFLOAD_EXEC_TRANSPORT="$PAP_OFFLOAD_EXEC_TRANSPORT" \
    PAP_OFFLOAD_EXEC_HOST=127.0.0.1 \
    PAP_OFFLOAD_EXEC_ZMQ_PORT="$projection_zmq_port" \
    PAP_OFFLOAD_EXEC_LOCAL_RANK=0 \
    PAP_OFFLOAD_EXEC_TRACE="$PAP_OFFLOAD_EXEC_TRACE" \
    PAP_REMOTE_ATTENTION_PARALLELISM="$PAP_REMOTE_ATTENTION_PARALLELISM" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$projection_nixl_port" \
    .venv/bin/vllm serve "$MODEL_PATH" \
        --port "$projection_port" \
        --host 127.0.0.1 \
        --enforce-eager \
        --generation-config vllm \
        --enable-request-id-headers \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$PROJECTION_GPU_MEMORY_UTILIZATION" \
        --kv-transfer-config "$projection_kv_transfer_config" \
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
    echo "Running one PAP 6PA:2P request through proxy"
    .venv/bin/python examples/pap/run_one_request.py \
        --host 127.0.0.1 \
        --port "$PROXY_PORT" \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-tokens "${PAP_MAX_TOKENS:-8}" \
        --conversation-id "pap-6pa2p-one-turn" \
        --output "$RESULT_PATH" \
        | tee "$LOG_DIR/one_request.log"
fi

if [[ "${PAP_KEEP_SERVERS:-0}" == "1" ]]; then
    echo "PAP_KEEP_SERVERS=1; services remain running. Press Ctrl-C to stop. Logs are in $LOG_DIR"
    wait
fi

echo "PAP 6PA:2P experiment finished. Logs are in $LOG_DIR"
