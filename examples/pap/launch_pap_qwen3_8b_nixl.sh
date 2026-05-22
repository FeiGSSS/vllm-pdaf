#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
PREFILL_PORT="${PAP_PREFILL_PORT:-8100}"
PROJECTION_PORT="${PAP_PROJECTION_PORT:-8200}"
ATTENTION_PORT="${PAP_ATTENTION_PORT:-8300}"
PROXY_PORT="${PAP_PROXY_PORT:-9000}"
MAX_MODEL_LEN="${PAP_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${PAP_MAX_NUM_SEQS:-8}"
PREFILL_GPU="${PAP_PREFILL_GPU:-0}"
PROJECTION_GPU="${PAP_PROJECTION_GPU:-1}"
PREFILL_NIXL_PORT="${PAP_PREFILL_NIXL_PORT:-5559}"
PROJECTION_NIXL_PORT="${PAP_PROJECTION_NIXL_PORT:-6000}"
PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.45}"
PROJECTION_GPU_MEMORY_UTILIZATION="${PAP_PROJECTION_GPU_MEMORY_UTILIZATION:-0.80}"
PREFILL_MPS_PERCENT="${PAP_PREFILL_MPS_PERCENT:-70}"
ATTENTION_MPS_PERCENT="${PAP_ATTENTION_MPS_PERCENT:-30}"
ENABLE_MPS="${PAP_ENABLE_MPS:-0}"
LOG_DIR="${PAP_LOG_DIR:-$ROOT_DIR/examples/pap/logs}"
MPS_PIPE_DIR="${PAP_MPS_PIPE_DIR:-$LOG_DIR/mps-pipe}"
MPS_LOG_DIR="${PAP_MPS_LOG_DIR:-$LOG_DIR/mps-log}"
RESULT_PATH="${PAP_RESULT_PATH:-$LOG_DIR/result.json}"
PROMPT="${PAP_PROMPT:-Briefly explain what PAP does.}"

mkdir -p "$LOG_DIR"

PIDS=()
MPS_STARTED=0

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
    if [[ "$MPS_STARTED" == "1" ]]; then
        timeout 5 bash -c 'echo quit | CUDA_MPS_PIPE_DIRECTORY="$0" nvidia-cuda-mps-control' "$MPS_PIPE_DIR" >/dev/null 2>&1 || true
    fi
    exit "$code"
}
trap cleanup INT TERM EXIT

start_mps_if_requested() {
    if [[ "$ENABLE_MPS" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "PAP_ENABLE_MPS=1 but nvidia-cuda-mps-control was not found" >&2
        return 1
    fi
    mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
    local attempt
    for attempt in 1 2 3; do
        echo "Starting CUDA MPS daemon for Prefill/Attention on GPU $PREFILL_GPU (attempt $attempt)"
        if CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
            CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" \
            CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" \
            nvidia-cuda-mps-control -d; then
            MPS_STARTED=1
            return 0
        fi
        timeout 5 bash -c 'echo quit | CUDA_MPS_PIPE_DIRECTORY="$0" nvidia-cuda-mps-control' "$MPS_PIPE_DIR" >/dev/null 2>&1 || true
        sleep 2
    done
    echo "Failed to start CUDA MPS daemon after 3 attempts" >&2
    return 1
}

with_prefill_attention_mps_env() {
    if [[ "$ENABLE_MPS" == "1" ]]; then
        env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" "$@"
    else
        env "$@"
    fi
}

wait_for_http() {
    local url=$1
    local name=$2
    local timeout_seconds=${3:-600}
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

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

if ! .venv/bin/python -c 'import nixl' >/dev/null 2>&1; then
    echo "Python package 'nixl' is not installed in .venv; install it before running PAP NIXL." >&2
    exit 1
fi

cd "$ROOT_DIR"

start_mps_if_requested

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

echo "Starting PAP Attention internal executor on GPU $PREFILL_GPU"
with_prefill_attention_mps_env \
    CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$ATTENTION_MPS_PERCENT" \
    .venv/bin/python examples/pap/pap_attention_executor.py \
    --host 127.0.0.1 \
    --port "$ATTENTION_PORT" \
    >"$LOG_DIR/attention.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$ATTENTION_PORT/health" "PAP Attention"

echo "Starting PAP Prefill vLLM NIXL producer on GPU $PREFILL_GPU"
with_prefill_attention_mps_env \
    CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PREFILL_MPS_PERCENT" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT" \
    .venv/bin/vllm serve "$MODEL_PATH" \
    --port "$PREFILL_PORT" \
    --host 127.0.0.1 \
    --enforce-eager \
    --generation-config vllm \
    --enable-request-id-headers \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=("$!")

PROJECTION_KV_TRANSFER_CONFIG=$(printf '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"pap_shadow_attention":true,"pap_remote_attention":true,"pap_attention_endpoint":"http://127.0.0.1:%s"}}' "$ATTENTION_PORT")

echo "Starting PAP Projection vLLM NIXL consumer on GPU $PROJECTION_GPU"
CUDA_VISIBLE_DEVICES="$PROJECTION_GPU" \
PAP_SHADOW_ATTENTION="${PAP_SHADOW_ATTENTION:-1}" \
PAP_REMOTE_ATTENTION="${PAP_REMOTE_ATTENTION:-1}" \
PAP_ATTENTION_ENDPOINT="http://127.0.0.1:$ATTENTION_PORT" \
VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
VLLM_NIXL_SIDE_CHANNEL_PORT="$PROJECTION_NIXL_PORT" \
.venv/bin/vllm serve "$MODEL_PATH" \
    --port "$PROJECTION_PORT" \
    --host 127.0.0.1 \
    --enforce-eager \
    --generation-config vllm \
    --enable-request-id-headers \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$PROJECTION_GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config "$PROJECTION_KV_TRANSFER_CONFIG" \
    >"$LOG_DIR/projection.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$PREFILL_PORT/health" "PAP Prefill"
wait_for_http "http://127.0.0.1:$PROJECTION_PORT/health" "PAP Projection"

echo "Starting PAP proxy on port $PROXY_PORT"
.venv/bin/python examples/pap/pap_proxy_server.py \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --prefill-host 127.0.0.1 \
    --prefill-port "$PREFILL_PORT" \
    --prefill-nixl-port "$PREFILL_NIXL_PORT" \
    --attention-host 127.0.0.1 \
    --attention-port "$ATTENTION_PORT" \
    --projection-host 127.0.0.1 \
    --projection-port "$PROJECTION_PORT" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$PROXY_PORT/health" "PAP proxy"

echo "Running one PAP request through proxy"
.venv/bin/python examples/pap/run_one_request.py \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max-tokens "${PAP_MAX_TOKENS:-8}" \
    --conversation-id "pap-one-turn" \
    --output "$RESULT_PATH" \
    | tee "$LOG_DIR/one_request.log"

if [[ "${PAP_KEEP_SERVERS:-0}" == "1" ]]; then
    echo "PAP_KEEP_SERVERS=1; services remain running. Press Ctrl-C to stop. Logs are in $LOG_DIR"
    wait
fi

echo "PAP experiment finished. Logs are in $LOG_DIR"

