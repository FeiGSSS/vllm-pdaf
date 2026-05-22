#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
PREFILL_PORT="${PAP_NATIVE_PREFILL_PORT:-8110}"
DECODE_PORT="${PAP_NATIVE_DECODE_PORT:-8210}"
PROXY_PORT="${PAP_NATIVE_PROXY_PORT:-9010}"
MAX_MODEL_LEN="${PAP_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${PAP_MAX_NUM_SEQS:-8}"
PREFILL_GPU="${PAP_NATIVE_PREFILL_GPU:-0}"
DECODE_GPU="${PAP_NATIVE_DECODE_GPU:-1}"
PREFILL_NIXL_PORT="${PAP_NATIVE_PREFILL_NIXL_PORT:-5569}"
DECODE_NIXL_PORT="${PAP_NATIVE_DECODE_NIXL_PORT:-6010}"
PREFILL_GPU_MEMORY_UTILIZATION="${PAP_NATIVE_PREFILL_GPU_MEMORY_UTILIZATION:-0.45}"
DECODE_GPU_MEMORY_UTILIZATION="${PAP_NATIVE_DECODE_GPU_MEMORY_UTILIZATION:-0.80}"
LOG_DIR="${PAP_LOG_DIR:-$ROOT_DIR/examples/pap/logs/native_pd}"
RESULT_PATH="${PAP_RESULT_PATH:-$LOG_DIR/result.json}"
PROMPT="${PAP_PROMPT:-Briefly explain what PAP does.}"
MAX_TOKENS="${PAP_MAX_TOKENS:-8}"

mkdir -p "$LOG_DIR"

PIDS=()
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
    exit "$code"
}
trap cleanup INT TERM EXIT

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
    echo "Python package 'nixl' is not installed in .venv; install it before running native PD NIXL." >&2
    exit 1
fi

cd "$ROOT_DIR"

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
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

CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT" \
.venv/bin/vllm serve "$MODEL_PATH" \
    --port "$DECODE_PORT" \
    --host 127.0.0.1 \
    --enforce-eager \
    --generation-config vllm \
    --enable-request-id-headers \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
    >"$LOG_DIR/decode.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$PREFILL_PORT/health" "Native PD Prefill"
wait_for_http "http://127.0.0.1:$DECODE_PORT/health" "Native PD Decode"

.venv/bin/python examples/pap/native_pd_proxy_server.py \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --prefill-host 127.0.0.1 \
    --prefill-port "$PREFILL_PORT" \
    --prefill-nixl-port "$PREFILL_NIXL_PORT" \
    --decode-host 127.0.0.1 \
    --decode-port "$DECODE_PORT" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$PROXY_PORT/health" "Native PD proxy"

.venv/bin/python examples/pap/run_one_request.py \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --output "$RESULT_PATH" \
    | tee "$LOG_DIR/one_request.log"

echo "Native PD experiment finished. Logs are in $LOG_DIR"
