#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
PORT="${PAP_FUSED_PORT:-8010}"
GPU="${PAP_FUSED_GPU:-0}"
MAX_MODEL_LEN="${PAP_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${PAP_MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${PAP_FUSED_GPU_MEMORY_UTILIZATION:-0.80}"
LOG_DIR="${PAP_LOG_DIR:-$ROOT_DIR/examples/pap/logs/fused}"
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

cd "$ROOT_DIR"

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

CUDA_VISIBLE_DEVICES="$GPU" \
.venv/bin/vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --enforce-eager \
    --generation-config vllm \
    --enable-request-id-headers \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    >"$LOG_DIR/server.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://127.0.0.1:$PORT/health" "Fused vLLM"

.venv/bin/python examples/pap/run_one_request.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --output "$RESULT_PATH" \
    | tee "$LOG_DIR/one_request.log"

echo "Fused experiment finished. Logs are in $LOG_DIR"
