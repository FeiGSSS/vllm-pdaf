#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${PAP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${PAP_CONSISTENCY_DIR:-$ROOT_DIR/examples/pap/logs/consistency/$RUN_ID}"
MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
PROMPT="${PAP_PROMPT:-Briefly explain what PAP does.}"
MAX_TOKENS="${PAP_MAX_TOKENS:-8}"

mkdir -p "$OUT_DIR"
cd "$ROOT_DIR"

export PAP_MODEL_PATH="$MODEL_PATH"
export PAP_PROMPT="$PROMPT"
export PAP_MAX_TOKENS="$MAX_TOKENS"
export PAP_MAX_MODEL_LEN="${PAP_MAX_MODEL_LEN:-1024}"
export PAP_MAX_NUM_SEQS="${PAP_MAX_NUM_SEQS:-2}"

run_arch() {
    local arch=$1
    local script=$2
    local log_dir="$OUT_DIR/$arch"
    mkdir -p "$log_dir"
    echo "===== Running $arch ====="
    if [[ "$arch" == "pap" ]]; then
        PAP_LOG_DIR="$log_dir" \
        PAP_RESULT_PATH="$log_dir/result.json" \
        PAP_MPS_PIPE_DIR="${PAP_MPS_PIPE_DIR:-/tmp/pap-${RUN_ID}-mps-pipe}" \
        PAP_MPS_LOG_DIR="${PAP_MPS_LOG_DIR:-$log_dir/mps-log}" \
        bash "$script" \
            >"$log_dir/launcher.log" 2>&1
    else
        PAP_LOG_DIR="$log_dir" \
        PAP_RESULT_PATH="$log_dir/result.json" \
        bash "$script" \
            >"$log_dir/launcher.log" 2>&1
    fi
    sleep "${PAP_BETWEEN_RUN_SLEEP:-5}"
}

run_arch fused examples/pap/launch_fused_qwen3_8b.sh
run_arch native_pd examples/pap/launch_native_pd_qwen3_8b_nixl.sh
run_arch pap examples/pap/launch_pap_qwen3_8b_nixl.sh

set +e
.venv/bin/python examples/pap/compare_outputs.py \
    --fused "$OUT_DIR/fused/result.json" \
    --native-pd "$OUT_DIR/native_pd/result.json" \
    --pap "$OUT_DIR/pap/result.json" \
    --output "$OUT_DIR/compare.json" \
    >"$OUT_DIR/compare.log" 2>&1
COMPARE_STATUS=$?
set -e
cat "$OUT_DIR/compare.log"

echo "Consistency artifacts are in $OUT_DIR"
exit "$COMPARE_STATUS"
