#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
LANE="${1:-}"
DATASET="${PAP_S128_BASELINE_DATASET:-/data/ssd1/llm-datasets/aiperf-research/pap-synthetic-longctx-i28k-o50-200-t2-5-seed42/aiperf_multiturn_s128.jsonl}"
DATASET_SHA256="5421e2d4f9868d4b0dc3f36b5a9aa8e256fadfd929dffd789dbb62692591bd9a"

case "${LANE}" in
  pap | pd | dynamo | dp8) ;;
  *)
    echo "usage: $0 {pap|pd|dynamo|dp8}" >&2
    exit 2
    ;;
esac

[[ -f "${DATASET}" ]] || {
  echo "missing frozen S128 dataset: ${DATASET}" >&2
  exit 1
}
actual_sha256="$(sha256sum "${DATASET}" | awk '{print $1}')"
[[ "${actual_sha256}" == "${DATASET_SHA256}" ]] || {
  echo "S128 dataset digest mismatch: ${actual_sha256}" >&2
  exit 1
}

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${LANE}" == "pap" ]]; then
  run_id="${PAP_RUN_ID:-${timestamp}_baseline_pap_7pa1p_vllm026}"
  env \
    PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}" \
    VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}" \
    AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}" \
    MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}" \
    PAP_TOPOLOGY=7pa1p \
    PAP_AIPERF_INPUT_FILE="${DATASET}" \
    PAP_AIPERF_EXPECTED_REQUESTS=455 \
    PAP_AIPERF_SESSIONS=128 \
    PAP_AIPERF_VARIABLE_TURNS=1 \
    PAP_AIPERF_CONCURRENCY=32 \
    RUN_ID="${run_id}" \
    MAX_MODEL_LEN=32768 \
    PAP_PREFILL_MAX_NUM_BATCHED_TOKENS=2048 \
    PAP_PREFILL_MAX_NUM_SEQS=256 \
    PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS=256 \
    PAP_PROJECTION_MAX_NUM_SEQS=256 \
    PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90 \
    PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,16,32,64,128 \
    PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=64 \
    PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1 \
    BENCH_TIMEOUT=1200 \
    SERVER_START_TIMEOUT=900 \
    bash "${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
elif [[ "${LANE}" == "pd" ]]; then
  run_id="${PD_LOAD_RUN_ID:-${timestamp}_baseline_pd_6p2d_vllm026}"
  env \
    VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv-dynamo/bin/vllm}" \
    AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}" \
    MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}" \
    PD_EXPECTED_VLLM_VERSION=0.26.0 \
    PD_LOAD_TOPOLOGY=6p2d \
    PD_AIPERF_INPUT_FILE="${DATASET}" \
    PD_AIPERF_EXPECTED_REQUESTS=455 \
    PD_LOAD_CONVERSATIONS=128 \
    PD_AIPERF_CONCURRENCY=32 \
    PD_LOAD_RUN_ID="${run_id}" \
    PD_LOAD_MAX_MODEL_LEN=32768 \
    PD_LOAD_PREFILL_MAX_NUM_BATCHED_TOKENS=2048 \
    PD_LOAD_PREFILL_MAX_NUM_SEQS=256 \
    PD_LOAD_DECODE_MAX_NUM_BATCHED_TOKENS=2048 \
    PD_LOAD_DECODE_MAX_NUM_SEQS=256 \
    PD_LOAD_GPU_MEMORY_UTILIZATION=0.90 \
    PD_LOAD_MIN_KV_TRANSFER_MB_S=5000 \
    PD_LOAD_PREFILL_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,16,32,64,128 \
    PD_LOAD_DECODE_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,12,16,20,24,28,32 \
    PD_LOAD_REQUEST_TIMEOUT_SECONDS=1200 \
    bash "${ROOT_DIR}/benchmarks/pap/scripts/run_pd_multiturn_topology.sh" \
      oneway
elif [[ "${LANE}" == "dynamo" ]]; then
  run_id="${DYNAMO_PD_RUN_ID:-${timestamp}_baseline_dynamo_6p2d_vllm026}"
  env \
    DYNAMO_PYTHON="${DYNAMO_PYTHON:-${ROOT_DIR}/.venv-dynamo/bin/python}" \
    AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}" \
    MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}" \
    DYNAMO_PD_TOPOLOGY=6p2d \
    DYNAMO_PD_ROUTER_MODE=kv \
    DYNAMO_PD_DISCOVERY_BACKEND=etcd \
    DYNAMO_PD_START_ETCD=1 \
    DYNAMO_PD_AIPERF_INPUT_FILE="${DATASET}" \
    DYNAMO_PD_AIPERF_EXPECTED_REQUESTS=455 \
    DYNAMO_PD_AIPERF_SESSIONS=128 \
    DYNAMO_PD_AIPERF_CONCURRENCY=32 \
    DYNAMO_PD_RUN_ID="${run_id}" \
    DYNAMO_PD_MAX_MODEL_LEN=32768 \
    DYNAMO_PD_PREFILL_MAX_NUM_BATCHED_TOKENS=2048 \
    DYNAMO_PD_DECODE_MAX_NUM_BATCHED_TOKENS=2048 \
    DYNAMO_PD_MAX_NUM_SEQS=256 \
    DYNAMO_PD_GPU_MEMORY_UTILIZATION=0.90 \
    DYNAMO_PD_MIN_KV_TRANSFER_MB_S=5000 \
    DYNAMO_PD_PREFILL_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,16,32,64,128 \
    DYNAMO_PD_DECODE_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,12,16,20,24,28,32 \
    DYNAMO_PD_AIPERF_REQUEST_TIMEOUT_SECONDS=1200 \
    bash "${ROOT_DIR}/benchmarks/pap/scripts/run_dynamo_pd_workload.sh"
else
  run_id="${DP_LOAD_RUN_ID:-${timestamp}_baseline_dp8_vllm026}"
  env \
    PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-dynamo/bin/python}" \
    VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv-dynamo/bin/vllm}" \
    AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}" \
    MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}" \
    DP_LOAD_SIZE=8 \
    DP_LOAD_RUN_ID="${run_id}" \
    DP_AIPERF_INPUT_FILE="${DATASET}" \
    DP_LOAD_CONVERSATIONS=128 \
    DP_AIPERF_CONCURRENCY=32 \
    DP_LOAD_MAX_MODEL_LEN=32768 \
    DP_LOAD_MAX_NUM_BATCHED_TOKENS=2048 \
    DP_LOAD_MAX_NUM_SEQS=256 \
    DP_LOAD_GPU_MEMORY_UTILIZATION=0.90 \
    DP_LOAD_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,12,16,20,24,28,32,64,128 \
    DP_LOAD_REQUEST_TIMEOUT_SECONDS=1200 \
    bash "${ROOT_DIR}/benchmarks/pap/scripts/run_dp_multiturn.sh"
fi
