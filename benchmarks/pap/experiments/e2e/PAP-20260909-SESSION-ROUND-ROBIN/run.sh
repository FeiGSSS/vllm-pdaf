#!/usr/bin/env bash

set -euo pipefail

POLICY="${1:?usage: run.sh POLICY RUN_ID}"
RUN_ID="${2:?usage: run.sh POLICY RUN_ID}"
case "${POLICY}" in
  dynamo | session_round_robin) ;;
  *)
    echo "unsupported policy: ${POLICY}" >&2
    exit 2
    ;;
esac

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
RUN_ROOT="${ROOT_DIR}/benchmarks/pap/experiments/_staging/session_round_robin/${RUN_ID}"
DATASET="${ROOT_DIR}/benchmarks/pap/datasets/agentic-code/s60-t3-native32k-stratified-seed42/dataset.jsonl"

env \
  MODEL_PATH=/data/ssd1/llm-models/Qwen3-8B \
  PAP_TOPOLOGY=7pa1p \
  PAP_PREFILL_GPUS=0,1,2,3,4,5,6 \
  PAP_PROJECTION_GPUS=7 \
  PAP_ROUTING_POLICY="${POLICY}" \
  PAP_SESSION_ROUTER_MAX_CONVERSATIONS=60 \
  PAP_VLLM_DTYPE=float16 \
  MAX_MODEL_LEN=32768 \
  PAP_PREFILL_MAX_NUM_BATCHED_TOKENS=2048 \
  PAP_PREFILL_MAX_NUM_SEQS=256 \
  PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS=256 \
  PAP_PROJECTION_MAX_NUM_SEQS=256 \
  PAP_PREFILL_ASYNC_SCHEDULING=off \
  PAP_PROJECTION_ASYNC_SCHEDULING=on \
  PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90 \
  PAP_BLOCK_SIZE=16 \
  PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=256 \
  PAP_AIPERF_INPUT_FILE="${DATASET}" \
  PAP_AIPERF_SESSIONS=60 \
  PAP_AIPERF_EXPECTED_REQUESTS=180 \
  PAP_AIPERF_CONCURRENCY=30 \
  PAP_AIPERF_TIMING_MODE=concurrency \
  PAP_AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
  PAP_PREFILL_IPC_PROFILE=0 \
  PAP_RUNTIME_CUDA_CONTEXT_AUDIT=1 \
  BENCH_TIMEOUT=1800 \
  PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=0 \
  PAP_BENCH_STRICT_CORRECTNESS_AUDIT=1 \
  RUN_ID="${RUN_ID}" \
  RUN_ROOT="${RUN_ROOT}" \
  bash "${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"

gpu_processes="$(
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
    --format=csv,noheader,nounits
)"
if [[ -n "${gpu_processes}" ]]; then
  jq -n --arg processes "${gpu_processes}" \
    '{status:"failed",processes:$processes}' >"${RUN_ROOT}/cleanup_audit.json"
  echo "GPU processes remain after cleanup" >&2
  exit 1
fi
printf '{"status":"passed","gpu_processes":0}\n' \
  >"${RUN_ROOT}/cleanup_audit.json"
