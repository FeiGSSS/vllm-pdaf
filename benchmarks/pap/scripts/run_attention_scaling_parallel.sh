#!/usr/bin/env bash
set -euo pipefail

# Run a measured Attention workload/config matrix across independent L20 shards.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SINGLE_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_attention_scaling.sh"
MERGER="${ROOT_DIR}/benchmarks/pap/tooling/merge_attention_scaling.py"
TABLE_EXPORTER="${ROOT_DIR}/benchmarks/pap/tooling/attention_latency_table.py"
GPU_LIST="${PAP_ATTENTION_SCALING_GPUS:-0,1,2,3,4,5,6,7}"
OUTPUT_ROOT="${PAP_ATTENTION_SCALING_OUTPUT_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/microbench/_runs/$(date +%Y%m%d_%H%M%S)_attention_scaling_expanded}"

IFS=',' read -r -a gpu_indices <<< "${GPU_LIST}"
shard_count="${#gpu_indices[@]}"
(( shard_count > 0 )) || { echo "ERROR: empty GPU list" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/shards"

pids=()
for shard_index in "${!gpu_indices[@]}"; do
  gpu_index="${gpu_indices[${shard_index}]}"
  shard_root="${OUTPUT_ROOT}/shards/shard_${shard_index}"
  env \
    PAP_ATTENTION_SCALING_GPU="${gpu_index}" \
    PAP_ATTENTION_SCALING_GROUPS=expanded \
    PAP_ATTENTION_SCALING_KERNEL_SET=practical \
    PAP_ATTENTION_SCALING_SHARD_COUNT="${shard_count}" \
    PAP_ATTENTION_SCALING_SHARD_INDEX="${shard_index}" \
    PAP_ATTENTION_SCALING_RUN_NSYS=0 \
    PAP_ATTENTION_SCALING_OUTPUT_ROOT="${shard_root}" \
    bash "${SINGLE_RUNNER}" \
    > "${OUTPUT_ROOT}/logs/shard_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  for log in "${OUTPUT_ROOT}"/logs/*.log; do
    echo "===== ${log}" >&2
    tail -n 80 "${log}" >&2
  done
  exit 1
fi

shard_results=()
for shard_index in "${!gpu_indices[@]}"; do
  shard_results+=(
    "${OUTPUT_ROOT}/shards/shard_${shard_index}/timing/result.json"
  )
done
{
  printf 'GPU_LIST=%q\n' "${GPU_LIST}"
  printf 'SHARD_COUNT=%q\n' "${shard_count}"
  printf 'SHAPE_GROUPS=expanded\n'
  printf 'KERNEL_SET=practical\n'
  printf 'MODEL_FITTED=0\n'
} > "${OUTPUT_ROOT}/parallel_run.env"
"${PYTHON_BIN}" "${MERGER}" \
  --output "${OUTPUT_ROOT}/timing/result.json" "${shard_results[@]}"
"${PYTHON_BIN}" "${TABLE_EXPORTER}" \
  "${OUTPUT_ROOT}/timing/result.json" "${OUTPUT_ROOT}/tables"
printf 'STATUS=passed\n' > "${OUTPUT_ROOT}/status.env"
echo "PAP_ATTENTION_SCALING_OUTPUT_ROOT=${OUTPUT_ROOT}"
