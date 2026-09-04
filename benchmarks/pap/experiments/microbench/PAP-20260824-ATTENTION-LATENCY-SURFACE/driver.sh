#!/usr/bin/env bash
set -euo pipefail

# Run a measured Attention workload/config matrix across independent L20 shards.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_RUNNER="${EXPERIMENT_DIR}/run_one.sh"
MERGER="${ROOT_DIR}/benchmarks/pap/tooling/merge_attention_scaling.py"
TABLE_EXPORTER="${ROOT_DIR}/benchmarks/pap/tooling/attention_latency_table.py"
MODEL_CONFIG="${MODEL_CONFIG:-/data/ssd1/llm-models/Qwen3-8B/config.json}"
GPU_LIST="${PAP_ATTENTION_SCALING_GPUS:-0,1,2,3,4,5,6,7}"
SHAPE_GROUPS="${PAP_ATTENTION_SCALING_GROUPS:-expanded}"
KERNEL_SET="${PAP_ATTENTION_SCALING_KERNEL_SET:-practical}"
RUN_NSYS="${PAP_ATTENTION_SCALING_RUN_NSYS:-0}"
EXPERIMENT_CONFIG="${PAP_ATTENTION_SCALING_EXPERIMENT_CONFIG:-}"
VALIDATE_ONLY="${PAP_ATTENTION_SCALING_VALIDATE_ONLY:-0}"
OUTPUT_ROOT="${PAP_ATTENTION_SCALING_OUTPUT_ROOT:-${EXPERIMENT_DIR}/runs/$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${EXPERIMENT_CONFIG}" && ! -f "${EXPERIMENT_CONFIG}" ]]; then
  echo "ERROR: experiment config is missing: ${EXPERIMENT_CONFIG}" >&2
  exit 2
fi
[[ "${VALIDATE_ONLY}" =~ ^[01]$ ]] || {
  echo "ERROR: PAP_ATTENTION_SCALING_VALIDATE_ONLY must be 0 or 1" >&2
  exit 2
}
for path in "${PYTHON_BIN}" "${SINGLE_RUNNER}" "${MERGER}" \
  "${TABLE_EXPORTER}" "${MODEL_CONFIG}"; do
  [[ -e "${path}" ]] || {
    echo "ERROR: required path is missing: ${path}" >&2
    exit 2
  }
done
if (( VALIDATE_ONLY == 1 )); then
  echo "Attention latency experiment configuration is valid"
  exit 0
fi

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
    PAP_ATTENTION_SCALING_GROUPS="${SHAPE_GROUPS}" \
    PAP_ATTENTION_SCALING_KERNEL_SET="${KERNEL_SET}" \
    PAP_ATTENTION_SCALING_SHARD_COUNT="${shard_count}" \
    PAP_ATTENTION_SCALING_SHARD_INDEX="${shard_index}" \
    PAP_ATTENTION_SCALING_RUN_NSYS="${RUN_NSYS}" \
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
  printf 'SHAPE_GROUPS=%q\n' "${SHAPE_GROUPS}"
  printf 'KERNEL_SET=%q\n' "${KERNEL_SET}"
  printf 'RUN_NSYS=%q\n' "${RUN_NSYS}"
  if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
    printf 'EXPERIMENT_CONFIG=%q\n' "${EXPERIMENT_CONFIG}"
    printf 'EXPERIMENT_CONFIG_SHA256=%q\n' \
      "$(sha256sum "${EXPERIMENT_CONFIG}" | cut -d' ' -f1)"
  fi
  printf 'MODEL_FITTED=0\n'
} > "${OUTPUT_ROOT}/parallel_run.env"
"${PYTHON_BIN}" "${MERGER}" \
  --output "${OUTPUT_ROOT}/timing/result.json" "${shard_results[@]}"
"${PYTHON_BIN}" "${TABLE_EXPORTER}" \
  "${OUTPUT_ROOT}/timing/result.json" "${OUTPUT_ROOT}/tables"
printf 'STATUS=passed\n' > "${OUTPUT_ROOT}/status.env"
echo "PAP_ATTENTION_SCALING_OUTPUT_ROOT=${OUTPUT_ROOT}"
