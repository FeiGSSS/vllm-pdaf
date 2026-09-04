#!/usr/bin/env bash
set -euo pipefail

# Current 80-SM PAP Prefill utilization at selected post-knee lengths.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_prefill_saturation.sh"
METRICS="${ROOT_DIR}/benchmarks/pap/tooling/component_gpu_metrics.py"
GPU_INDEX="${PAP_PREFILL_UTILIZATION_GPU:-5}"
SHAPE_GROUPS="${PAP_PREFILL_UTILIZATION_GROUPS:-utilization}"
OUTPUT_ROOT="${PAP_PREFILL_UTILIZATION_OUTPUT_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/microbench/_runs/$(date +%Y%m%d_%H%M%S)_prefill_utilization}"
NSYS_IMPORTER="${PAP_NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}"
PREFIX="${OUTPUT_ROOT}/nsys/prefill"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

main() {
  [[ -x "${PYTHON_BIN}" && -x "${RUNNER}" && -f "${METRICS}" ]] \
    || die "missing Prefill benchmark environment"
  mkdir -p "${OUTPUT_ROOT}/nsys"
  local pids
  pids="$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${pids//[[:space:]]/}" ]] \
    || die "GPU ${GPU_INDEX} has active compute processes: ${pids}"
  env \
    PAP_PREFILL_MICROBENCH_GPU="${GPU_INDEX}" \
    PAP_PREFILL_MICROBENCH_CHUNKS=20 \
    PAP_PREFILL_MICROBENCH_EXPECTED_SMS=80 \
    PAP_PREFILL_MICROBENCH_GROUPS="${SHAPE_GROUPS}" \
    PAP_PREFILL_MICROBENCH_RUN_ROOT="${OUTPUT_ROOT}/run" \
    VLLM_QWEN3_COMPONENT_NVTX=prefill \
    VLLM_QWEN3_COMPONENT_NVTX_SYNC=1 \
    VLLM_QWEN3_COMPONENT_NVTX_GATE_FILE="${OUTPUT_ROOT}/measurement.nvtx_gate" \
    nsys profile \
      --force-overwrite=true \
      --sample=none \
      --trace=nvtx \
      --trace-fork-before-exec=true \
      --gpu-metrics-device="${GPU_INDEX}" \
      --gpu-metrics-set=5 \
      --gpu-metrics-frequency=10000 \
      --output="${PREFIX}" \
      bash "${RUNNER}"
  if [[ ! -f "${PREFIX}.nsys-rep" ]]; then
    [[ -x "${NSYS_IMPORTER}" && -f "${PREFIX}.qdstrm" ]] \
      || die "nsys report is missing"
    "${NSYS_IMPORTER}" --force-overwrite \
      --input-file "${PREFIX}.qdstrm" \
      --output-file "${PREFIX}.nsys-rep"
  fi
  nsys export --force-overwrite=true --type=sqlite \
    --output="${PREFIX}.sqlite" "${PREFIX}.nsys-rep"
  "${PYTHON_BIN}" "${METRICS}" "${PREFIX}.sqlite" \
    --component prefill --output "${OUTPUT_ROOT}/metrics.json"
  {
    printf 'STATUS=passed\n'
    printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'GPU_INDEX=%q\n' "${GPU_INDEX}"
    printf 'VISIBLE_SMS=80\n'
  } > "${OUTPUT_ROOT}/status.env"
  echo "PAP_PREFILL_UTILIZATION_OUTPUT_ROOT=${OUTPUT_ROOT}"
}

main "$@"
