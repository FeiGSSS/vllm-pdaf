#!/usr/bin/env bash
set -euo pipefail

# Real Qwen3/vLLM non-Attention decode-stage scaling on one Projection GPU.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PROBE="${ROOT_DIR}/benchmarks/pap/microbench/projection_scaling.py"
METRICS="${ROOT_DIR}/benchmarks/pap/tooling/component_gpu_metrics.py"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
MODEL_CONFIG="${MODEL_CONFIG:-${MODEL_PATH}/config.json}"
GPU_INDEX="${PAP_PROJECTION_SCALING_GPU:-7}"
RUN_TIMING="${PAP_PROJECTION_SCALING_RUN_TIMING:-1}"
RUN_NSYS="${PAP_PROJECTION_SCALING_RUN_NSYS:-1}"
BATCHES="${PAP_PROJECTION_SCALING_BATCHES:-1,2,4,8,16,32,64,128,256}"
NSYS_BATCHES="${PAP_PROJECTION_SCALING_NSYS_BATCHES:-1,32,256}"
OUTPUT_ROOT="${PAP_PROJECTION_SCALING_OUTPUT_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/_staging/microbench/$(date +%Y%m%d_%H%M%S)_projection_scaling}"
NSYS_IMPORTER="${PAP_NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

ensure_gpu_idle() {
  local pids
  pids="$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${pids//[[:space:]]/}" ]] \
    || die "GPU ${GPU_INDEX} has active compute processes: ${pids}"
}

import_nsys() {
  local prefix="$1"
  if [[ ! -f "${prefix}.nsys-rep" ]]; then
    [[ -x "${NSYS_IMPORTER}" && -f "${prefix}.qdstrm" ]] \
      || die "nsys report is missing: ${prefix}"
    "${NSYS_IMPORTER}" --force-overwrite \
      --input-file "${prefix}.qdstrm" --output-file "${prefix}.nsys-rep"
  fi
  nsys export --force-overwrite=true --type=sqlite \
    --output="${prefix}.sqlite" "${prefix}.nsys-rep"
}

run_nsys_case() {
  local batch="$1"
  local case_root="${OUTPUT_ROOT}/nsys/b${batch}"
  local prefix="${case_root}/profile"
  mkdir -p "${case_root}"
  env \
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    VLLM_QWEN3_COMPONENT_NVTX=projection \
    VLLM_QWEN3_COMPONENT_NVTX_SYNC=1 \
    VLLM_QWEN3_COMPONENT_NVTX_GATE_FILE="${case_root}/measurement.nvtx_gate" \
    VLLM_QWEN3_COMPONENT_NVTX_LAYER=18 \
    nsys profile \
      --force-overwrite=true \
      --sample=none \
      --trace=nvtx \
      --trace-fork-before-exec=true \
      --gpu-metrics-device="${GPU_INDEX}" \
      --gpu-metrics-set=5 \
      --gpu-metrics-frequency=10000 \
      --output="${prefix}" \
      "${PYTHON_BIN}" "${PROBE}" \
        --model "${MODEL_PATH}" \
        --model-config "${MODEL_CONFIG}" \
        --batch-sizes "${batch}" \
        --measure-output-tokens 16 \
        --output-root "${case_root}/run"
  import_nsys "${prefix}"
  "${PYTHON_BIN}" "${METRICS}" "${prefix}.sqlite" \
    --component projection --output "${case_root}/metrics.json"
}

main() {
  [[ -x "${PYTHON_BIN}" && -f "${PROBE}" && -f "${METRICS}" ]] \
    || die "missing Projection benchmark environment"
  [[ -d "${MODEL_PATH}" && -f "${MODEL_CONFIG}" ]] \
    || die "missing model"
  ensure_gpu_idle
  mkdir -p "${OUTPUT_ROOT}"
  local tracked_worktree_dirty=0
  git -C "${ROOT_DIR}" diff --quiet || tracked_worktree_dirty=1
  git -C "${ROOT_DIR}" diff --cached --quiet || tracked_worktree_dirty=1
  {
    printf 'RUN_SCRIPT=%q\n' "${BASH_SOURCE[0]}"
    printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'GIT_TRACKED_WORKTREE_DIRTY=%q\n' "${tracked_worktree_dirty}"
    printf 'MODEL_CONFIG_SHA256=%q\n' "$(sha256sum "${MODEL_CONFIG}" | cut -d' ' -f1)"
    printf 'GPU_INDEX=%q\n' "${GPU_INDEX}"
    printf 'GPU_UUID=%q\n' "$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=uuid --format=csv,noheader)"
    printf 'GPU_NAME=%q\n' "$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader)"
    printf 'DRIVER_VERSION=%q\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    printf 'BATCHES=%q\n' "${BATCHES}"
    printf 'NSYS_BATCHES=%q\n' "${NSYS_BATCHES}"
    printf 'RUN_TIMING=%q\n' "${RUN_TIMING}"
    printf 'RUN_NSYS=%q\n' "${RUN_NSYS}"
    printf 'PYTORCH_CUDA=%q\n' "$(env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    if [[ "${RUN_NSYS}" == "1" ]]; then
      printf 'NSYS_VERSION=%q\n' "$(nsys --version)"
    fi
  } > "${OUTPUT_ROOT}/run.env"
  if [[ "${RUN_TIMING}" == "1" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
      "${PYTHON_BIN}" "${PROBE}" \
        --model "${MODEL_PATH}" \
        --model-config "${MODEL_CONFIG}" \
        --batch-sizes "${BATCHES}" \
        --output-root "${OUTPUT_ROOT}/timing"
  fi
  if [[ "${RUN_NSYS}" == "1" ]]; then
    local -a nsys_batches
    IFS=',' read -r -a nsys_batches <<< "${NSYS_BATCHES}"
    local batch
    for batch in "${nsys_batches[@]}"; do
      run_nsys_case "${batch}"
    done
  fi
  printf 'STATUS=passed\n' > "${OUTPUT_ROOT}/status.env"
  echo "PAP_PROJECTION_SCALING_OUTPUT_ROOT=${OUTPUT_ROOT}"
}

main "$@"
