#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE="${ROOT_DIR}/benchmarks/pap/microbench/torch_device_graph_bridge.cu"
LIBRARY="${TMPDIR:-/tmp}/libpap_torch_device_graph_smoke.so"
NVCC="${PAP_NVSHMEM_NVCC:-${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc}"
CUDA_BIN="$(dirname "${NVCC}")"

if [[ ! -x "${NVCC}" ]]; then
  echo "CUDA 13 nvcc is missing: ${NVCC}" >&2
  exit 2
fi
if [[ ! -f "${LIBRARY}" || "${SOURCE}" -nt "${LIBRARY}" ]]; then
  PATH="${CUDA_BIN}:${PATH}" "${NVCC}" \
    -std=c++17 \
    -O3 \
    -lineinfo \
    -rdc=true \
    -shared \
    -Xcompiler=-fPIC \
    -arch="${PAP_NVSHMEM_GPU_ARCH:-sm_89}" \
    "${SOURCE}" \
    -lcudadevrt \
    -o "${LIBRARY}"
fi

exec "${ROOT_DIR}/.venv/bin/python" \
  "${ROOT_DIR}/benchmarks/pap/microbench/torch_device_graph_smoke.py" \
  --library "${LIBRARY}" \
  "$@"
