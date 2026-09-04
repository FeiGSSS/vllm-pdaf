#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NVSHMEM_PREFIX="${PAP_NVSHMEM_PREFIX:-${ROOT_DIR}/.local/nvshmem-3.3.24-cuda13}"
SOURCE="${ROOT_DIR}/vllm/pap/transport/nvshmem/device_bridge.cu"
VERSION_SCRIPT="${ROOT_DIR}/vllm/pap/transport/nvshmem/device_bridge.map"
OUTPUT="${NVSHMEM_PREFIX}/lib/libpap_nvshmem_device.so"
VENV_NVCC="${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
NVCC="${PAP_NVSHMEM_NVCC:-${VENV_NVCC}}"
CUDA_BIN="$(dirname "${NVCC}")"

if [[ ! -f "${NVSHMEM_PREFIX}/lib/libnvshmem_device.a" ]]; then
  echo "NVSHMEM device library is missing under ${NVSHMEM_PREFIX}" >&2
  exit 2
fi
if [[ ! -x "${NVCC}" ]]; then
  echo "CUDA 13 nvcc is missing: ${NVCC}" >&2
  exit 2
fi
if [[ "$("${NVCC}" --version)" != *"release 13."* ]]; then
  echo "PAP NVSHMEM bridge requires CUDA 13 nvcc: ${NVCC}" >&2
  exit 2
fi

if [[ -f "${OUTPUT}" \
  && "${OUTPUT}" -nt "${SOURCE}" \
  && "${OUTPUT}" -nt "${VERSION_SCRIPT}" ]]; then
  exit 0
fi

PATH="${CUDA_BIN}:${PATH}" "${NVCC}" \
  -std=c++17 \
  -O3 \
  -lineinfo \
  -rdc=true \
  -shared \
  -Xcompiler=-fPIC \
  -Xlinker="--version-script=${VERSION_SCRIPT}" \
  -arch="${PAP_NVSHMEM_GPU_ARCH:-sm_89}" \
  -I"${NVSHMEM_PREFIX}/include" \
  "${SOURCE}" \
  -L"${NVSHMEM_PREFIX}/lib" \
  -lcuda \
  -lnvshmem_host \
  -lnvshmem_device \
  -o "${OUTPUT}"

echo "Built PAP NVSHMEM device bridge: ${OUTPUT}"
