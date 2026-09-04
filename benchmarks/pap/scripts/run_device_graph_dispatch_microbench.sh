#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NVSHMEM_PREFIX="${PAP_NVSHMEM_PREFIX:-${ROOT_DIR}/.local/nvshmem-3.3.24-cuda13}"
SOURCE="${ROOT_DIR}/benchmarks/pap/microbench/device_graph_dispatch.cu"
BINARY="${TMPDIR:-/tmp}/pap_device_graph_dispatch"
NVCC="${PAP_NVSHMEM_NVCC:-${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc}"
CUDA_BIN="$(dirname "${NVCC}")"

if [[ ! -x "${NVSHMEM_PREFIX}/bin/nvshmrun.hydra" ]]; then
  echo "NVSHMEM launcher is missing under ${NVSHMEM_PREFIX}" >&2
  exit 2
fi
if [[ ! -x "${NVCC}" ]]; then
  echo "CUDA 13 nvcc is missing: ${NVCC}" >&2
  exit 2
fi

if [[ ! -x "${BINARY}" || "${SOURCE}" -nt "${BINARY}" ]]; then
  PATH="${CUDA_BIN}:${PATH}" "${NVCC}" \
    -std=c++17 \
    -O3 \
    -lineinfo \
    -rdc=true \
    -arch="${PAP_NVSHMEM_GPU_ARCH:-sm_89}" \
    -I"${NVSHMEM_PREFIX}/include" \
    "${SOURCE}" \
    -L"${NVSHMEM_PREFIX}/lib" \
    -lnvshmem_host \
    -lnvshmem_device \
    -lcudadevrt \
    -lcuda \
    -o "${BINARY}"
fi

export LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_BOOTSTRAP=PMI
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-16M}"
export NVSHMEM_DISABLE_CUDA_VMM=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec "${NVSHMEM_PREFIX}/bin/nvshmrun.hydra" \
  -launcher fork \
  -np 2 \
  "${BINARY}" \
  "$@"
