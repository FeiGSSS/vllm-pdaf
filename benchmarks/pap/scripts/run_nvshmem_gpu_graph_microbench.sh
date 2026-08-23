#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NVSHMEM_PREFIX="${PAP_NVSHMEM_PREFIX:-${ROOT_DIR}/.local/nvshmem-3.2.5-cuda12}"
SOURCE="${ROOT_DIR}/benchmarks/pap/microbench/nvshmem_gpu_graph.cu"
BINARY="${TMPDIR:-/tmp}/pap_nvshmem_gpu_graph"
BRIDGE_SOURCE="${ROOT_DIR}/vllm/pap/transport/nvshmem/device_bridge.cu"
BRIDGE_LIBRARY="${NVSHMEM_PREFIX}/lib/libpap_nvshmem_device.so"

if [[ ! -x "${NVSHMEM_PREFIX}/bin/nvshmrun.hydra" ]]; then
  echo "NVSHMEM launcher is missing under ${NVSHMEM_PREFIX}" >&2
  exit 2
fi

PAP_NVSHMEM_PREFIX="${NVSHMEM_PREFIX}" \
  "${ROOT_DIR}/benchmarks/pap/scripts/build_nvshmem_device_bridge.sh"

if [[ ! -x "${BINARY}" \
  || "${SOURCE}" -nt "${BINARY}" \
  || "${BRIDGE_SOURCE}" -nt "${BINARY}" \
  || "${BRIDGE_LIBRARY}" -nt "${BINARY}" ]]; then
  nvcc \
    -std=c++17 \
    -O3 \
    -lineinfo \
    -rdc=true \
    -arch="${PAP_NVSHMEM_GPU_ARCH:-sm_89}" \
    -I"${NVSHMEM_PREFIX}/include" \
    "${SOURCE}" \
    -L"${NVSHMEM_PREFIX}/lib" \
    -lpap_nvshmem_device \
    -lnvshmem_host \
    -lnvshmem_device \
    -o "${BINARY}"
fi

export LD_LIBRARY_PATH="${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_BOOTSTRAP=PMI
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-16M}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec "${NVSHMEM_PREFIX}/bin/nvshmrun.hydra" \
  -launcher fork \
  -np 2 \
  "${BINARY}" \
  "$@"
