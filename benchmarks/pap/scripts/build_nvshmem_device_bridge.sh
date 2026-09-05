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
BUILD_RECORD="${OUTPUT}.build.txt"

if [[ ! -f "${NVSHMEM_PREFIX}/lib/libnvshmem_device.a" ]]; then
  echo "NVSHMEM device library is missing under ${NVSHMEM_PREFIX}" >&2
  exit 2
fi
if [[ ! -x "${NVCC}" ]]; then
  echo "CUDA 13 nvcc is missing: ${NVCC}" >&2
  exit 2
fi
NVCC_VERSION="$("${NVCC}" --version)"
if [[ "${NVCC_VERSION}" != *"release 13."* ]]; then
  echo "PAP NVSHMEM bridge requires CUDA 13 nvcc: ${NVCC}" >&2
  exit 2
fi

BUILD_ARGS=(
  -std=c++17 -O3 -lineinfo -rdc=true -shared -Xcompiler=-fPIC
  "-Xlinker=--version-script=${VERSION_SCRIPT}"
  "-arch=${PAP_NVSHMEM_GPU_ARCH:-sm_89}"
  "-I${NVSHMEM_PREFIX}/include" "${SOURCE}"
  "-L${NVSHMEM_PREFIX}/lib" -lcuda -lnvshmem_host -lnvshmem_device
)

# Serialize both cache inspection and publication across concurrent launchers.
exec 9> "${OUTPUT}.lock"
flock 9
BUILD_INPUTS="$(
  set -e
  printf 'compiler=%s\n%s\n' "${NVCC}" "${NVCC_VERSION}"
  printf 'argument=%q\n' "${BUILD_ARGS[@]}"
  for variable in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS NVCC_CCBIN \
    CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%q\n' "${variable}" "${!variable:-}"
  done
  HOST_COMPILER="${NVCC_CCBIN:-g++}"
  if [[ -d "${HOST_COMPILER}" ]]; then
    HOST_COMPILER="${HOST_COMPILER}/g++"
  fi
  "${HOST_COMPILER}" --version
  sha256sum "${NVCC}" "$(command -v "${HOST_COMPILER}")" \
    "${BASH_SOURCE[0]}" "${SOURCE}" "${VERSION_SCRIPT}" \
    "${NVSHMEM_PREFIX}/lib/libnvshmem_device.a" \
    "${NVSHMEM_PREFIX}/lib/libnvshmem_host.so"
  find -L "${NVSHMEM_PREFIX}/include" -type f -print0 \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum
)"
if [[ -f "${OUTPUT}" && -f "${BUILD_RECORD}" ]]; then
  EXPECTED_RECORD="$(printf '%s\n' "${BUILD_INPUTS}"; sha256sum "${OUTPUT}")"
  if [[ "$(< "${BUILD_RECORD}")" == "${EXPECTED_RECORD}" ]]; then
    exit 0
  fi
fi

BUILD_DIR="$(mktemp -d "${NVSHMEM_PREFIX}/lib/.pap-bridge-build.XXXXXX")"
trap 'rm -f "${BUILD_DIR}/bridge.so" "${BUILD_DIR}/build.txt"; rmdir "${BUILD_DIR}"' EXIT
PATH="${CUDA_BIN}:${PATH}" "${NVCC}" "${BUILD_ARGS[@]}" \
  -o "${BUILD_DIR}/bridge.so"
# A failed compile leaves the previous usable library untouched. Publishing the
# record last makes interruption detectable as a cache miss on the next build.
mv "${BUILD_DIR}/bridge.so" "${OUTPUT}"
{ printf '%s\n' "${BUILD_INPUTS}"; sha256sum "${OUTPUT}"; } \
  > "${BUILD_DIR}/build.txt"
mv "${BUILD_DIR}/build.txt" "${BUILD_RECORD}"

echo "Built PAP NVSHMEM device bridge: ${OUTPUT}"
