#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
PAT_SOURCE="${PAP_PAT_SOURCE:-${PROJECT_ROOT}/third_party/pat}"
CUTLASS_ROOT="${CUTLASS_ROOT:-$(
  "${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_path("purelib") + "/flashinfer/data/cutlass")'
)}"
PAT_PATCH="${SCRIPT_DIR}/pat-sm89-pap.patch"
PAT_COMMIT=b61e589cc8775930931157ff3bb107ba28bafd77
CUDA_ARCH="${PAP_PAT_CUDA_ARCH:-8.9}"
MAX_JOBS="${MAX_JOBS:-32}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

cleanup_vendored_build() {
  if [[ "${PAT_SOURCE}" == "${PROJECT_ROOT}/third_party/pat" ]]; then
    rm -rf \
      "${PROJECT_ROOT}/third_party/pat/build" \
      "${PROJECT_ROOT}/third_party/pat/dist" \
      "${PROJECT_ROOT}/third_party/pat/prefix_attn.egg-info"
  fi
}
trap cleanup_vendored_build EXIT

[[ -x "${PYTHON_BIN}" ]] || die "missing project Python: ${PYTHON_BIN}"
[[ -d "${PAT_SOURCE}" ]] || die "missing PAT source: ${PAT_SOURCE}"
[[ -d "${CUTLASS_ROOT}/include" ]] || die "missing CUTLASS: ${CUTLASS_ROOT}"
[[ -f "${PAT_PATCH}" ]] || die "missing PAT patch: ${PAT_PATCH}"

if [[ -d "${PAT_SOURCE}/.git" ]]; then
  actual_commit="$(git -C "${PAT_SOURCE}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${PAT_COMMIT}" ]] \
    || die "PAT commit is ${actual_commit}; expected ${PAT_COMMIT}"
  if git -C "${PAT_SOURCE}" apply --reverse --check "${PAT_PATCH}" \
    >/dev/null 2>&1; then
    echo "PAT SM89/PAP patch is already applied"
  elif git -C "${PAT_SOURCE}" apply --check "${PAT_PATCH}"; then
    git -C "${PAT_SOURCE}" apply "${PAT_PATCH}"
  else
    die "PAT checkout does not accept the required patch"
  fi
else
  vendored_commit="$(tr -d '[:space:]' < "${PAT_SOURCE}/UPSTREAM_COMMIT")"
  [[ "${vendored_commit}" == "${PAT_COMMIT}" ]] \
    || die "vendored PAT commit is ${vendored_commit}; expected ${PAT_COMMIT}"
fi

if [[ -n "${PAP_PAT_CUDA_HOME:-}" ]]; then
  CUDA_HOME="${PAP_PAT_CUDA_HOME}"
else
  CUDA_HOME="$(
    "${PYTHON_BIN}" -c \
      'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")'
  )"
fi
[[ -x "${CUDA_HOME}/bin/nvcc" ]] \
  || die "missing CUDA compiler under CUDA_HOME: ${CUDA_HOME}"

env \
  CUDA_HOME="${CUDA_HOME}" \
  TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
  CUTLASS_ROOT="${CUTLASS_ROOT}" \
  MAX_JOBS="${MAX_JOBS}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  uv pip install \
    --python "${PYTHON_BIN}" \
    --no-deps \
    --no-build-isolation \
    --reinstall \
    "${PAT_SOURCE}"

DISABLE_STREAM=1 "${PYTHON_BIN}" -c \
  'from prefix_attn import PrefixTreeCPP, prefix_attn_with_kvcache; print("PAT import passed")'
