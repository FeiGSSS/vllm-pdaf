#!/usr/bin/env bash

pap_configure_nvshmem() {
  local root_dir="${1:?repository root is required}"
  local prefix="${PAP_NVSHMEM_PREFIX:-${root_dir}/.local/nvshmem-3.3.24-cuda13}"
  local info_tool="${prefix}/bin/nvshmem-info"
  local info

  if [[ ! -x "${info_tool}" ]]; then
    echo "NVSHMEM runtime is missing: ${info_tool}" >&2
    return 1
  fi
  info="$(
    LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}" \
      "${info_tool}" -a 2>&1
  )"
  if [[ "${info}" != *"NVSHMEM v3.3.24"* \
    || "${info}" != *"CUDA API                     13000"* ]]; then
    echo "Expected CUDA 13 NVSHMEM 3.3.24 at ${prefix}" >&2
    return 1
  fi

  PAP_NVSHMEM_PREFIX="${prefix}" \
    "${root_dir}/benchmarks/pap/scripts/build_nvshmem_device_bridge.sh"

  export PAP_NVSHMEM_PREFIX="${prefix}"
  export NVSHMEM_HOME="${prefix}"
  export PATH="${prefix}/bin:${PATH}"
  export LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}"
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-64M}"
  # PAP ranks intentionally expose different physical GPUs and MPS partitions.
  # NVSHMEM's VMM heap requires a uniform CUDA_VISIBLE_DEVICES topology; the
  # CUDA IPC heap supports this same-host mapping without changing device RMA.
  export NVSHMEM_DISABLE_CUDA_VMM=1
  export NVSHMEM_REMOTE_TRANSPORT="none"
  export NVSHMEM_IB_ENABLE_IBGDA=0
  export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1

  echo "NVSHMEM runtime: version=3.3.24 cuda=13 transport=p2p graph=whole-step prefix=${prefix}"
}
