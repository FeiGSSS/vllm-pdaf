#!/usr/bin/env bash

pap_configure_nvshmem() {
  local root_dir="${1:?repository root is required}"
  local prefix="${PAP_NVSHMEM_PREFIX:-${root_dir}/.local/nvshmem-3.2.5-cuda12}"
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
  if [[ "${info}" != *"NVSHMEM v3.2.5"* ]]; then
    echo "Expected NVSHMEM 3.2.5 at ${prefix}" >&2
    return 1
  fi

  PAP_NVSHMEM_PREFIX="${prefix}" \
    "${root_dir}/benchmarks/pap/scripts/build_nvshmem_device_bridge.sh"

  export PAP_NVSHMEM_PREFIX="${prefix}"
  export NVSHMEM_HOME="${prefix}"
  export PATH="${prefix}/bin:${PATH}"
  export LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}"
  export NVSHMEM_BOOTSTRAP="UID"
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-64M}"
  export NVSHMEM_REMOTE_TRANSPORT="none"
  export NVSHMEM_IB_ENABLE_IBGDA=0
  export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1

  echo "NVSHMEM runtime: version=3.2.5 transport=p2p graph=whole-step prefix=${prefix}"
}
