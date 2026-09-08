#!/usr/bin/env bash

pap_configure_nvshmem() {
  local root_dir="${1:?repository root is required}"
  local mode="${2:-${PAP_NVSHMEM_MODE:-local}}"
  local prefix="${PAP_NVSHMEM_PREFIX:-${root_dir}/.local/nvshmem-3.3.24-cuda13}"
  local info_tool="${prefix}/bin/nvshmem-info"
  local info hca registry_dwords

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
  export NVSHMEM_BOOTSTRAP=UID
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-64M}"

  case "${mode}" in
    local)
      export PAP_NVSHMEM_MODE=local
      unset PAP_NVSHMEM_UID_HOST PAP_NVSHMEM_UID_PORT
      unset PAP_NVSHMEM_UID_BIND_HOST
      export NVSHMEM_DISABLE_CUDA_VMM=1
      export NVSHMEM_REMOTE_TRANSPORT=none
      export NVSHMEM_IB_ENABLE_IBGDA=0
      export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
      ;;
    ibgda)
      [[ "${info}" == *"NVSHMEM_IBGDA_SUPPORT=ON"* ]] || {
        echo "NVSHMEM runtime was not built with IBGDA support" >&2
        return 1
      }
      [[ -r /proc/driver/nvidia/params ]] || {
        echo "Cannot verify NVIDIA PeerMappingOverride" >&2
        return 1
      }
      registry_dwords="$(
        sed -n 's/^RegistryDwords: //p' /proc/driver/nvidia/params
      )"
      [[ "${registry_dwords}" == *"PeerMappingOverride=1"* ]] || {
        echo "IBGDA requires NVIDIA PeerMappingOverride=1" >&2
        return 1
      }
      hca="${PAP_NVSHMEM_HCA:-mlx5_0:1}"
      [[ -d "/sys/class/infiniband/${hca%%:*}" ]] || {
        echo "IBGDA HCA is missing: ${hca}" >&2
        return 1
      }
      export PAP_NVSHMEM_MODE=ibgda
      unset PAP_NVSHMEM_UID_FILE
      unset NVSHMEM_DISABLE_CUDA_VMM
      export NVSHMEM_REMOTE_TRANSPORT=ibrc
      export NVSHMEM_IB_ENABLE_IBGDA=1
      export NVSHMEM_IBGDA_NIC_HANDLER=gpu
      export NVSHMEM_HCA_LIST="${hca}"
      unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY
      ;;
    *)
      echo "PAP NVSHMEM mode must be local or ibgda, got ${mode}" >&2
      return 1
      ;;
  esac

  echo "NVSHMEM runtime: version=3.3.24 cuda=13 mode=${mode} graph=whole-step prefix=${prefix}"
}
