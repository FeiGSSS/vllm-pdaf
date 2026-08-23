#!/usr/bin/env bash

pap_configure_same_node_nixl() {
  local root_dir="${1:?repository root is required}"
  local ucx_prefix="${PAP_UCX_PREFIX:-${root_dir}/.local/ucx-1.22}"
  local plugin_dir="${PAP_NIXL_PLUGIN_DIR:-${root_dir}/.local/nixl-ucx122/src/plugins/ucx}"
  local ucx_info="${ucx_prefix}/bin/ucx_info"
  local plugin="${plugin_dir}/libplugin_UCX.so"
  local ldd_output library version

  if [[ ! -x "${ucx_info}" ]]; then
    echo "UCX 1.22 runtime is missing: ${ucx_info}" >&2
    return 1
  fi
  if [[ ! -f "${plugin}" ]]; then
    echo "NIXL UCX 1.22 plugin is missing: ${plugin}" >&2
    return 1
  fi

  version="$("${ucx_info}" -v | sed -n 's/^# Library version: //p' | head -n 1)"
  if [[ "${version}" != "1.22.0" ]]; then
    echo "Expected UCX 1.22.0, found ${version:-unknown} at ${ucx_prefix}" >&2
    return 1
  fi
  if ! "${ucx_info}" -v 2>&1 | grep -F -- "--enable-mt" >/dev/null; then
    echo "UCX 1.22.0 must be built with --enable-mt" >&2
    return 1
  fi

  ldd_output="$(
    LD_LIBRARY_PATH="${ucx_prefix}/lib:${LD_LIBRARY_PATH:-}" ldd "${plugin}"
  )"
  for library in libucp.so.0 libuct.so.0 libucs.so.0; do
    if [[ "${ldd_output}" != *"${library} => ${ucx_prefix}/lib/"* ]]; then
      echo "${library} does not resolve to ${ucx_prefix}" >&2
      return 1
    fi
  done

  if [[ "${UCX_PROTO_EMULATION_ENABLE:-n}" != "n" ]]; then
    echo "Same-node NIXL benchmarks require UCX_PROTO_EMULATION_ENABLE=n" >&2
    return 1
  fi
  if [[ "${UCX_CUDA_IPC_ENABLE_GET_ZCOPY:-y}" != "y" ]]; then
    echo "Same-node NIXL benchmarks require CUDA IPC GET zero-copy" >&2
    return 1
  fi

  export PAP_NIXL_RUNTIME_MODE="same_node_ucx122_strict"
  export PAP_NIXL_UCX_VERSION="${version}"
  export PAP_UCX_PREFIX="${ucx_prefix}"
  export PAP_NIXL_PLUGIN_DIR="${plugin_dir}"
  export NIXL_PLUGIN_DIR="${plugin_dir}"
  export UCX_MODULE_DIR="${ucx_prefix}/lib/ucx"
  export LD_LIBRARY_PATH="${ucx_prefix}/lib:${LD_LIBRARY_PATH:-}"
  export UCX_PROTO_EMULATION_ENABLE=n
  export UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y
  export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
  export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
  export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"

  echo "NIXL runtime: UCX ${version}, plugin=${plugin}, emulation=disabled"
}
