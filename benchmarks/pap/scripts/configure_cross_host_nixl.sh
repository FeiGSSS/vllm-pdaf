#!/usr/bin/env bash

pap_configure_cross_host_nixl() {
  local root_dir="${1:?repository root is required}"
  local host_ip="${2:?routable host IP is required}"
  local net_device="${3:?UCX network device is required}"
  local ucx_prefix="${PAP_UCX_PREFIX:-${root_dir}/.local/ucx-1.22}"
  local plugin_dir="${PAP_NIXL_PLUGIN_DIR:-${root_dir}/.local/nixl-ucx122/src/plugins/ucx}"
  local ucx_info="${ucx_prefix}/bin/ucx_info"
  local plugin="${plugin_dir}/libplugin_UCX.so"
  local devices ldd_output library version

  [[ -x "${ucx_info}" ]] || {
    echo "UCX runtime is missing: ${ucx_info}" >&2
    return 1
  }
  [[ -f "${plugin}" ]] || {
    echo "NIXL UCX plugin is missing: ${plugin}" >&2
    return 1
  }
  version="$("${ucx_info}" -v | sed -n 's/^# Library version: //p' | head -n 1)"
  [[ "${version}" == "1.22.0" ]] || {
    echo "Expected UCX 1.22.0, found ${version:-unknown}" >&2
    return 1
  }
  "${ucx_info}" -v 2>&1 | grep -F -- "--enable-mt" >/dev/null || {
    echo "UCX 1.22.0 must be built with --enable-mt" >&2
    return 1
  }
  devices="$("${ucx_info}" -d)"
  [[ "${devices}" == *"Transport: tcp"* ]] || {
    echo "UCX TCP transport is unavailable" >&2
    return 1
  }
  [[ "${devices}" == *"Transport: cuda_copy"* ]] || {
    echo "UCX CUDA copy transport is unavailable" >&2
    return 1
  }
  [[ "$(cat "/sys/class/net/${net_device}/operstate" 2>/dev/null)" == up ]] || {
    echo "UCX network device is not up: ${net_device}" >&2
    return 1
  }

  ldd_output="$(
    LD_LIBRARY_PATH="${ucx_prefix}/lib:${LD_LIBRARY_PATH:-}" ldd "${plugin}"
  )"
  for library in libucp.so.0 libuct.so.0 libucs.so.0; do
    [[ "${ldd_output}" == *"${library} => ${ucx_prefix}/lib/"* ]] || {
      echo "${library} does not resolve to ${ucx_prefix}" >&2
      return 1
    }
  done

  export PAP_NIXL_RUNTIME_MODE=cross_host_ucx122_tcp
  export PAP_NIXL_UCX_VERSION="${version}"
  export PAP_UCX_PREFIX="${ucx_prefix}"
  export PAP_NIXL_PLUGIN_DIR="${plugin_dir}"
  export NIXL_PLUGIN_DIR="${plugin_dir}"
  export UCX_MODULE_DIR="${ucx_prefix}/lib/ucx"
  export LD_LIBRARY_PATH="${ucx_prefix}/lib:${LD_LIBRARY_PATH:-}"
  export UCX_PROTO_EMULATION_ENABLE=n
  export UCX_CUDA_IPC_ENABLE_GET_ZCOPY=n
  export UCX_TLS=tcp,cuda_copy
  export UCX_NET_DEVICES="${net_device}"
  export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"
  export VLLM_NIXL_SIDE_CHANNEL_HOST="${host_ip}"
}
