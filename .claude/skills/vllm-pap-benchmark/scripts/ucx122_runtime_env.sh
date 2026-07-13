#!/usr/bin/env bash

_ucx122_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd
}

configure_ucx122_runtime() {
  local root_dir
  root_dir="$(_ucx122_repo_root)"
  export PAP_UCX122_ROOT="${PAP_UCX122_ROOT:-${root_dir}/.local/ucx-1.22}"
  export PAP_NIXL_UCX122_ROOT="${PAP_NIXL_UCX122_ROOT:-${root_dir}/.local/nixl-ucx122}"
  export UCX_TLS=cuda_ipc,cuda_copy,tcp
  export UCX_PROTO_EMULATION_ENABLE=n
  export UCX_MODULE_DIR="${PAP_UCX122_ROOT}/lib/ucx"
  export NIXL_PLUGIN_DIR="${PAP_NIXL_UCX122_ROOT}/src/plugins/ucx"
  export LD_LIBRARY_PATH="${PAP_UCX122_ROOT}/lib:${PAP_UCX122_ROOT}/lib/ucx${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

verify_ucx122_runtime() {
  local root_dir ucx_info plugin version_output ldd_output library
  root_dir="$(_ucx122_repo_root)"
  ucx_info="${PAP_UCX122_ROOT}/bin/ucx_info"
  plugin="${NIXL_PLUGIN_DIR}/libplugin_UCX.so"

  [[ -x "${ucx_info}" ]] || {
    echo "UCX 1.22 is not installed: ${ucx_info}" >&2
    return 1
  }
  [[ -f "${plugin}" ]] || {
    echo "NIXL UCX 1.22 plugin is not installed: ${plugin}" >&2
    return 1
  }
  version_output="$("${ucx_info}" -v)"
  [[ "${version_output}" == *"Library version: 1.22.0"* ]] || {
    echo "unexpected UCX version" >&2
    echo "${version_output}" >&2
    return 1
  }
  [[ "${version_output}" == *"--enable-mt"* ]] || {
    echo "UCX 1.22 was built without multi-threading" >&2
    return 1
  }
  [[ "${UCX_PROTO_EMULATION_ENABLE:-}" == "n" ]] || {
    echo "UCX protocol emulation must be disabled" >&2
    return 1
  }

  ldd_output="$(ldd "${plugin}")"
  for library in libucp.so.0 libuct.so.0 libucs.so.0; do
    if ! grep -F "${library} => ${PAP_UCX122_ROOT}/lib/" \
      <<< "${ldd_output}" >/dev/null; then
      echo "${library} does not resolve to ${PAP_UCX122_ROOT}" >&2
      echo "${ldd_output}" >&2
      return 1
    fi
  done

  "${root_dir}/.venv/bin/python" -c \
    'from nixl._api import nixl_agent, nixl_agent_config; nixl_agent("ucx122_verify", nixl_agent_config(backends=["UCX"])); print("NIXL_UCX122_AGENT_OK")'
}
