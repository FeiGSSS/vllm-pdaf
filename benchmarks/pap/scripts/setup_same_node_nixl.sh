#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
)"
LOCAL_ROOT="${PAP_LOCAL_RUNTIME_ROOT:-${ROOT_DIR}/.local}"
UCX_VERSION=1.22.0
NIXL_VERSION=1.3.0
UCX_SOURCE="${PAP_UCX_SOURCE_DIR:-${LOCAL_ROOT}/src/ucx-${UCX_VERSION}}"
UCX_BUILD="${PAP_UCX_BUILD_DIR:-${LOCAL_ROOT}/build/ucx-${UCX_VERSION}}"
NIXL_SOURCE="${PAP_NIXL_SOURCE_DIR:-${LOCAL_ROOT}/src/nixl-${NIXL_VERSION}}"
PAP_UCX_PREFIX="${PAP_UCX_PREFIX:-${LOCAL_ROOT}/ucx-1.22}"
NIXL_BUILD_ROOT="${PAP_NIXL_BUILD_ROOT:-${LOCAL_ROOT}/nixl-ucx122}"
PAP_NIXL_PLUGIN_DIR="${PAP_NIXL_PLUGIN_DIR:-${NIXL_BUILD_ROOT}/src/plugins/ucx}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

require_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required build tool is missing: $1" >&2
    exit 1
  }
}

download_source() {
  local url="$1"
  local destination="$2"
  local archive="$3"
  [[ -f "${destination}/meson.build" \
    || -f "${destination}/configure" ]] && return
  mkdir -p "${destination}" "$(dirname "${archive}")"
  curl -L --fail --retry 3 "${url}" -o "${archive}"
  tar -xzf "${archive}" --strip-components=1 -C "${destination}"
}

install_ucx() {
  if [[ -x "${PAP_UCX_PREFIX}/bin/ucx_info" ]] \
    && "${PAP_UCX_PREFIX}/bin/ucx_info" -v 2>&1 \
      | grep -F "Library version: ${UCX_VERSION}" >/dev/null \
    && "${PAP_UCX_PREFIX}/bin/ucx_info" -v 2>&1 \
      | grep -F -- "--enable-mt" >/dev/null; then
    echo "UCX ${UCX_VERSION} already installed at ${PAP_UCX_PREFIX}"
    return
  fi

  download_source \
    "https://github.com/openucx/ucx/archive/refs/tags/v${UCX_VERSION}.tar.gz" \
    "${UCX_SOURCE}" \
    "${LOCAL_ROOT}/downloads/ucx-${UCX_VERSION}.tar.gz"
  if [[ ! -x "${UCX_SOURCE}/configure" ]]; then
    (cd "${UCX_SOURCE}" && ./autogen.sh)
  fi
  mkdir -p "${UCX_BUILD}" "${PAP_UCX_PREFIX}"
  (
    cd "${UCX_BUILD}"
    "${UCX_SOURCE}/configure" \
      --prefix="${PAP_UCX_PREFIX}" \
      --enable-shared \
      --disable-static \
      --enable-cma \
      --enable-devel-headers \
      --enable-mt \
      --with-cuda=/usr \
      --without-verbs \
      --without-rdmacm \
      --without-gdrcopy
    make -j"$(nproc)"
    make install
  )
}

install_nixl_plugin() {
  local plugin="${PAP_NIXL_PLUGIN_DIR}/libplugin_UCX.so"
  if [[ -f "${plugin}" ]]; then
    echo "NIXL UCX plugin already installed at ${plugin}"
    return
  fi

  download_source \
    "https://github.com/ai-dynamo/nixl/archive/refs/tags/v${NIXL_VERSION}.tar.gz" \
    "${NIXL_SOURCE}" \
    "${LOCAL_ROOT}/downloads/nixl-${NIXL_VERSION}.tar.gz"
  if ! "${PYTHON_BIN}" -c 'import pybind11' >/dev/null 2>&1; then
    "${UV_BIN}" pip install --python "${PYTHON_BIN}" pybind11
  fi
  PATH="${ROOT_DIR}/.venv/bin:${PATH}" meson setup \
    "${NIXL_BUILD_ROOT}" "${NIXL_SOURCE}" \
    --buildtype=release \
    --prefix="${LOCAL_ROOT}/nixl-ucx122-install" \
    -Ducx_path="${PAP_UCX_PREFIX}" \
    -Denable_plugins=UCX \
    -Dbuild_tests=false \
    -Dbuild_examples=false \
    -Dbuild_nixl_ep=false \
    -Dinstall_headers=false \
    -Ddisable_gds_backend=true \
    -Ddisable_mooncake_backend=true \
    -Ddisable_infinia_backend=true \
    -Dcudapath_inc=/usr/include \
    -Dcudapath_lib=/usr/lib/x86_64-linux-gnu \
    -Dcudapath_stub=/usr/lib/x86_64-linux-gnu/stubs \
    -Dnixl_cuda_arch_list=89
  ninja -C "${NIXL_BUILD_ROOT}" src/plugins/ucx/libplugin_UCX.so
}

verify_runtime() {
  export PAP_UCX_PREFIX PAP_NIXL_PLUGIN_DIR
  source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
  pap_configure_same_node_nixl "${ROOT_DIR}"
  "${PYTHON_BIN}" -c \
    'from nixl._api import nixl_agent, nixl_agent_config; nixl_agent("pap_ucx122_verify", nixl_agent_config(backends=["UCX"])); print("NIXL_UCX122_AGENT_OK")'
}

case "${1:-verify}" in
  install)
    for tool in curl tar make meson ninja; do
      require_tool "${tool}"
    done
    [[ -x "${PYTHON_BIN}" ]] || {
      echo "repo Python is missing: ${PYTHON_BIN}" >&2
      exit 1
    }
    install_ucx
    install_nixl_plugin
    verify_runtime
    ;;
  verify)
    verify_runtime
    ;;
  *)
    echo "usage: $0 [install|verify]" >&2
    exit 2
    ;;
esac
