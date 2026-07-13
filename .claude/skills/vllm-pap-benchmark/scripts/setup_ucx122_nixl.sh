#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
RUNTIME_HELPER="${ROOT_DIR}/.claude/skills/vllm-pap-benchmark/scripts/ucx122_runtime_env.sh"
LOCAL_ROOT="${PAP_LOCAL_RUNTIME_ROOT:-${ROOT_DIR}/.local}"
UCX_VERSION="1.22.0"
NIXL_VERSION="1.3.0"
UCX_SOURCE="${PAP_UCX122_SOURCE_DIR:-${LOCAL_ROOT}/src/ucx-${UCX_VERSION}}"
UCX_BUILD="${PAP_UCX122_BUILD_DIR:-${LOCAL_ROOT}/build/ucx-${UCX_VERSION}}"
NIXL_SOURCE="${PAP_NIXL130_SOURCE_DIR:-${LOCAL_ROOT}/src/nixl-${NIXL_VERSION}}"
export PAP_UCX122_ROOT="${PAP_UCX122_ROOT:-${LOCAL_ROOT}/ucx-1.22}"
export PAP_NIXL_UCX122_ROOT="${PAP_NIXL_UCX122_ROOT:-${LOCAL_ROOT}/nixl-ucx122}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

source "${RUNTIME_HELPER}"

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
  local version_output=""
  if [[ -x "${PAP_UCX122_ROOT}/bin/ucx_info" ]]; then
    version_output="$("${PAP_UCX122_ROOT}/bin/ucx_info" -v)"
    if [[ "${version_output}" == *"Library version: ${UCX_VERSION}"* \
      && "${version_output}" == *"--enable-mt"* ]]; then
      echo "UCX ${UCX_VERSION} already installed at ${PAP_UCX122_ROOT}"
      return
    fi
  fi
  download_source \
    "https://github.com/openucx/ucx/archive/refs/tags/v${UCX_VERSION}.tar.gz" \
    "${UCX_SOURCE}" "${LOCAL_ROOT}/downloads/ucx-${UCX_VERSION}.tar.gz"
  if [[ ! -x "${UCX_SOURCE}/configure" ]]; then
    (cd "${UCX_SOURCE}" && ./autogen.sh)
  fi
  mkdir -p "${UCX_BUILD}" "${PAP_UCX122_ROOT}"
  (
    cd "${UCX_BUILD}"
    "${UCX_SOURCE}/configure" \
      --prefix="${PAP_UCX122_ROOT}" \
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
  local plugin
  plugin="${PAP_NIXL_UCX122_ROOT}/src/plugins/ucx/libplugin_UCX.so"
  [[ -f "${plugin}" ]] && {
    echo "NIXL UCX plugin already installed at ${plugin}"
    return
  }
  download_source \
    "https://github.com/ai-dynamo/nixl/archive/refs/tags/v${NIXL_VERSION}.tar.gz" \
    "${NIXL_SOURCE}" "${LOCAL_ROOT}/downloads/nixl-${NIXL_VERSION}.tar.gz"
  if ! "${PYTHON_BIN}" -c 'import pybind11' >/dev/null 2>&1; then
    "${UV_BIN}" pip install --python "${PYTHON_BIN}" pybind11
  fi
  PATH="${ROOT_DIR}/.venv/bin:${PATH}" meson setup \
    "${PAP_NIXL_UCX122_ROOT}" "${NIXL_SOURCE}" \
    --buildtype=release \
    --prefix="${LOCAL_ROOT}/nixl-ucx122-install" \
    -Ducx_path="${PAP_UCX122_ROOT}" \
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
  ninja -C "${PAP_NIXL_UCX122_ROOT}" \
    src/plugins/ucx/libplugin_UCX.so
}

install_runtime() {
  local tool
  for tool in curl tar make meson ninja; do
    require_tool "${tool}"
  done
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "repo Python is missing: ${PYTHON_BIN}" >&2
    exit 1
  }
  install_ucx
  install_nixl_plugin
  configure_ucx122_runtime
  verify_ucx122_runtime
  {
    printf 'UCX_VERSION=%s\n' "${UCX_VERSION}"
    printf 'NIXL_VERSION=%s\n' "${NIXL_VERSION}"
    printf 'UCX_ROOT=%s\n' "${PAP_UCX122_ROOT}"
    printf 'NIXL_BUILD_ROOT=%s\n' "${PAP_NIXL_UCX122_ROOT}"
  } > "${LOCAL_ROOT}/ucx122-nixl-runtime.env"
}

case "${1:-verify}" in
  install)
    install_runtime
    ;;
  verify)
    configure_ucx122_runtime
    verify_ucx122_runtime
    ;;
  *)
    echo "usage: $0 [install|verify]" >&2
    exit 2
    ;;
esac
