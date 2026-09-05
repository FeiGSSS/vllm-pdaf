#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.local/build/pap-dynamo-router"
NATIVE_DIR="${ROOT_DIR}/vllm/pap/gateway/dynamo_native"
INSTALL_DIR="${ROOT_DIR}/.local/pap-dynamo-router"
REVISION=2112d6ba74da72e2715ae69f4b76458b7691380d
SOURCE_SHA=e28fb6508a878969c69fbec2fd075db332f166fe4680d52c25a847c66eca8293
# Matches the compiler used to resolve the committed Cargo.lock.
export RUSTUP_TOOLCHAIN=1.94.1
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
exec 9>"${BUILD_DIR}/build.lock"
flock 9
if [[ ! -f "${BUILD_DIR}/upstream.tar.gz" ]]; then
  curl --fail --location --max-time 180 \
    "https://api.github.com/repos/ai-dynamo/dynamo/tarball/${REVISION}" \
    -o "${BUILD_DIR}/upstream.tar.gz.partial"
  mv "${BUILD_DIR}/upstream.tar.gz.partial" "${BUILD_DIR}/upstream.tar.gz"
fi
test "$(sha256sum "${BUILD_DIR}/upstream.tar.gz" | cut -d ' ' -f1)" = "${SOURCE_SHA}"
# Fresh extraction prevents incremental source edits from entering the build.
STAGE_DIR="$(mktemp -d "${BUILD_DIR}/stage.XXXXXX")"
mkdir "${STAGE_DIR}/source" "${STAGE_DIR}/binding"
tar -xzf "${BUILD_DIR}/upstream.tar.gz" -C "${STAGE_DIR}/source" --strip-components=1
patch --batch --fuzz=0 -d "${STAGE_DIR}/source" -p1 < "${NATIVE_DIR}/explicit-owner.patch"
cp "${NATIVE_DIR}/Cargo.toml" "${NATIVE_DIR}/Cargo.lock" "${NATIVE_DIR}/lib.rs" "${STAGE_DIR}/binding/"
export CARGO_TARGET_DIR="${BUILD_DIR}/target"
cargo build --locked --release -j 24 --manifest-path "${STAGE_DIR}/binding/Cargo.toml"
cp "${CARGO_TARGET_DIR}/release/libpap_dynamo_router.so" "${INSTALL_DIR}/pap_dynamo_router.abi3.so.new"
mv "${INSTALL_DIR}/pap_dynamo_router.abi3.so.new" "${INSTALL_DIR}/pap_dynamo_router.abi3.so"
{
  echo "upstream_revision=${REVISION}"
  echo "upstream_archive_sha256=${SOURCE_SHA}"
  echo "reservation_lifetime=explicit-owner-v1"
  rustc --version
  cargo --version
  sha256sum "${NATIVE_DIR}/Cargo.toml" "${NATIVE_DIR}/Cargo.lock" \
    "${NATIVE_DIR}/lib.rs" "${NATIVE_DIR}/explicit-owner.patch" \
    "${INSTALL_DIR}/pap_dynamo_router.abi3.so"
} > "${INSTALL_DIR}/build.txt"
"${ROOT_DIR}/.venv/bin/python" - "${INSTALL_DIR}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pap_dynamo_router import SelectionService
service = SelectionService(indexer_threads=1)
assert service.reservation_lifetime == "explicit-owner-v1"
service.shutdown()
print("PAP Dynamo explicit-owner runtime installed:", sys.argv[1])
PY
