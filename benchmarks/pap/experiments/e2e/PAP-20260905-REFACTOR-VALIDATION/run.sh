#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env -i HOME="${HOME}" USER="${USER:-fei}" PATH="${PATH}" LANG=C.UTF-8 \
  bash "${EXPERIMENT_DIR}/driver.sh" "$@"
