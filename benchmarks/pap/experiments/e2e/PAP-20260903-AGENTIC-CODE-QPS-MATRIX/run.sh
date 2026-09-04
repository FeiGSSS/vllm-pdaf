#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_CONFIG="${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${EXPERIMENT_CONFIG}"
export PAP_QPS_SCAN_EXPERIMENT_DIR="${EXPERIMENT_DIR}"
export PAP_QPS_SCAN_EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}"
exec bash "${ROOT_DIR}/benchmarks/pap/scripts/run_agentic_code_qps_matrix.sh"
