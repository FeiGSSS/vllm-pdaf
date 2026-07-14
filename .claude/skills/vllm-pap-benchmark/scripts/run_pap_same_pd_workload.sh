#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
exec bash "${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh" "$@"
