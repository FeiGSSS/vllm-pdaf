#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 0 ]]; then
  echo "Usage: bash run.sh" >&2
  exit 2
fi
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env -i HOME="${HOME}" USER="${USER:-fei}" PATH="${PATH}" LANG=C.UTF-8 \
  bash -s -- "${EXPERIMENT_DIR}" <<'DIAGNOSTIC'
set -euo pipefail
EXPERIMENT_DIR="$1"
PAP_ROOT="$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"
printf '%s  %s\n' "${DATASET_SHA256}" "${PAP_AIPERF_INPUT_FILE}" \
  | sha256sum --check --status

mkdir -p "${EXPERIMENT_DIR}/runs"
RUN_ROOT="$(mktemp -d "${EXPERIMENT_DIR}/runs/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export RUN_ROOT
export RUN_LOG_DIR="${RUN_ROOT}/service_logs"
export PAP_AIPERF_OUTPUT_DIR="${RUN_ROOT}/aiperf"
cp "${EXPERIMENT_DIR}/experiment.env" "${RUN_ROOT}/experiment.env"
cp "${EXPERIMENT_DIR}/run.sh" "${RUN_ROOT}/"
cp "${EXPERIMENT_DIR}/collect.py" "${RUN_ROOT}/"
"${PAP_ROOT}/.venv/bin/python" "${EXPERIMENT_DIR}/collect.py" \
  "${RUN_ROOT}/prefill_metrics.jsonl" "${RUN_ROOT}/collector.stop" \
  > "${RUN_ROOT}/collector.log" 2>&1 &
collector=$!
trap 'touch "${RUN_ROOT}/collector.stop"; wait "${collector}" || true' EXIT
cd "${PAP_ROOT}"
set +e
bash benchmarks/pap/scripts/run_pap_workload.sh > "${RUN_ROOT}/runner.log" 2>&1
result=$?
set -e
printf '%s\n' "${result}" > "${RUN_ROOT}/exit_code"
echo "PAP diagnostic exit=${result} artifacts=${RUN_ROOT}"
exit "${result}"
DIAGNOSTIC
