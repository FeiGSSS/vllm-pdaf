#!/usr/bin/env bash
# Run the traced 7PA1P C30 point and freeze an aligned PA/Projection window
# with capture_aligned_trace.py while the service is still serving.
#
# Usage: bash run_trace.sh <repetition>
set -uo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)"
repetition="${1:?repetition is required}"
count_runs() { find "${EXPERIMENT_DIR}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l; }
before="$(count_runs)"
bash "${EXPERIMENT_DIR}/run.sh" trace-c30 "${repetition}" &
run_pid=$!
run_root=""
while kill -0 "${run_pid}" 2>/dev/null; do
  if (( $(count_runs) > before )); then
    run_root="$(find "${EXPERIMENT_DIR}/runs" -mindepth 1 -maxdepth 1 -type d \
      -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2)"
    break
  fi
  sleep 1
done
if [[ -n "${run_root}" ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/benchmarks/pap/tooling/capture_aligned_trace.py" \
    "${run_root}/projection_trace.pt" --output "${run_root}/aligned_trace.pt" \
    --pa-count 7 --timeout 1800 > "${run_root}/capture_aligned_trace.log" 2>&1 &
fi
wait "${run_pid}"
status=$?
wait
exit "${status}"
