#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

OUTPUT_FILE="${1:?usage: $0 OUTPUT_FILE EXPECTED_MODE LOG...}"
EXPECTED_MODE="${2:?usage: $0 OUTPUT_FILE EXPECTED_MODE LOG...}"
shift 2
(( $# > 0 )) || {
  echo "ERROR: at least one vLLM worker log is required" >&2
  exit 2
}

errors=()
capture_count=0
for log in "$@"; do
  if [[ ! -s "${log}" ]]; then
    errors+=("missing or empty log: ${log}")
    continue
  fi
  rg -q 'enforce_eager=False' "${log}" \
    || errors+=("${log}: enforce_eager=False was not recorded")
  if rg -q 'Enforce eager set|enforce_eager=True' "${log}"; then
    errors+=("${log}: eager execution was enabled")
  fi
  rg -q "cudagraph_mode.*${EXPECTED_MODE}" "${log}" \
    || errors+=("${log}: expected CUDA Graph mode ${EXPECTED_MODE} was not recorded")
  if rg -q 'Graph capturing finished' "${log}"; then
    (( capture_count += 1 ))
  else
    errors+=("${log}: CUDA Graph capture did not finish")
  fi
done

mkdir -p "$(dirname "${OUTPUT_FILE}")"
{
  if (( ${#errors[@]} == 0 )); then
    printf 'STATUS=passed\n'
  else
    printf 'STATUS=failed\n'
  fi
  printf 'EXPECTED_MODE=%q\n' "${EXPECTED_MODE}"
  printf 'EXPECTED_LOG_COUNT=%q\n' "$#"
  printf 'CAPTURED_LOG_COUNT=%q\n' "${capture_count}"
  printf 'ERROR_COUNT=%q\n' "${#errors[@]}"
} > "${OUTPUT_FILE}"

if (( ${#errors[@]} > 0 )); then
  printf 'ERROR: CUDA Graph audit failed:\n' >&2
  printf '  - %s\n' "${errors[@]}" >&2
  exit 1
fi

echo "CUDA Graph audit passed for ${capture_count} vLLM workers"
