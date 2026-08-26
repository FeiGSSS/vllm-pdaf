#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PROFILE_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
source "${ROOT_DIR}/benchmarks/pap/aiperf/agentic_code_baseline.sh"

REQUEST_RATE="${1:-${AIPERF_REQUEST_RATE:-${PAP_AGENTIC_BASELINE_REQUEST_RATE}}}"
NUM_CONVERSATIONS="${2:-${AIPERF_NUM_CONVERSATIONS:-${PAP_AGENTIC_BASELINE_SESSIONS}}}"
CONCURRENCY="${3:-${AIPERF_CONCURRENCY:-${PAP_AGENTIC_BASELINE_CONCURRENCY}}}"
DURATION_SECONDS="${4:-${AIPERF_BENCHMARK_DURATION_SECONDS:-${PAP_AGENTIC_BASELINE_DURATION_SECONDS}}}"
WARMUP_SECONDS="${AIPERF_WARMUP_DURATION_SECONDS:-${PAP_AGENTIC_BASELINE_WARMUP_SECONDS}}"
GRACE_SECONDS="${AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS:-${PAP_AGENTIC_BASELINE_GRACE_SECONDS}}"

if (( $# > 4 )); then
  echo "usage: $0 [request_rate] [num_conversations] [concurrency] [duration_seconds]" >&2
  exit 2
fi
for value in "${REQUEST_RATE}" "${DURATION_SECONDS}" "${WARMUP_SECONDS}" \
  "${GRACE_SECONDS}"; do
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "rate and duration controls must be non-negative numbers" >&2
    exit 2
  }
done
[[ "${REQUEST_RATE}" =~ [1-9] ]] || {
  echo "request_rate must be positive" >&2
  exit 2
}
for value in "${NUM_CONVERSATIONS}" "${CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "conversation and concurrency controls must be positive integers" >&2
    exit 2
  }
done
(( CONCURRENCY <= NUM_CONVERSATIONS )) || {
  echo "concurrency must not exceed num_conversations" >&2
  exit 2
}

DEFAULT_DATASET="${ROOT_DIR}/${PAP_AGENTIC_BASELINE_DATASET_REL}"
SOURCE_DATASET="${AIPERF_AGENTIC_SOURCE_DATASET:-${DEFAULT_DATASET}}"
EXPECTED_SHA256="${AIPERF_AGENTIC_DATASET_SHA256:-${PAP_AGENTIC_BASELINE_DATASET_SHA256}}"
[[ -f "${SOURCE_DATASET}" ]] || {
  echo "Agentic Coding dataset does not exist: ${SOURCE_DATASET}" >&2
  exit 1
}
[[ -x "${PROFILE_RUNNER}" ]] || {
  echo "AIPerf profile runner is not executable: ${PROFILE_RUNNER}" >&2
  exit 1
}
actual_sha256="$(sha256sum "${SOURCE_DATASET}" | cut -d' ' -f1)"
[[ "${actual_sha256}" == "${EXPECTED_SHA256}" ]] || {
  echo "Agentic Coding dataset SHA-256 mismatch: ${actual_sha256}" >&2
  exit 1
}
if jq -e 'select(has("delay"))' "${SOURCE_DATASET}" >/dev/null; then
  echo "steady-state Agentic Coding input must not contain turn delays" >&2
  exit 1
fi

RUN_ID="${AIPERF_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${AIPERF_OUTPUT_DIR:-${ROOT_DIR}/aiperf-artifacts/agentic-code-steady/rate${REQUEST_RATE}_s${NUM_CONVERSATIONS}_c${CONCURRENCY}_${DURATION_SECONDS}s_${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}"
{
  printf 'DATASET=%q\nDATASET_SHA256=%q\n' \
    "${SOURCE_DATASET}" "${actual_sha256}"
  printf 'REQUEST_RATE=%q\nSESSIONS=%q\nCONCURRENCY=%q\n' \
    "${REQUEST_RATE}" "${NUM_CONVERSATIONS}" "${CONCURRENCY}"
  printf 'WARMUP_SECONDS=%q\nDURATION_SECONDS=%q\nGRACE_SECONDS=%q\n' \
    "${WARMUP_SECONDS}" "${DURATION_SECONDS}" "${GRACE_SECONDS}"
  printf 'ARRIVAL_PATTERN=%q\nRANDOM_SEED=%q\nINTER_TURN_DELAY=absent\n' \
    "${PAP_AGENTIC_BASELINE_ARRIVAL_PATTERN}" \
    "${PAP_AGENTIC_BASELINE_RANDOM_SEED}"
} > "${OUTPUT_DIR}/workload.env"

echo "Agentic Coding steady-state AIPerf profile"
echo "  request rate:      ${REQUEST_RATE} turn/s"
echo "  conversations:     ${NUM_CONVERSATIONS}"
echo "  concurrency cap:   ${CONCURRENCY}"
echo "  warmup:            ${WARMUP_SECONDS}s"
echo "  measurement:       ${DURATION_SECONDS}s"
echo "  grace/drain:       ${GRACE_SECONDS}s"
echo "  output:            ${OUTPUT_DIR}"

export AIPERF_INPUT_FILE="${SOURCE_DATASET}"
export AIPERF_CUSTOM_DATASET_TYPE="mooncake-trace"
export AIPERF_OUTPUT_DIR="${OUTPUT_DIR}"
export AIPERF_SESSIONS="${NUM_CONVERSATIONS}"
export AIPERF_CONCURRENCY="${CONCURRENCY}"
export AIPERF_TIMING_MODE="request_rate"
export AIPERF_REQUEST_RATE="${REQUEST_RATE}"
export AIPERF_ARRIVAL_PATTERN="${PAP_AGENTIC_BASELINE_ARRIVAL_PATTERN}"
export AIPERF_RANDOM_SEED="${PAP_AGENTIC_BASELINE_RANDOM_SEED}"
export AIPERF_WARMUP_DURATION_SECONDS="${WARMUP_SECONDS}"
export AIPERF_WARMUP_CONCURRENCY="${CONCURRENCY}"
export AIPERF_WARMUP_REQUEST_RATE="${REQUEST_RATE}"
export AIPERF_WARMUP_ARRIVAL_PATTERN="${PAP_AGENTIC_BASELINE_ARRIVAL_PATTERN}"
export AIPERF_BENCHMARK_DURATION_SECONDS="${DURATION_SECONDS}"
export AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${GRACE_SECONDS}"
export AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_REQUEST_TIMEOUT_SECONDS:-21600}"

exec "${PROFILE_RUNNER}"
