#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PROFILE_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
DEFAULT_DATASET="/data/ssd1/llm-datasets/aiperf-research/agentic-code-balanced-131k/filtered_s128_t5-32_seed42/dataset.jsonl"

REQUEST_RATE="${1:-${AIPERF_REQUEST_RATE:-2}}"
NUM_CONVERSATIONS="${2:-${AIPERF_NUM_CONVERSATIONS:-128}}"
CONCURRENCY="${3:-${AIPERF_CONCURRENCY:-64}}"

if (( $# > 3 )); then
  echo "usage: $0 [request_rate] [num_conversations] [concurrency]" >&2
  exit 2
fi
if [[ ! "${REQUEST_RATE}" =~ ^[0-9]+([.][0-9]+)?$ \
  || ! "${REQUEST_RATE}" =~ [1-9] ]]; then
  echo "request_rate must be a positive number" >&2
  exit 2
fi
if [[ ! "${NUM_CONVERSATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_conversations must be a positive integer" >&2
  exit 2
fi
if [[ ! "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "concurrency must be a positive integer" >&2
  exit 2
fi
if (( CONCURRENCY > NUM_CONVERSATIONS )); then
  echo "concurrency must not exceed num_conversations" >&2
  exit 2
fi

SOURCE_DATASET="${AIPERF_AGENTIC_SOURCE_DATASET:-${DEFAULT_DATASET}}"
[[ -f "${SOURCE_DATASET}" ]] || {
  echo "Agentic Coding dataset does not exist: ${SOURCE_DATASET}" >&2
  exit 1
}
[[ -x "${PROFILE_RUNNER}" ]] || {
  echo "AIPerf profile runner is not executable: ${PROFILE_RUNNER}" >&2
  exit 1
}
command -v jq >/dev/null || {
  echo "jq is required to build the no-delay replay input" >&2
  exit 1
}

RUN_ID="${AIPERF_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${AIPERF_OUTPUT_DIR:-${ROOT_DIR}/aiperf-artifacts/agentic-code/rate${REQUEST_RATE}_conv${NUM_CONVERSATIONS}_c${CONCURRENCY}_${RUN_ID}}"
NO_DELAY_DATASET="${OUTPUT_DIR}/dataset.no_delay.jsonl"
mkdir -p "${OUTPUT_DIR}"

temporary_dataset="$(mktemp "${OUTPUT_DIR}/.dataset.no_delay.XXXXXX")"
trap 'rm -f "${temporary_dataset}"' EXIT
jq -c 'del(.delay)' "${SOURCE_DATASET}" > "${temporary_dataset}"
if jq -e 'select(has("delay"))' "${temporary_dataset}" >/dev/null; then
  echo "failed to remove all delay fields from the replay input" >&2
  exit 1
fi
mv "${temporary_dataset}" "${NO_DELAY_DATASET}"

SOURCE_SHA256="$(sha256sum "${SOURCE_DATASET}" | cut -d' ' -f1)"
INPUT_SHA256="$(sha256sum "${NO_DELAY_DATASET}" | cut -d' ' -f1)"
{
  printf 'SOURCE_DATASET=%q\n' "${SOURCE_DATASET}"
  printf 'SOURCE_SHA256=%q\n' "${SOURCE_SHA256}"
  printf 'INPUT_DATASET=%q\n' "${NO_DELAY_DATASET}"
  printf 'INPUT_SHA256=%q\n' "${INPUT_SHA256}"
  printf 'REQUEST_RATE=%q\n' "${REQUEST_RATE}"
  printf 'NUM_CONVERSATIONS=%q\n' "${NUM_CONVERSATIONS}"
  printf 'CONCURRENCY=%q\n' "${CONCURRENCY}"
  printf 'ARRIVAL_PATTERN=poisson\n'
  printf 'RANDOM_SEED=42\n'
  printf 'INTER_TURN_DELAY=removed\n'
} > "${OUTPUT_DIR}/workload.env"

echo "Agentic Coding AIPerf profile"
echo "  request rate:      ${REQUEST_RATE} turn/s"
echo "  conversations:     ${NUM_CONVERSATIONS}"
echo "  concurrency cap:   ${CONCURRENCY}"
echo "  arrival pattern:   poisson"
echo "  inter-turn delay:  removed"
echo "  output:            ${OUTPUT_DIR}"

export AIPERF_INPUT_FILE="${NO_DELAY_DATASET}"
export AIPERF_CUSTOM_DATASET_TYPE="mooncake-trace"
export AIPERF_OUTPUT_DIR="${OUTPUT_DIR}"
export AIPERF_SESSIONS="${NUM_CONVERSATIONS}"
export AIPERF_CONCURRENCY="${CONCURRENCY}"
export AIPERF_TIMING_MODE="request_rate"
export AIPERF_REQUEST_RATE="${REQUEST_RATE}"
export AIPERF_ARRIVAL_PATTERN="poisson"
export AIPERF_RANDOM_SEED="42"
export AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_REQUEST_TIMEOUT_SECONDS:-21600}"

exec "${PROFILE_RUNNER}"
