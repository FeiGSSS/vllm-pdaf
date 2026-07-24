#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${AIPERF_ROOT}/.venv/bin/aiperf}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
AIPERF_INPUT_FILE="${AIPERF_INPUT_FILE:?set AIPERF_INPUT_FILE}"
AIPERF_TARGET_URL="${AIPERF_TARGET_URL:-http://127.0.0.1:9460}"
AIPERF_TARGET_URLS="${AIPERF_TARGET_URLS:-}"
AIPERF_CONNECTION_REUSE_STRATEGY="${AIPERF_CONNECTION_REUSE_STRATEGY:-pooled}"
AIPERF_OUTPUT_DIR="${AIPERF_OUTPUT_DIR:-${ROOT_DIR}/aiperf-artifacts}"
AIPERF_SESSIONS="${AIPERF_SESSIONS:-12}"
AIPERF_CONCURRENCY="${AIPERF_CONCURRENCY:-12}"
AIPERF_TIMING_MODE="${AIPERF_TIMING_MODE:-concurrency}"
AIPERF_REQUEST_RATE="${AIPERF_REQUEST_RATE:-}"
AIPERF_ARRIVAL_PATTERN="${AIPERF_ARRIVAL_PATTERN:-poisson}"
AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_REQUEST_TIMEOUT_SECONDS:-180}"
AIPERF_EXPORT_LEVEL="${AIPERF_EXPORT_LEVEL:-records}"
AIPERF_SLICE_DURATION="${AIPERF_SLICE_DURATION:-5}"
AIPERF_RANDOM_SEED="${AIPERF_RANDOM_SEED:-42}"

[[ -x "${AIPERF_BIN}" ]] || {
  echo "AIPerf is not installed at ${AIPERF_BIN}" >&2
  exit 1
}
[[ -d "${MODEL_PATH}" ]] || {
  echo "model path does not exist: ${MODEL_PATH}" >&2
  exit 1
}
[[ -f "${AIPERF_INPUT_FILE}" ]] || {
  echo "AIPerf dataset does not exist: ${AIPERF_INPUT_FILE}" >&2
  exit 1
}
case "${AIPERF_TIMING_MODE}" in
  concurrency)
    if [[ -n "${AIPERF_REQUEST_RATE}" ]]; then
      echo "AIPERF_REQUEST_RATE must be empty in concurrency mode" >&2
      exit 2
    fi
    ;;
  request_rate)
    if [[ -z "${AIPERF_REQUEST_RATE}" ]]; then
      echo "AIPERF_REQUEST_RATE is required in request_rate mode" >&2
      exit 2
    fi
    ;;
  *)
    echo "unsupported AIPERF_TIMING_MODE=${AIPERF_TIMING_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${AIPERF_OUTPUT_DIR}"
"${AIPERF_BIN}" --version > "${AIPERF_OUTPUT_DIR}/aiperf_version.txt"
git -C "${AIPERF_ROOT}" rev-parse HEAD \
  > "${AIPERF_OUTPUT_DIR}/aiperf_commit.txt"

args=(
  profile
  --model "${MODEL_PATH}"
  --tokenizer "${MODEL_PATH}"
  --endpoint-type chat
  --input-file "${AIPERF_INPUT_FILE}"
  --custom-dataset-type multi-turn
  --streaming
  --use-server-token-count
  --use-legacy-max-tokens
  --num-sessions "${AIPERF_SESSIONS}"
  --concurrency "${AIPERF_CONCURRENCY}"
  --request-timeout-seconds "${AIPERF_REQUEST_TIMEOUT_SECONDS}"
  --dataset-sampling-strategy sequential
  --random-seed "${AIPERF_RANDOM_SEED}"
  --output-artifact-dir "${AIPERF_OUTPUT_DIR}"
  --profile-export-prefix profile
  --export-level "${AIPERF_EXPORT_LEVEL}"
  --slice-duration "${AIPERF_SLICE_DURATION}"
  --connection-reuse-strategy "${AIPERF_CONNECTION_REUSE_STRATEGY}"
)

if [[ -n "${AIPERF_TARGET_URLS}" ]]; then
  IFS=, read -r -a target_urls <<< "${AIPERF_TARGET_URLS}"
  for target_url in "${target_urls[@]}"; do
    args+=(--url "${target_url}")
  done
else
  args+=(--url "${AIPERF_TARGET_URL}")
fi

if [[ "${AIPERF_TIMING_MODE}" == "request_rate" ]]; then
  args+=(
    --request-rate "${AIPERF_REQUEST_RATE}"
    --arrival-pattern "${AIPERF_ARRIVAL_PATTERN}"
  )
fi

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
exec "${AIPERF_BIN}" "${args[@]}"
