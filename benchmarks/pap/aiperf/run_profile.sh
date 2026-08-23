#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${ROOT_DIR}/.venv-aiperf/bin/aiperf}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
AIPERF_INPUT_FILE="${AIPERF_INPUT_FILE:?set AIPERF_INPUT_FILE}"
AIPERF_CUSTOM_DATASET_TYPE="${AIPERF_CUSTOM_DATASET_TYPE:-multi-turn}"
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
AIPERF_NUM_PROFILE_RUNS="${AIPERF_NUM_PROFILE_RUNS:-1}"
AIPERF_PROFILE_RUN_COOLDOWN_SECONDS="${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS:-0}"
AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS="${AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS:-0}"
AIPERF_PARAMETER_SWEEP_MODE="${AIPERF_PARAMETER_SWEEP_MODE:-repeated}"
AIPERF_WARMUP_SESSIONS="${AIPERF_WARMUP_SESSIONS:-0}"
AIPERF_WARMUP_CONCURRENCY="${AIPERF_WARMUP_CONCURRENCY:-}"
AIPERF_GOODPUT_SLO="${AIPERF_GOODPUT_SLO:-}"

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
if [[ ! "${AIPERF_CONCURRENCY}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "AIPERF_CONCURRENCY must be a positive integer or CSV list" >&2
  exit 2
fi
IFS=, read -r -a concurrency_points <<< "${AIPERF_CONCURRENCY}"
for concurrency in "${concurrency_points[@]}"; do
  if (( concurrency > AIPERF_SESSIONS )); then
    echo "AIPerf concurrency exceeds total sessions: ${concurrency}" >&2
    exit 2
  fi
done
if [[ ! "${AIPERF_NUM_PROFILE_RUNS}" =~ ^[1-9][0-9]*$ ]] \
  || (( AIPERF_NUM_PROFILE_RUNS > 10 )); then
  echo "AIPERF_NUM_PROFILE_RUNS must be between 1 and 10" >&2
  exit 2
fi
for value in "${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}" \
  "${AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS}"; do
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "AIPerf cooldown values must be non-negative numbers" >&2
    exit 2
  fi
done
if [[ ! "${AIPERF_WARMUP_SESSIONS}" =~ ^[0-9]+$ ]]; then
  echo "AIPERF_WARMUP_SESSIONS must be non-negative" >&2
  exit 2
fi
case "${AIPERF_PARAMETER_SWEEP_MODE}" in
  repeated | independent) ;;
  *)
    echo "unsupported AIPERF_PARAMETER_SWEEP_MODE" >&2
    exit 2
    ;;
esac
case "${AIPERF_CUSTOM_DATASET_TYPE}" in
  multi-turn | mooncake-trace) ;;
  *)
    echo "unsupported AIPERF_CUSTOM_DATASET_TYPE" >&2
    exit 2
    ;;
esac
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
AIPERF_VERSION="$("${AIPERF_BIN}" --version)"
printf '%s\n' "${AIPERF_VERSION}" \
  > "${AIPERF_OUTPUT_DIR}/aiperf_version.txt"
{
  printf 'SOURCE=pypi\nPACKAGE=aiperf\n'
  printf 'VERSION=%q\nAIPERF_BIN=%q\n' \
    "${AIPERF_VERSION}" "${AIPERF_BIN}"
} > "${AIPERF_OUTPUT_DIR}/aiperf_install.env"
if git -C "${AIPERF_ROOT}" rev-parse HEAD \
  > "${AIPERF_OUTPUT_DIR}/aiperf_source_checkout_commit.txt" 2>/dev/null; then
  :
fi

args=(
  profile
  --model "${MODEL_PATH}"
  --tokenizer "${MODEL_PATH}"
  --endpoint-type chat
  --input-file "${AIPERF_INPUT_FILE}"
  --custom-dataset-type "${AIPERF_CUSTOM_DATASET_TYPE}"
  --streaming
  --use-server-token-count
  --use-legacy-max-tokens
  --num-sessions "${AIPERF_SESSIONS}"
  --concurrency "${AIPERF_CONCURRENCY}"
  --request-timeout-seconds "${AIPERF_REQUEST_TIMEOUT_SECONDS}"
  --dataset-sampling-strategy sequential
  --random-seed "${AIPERF_RANDOM_SEED}"
  --set-consistent-seed
  --output-artifact-dir "${AIPERF_OUTPUT_DIR}"
  --profile-export-prefix profile
  --export-level "${AIPERF_EXPORT_LEVEL}"
  --slice-duration "${AIPERF_SLICE_DURATION}"
  --connection-reuse-strategy "${AIPERF_CONNECTION_REUSE_STRATEGY}"
)

if [[ "${AIPERF_CUSTOM_DATASET_TYPE}" == "mooncake-trace" ]]; then
  args+=(--no-fixed-schedule)
fi

if (( AIPERF_NUM_PROFILE_RUNS > 1 )); then
  args+=(
    --num-profile-runs "${AIPERF_NUM_PROFILE_RUNS}"
    --profile-run-cooldown-seconds
    "${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}"
  )
fi

if (( ${#concurrency_points[@]} > 1 )); then
  args+=(
    --parameter-sweep-same-seed
    --parameter-sweep-mode "${AIPERF_PARAMETER_SWEEP_MODE}"
    --parameter-sweep-cooldown-seconds
    "${AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS}"
  )
fi

if (( AIPERF_WARMUP_SESSIONS > 0 )); then
  args+=(--num-warmup-sessions "${AIPERF_WARMUP_SESSIONS}")
  if [[ -n "${AIPERF_WARMUP_CONCURRENCY}" ]]; then
    args+=(--warmup-concurrency "${AIPERF_WARMUP_CONCURRENCY}")
  fi
fi

if [[ -n "${AIPERF_GOODPUT_SLO}" ]]; then
  args+=(--goodput "${AIPERF_GOODPUT_SLO}")
fi

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
