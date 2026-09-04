#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${ROOT_DIR}/benchmarks/pap/aiperf/agentic_code_baseline.sh"

DYNAMO_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_dynamo_workload.sh"
PAP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
DATASET="${PAP_STEADY_DATASET:-${ROOT_DIR}/${PAP_AGENTIC_BASELINE_DATASET_REL}}"
EXPECTED_SHA256="${PAP_STEADY_DATASET_SHA256:-${PAP_AGENTIC_BASELINE_DATASET_SHA256}}"
SESSIONS="${PAP_STEADY_SESSIONS:-${PAP_AGENTIC_BASELINE_SESSIONS}}"
CONCURRENCY="${PAP_STEADY_CONCURRENCY:-${PAP_AGENTIC_BASELINE_CONCURRENCY}}"
REQUEST_RATE="${PAP_STEADY_REQUEST_RATE:-${PAP_AGENTIC_BASELINE_REQUEST_RATE}}"
ARRIVAL_PATTERN="${PAP_STEADY_ARRIVAL_PATTERN:-${PAP_AGENTIC_BASELINE_ARRIVAL_PATTERN}}"
WARMUP_SECONDS="${PAP_STEADY_WARMUP_SECONDS:-${PAP_AGENTIC_BASELINE_WARMUP_SECONDS}}"
DURATION_SECONDS="${PAP_STEADY_DURATION_SECONDS:-${PAP_AGENTIC_BASELINE_DURATION_SECONDS}}"
GRACE_SECONDS="${PAP_STEADY_GRACE_SECONDS:-${PAP_AGENTIC_BASELINE_GRACE_SECONDS}}"
PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${PAP_STEADY_PAP_PREFILL_MAX_NUM_BATCHED_TOKENS:-2048}"
PD_PREFILL_MAX_NUM_BATCHED_TOKENS="${PAP_STEADY_PD_PREFILL_MAX_NUM_BATCHED_TOKENS:-2048}"
DP_MAX_NUM_BATCHED_TOKENS="${PAP_STEADY_DP_MAX_NUM_BATCHED_TOKENS:-32768}"
LANES_CSV="${PAP_STEADY_LANES:-dynamo_dp8,dynamo_6p2d,dynamo_4p4d,pap_7pa1p}"
RESUME="${PAP_STEADY_RESUME:-1}"
VALIDATE_ONLY="${PAP_STEADY_VALIDATE_ONLY:-0}"
LANE_TIMEOUT_SECONDS="${PAP_STEADY_LANE_TIMEOUT_SECONDS:-5400}"
AIPERF_TIMEOUT_SECONDS="${PAP_STEADY_AIPERF_TIMEOUT_SECONDS:-4500}"
RUN_ID="${PAP_STEADY_RUN_ID:-$(date +%Y%m%d_%H%M%S)_agentic_steady_q${REQUEST_RATE}_c${CONCURRENCY}}"
MATRIX_ROOT="${PAP_STEADY_RUN_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/e2e/_runs/steady/${RUN_ID}}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

profile_is_complete() {
  local profile="$1"
  [[ -s "${profile}" ]] || return 1
  jq -e --argjson duration "${DURATION_SECONDS}" '
    ((.error_summary // []) | length) == 0
    and any(
      .input_config.phases[]?;
      .name == "profiling" and .duration == $duration
    )
  ' "${profile}" >/dev/null
}

wait_for_idle_gpus() {
  local deadline=$((SECONDS + 300))
  local processes
  while true; do
    processes="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d' || true
    )"
    [[ -z "${processes}" ]] && return
    (( SECONDS < deadline )) \
      || die "GPUs stayed occupied by PIDs: ${processes//$'\n'/,}"
    sleep 5
  done
}

run_dynamo() {
  local architecture="$1"
  local lane="dynamo_${architecture}"
  local run_root="${MATRIX_ROOT}/${lane}"
  if (( RESUME == 1 )) && profile_is_complete "${run_root}/aiperf/profile.json"; then
    echo "Skipping completed lane: ${lane}"
    return
  fi
  wait_for_idle_gpus
  echo "=== Running ${lane}: ${run_root} ==="
  timeout --foreground "${LANE_TIMEOUT_SECONDS}" env \
    PAP_ROOT="${ROOT_DIR}" \
    DYNAMO_ARCHITECTURE="${architecture}" \
    DYNAMO_ROUTER_MODE=kv \
    DYNAMO_RUN_ID="${RUN_ID}_${lane}" \
    DYNAMO_RUN_ROOT="${run_root}" \
    DYNAMO_AIPERF_INPUT_FILE="${DATASET}" \
    DYNAMO_AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
    DYNAMO_AIPERF_SESSIONS="${SESSIONS}" \
    DYNAMO_AIPERF_CONCURRENCY="${CONCURRENCY}" \
    DYNAMO_AIPERF_TIMING_MODE=request_rate \
    DYNAMO_AIPERF_REQUEST_RATE="${REQUEST_RATE}" \
    DYNAMO_AIPERF_ARRIVAL_PATTERN="${ARRIVAL_PATTERN}" \
    DYNAMO_AIPERF_EXPECTED_REQUESTS= \
    DYNAMO_AIPERF_REQUEST_TIMEOUT_SECONDS="${AIPERF_TIMEOUT_SECONDS}" \
    DYNAMO_AIPERF_WARMUP_DURATION_SECONDS="${WARMUP_SECONDS}" \
    DYNAMO_AIPERF_BENCHMARK_DURATION_SECONDS="${DURATION_SECONDS}" \
    DYNAMO_AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${GRACE_SECONDS}" \
    DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS="${PD_PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS="${DP_MAX_NUM_BATCHED_TOKENS}" \
    bash "${DYNAMO_RUNNER}" 2>&1 | tee "${MATRIX_ROOT}/${lane}.log"
}

run_pap() {
  local lane="pap_7pa1p"
  local run_root="${MATRIX_ROOT}/${lane}"
  if (( RESUME == 1 )) && profile_is_complete "${run_root}/aiperf/profile.json"; then
    echo "Skipping completed lane: ${lane}"
    return
  fi
  wait_for_idle_gpus
  echo "=== Running ${lane}: ${run_root} ==="
  timeout --foreground "${LANE_TIMEOUT_SECONDS}" env \
    PAP_ROOT="${ROOT_DIR}" \
    PAP_TOPOLOGY=7pa1p \
    PAP_ROUTING_POLICY=conversation_affinity \
    RUN_ID="${RUN_ID}_${lane}" \
    RUN_ROOT="${run_root}" \
    PAP_AIPERF_INPUT_FILE="${DATASET}" \
    PAP_AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
    PAP_AIPERF_VARIABLE_TURNS=1 \
    PAP_AIPERF_EXPECTED_REQUESTS="${PAP_AGENTIC_BASELINE_AVAILABLE_REQUESTS}" \
    PAP_AIPERF_SESSIONS="${SESSIONS}" \
    PAP_AIPERF_CONCURRENCY="${CONCURRENCY}" \
    PAP_AIPERF_TIMING_MODE=request_rate \
    PAP_AIPERF_REQUEST_RATE="${REQUEST_RATE}" \
    PAP_AIPERF_ARRIVAL_PATTERN="${ARRIVAL_PATTERN}" \
    PAP_AIPERF_WARMUP_DURATION_SECONDS="${WARMUP_SECONDS}" \
    PAP_AIPERF_BENCHMARK_DURATION_SECONDS="${DURATION_SECONDS}" \
    PAP_AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${GRACE_SECONDS}" \
    PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    BENCH_TIMEOUT="${AIPERF_TIMEOUT_SECONDS}" \
    bash "${PAP_RUNNER}" 2>&1 | tee "${MATRIX_ROOT}/${lane}.log"
}

write_summary() {
  local output="${MATRIX_ROOT}/summary.tsv"
  printf 'architecture\trequests\tduration_s\treq_per_s\toutput_tok_per_s\tttft_mean_ms\tttft_p99_ms\titl_mean_ms\titl_p99_ms\n' \
    > "${output}"
  local lane profile
  for lane in dynamo_dp8 dynamo_6p2d dynamo_4p4d dynamo_2p6d pap_7pa1p; do
    profile="${MATRIX_ROOT}/${lane}/aiperf/profile.json"
    [[ -s "${profile}" ]] || continue
    jq -r --arg lane "${lane}" '[
      $lane,
      .request_count.avg,
      .benchmark_duration.avg,
      .request_throughput.avg,
      .output_token_throughput.avg,
      .time_to_first_token.avg,
      .time_to_first_token.p99,
      .inter_token_latency.avg,
      .inter_token_latency.p99
    ] | @tsv' "${profile}" >> "${output}"
  done
  echo "Summary: ${output}"
}

for required in "${DYNAMO_RUNNER}" "${PAP_RUNNER}" "${DATASET}"; do
  [[ -e "${required}" ]] || die "missing required path: ${required}"
done
for required_command in jq sha256sum nvidia-smi timeout; do
  command -v "${required_command}" >/dev/null \
    || die "missing required command: ${required_command}"
done
for value in "${SESSIONS}" "${CONCURRENCY}" "${LANE_TIMEOUT_SECONDS}" \
  "${AIPERF_TIMEOUT_SECONDS}" "${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS}" \
  "${PD_PREFILL_MAX_NUM_BATCHED_TOKENS}" "${DP_MAX_NUM_BATCHED_TOKENS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "integer controls must be positive"
done
(( CONCURRENCY <= SESSIONS )) || die "concurrency exceeds session limit"
[[ "${RESUME}" =~ ^[01]$ ]] || die "PAP_STEADY_RESUME must be 0 or 1"
[[ "${VALIDATE_ONLY}" =~ ^[01]$ ]] \
  || die "PAP_STEADY_VALIDATE_ONLY must be 0 or 1"
for value in "${REQUEST_RATE}" "${WARMUP_SECONDS}" "${DURATION_SECONDS}" \
  "${GRACE_SECONDS}"; do
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || die "rate and duration controls must be non-negative numbers"
done

actual_sha256="$(sha256sum "${DATASET}" | cut -d' ' -f1)"
[[ "${actual_sha256}" == "${EXPECTED_SHA256}" ]] \
  || die "dataset SHA-256 mismatch: ${actual_sha256}"
if jq -e 'select(has("delay"))' "${DATASET}" >/dev/null; then
  die "steady-state dataset contains turn delays"
fi

mkdir -p "${MATRIX_ROOT}"
{
  printf 'RUN_ID=%q\nMATRIX_ROOT=%q\n' "${RUN_ID}" "${MATRIX_ROOT}"
  printf 'DATASET=%q\nDATASET_SHA256=%q\n' "${DATASET}" "${actual_sha256}"
  printf 'SESSIONS=%q\nCONCURRENCY=%q\nREQUEST_RATE=%q\n' \
    "${SESSIONS}" "${CONCURRENCY}" "${REQUEST_RATE}"
  printf 'WARMUP_SECONDS=%q\nDURATION_SECONDS=%q\nGRACE_SECONDS=%q\n' \
    "${WARMUP_SECONDS}" "${DURATION_SECONDS}" "${GRACE_SECONDS}"
  printf 'PAP_PREFILL_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${PAP_PREFILL_MAX_NUM_BATCHED_TOKENS}"
  printf 'PD_PREFILL_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${PD_PREFILL_MAX_NUM_BATCHED_TOKENS}"
  printf 'DP_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${DP_MAX_NUM_BATCHED_TOKENS}"
  printf 'ARRIVAL_PATTERN=%q\nLANES=%q\n' \
    "${ARRIVAL_PATTERN}" "${LANES_CSV}"
} > "${MATRIX_ROOT}/matrix.env"

if (( VALIDATE_ONLY == 1 )); then
  echo "Steady-state matrix configuration is valid: ${MATRIX_ROOT}/matrix.env"
  exit 0
fi

IFS=, read -r -a lanes <<< "${LANES_CSV}"
for lane in "${lanes[@]}"; do
  case "${lane}" in
    dynamo_dp8) run_dynamo dp8 ;;
    dynamo_6p2d) run_dynamo 6p2d ;;
    dynamo_4p4d) run_dynamo 4p4d ;;
    dynamo_2p6d) run_dynamo 2p6d ;;
    pap_7pa1p) run_pap ;;
    *) die "unsupported steady-state lane: ${lane}" ;;
  esac
done

write_summary
echo "MATRIX_ROOT=${MATRIX_ROOT}"
