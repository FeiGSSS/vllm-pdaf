#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPERIMENT_DIR="${PAP_QPS_SCAN_EXPERIMENT_DIR:-${ROOT_DIR}/benchmarks/pap/experiments/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX}"
MATRIX_ROOT="${PAP_QPS_SCAN_RUN_ROOT:-${EXPERIMENT_DIR}/results}"
DYNAMO_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_dynamo_workload.sh"
PAP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
PLOTTER="${ROOT_DIR}/benchmarks/pap/tooling/plot_qps_matrix.py"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/run_profile.sh"
DATASET="${ROOT_DIR}/benchmarks/pap/datasets/agentic-code/s60-t3-half-seed42/dataset.jsonl"
DATASET_SHA256="258b72c85772c9d372f1b63ee0bf6d710f27cb00234027e2c750c82a5fa9563c"

CANONICAL_ARCHITECTURES=(
  dp8
  2p6d
  4p4d
  6p2d
  pap_7pa1p_2k
  pap_7pa1p_32k
  pap_6pa2p_2k
  pap_6pa2p_32k
)
CANONICAL_QPS=(0.6 0.9 1.2 1.5 1.8)
ARCHITECTURES_CSV="${PAP_QPS_SCAN_ONLY_ARCHITECTURES:-dp8,2p6d,4p4d,6p2d,pap_7pa1p_2k,pap_7pa1p_32k,pap_6pa2p_2k,pap_6pa2p_32k}"
QPS_CSV="${PAP_QPS_SCAN_ONLY_QPS:-0.6,0.9,1.2,1.5,1.8}"
RESUME="${PAP_QPS_SCAN_RESUME:-1}"
VALIDATE_ONLY="${PAP_QPS_SCAN_VALIDATE_ONLY:-0}"
CONTINUE_ON_FAILURE="${PAP_QPS_SCAN_CONTINUE_ON_FAILURE:-1}"
RUN_TIMEOUT_SECONDS="${PAP_QPS_SCAN_RUN_TIMEOUT_SECONDS:-3600}"

SESSIONS=60
CONCURRENCY=60
EXPECTED_REQUESTS=180
ARRIVAL_PATTERN=poisson
REQUEST_TIMEOUT_SECONDS=1800
WARMUP_SECONDS=0
BENCHMARK_DURATION_SECONDS=0
GRACE_SECONDS=0
DYNAMO_MAX_NUM_BATCHED_TOKENS=32768
PAP_2K_MAX_NUM_BATCHED_TOKENS=2048
PAP_32K_MAX_NUM_BATCHED_TOKENS=32768
DECODE_MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=256

die() {
  echo "ERROR: $*" >&2
  exit 2
}

contains() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "${value}" == "${needle}" ]] && return 0
  done
  return 1
}

qps_tag() {
  printf '%s' "${1/./p}"
}

profile_is_complete() {
  local profile="$1"
  local qps="$2"
  [[ -s "${profile}" ]] || return 1
  jq -e \
    --arg dataset "${DATASET}" \
    --argjson expected "${EXPECTED_REQUESTS}" \
    --argjson sessions "${SESSIONS}" \
    --argjson concurrency "${CONCURRENCY}" \
    --argjson qps "${qps}" '
      .request_count.avg == $expected
      and ((.error_summary // []) | length) == 0
      and .was_cancelled == false
      and any(
        .input_config.datasets[]?;
        .path == $dataset and .format == "mooncake_trace"
      )
      and any(
        .input_config.phases[]?;
        .name == "profiling"
        and .type == "poisson"
        and .sessions == $sessions
        and .concurrency == $concurrency
        and .rate == $qps
      )
    ' "${profile}" >/dev/null
}

attempt_is_complete() {
  local architecture="$1"
  local qps="$2"
  local attempt="$3"
  profile_is_complete "${attempt}/aiperf/profile.json" "${qps}" \
    || return 1
  grep -qx 'STATUS=passed' "${attempt}/correctness_audit.env" \
    || return 1
  if [[ "${architecture}" == pap_* ]]; then
    grep -qx 'STATUS=passed' "${attempt}/pap_whole_step_graph_audit.env" \
      || return 1
  else
    grep -qx 'STATUS=passed' "${attempt}/vllm_cuda_graph_audit.env" \
      || return 1
  fi
}

completed_attempt() {
  local architecture="$1"
  local qps="$2"
  local point_root="$3"
  local attempt
  shopt -s nullglob
  for attempt in "${point_root}"/attempt_*; do
    if attempt_is_complete "${architecture}" "${qps}" "${attempt}"; then
      printf '%s\n' "${attempt}"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

next_attempt() {
  local point_root="$1"
  local maximum=0
  local attempt suffix
  shopt -s nullglob
  for attempt in "${point_root}"/attempt_*; do
    suffix="${attempt##*_}"
    [[ "${suffix}" =~ ^[0-9]+$ ]] || continue
    (( 10#${suffix} > maximum )) && maximum=$((10#${suffix}))
  done
  shopt -u nullglob
  printf '%s/attempt_%03d\n' "${point_root}" "$((maximum + 1))"
}

write_completed_status() {
  local architecture="$1"
  local attempt="$2"
  local launcher_code=0
  if [[ -f "${attempt}/matrix_status.env" ]]; then
    launcher_code="$(
      sed -n 's/^LAUNCHER_EXIT_CODE=//p' "${attempt}/matrix_status.env" \
        | tail -n 1
    )"
    [[ "${launcher_code}" =~ ^[0-9]+$ ]] || launcher_code=0
  fi
  if [[ "${architecture}" != dp8 \
    && "${architecture}" != pap_* ]] \
    && grep -qx 'STATUS=failed' "${attempt}/kv_transfer_audit.env"; then
    printf 'STATUS=completed_with_kv_transfer_warning\n' \
      > "${attempt}/matrix_status.env"
  else
    printf 'STATUS=passed\n' > "${attempt}/matrix_status.env"
  fi
  printf 'LAUNCHER_EXIT_CODE=%q\n' "${launcher_code}" \
    >> "${attempt}/matrix_status.env"
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
  local qps="$2"
  local attempt="$3"
  timeout --foreground "${RUN_TIMEOUT_SECONDS}" env \
    PAP_ROOT="${ROOT_DIR}" \
    DYNAMO_ARCHITECTURE="${architecture}" \
    DYNAMO_ROUTER_MODE=kv \
    DYNAMO_RUN_ID="qps_matrix_${architecture}_$(qps_tag "${qps}")" \
    DYNAMO_RUN_ROOT="${attempt}" \
    DYNAMO_AIPERF_INPUT_FILE="${DATASET}" \
    DYNAMO_AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
    DYNAMO_AIPERF_SESSIONS="${SESSIONS}" \
    DYNAMO_AIPERF_CONCURRENCY="${CONCURRENCY}" \
    DYNAMO_AIPERF_TIMING_MODE=request_rate \
    DYNAMO_AIPERF_REQUEST_RATE="${qps}" \
    DYNAMO_AIPERF_ARRIVAL_PATTERN="${ARRIVAL_PATTERN}" \
    DYNAMO_AIPERF_EXPECTED_REQUESTS="${EXPECTED_REQUESTS}" \
    DYNAMO_AIPERF_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
    DYNAMO_AIPERF_WARMUP_DURATION_SECONDS="${WARMUP_SECONDS}" \
    DYNAMO_AIPERF_BENCHMARK_DURATION_SECONDS="${BENCHMARK_DURATION_SECONDS}" \
    DYNAMO_AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${GRACE_SECONDS}" \
    DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS="${DYNAMO_MAX_NUM_BATCHED_TOKENS}" \
    DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS="${DYNAMO_MAX_NUM_BATCHED_TOKENS}" \
    DYNAMO_DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS}" \
    DYNAMO_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    bash "${DYNAMO_RUNNER}"
}

run_pap() {
  local architecture="$1"
  local qps="$2"
  local attempt="$3"
  local topology token_budget
  case "${architecture}" in
    pap_7pa1p_2k)
      topology=7pa1p
      token_budget="${PAP_2K_MAX_NUM_BATCHED_TOKENS}"
      ;;
    pap_7pa1p_32k)
      topology=7pa1p
      token_budget="${PAP_32K_MAX_NUM_BATCHED_TOKENS}"
      ;;
    pap_6pa2p_2k)
      topology=6pa2p
      token_budget="${PAP_2K_MAX_NUM_BATCHED_TOKENS}"
      ;;
    pap_6pa2p_32k)
      topology=6pa2p
      token_budget="${PAP_32K_MAX_NUM_BATCHED_TOKENS}"
      ;;
    *) die "unsupported PAP architecture: ${architecture}" ;;
  esac
  timeout --foreground "${RUN_TIMEOUT_SECONDS}" env \
    PAP_ROOT="${ROOT_DIR}" \
    PAP_TOPOLOGY="${topology}" \
    PAP_ROUTING_POLICY=conversation_affinity \
    RUN_ID="qps_matrix_${architecture}_$(qps_tag "${qps}")" \
    RUN_ROOT="${attempt}" \
    PAP_AIPERF_INPUT_FILE="${DATASET}" \
    PAP_AIPERF_CUSTOM_DATASET_TYPE=mooncake-trace \
    PAP_AIPERF_VARIABLE_TURNS=1 \
    PAP_AIPERF_EXPECTED_REQUESTS="${EXPECTED_REQUESTS}" \
    PAP_AIPERF_SESSIONS="${SESSIONS}" \
    PAP_AIPERF_CONCURRENCY="${CONCURRENCY}" \
    PAP_AIPERF_TIMING_MODE=request_rate \
    PAP_AIPERF_REQUEST_RATE="${qps}" \
    PAP_AIPERF_ARRIVAL_PATTERN="${ARRIVAL_PATTERN}" \
    PAP_AIPERF_WARMUP_DURATION_SECONDS="${WARMUP_SECONDS}" \
    PAP_AIPERF_BENCHMARK_DURATION_SECONDS="${BENCHMARK_DURATION_SECONDS}" \
    PAP_AIPERF_BENCHMARK_GRACE_PERIOD_SECONDS="${GRACE_SECONDS}" \
    PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${token_budget}" \
    PAP_PREFILL_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_PROJECTION_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    BENCH_TIMEOUT="${REQUEST_TIMEOUT_SECONDS}" \
    bash "${PAP_RUNNER}"
}

write_matrix_manifest() {
  local aiperf_sha256 dataset_sha256
  aiperf_sha256="$(sha256sum "${AIPERF_RUNNER}" | cut -d' ' -f1)"
  dataset_sha256="$(sha256sum "${DATASET}" | cut -d' ' -f1)"
  [[ "${dataset_sha256}" == "${DATASET_SHA256}" ]] \
    || die "dataset SHA-256 mismatch: ${dataset_sha256}"
  {
    printf 'SCHEMA_VERSION=1\n'
    printf 'DATASET=%q\nDATASET_SHA256=%q\n' \
      "${DATASET}" "${dataset_sha256}"
    printf 'AIPERF_RUNNER=%q\nAIPERF_RUNNER_SHA256=%q\n' \
      "${AIPERF_RUNNER}" "${aiperf_sha256}"
    printf 'ARCHITECTURES=%q\nQPS_POINTS=%q\n' \
      "${CANONICAL_ARCHITECTURES[*]}" "${CANONICAL_QPS[*]}"
    printf 'SESSIONS=%q\nCONCURRENCY=%q\nEXPECTED_REQUESTS=%q\n' \
      "${SESSIONS}" "${CONCURRENCY}" "${EXPECTED_REQUESTS}"
    printf 'ARRIVAL_PATTERN=%q\nREQUEST_TIMEOUT_SECONDS=%q\n' \
      "${ARRIVAL_PATTERN}" "${REQUEST_TIMEOUT_SECONDS}"
    printf 'WARMUP_SECONDS=%q\nBENCHMARK_DURATION_SECONDS=%q\n' \
      "${WARMUP_SECONDS}" "${BENCHMARK_DURATION_SECONDS}"
    printf 'GRACE_SECONDS=%q\nDYNAMO_MAX_NUM_BATCHED_TOKENS=%q\n' \
      "${GRACE_SECONDS}" "${DYNAMO_MAX_NUM_BATCHED_TOKENS}"
    printf 'PAP_2K_MAX_NUM_BATCHED_TOKENS=%q\n' \
      "${PAP_2K_MAX_NUM_BATCHED_TOKENS}"
    printf 'PAP_32K_MAX_NUM_BATCHED_TOKENS=%q\n' \
      "${PAP_32K_MAX_NUM_BATCHED_TOKENS}"
    printf 'DECODE_MAX_NUM_BATCHED_TOKENS=%q\nMAX_NUM_SEQS=%q\n' \
      "${DECODE_MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}"
  } > "${MATRIX_ROOT}/matrix.env"
}

for path in "${DYNAMO_RUNNER}" "${PAP_RUNNER}" "${PLOTTER}" \
  "${PYTHON_BIN}" "${AIPERF_RUNNER}" "${DATASET}"; do
  [[ -e "${path}" ]] || die "missing required path: ${path}"
done
for command in jq nvidia-smi sha256sum timeout; do
  command -v "${command}" >/dev/null || die "missing command: ${command}"
done
for value in "${RESUME}" "${VALIDATE_ONLY}" "${CONTINUE_ON_FAILURE}"; do
  [[ "${value}" =~ ^[01]$ ]] || die "boolean controls must be 0 or 1"
done

IFS=, read -r -a selected_architectures <<< "${ARCHITECTURES_CSV}"
IFS=, read -r -a selected_qps <<< "${QPS_CSV}"
for architecture in "${selected_architectures[@]}"; do
  contains "${architecture}" "${CANONICAL_ARCHITECTURES[@]}" \
    || die "unsupported architecture: ${architecture}"
done
for qps in "${selected_qps[@]}"; do
  contains "${qps}" "${CANONICAL_QPS[@]}" \
    || die "QPS must be one of: ${CANONICAL_QPS[*]}"
done

mkdir -p "${MATRIX_ROOT}"
write_matrix_manifest
if (( VALIDATE_ONLY == 1 )); then
  echo "QPS matrix configuration is valid: ${MATRIX_ROOT}/matrix.env"
  exit 0
fi

failures=0
for architecture in "${selected_architectures[@]}"; do
  for qps in "${selected_qps[@]}"; do
    tag="$(qps_tag "${qps}")"
    point_root="${MATRIX_ROOT}/${architecture}/qps_${tag}"
    if (( RESUME == 1 )) \
      && complete="$(completed_attempt \
        "${architecture}" "${qps}" "${point_root}")"; then
      write_completed_status "${architecture}" "${complete}"
      echo "Skipping completed ${architecture} QPS=${qps}: ${complete}"
      continue
    fi
    attempt="$(next_attempt "${point_root}")"
    mkdir -p "${attempt}"
    wait_for_idle_gpus
    echo "=== ${architecture} QPS=${qps}: ${attempt} ==="
    set +e
    if [[ "${architecture}" == pap_* ]]; then
      run_pap "${architecture}" "${qps}" "${attempt}" \
        2>&1 | tee "${attempt}/launcher.log"
    else
      run_dynamo "${architecture}" "${qps}" "${attempt}" \
        2>&1 | tee "${attempt}/launcher.log"
    fi
    launcher_code="${PIPESTATUS[0]}"
    set -e
    wait_for_idle_gpus
    if attempt_is_complete "${architecture}" "${qps}" "${attempt}"; then
      if (( launcher_code == 0 )); then
        printf 'STATUS=passed\nLAUNCHER_EXIT_CODE=0\n' \
          > "${attempt}/matrix_status.env"
      elif [[ "${architecture}" != dp8 \
        && "${architecture}" != pap_* ]] \
        && grep -qx 'STATUS=failed' \
          "${attempt}/kv_transfer_audit.env"; then
        printf 'STATUS=completed_with_kv_transfer_warning\n' \
          > "${attempt}/matrix_status.env"
        printf 'LAUNCHER_EXIT_CODE=%q\n' "${launcher_code}" \
          >> "${attempt}/matrix_status.env"
      else
        printf 'STATUS=failed\nLAUNCHER_EXIT_CODE=%q\n' \
          "${launcher_code}" > "${attempt}/matrix_status.env"
        failures=$((failures + 1))
        (( CONTINUE_ON_FAILURE == 1 )) || exit "${launcher_code}"
      fi
    else
      printf 'STATUS=failed\nLAUNCHER_EXIT_CODE=%q\n' "${launcher_code}" \
        > "${attempt}/matrix_status.env"
      failures=$((failures + 1))
      (( CONTINUE_ON_FAILURE == 1 )) || exit "${launcher_code}"
    fi
    MPLCONFIGDIR="${MATRIX_ROOT}/.matplotlib" \
      "${PYTHON_BIN}" "${PLOTTER}" --matrix-root "${MATRIX_ROOT}"
  done
done

MPLCONFIGDIR="${MATRIX_ROOT}/.matplotlib" \
  "${PYTHON_BIN}" "${PLOTTER}" --matrix-root "${MATRIX_ROOT}"
echo "MATRIX_ROOT=${MATRIX_ROOT}"
(( failures == 0 )) || exit 1
