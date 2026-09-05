#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
EXPERIMENT_DIR="${PAP_QPS_SCAN_EXPERIMENT_DIR:?set by the experiment run.sh}"
EXPERIMENT_CONFIG="${PAP_QPS_SCAN_EXPERIMENT_CONFIG:?set by the experiment run.sh}"
MATRIX_ROOT="${PAP_QPS_SCAN_RUN_ROOT:-${EXPERIMENT_DIR}/runs/$(date +%Y%m%d_%H%M%S)}"
DYNAMO_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_dynamo_workload.sh"
PAP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
PLOTTER="${ROOT_DIR}/benchmarks/pap/tooling/plot_qps_matrix.py"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
AIPERF_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_aiperf_profile.sh"
ENVIRONMENT_CAPTURE="${ROOT_DIR}/benchmarks/pap/tooling/capture_run_environment.py"
DATASET_ID="${PAP_QPS_SCAN_DATASET_ID:?missing experiment dataset ID}"
DATASET_REL="${PAP_QPS_SCAN_DATASET_REL:?missing experiment dataset path}"
DATASET="${ROOT_DIR}/${DATASET_REL}"
DATASET_SHA256="${PAP_QPS_SCAN_DATASET_SHA256:?missing dataset checksum}"
DATASET_TYPE="${PAP_QPS_SCAN_DATASET_TYPE:?missing AIPerf dataset type}"
MODEL_PATH="${PAP_QPS_SCAN_MODEL_PATH:?missing model path}"
MAX_MODEL_LEN="${PAP_QPS_SCAN_MAX_MODEL_LEN:?missing model length}"
HF_OVERRIDES="${PAP_QPS_SCAN_HF_OVERRIDES:?missing model overrides}"
GPU_MEMORY_UTILIZATION="${PAP_QPS_SCAN_GPU_MEMORY_UTILIZATION:?missing GPU memory utilization}"
BLOCK_SIZE="${PAP_QPS_SCAN_BLOCK_SIZE:?missing KV block size}"
DYNAMO_ROUTER_MODE="${PAP_QPS_SCAN_DYNAMO_ROUTER_MODE:?missing Dynamo router mode}"
PAP_ROUTING_POLICY="${PAP_QPS_SCAN_PAP_ROUTING_POLICY:?missing PAP routing policy}"
if [[ "${PAP_ROUTING_POLICY}" != "dynamo" ]]; then
  echo "ERROR: current PAP supports only Dynamo routing; replay this frozen experiment with its recorded source revision" >&2
  exit 2
fi
TIMING_MODE="${PAP_QPS_SCAN_TIMING_MODE:?missing AIPerf timing mode}"

SUPPORTED_ARCHITECTURES=(
  dp8
  2p6d
  4p4d
  6p2d
  pap_7pa1p_2k
  pap_7pa1p_32k
  pap_6pa2p_2k
  pap_6pa2p_32k
)
CONFIGURED_ARCHITECTURES_CSV="${PAP_QPS_SCAN_ARCHITECTURES:?missing architectures}"
CONFIGURED_QPS_CSV="${PAP_QPS_SCAN_QPS_POINTS:?missing QPS points}"
ARCHITECTURES_CSV="${PAP_QPS_SCAN_ONLY_ARCHITECTURES:-${CONFIGURED_ARCHITECTURES_CSV}}"
QPS_CSV="${PAP_QPS_SCAN_ONLY_QPS:-${CONFIGURED_QPS_CSV}}"
RESUME="${PAP_QPS_SCAN_RESUME:-1}"
VALIDATE_ONLY="${PAP_QPS_SCAN_VALIDATE_ONLY:-0}"
CONTINUE_ON_FAILURE="${PAP_QPS_SCAN_CONTINUE_ON_FAILURE:-1}"
RUN_TIMEOUT_SECONDS="${PAP_QPS_SCAN_RUN_TIMEOUT_SECONDS:?missing run timeout}"

SESSIONS="${PAP_QPS_SCAN_SESSIONS:?missing session count}"
CONCURRENCY="${PAP_QPS_SCAN_CONCURRENCY:?missing concurrency}"
EXPECTED_REQUESTS="${PAP_QPS_SCAN_EXPECTED_REQUESTS:?missing request count}"
ARRIVAL_PATTERN="${PAP_QPS_SCAN_ARRIVAL_PATTERN:?missing arrival pattern}"
REQUEST_TIMEOUT_SECONDS="${PAP_QPS_SCAN_REQUEST_TIMEOUT_SECONDS:?missing request timeout}"
WARMUP_SECONDS="${PAP_QPS_SCAN_WARMUP_SECONDS:?missing warmup duration}"
BENCHMARK_DURATION_SECONDS="${PAP_QPS_SCAN_BENCHMARK_DURATION_SECONDS:?missing benchmark duration}"
GRACE_SECONDS="${PAP_QPS_SCAN_GRACE_SECONDS:?missing grace duration}"
DYNAMO_MAX_NUM_BATCHED_TOKENS="${PAP_QPS_SCAN_DYNAMO_MAX_NUM_BATCHED_TOKENS:?missing Dynamo token budget}"
PAP_2K_MAX_NUM_BATCHED_TOKENS="${PAP_QPS_SCAN_PAP_2K_MAX_NUM_BATCHED_TOKENS:?missing PAP 2K token budget}"
PAP_32K_MAX_NUM_BATCHED_TOKENS="${PAP_QPS_SCAN_PAP_32K_MAX_NUM_BATCHED_TOKENS:?missing PAP 32K token budget}"
DECODE_MAX_NUM_BATCHED_TOKENS="${PAP_QPS_SCAN_DECODE_MAX_NUM_BATCHED_TOKENS:?missing Decode token budget}"
MAX_NUM_SEQS="${PAP_QPS_SCAN_MAX_NUM_SEQS:?missing sequence limit}"

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
        | sed '/^[[:space:]]*$/d'
    )" || die "cannot inspect GPU processes; refusing to treat an unknown state as idle"
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
    MODEL_PATH="${MODEL_PATH}" \
    DYNAMO_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    DYNAMO_HF_OVERRIDES="${HF_OVERRIDES}" \
    DYNAMO_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    DYNAMO_BLOCK_SIZE="${BLOCK_SIZE}" \
    DYNAMO_ROUTER_MODE="${DYNAMO_ROUTER_MODE}" \
    DYNAMO_RUN_ID="qps_matrix_${architecture}_$(qps_tag "${qps}")" \
    DYNAMO_RUN_ROOT="${attempt}" \
    DYNAMO_AIPERF_INPUT_FILE="${DATASET}" \
    DYNAMO_AIPERF_CUSTOM_DATASET_TYPE="${DATASET_TYPE}" \
    DYNAMO_AIPERF_SESSIONS="${SESSIONS}" \
    DYNAMO_AIPERF_CONCURRENCY="${CONCURRENCY}" \
    DYNAMO_AIPERF_TIMING_MODE="${TIMING_MODE}" \
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
    MODEL_PATH="${MODEL_PATH}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    PAP_HF_OVERRIDES="${HF_OVERRIDES}" \
    PAP_PREFILL_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    PROJECTION_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    PAP_BLOCK_SIZE="${BLOCK_SIZE}" \
    PAP_TOPOLOGY="${topology}" \
    PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY}" \
    RUN_ID="qps_matrix_${architecture}_$(qps_tag "${qps}")" \
    RUN_ROOT="${attempt}" \
    PAP_AIPERF_INPUT_FILE="${DATASET}" \
    PAP_AIPERF_CUSTOM_DATASET_TYPE="${DATASET_TYPE}" \
    PAP_AIPERF_EXPECTED_REQUESTS="${EXPECTED_REQUESTS}" \
    PAP_AIPERF_SESSIONS="${SESSIONS}" \
    PAP_AIPERF_CONCURRENCY="${CONCURRENCY}" \
    PAP_AIPERF_TIMING_MODE="${TIMING_MODE}" \
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

render_matrix_manifest() {
  local aiperf_sha256 config_sha256 dataset_sha256
  aiperf_sha256="$(sha256sum "${AIPERF_RUNNER}" | cut -d' ' -f1)"
  config_sha256="$(sha256sum "${EXPERIMENT_CONFIG}" | cut -d' ' -f1)"
  dataset_sha256="$(sha256sum "${DATASET}" | cut -d' ' -f1)"
  [[ "${dataset_sha256}" == "${DATASET_SHA256}" ]] \
    || die "dataset SHA-256 mismatch: ${dataset_sha256}"
  {
    printf 'SCHEMA_VERSION=3\n'
    printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'GIT_DIFF_SHA256=%q\n' \
      "$(git -C "${ROOT_DIR}" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
    printf 'GIT_UNTRACKED_SHA256=%q\n' "$(
      cd "${ROOT_DIR}"
      git ls-files --others --exclude-standard -z \
        | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1
    )"
    printf 'DRIVER_SHA256=%q\n' "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
    printf 'PAP_RUNNER_SHA256=%q\nDYNAMO_RUNNER_SHA256=%q\n' \
      "$(sha256sum "${PAP_RUNNER}" | cut -d' ' -f1)" \
      "$(sha256sum "${DYNAMO_RUNNER}" | cut -d' ' -f1)"
    printf 'EXPERIMENT_CONFIG=%q\nEXPERIMENT_CONFIG_SHA256=%q\n' \
      "${EXPERIMENT_CONFIG}" "${config_sha256}"
    printf 'DATASET_ID=%q\nDATASET_REL=%q\nDATASET=%q\n' \
      "${DATASET_ID}" "${DATASET_REL}" "${DATASET}"
    printf 'DATASET_SHA256=%q\nDATASET_TYPE=%q\n' \
      "${dataset_sha256}" "${DATASET_TYPE}"
    printf 'MODEL_PATH=%q\nMAX_MODEL_LEN=%q\nHF_OVERRIDES=%q\n' \
      "${MODEL_PATH}" "${MAX_MODEL_LEN}" "${HF_OVERRIDES}"
    printf 'GPU_MEMORY_UTILIZATION=%q\nBLOCK_SIZE=%q\n' \
      "${GPU_MEMORY_UTILIZATION}" "${BLOCK_SIZE}"
    printf 'DYNAMO_ROUTER_MODE=%q\nPAP_ROUTING_POLICY=%q\n' \
      "${DYNAMO_ROUTER_MODE}" "${PAP_ROUTING_POLICY}"
    printf 'TIMING_MODE=%q\n' "${TIMING_MODE}"
    printf 'AIPERF_RUNNER=%q\nAIPERF_RUNNER_SHA256=%q\n' \
      "${AIPERF_RUNNER}" "${aiperf_sha256}"
    printf 'ARCHITECTURES=%q\nQPS_POINTS=%q\n' \
      "${configured_architectures[*]}" "${configured_qps[*]}"
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
  }
}

for path in "${DYNAMO_RUNNER}" "${PAP_RUNNER}" "${PLOTTER}" \
  "${PYTHON_BIN}" "${AIPERF_RUNNER}" "${EXPERIMENT_CONFIG}" \
  "${DATASET}" "${ENVIRONMENT_CAPTURE}"; do
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
IFS=, read -r -a configured_architectures \
  <<< "${CONFIGURED_ARCHITECTURES_CSV}"
IFS=, read -r -a configured_qps <<< "${CONFIGURED_QPS_CSV}"
for architecture in "${selected_architectures[@]}"; do
  contains "${architecture}" "${SUPPORTED_ARCHITECTURES[@]}" \
    || die "unsupported architecture: ${architecture}"
  contains "${architecture}" "${configured_architectures[@]}" \
    || die "architecture is not part of this experiment: ${architecture}"
done
for qps in "${selected_qps[@]}"; do
  contains "${qps}" "${configured_qps[@]}" \
    || die "QPS is not part of this experiment: ${qps}"
done

requested_manifest="$(render_matrix_manifest)"
if (( VALIDATE_ONLY == 1 )); then
  echo "QPS matrix configuration is valid; no files written"
  exit 0
fi

hardware_identity="$(
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id \
    --format=csv,noheader
)"
mkdir -p "${MATRIX_ROOT}"
if [[ -e "${MATRIX_ROOT}/matrix.env" ]]; then
  [[ "$(< "${MATRIX_ROOT}/matrix.env")" == "${requested_manifest}" ]] \
    || die "run configuration or source changed: ${MATRIX_ROOT}/matrix.env; create a new run"
  [[ -f "${MATRIX_ROOT}/provenance/COMPLETE" ]] \
    || die "run has no complete environment snapshot; create a new run"
  [[ -f "${MATRIX_ROOT}/hardware_identity.csv" \
    && "$(< "${MATRIX_ROOT}/hardware_identity.csv")" == "${hardware_identity}" ]] \
    || die "GPU hardware or driver changed; create a new run"
else
  (set -o noclobber; printf '%s\n' "${requested_manifest}" > "${MATRIX_ROOT}/matrix.env")
  cp "${EXPERIMENT_CONFIG}" "${MATRIX_ROOT}/experiment.env"
  cp "${BASH_SOURCE[0]}" "${MATRIX_ROOT}/driver.snapshot.sh"
  git -C "${ROOT_DIR}" diff --binary HEAD > "${MATRIX_ROOT}/source.patch"
  printf '%s\n' "${hardware_identity}" > "${MATRIX_ROOT}/hardware_identity.csv"
  wait_for_idle_gpus
  "${PYTHON_BIN}" "${ENVIRONMENT_CAPTURE}" "${MATRIX_ROOT}/provenance" \
    --model "${MODEL_PATH}" --environments .venv .venv-aiperf .venv-dynamo
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
    mkdir -p "${point_root}"
    mkdir "${attempt}"
    cp "${MATRIX_ROOT}/matrix.env" "${attempt}/matrix.env"
    cp "${MATRIX_ROOT}/experiment.env" "${attempt}/experiment.env"
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
      (( CONTINUE_ON_FAILURE == 1 )) || exit 1
    fi
    MPLCONFIGDIR="${MATRIX_ROOT}/.matplotlib" \
      "${PYTHON_BIN}" "${PLOTTER}" --matrix-root "${MATRIX_ROOT}"
  done
done

MPLCONFIGDIR="${MATRIX_ROOT}/.matplotlib" \
  "${PYTHON_BIN}" "${PLOTTER}" --matrix-root "${MATRIX_ROOT}"
echo "MATRIX_ROOT=${MATRIX_ROOT}"
(( failures == 0 )) || exit 1
