#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)"
export PAP_ROOT="${ROOT_DIR}"
source "${EXPERIMENT_DIR}/experiment.env"
SELECTED_CASES=("$@")
TRACE_CAPTURE_PID=""
cleanup_trace_capture() {
  if [[ -n "${TRACE_CAPTURE_PID}" ]]; then
    if jobs -pr | grep -Fxq "${TRACE_CAPTURE_PID}"; then
      kill -TERM "${TRACE_CAPTURE_PID}" 2>/dev/null || true
    fi
    wait "${TRACE_CAPTURE_PID}" 2>/dev/null || true
  fi
}
trap cleanup_trace_capture EXIT
if (( ${#SELECTED_CASES[@]} == 0 )); then
  mapfile -t SELECTED_CASES < <(awk '{print $1}' "${EXPERIMENT_DIR}/workloads.tsv")
fi
declare -A SEEN_CASES=()
for selected_case in "${SELECTED_CASES[@]}"; do
  if ! awk '{print $1}' "${EXPERIMENT_DIR}/workloads.tsv" | grep -Fxq "${selected_case}"; then
    echo "Unknown validation case: ${selected_case}" >&2
    exit 2
  fi
  [[ ! -v "SEEN_CASES[${selected_case}]" ]] \
    || { echo "Duplicate validation case: ${selected_case}" >&2; exit 2; }
  SEEN_CASES["${selected_case}"]=1
done
assert_idle() {
  local active
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  [[ -z "${active}" ]] || { echo "GPU processes are active; stopping validation" >&2; exit 1; }
}
assert_idle
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ "$(grep -c '^NVIDIA L20$' <<< "${GPU_NAMES}")" == 8 ]] \
  || { echo "This validation protocol requires eight NVIDIA L20 GPUs" >&2; exit 1; }
(
  cd "${ROOT_DIR}/benchmarks/pap/datasets"
  sha256sum --check SHA256SUMS
)
SUITE_ROOT="${EXPERIMENT_DIR}/runs/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${SUITE_ROOT}"
cp "${EXPERIMENT_DIR}/"{experiment.env,workloads.tsv,run.sh,driver.sh,README.md} "${SUITE_ROOT}/"
printf '%s\n' "${SELECTED_CASES[@]}" > "${SUITE_ROOT}/selected_cases.txt"
echo "SUITE_ROOT=${SUITE_ROOT}"
"${ROOT_DIR}/.venv/bin/python" \
  "${ROOT_DIR}/benchmarks/pap/tooling/capture_run_environment.py" \
  "${SUITE_ROOT}/provenance" --model "${MODEL_PATH}" \
  --environments .venv .venv-aiperf .venv-dynamo
for selected_case in "${SELECTED_CASES[@]}"; do
  read -r name dataset schema sessions requests concurrency duration trace <<< "$(
    awk -v name="${selected_case}" '$1 == name {print}' "${EXPERIMENT_DIR}/workloads.tsv"
  )"
  assert_idle
  export RUN_ROOT="${SUITE_ROOT}/${name}"
  export RUN_ID="refactor_${name}"
  export PAP_AIPERF_INPUT_FILE="${ROOT_DIR}/benchmarks/pap/datasets/${dataset}"
  export PAP_AIPERF_CUSTOM_DATASET_TYPE="${schema}"
  export PAP_AIPERF_SESSIONS="${sessions}"
  export PAP_AIPERF_EXPECTED_REQUESTS="${requests}"
  export PAP_AIPERF_CONCURRENCY="${concurrency}"
  export PAP_AIPERF_BENCHMARK_DURATION_SECONDS="${duration}"
  case "${trace:-0}" in
    0) unset PAP_PROJECTION_PA_TRACE_OUTPUT ;;
    1)
      export PAP_PROJECTION_PA_TRACE_OUTPUT="${RUN_ROOT}/trace/projection.pt"
      export PAP_PROJECTION_PA_TRACE_RING_STEPS=2048
      export PAP_PROJECTION_PA_TRACE_SAMPLES=512
      export PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS=5
      ;;
    *) echo "Invalid trace setting for ${name}" >&2; exit 2 ;;
  esac
  mkdir "${RUN_ROOT}"
  if [[ "${trace:-0}" == 1 ]]; then
    "${ROOT_DIR}/.venv/bin/python" \
      "${ROOT_DIR}/benchmarks/pap/tooling/capture_aligned_trace.py" \
      "${PAP_PROJECTION_PA_TRACE_OUTPUT}" \
      --output "${RUN_ROOT}/trace_capture" \
      --pa-count "${PAP_TOPOLOGY%%pa*}" --timeout "${BENCH_TIMEOUT}" \
      > "${RUN_ROOT}/trace_capture.log" 2>&1 &
    TRACE_CAPTURE_PID=$!
  fi
  set +e
  bash "${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh" \
    > "${RUN_ROOT}/launcher.log" 2>&1
  result=$?
  set -e
  if [[ -n "${TRACE_CAPTURE_PID}" ]]; then
    if (( result == 0 )); then
      if ! wait "${TRACE_CAPTURE_PID}"; then result=1; fi
    else
      cleanup_trace_capture
    fi
    TRACE_CAPTURE_PID=""
  fi
  if [[ -f "${PAP_NVSHMEM_PREFIX}/lib/libpap_nvshmem_device.so.build.txt" ]]; then
    cp "${PAP_NVSHMEM_PREFIX}/lib/libpap_nvshmem_device.so.build.txt" \
      "${RUN_ROOT}/nvshmem_bridge_build.txt"
  fi
  printf '%s\t%s\n' "${name}" "${result}" >> "${SUITE_ROOT}/exit_codes.tsv"
  (( result == 0 )) || { echo "Case failed: ${RUN_ROOT}" >&2; exit "${result}"; }
  assert_idle
done
printf 'passed\n' > "${SUITE_ROOT}/COMPLETE"
