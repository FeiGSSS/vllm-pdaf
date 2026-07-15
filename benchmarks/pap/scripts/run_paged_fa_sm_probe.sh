#!/usr/bin/env bash
set -euo pipefail

# Standalone P17 paged-FlashAttention SM/bandwidth experiment. This script
# deliberately does not launch PAP services or modify the runtime data path.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PROBE="${ROOT_DIR}/benchmarks/pap/tooling/paged_fa_sm_probe.py"
GPU="${PAP_FA_PROBE_GPU:-1}"
OUTPUT_ROOT="${PAP_FA_PROBE_OUTPUT_ROOT:-${ROOT_DIR}/test/baseline/pap/results/runs/$(date +%Y%m%d_%H%M%S)_paged_fa_sm_probe}"
RUN_NCU="${PAP_FA_PROBE_RUN_NCU:-1}"
RUN_FULL="${PAP_FA_PROBE_RUN_FULL:-1}"
RUN_MPS="${PAP_FA_PROBE_RUN_MPS:-1}"
RUN_TIMING="${PAP_FA_PROBE_RUN_TIMING:-1}"
RUN_NSYS="${PAP_FA_PROBE_RUN_NSYS:-1}"
RUN_TORCH_TRACE="${PAP_FA_PROBE_RUN_TORCH_TRACE:-0}"
PROFILE_SECONDS="${PAP_FA_PROBE_PROFILE_SECONDS:-2}"
NCU_LAUNCH_SKIP="${PAP_FA_PROBE_NCU_LAUNCH_SKIP:-1}"
NSYS_IMPORTER="${PAP_FA_PROBE_NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}"
MPS_CHUNKS="${PAP_FA_PROBE_MPS_CHUNKS:-7}"
EXPECTED_FULL_SMS="${PAP_FA_PROBE_FULL_SMS:-92}"
EXPECTED_MPS_SMS="${PAP_FA_PROBE_MPS_SMS:-28}"
MPS_SESSION_ID="${PAP_FA_PROBE_MPS_SESSION_ID:-${BASHPID}}"
MPS_PIPE_DIR="${PAP_FA_PROBE_MPS_PIPE_DIR:-/tmp/pap-fa-mps-${UID}-${MPS_SESSION_ID}}"
MPS_LOG_DIR="${OUTPUT_ROOT}/mps/log-${MPS_SESSION_ID}"
NCU_HOME="${OUTPUT_ROOT}/ncu-home"
GPU_UUID=""
MPS_PARTITION=""
MPS_STARTED=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

mps_control() {
  local command="$1"
  timeout 10 env \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    nvidia-cuda-mps-control <<< "${command}"
}

stop_static_mps() {
  set +e
  if [[ -n "${MPS_PARTITION}" && -n "${GPU_UUID}" ]]; then
    local partition_id="${MPS_PARTITION#"${GPU_UUID}/"}"
    mps_control "sm_partition rm ${GPU_UUID} ${partition_id}" \
      > "${OUTPUT_ROOT}/mps/remove.log" 2>&1
    MPS_PARTITION=""
  fi
  if (( MPS_STARTED != 0 )); then
    mps_control quit > "${OUTPUT_ROOT}/mps/quit.log" 2>&1
    MPS_STARTED=0
  fi
  set -e
}

cleanup() {
  local code=$?
  stop_static_mps
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

require_tools() {
  [[ -x "${PYTHON_BIN}" ]] || die "missing Python environment: ${PYTHON_BIN}"
  [[ -f "${PROBE}" ]] || die "missing probe: ${PROBE}"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v nvidia-cuda-mps-control >/dev/null \
    || die "nvidia-cuda-mps-control is unavailable"
  if [[ "${RUN_NSYS}" == "1" ]]; then
    command -v nsys >/dev/null || die "nsys is unavailable"
    [[ -x "${NSYS_IMPORTER}" ]] \
      || die "NSYS importer is unavailable: ${NSYS_IMPORTER}"
  fi
  if [[ "${RUN_NCU}" == "1" ]]; then
    command -v ncu >/dev/null || die "ncu is unavailable"
  fi
}

ensure_gpu_idle() {
  local pids
  pids="$(
    nvidia-smi -i "${GPU}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  [[ -z "${pids//[[:space:]]/}" ]] \
    || die "GPU ${GPU} has active compute processes: ${pids}"
}

run_timing() {
  local placement="$1"
  local num_splits="$2"
  local expected_sms="$3"
  local output="${OUTPUT_ROOT}/timing/${placement}_splits${num_splits}.json"
  shift 3
  env "$@" "${PYTHON_BIN}" "${PROBE}" \
    --mode timing \
    --num-splits "${num_splits}" \
    --expected-sms "${expected_sms}" \
    --output "${output}"
}

run_nsys() {
  local placement="$1"
  local num_splits="$2"
  local expected_sms="$3"
  local prefix="${OUTPUT_ROOT}/nsys/${placement}_splits${num_splits}"
  shift 3
  env "$@" nsys profile \
    --force-overwrite=true \
    --sample=none \
    --trace=nvtx \
    --gpu-metrics-device="${GPU}" \
    --gpu-metrics-set=5 \
    --gpu-metrics-frequency=10000 \
    --output="${prefix}" \
    "${PYTHON_BIN}" "${PROBE}" \
      --mode profile \
      --num-splits "${num_splits}" \
      --expected-sms "${expected_sms}" \
      --profile-seconds "${PROFILE_SECONDS}" \
      --output "${prefix}_metadata.json"
  if [[ ! -f "${prefix}.nsys-rep" ]]; then
    [[ -f "${prefix}.qdstrm" ]] \
      || die "NSYS did not produce a report or raw stream: ${prefix}"
    "${NSYS_IMPORTER}" \
      --force-overwrite \
      --input-file "${prefix}.qdstrm" \
      --output-file "${prefix}.nsys-rep"
  fi
  nsys export \
    --force-overwrite=true \
    --type=sqlite \
    --output="${prefix}.sqlite" \
    "${prefix}.nsys-rep"
}

run_torch_trace() {
  local placement="$1"
  local num_splits="$2"
  local expected_sms="$3"
  local prefix="${OUTPUT_ROOT}/torch_trace/${placement}_splits${num_splits}"
  shift 3
  env "$@" "${PYTHON_BIN}" "${PROBE}" \
    --mode trace \
    --num-splits "${num_splits}" \
    --expected-sms "${expected_sms}" \
    --trace-output "${prefix}.trace.json" \
    --output "${prefix}_metadata.json"
}

run_ncu_full() {
  local num_splits="$1"
  local prefix="${OUTPUT_ROOT}/ncu/full92_splits${num_splits}"
  local status="${prefix}_status.env"
  if env \
    HOME="${NCU_HOME}" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    ncu \
      --force-overwrite \
      --replay-mode=kernel \
      --target-processes=all \
      --nvtx \
      --nvtx-include=pap_paged_fa_probe/ \
      --launch-skip="${NCU_LAUNCH_SKIP}" \
      --launch-count=1 \
      --section=SpeedOfLight \
      --section=MemoryWorkloadAnalysis \
      --section=LaunchStats \
      --section=Occupancy \
      --section=WarpStateStats \
      --export="${prefix}" \
      "${PYTHON_BIN}" "${PROBE}" \
        --mode profile \
        --num-splits "${num_splits}" \
        --expected-sms "${EXPECTED_FULL_SMS}" \
        --profile-seconds 0.1 \
        --output "${prefix}_metadata.json" \
      > "${prefix}.log" 2>&1 \
      && [[ -f "${prefix}.ncu-rep" ]] \
      && ! grep -q "No kernels were profiled" "${prefix}.log"; then
    HOME="${NCU_HOME}" ncu \
      --import "${prefix}.ncu-rep" \
      --page raw \
      --csv \
      --print-units base \
      --log-file "${prefix}.csv"
    printf 'STATUS=passed\n' > "${status}"
  else
    printf 'STATUS=failed\n' > "${status}"
    echo "WARNING: NCU collection failed for num_splits=${num_splits}" >&2
  fi
}

start_static_mps() {
  local response line start_output
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  GPU_UUID="$(
    nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader
  )"
  GPU_UUID="${GPU_UUID//[[:space:]]/}"
  [[ "${GPU_UUID}" == GPU-* ]] || die "invalid GPU UUID: ${GPU_UUID}"
  if ! start_output="$(
    env \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
      nvidia-cuda-mps-control -d -S 2>&1
  )"; then
    die "failed to start static MPS: ${start_output}"
  fi
  MPS_STARTED=1
  response="$(
    mps_control "sm_partition add ${GPU_UUID} ${MPS_CHUNKS}" 2>&1
  )" || die "failed to create static MPS partition: ${response}"
  while IFS= read -r line; do
    line="${line%$'\r'}"
    if [[ "${line}" == Partition\ *\ created ]]; then
      MPS_PARTITION="${line#Partition }"
      MPS_PARTITION="${MPS_PARTITION% created}"
      break
    fi
  done <<< "${response}"
  [[ -n "${MPS_PARTITION}" ]] \
    || die "unexpected static MPS response: ${response}"
  {
    printf 'GPU_INDEX=%q\n' "${GPU}"
    printf 'GPU_UUID=%q\n' "${GPU_UUID}"
    printf 'MPS_CHUNKS=%q\n' "${MPS_CHUNKS}"
    printf 'MPS_PARTITION=%q\n' "${MPS_PARTITION}"
    printf 'MPS_PIPE_DIRECTORY=%q\n' "${MPS_PIPE_DIR}"
    printf 'EXPECTED_VISIBLE_SMS=%q\n' "${EXPECTED_MPS_SMS}"
  } > "${OUTPUT_ROOT}/mps/partition.env"
}

main() {
  require_tools
  ensure_gpu_idle
  mkdir -p \
    "${OUTPUT_ROOT}/timing" \
    "${OUTPUT_ROOT}/nsys" \
    "${OUTPUT_ROOT}/torch_trace" \
    "${OUTPUT_ROOT}/ncu" \
    "${NCU_HOME}"
  {
    printf 'SCHEMA_VERSION=1\n'
    printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'GPU_INDEX=%q\n' "${GPU}"
    printf 'PROFILE_SECONDS=%q\n' "${PROFILE_SECONDS}"
    printf 'NCU_VERSION=%q\n' "$(ncu --version | tail -n 1)"
    printf 'NSYS_VERSION=%q\n' "$(nsys --version)"
  } > "${OUTPUT_ROOT}/run.env"

  if [[ "${RUN_MPS}" == "1" ]]; then
    start_static_mps
    local -a mps_env=(
      CUDA_VISIBLE_DEVICES=0
      CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
      CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
      CUDA_MPS_SM_PARTITION="${MPS_PARTITION}"
    )
    if [[ "${RUN_TIMING}" == "1" ]]; then
      run_timing mps28 0 "${EXPECTED_MPS_SMS}" "${mps_env[@]}"
      run_timing mps28 1 "${EXPECTED_MPS_SMS}" "${mps_env[@]}"
    fi
    if [[ "${RUN_TORCH_TRACE}" == "1" ]]; then
      run_torch_trace mps28 0 "${EXPECTED_MPS_SMS}" "${mps_env[@]}"
      run_torch_trace mps28 1 "${EXPECTED_MPS_SMS}" "${mps_env[@]}"
    fi
    if [[ "${RUN_NSYS}" == "1" ]]; then
      run_nsys mps28 0 "${EXPECTED_MPS_SMS}" "${mps_env[@]}"
      run_nsys mps28 1 "${EXPECTED_MPS_SMS}" "${mps_env[@]}"
    fi
    stop_static_mps
  fi

  if [[ "${RUN_FULL}" == "1" ]]; then
    if [[ "${RUN_TIMING}" == "1" ]]; then
      run_timing full92 0 "${EXPECTED_FULL_SMS}" \
        CUDA_VISIBLE_DEVICES="${GPU}"
      run_timing full92 1 "${EXPECTED_FULL_SMS}" \
        CUDA_VISIBLE_DEVICES="${GPU}"
    fi
    if [[ "${RUN_TORCH_TRACE}" == "1" ]]; then
      run_torch_trace full92 0 "${EXPECTED_FULL_SMS}" \
        CUDA_VISIBLE_DEVICES="${GPU}"
      run_torch_trace full92 1 "${EXPECTED_FULL_SMS}" \
        CUDA_VISIBLE_DEVICES="${GPU}"
    fi
    if [[ "${RUN_NSYS}" == "1" ]]; then
      run_nsys full92 0 "${EXPECTED_FULL_SMS}" \
        CUDA_VISIBLE_DEVICES="${GPU}"
      run_nsys full92 1 "${EXPECTED_FULL_SMS}" \
        CUDA_VISIBLE_DEVICES="${GPU}"
    fi
  fi

  if [[ "${RUN_NCU}" == "1" ]]; then
    run_ncu_full 0
    run_ncu_full 1
  fi

  printf 'STATUS=passed\n' > "${OUTPUT_ROOT}/status.env"
  echo "PAP_FA_PROBE_OUTPUT_ROOT=${OUTPUT_ROOT}"
}

main "$@"
