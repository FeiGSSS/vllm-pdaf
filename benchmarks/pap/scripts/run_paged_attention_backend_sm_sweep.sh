#!/usr/bin/env bash
set -euo pipefail

# Compare PAP decode-attention kernel configurations under static MPS slices.

ROOT_DIR="${PAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PROBE="${ROOT_DIR}/benchmarks/pap/tooling/paged_attention_backend_probe.py"
GPU="${PAP_ATTENTION_SWEEP_GPU:-1}"
EXPERIMENTS_ROOT="${PAP_EXPERIMENTS_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments}"
OUTPUT_ROOT="${PAP_ATTENTION_SWEEP_OUTPUT_ROOT:-${EXPERIMENTS_ROOT}/_staging/runs/$(date +%Y%m%d_%H%M%S)_attention_sm_sweep}"
PLACEMENTS="${PAP_ATTENTION_SWEEP_PLACEMENTS:-12:3 20:5}"
TRITON_SPLITS="${PAP_ATTENTION_SWEEP_SPLITS:-4,8}"
TRITON_WARPS="${PAP_ATTENTION_SWEEP_WARPS:-4,8}"
TRITON_BLOCK_HS="${PAP_ATTENTION_SWEEP_BLOCK_HS:-4,8,16}"
TRITON_STAGES="${PAP_ATTENTION_SWEEP_STAGES:-1,2}"
SEQ_LENS="${PAP_ATTENTION_SWEEP_SEQ_LENS:-17344,17334,17324}"
WARMUP_CALLS="${PAP_ATTENTION_SWEEP_WARMUP_CALLS:-20}"
SAMPLES="${PAP_ATTENTION_SWEEP_SAMPLES:-5}"
CALLS_PER_SAMPLE="${PAP_ATTENTION_SWEEP_CALLS_PER_SAMPLE:-60}"
MPS_SESSION_ID="${PAP_ATTENTION_SWEEP_SESSION_ID:-${BASHPID}}"
MPS_PIPE_DIR="${PAP_ATTENTION_SWEEP_PIPE_DIR:-/tmp/pap-attn-mps-${UID}-${MPS_SESSION_ID}}"
MPS_LOG_DIR="${OUTPUT_ROOT}/mps/log-${MPS_SESSION_ID}"
TRITON_CACHE_DIR="${OUTPUT_ROOT}/triton-cache"
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

remove_partition() {
  set +e
  if [[ -n "${MPS_PARTITION}" && -n "${GPU_UUID}" ]]; then
    local partition_id="${MPS_PARTITION#"${GPU_UUID}/"}"
    mps_control "sm_partition rm ${GPU_UUID} ${partition_id}" \
      > "${OUTPUT_ROOT}/mps/remove-${partition_id}.log" 2>&1
    MPS_PARTITION=""
  fi
  set -e
}

stop_mps() {
  set +e
  remove_partition
  if (( MPS_STARTED != 0 )); then
    mps_control quit > "${OUTPUT_ROOT}/mps/quit.log" 2>&1
    MPS_STARTED=0
  fi
  set -e
}

cleanup() {
  local code=$?
  stop_mps
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ensure_gpu_idle() {
  local pids
  if ! pids="$(
    nvidia-smi -i "${GPU}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null
  )"; then
    die "nvidia-smi failed while checking GPU ${GPU}"
  fi
  [[ -z "${pids//[[:space:]]/}" ]] \
    || die "GPU ${GPU} has active compute processes: ${pids}"
}

start_mps() {
  local start_output
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
}

create_partition() {
  local chunks="$1"
  local response line
  response="$(
    mps_control "sm_partition add ${GPU_UUID} ${chunks}" 2>&1
  )" || die "failed to create ${chunks}-chunk MPS partition: ${response}"
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
}

run_placement() {
  local expected_sms="$1"
  local chunks="$2"
  local output="${OUTPUT_ROOT}/mps${expected_sms}.json"
  create_partition "${chunks}"
  env \
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    CUDA_MPS_SM_PARTITION="${MPS_PARTITION}" \
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    "${PYTHON_BIN}" "${PROBE}" \
      --triton-splits "${TRITON_SPLITS}" \
      --triton-warps "${TRITON_WARPS}" \
      --triton-block-hs "${TRITON_BLOCK_HS}" \
      --triton-stages "${TRITON_STAGES}" \
      --seq-lens "${SEQ_LENS}" \
      --warmup-calls "${WARMUP_CALLS}" \
      --samples "${SAMPLES}" \
      --calls-per-sample "${CALLS_PER_SAMPLE}" \
      --expected-sms "${expected_sms}" \
      --output "${output}"
  remove_partition
}

main() {
  [[ -x "${PYTHON_BIN}" ]] || die "missing Python environment: ${PYTHON_BIN}"
  [[ -f "${PROBE}" ]] || die "missing probe: ${PROBE}"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v nvidia-cuda-mps-control >/dev/null \
    || die "nvidia-cuda-mps-control is unavailable"
  ensure_gpu_idle
  mkdir -p "${OUTPUT_ROOT}/mps" "${TRITON_CACHE_DIR}"
  {
    printf 'SCHEMA_VERSION=1\n'
    printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'GPU_INDEX=%q\n' "${GPU}"
    printf 'PLACEMENTS=%q\n' "${PLACEMENTS}"
    printf 'TRITON_SPLITS=%q\n' "${TRITON_SPLITS}"
    printf 'TRITON_WARPS=%q\n' "${TRITON_WARPS}"
    printf 'TRITON_BLOCK_HS=%q\n' "${TRITON_BLOCK_HS}"
    printf 'TRITON_STAGES=%q\n' "${TRITON_STAGES}"
    printf 'SEQ_LENS=%q\n' "${SEQ_LENS}"
  } > "${OUTPUT_ROOT}/run.env"
  start_mps
  local placement expected_sms chunks
  for placement in ${PLACEMENTS}; do
    expected_sms="${placement%%:*}"
    chunks="${placement##*:}"
    run_placement "${expected_sms}" "${chunks}"
  done
  printf 'STATUS=passed\n' > "${OUTPUT_ROOT}/status.env"
  echo "PAP_ATTENTION_SWEEP_OUTPUT_ROOT=${OUTPUT_ROOT}"
}

main "$@"
