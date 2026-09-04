#!/usr/bin/env bash
set -euo pipefail

# Standalone PAP Prefill-compute saturation microbenchmark.
#
# This deliberately excludes Decode, remote Attention, and NIXL data movement.
# It uses the production PAP Prefill static-MPS allocation (20 chunks / 80 SM)
# and audits the actual vLLM context-iteration shape for every sample.

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
GPU_INDEX="${PAP_PREFILL_MICROBENCH_GPU:-0}"
PREFILL_CHUNKS="${PAP_PREFILL_MICROBENCH_CHUNKS:-20}"
EXPECTED_SMS="${PAP_PREFILL_MICROBENCH_EXPECTED_SMS:-80}"
EXPERIMENT_GROUPS="${PAP_PREFILL_MICROBENCH_GROUPS:-step1,step2,step3}"
WARMUP_REPEATS="${PAP_PREFILL_MICROBENCH_WARMUPS:-1}"
MEASURE_REPEATS="${PAP_PREFILL_MICROBENCH_REPEATS:-3}"
MAX_MODEL_LEN="${PAP_PREFILL_MICROBENCH_MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${PAP_PREFILL_MICROBENCH_MAX_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${PAP_PREFILL_MICROBENCH_MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${PAP_PREFILL_MICROBENCH_GPU_MEMORY_UTILIZATION:-0.90}"
RUN_ID="$(
  date +%Y%m%d_%H%M%S
)_prefill_saturation_gpu${GPU_INDEX}_${EXPECTED_SMS}sm"
RUN_ROOT="${PAP_PREFILL_MICROBENCH_RUN_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments/microbench/_runs/${RUN_ID}}"
MICROBENCH="${ROOT_DIR}/benchmarks/pap/microbench/prefill_saturation.py"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

for required in "${PYTHON_BIN}" "${MODEL_PATH}" "${MICROBENCH}"; do
  [[ -e "${required}" ]] || die "required path is missing: ${required}"
done
for value in "${GPU_INDEX}" "${PREFILL_CHUNKS}" "${EXPECTED_SMS}" \
  "${WARMUP_REPEATS}" "${MEASURE_REPEATS}" "${MAX_MODEL_LEN}" \
  "${MAX_BATCHED_TOKENS}" "${MAX_NUM_SEQS}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || die "expected a non-negative integer: ${value}"
done
(( PREFILL_CHUNKS > 0 && EXPECTED_SMS > 0 && MEASURE_REPEATS > 0 )) \
  || die "MPS chunks, expected SMs, and measured repetitions must be positive"

# Required PAP benchmark runtime audit. This benchmark does not transfer data,
# but recording the validated runtime prevents accidental reuse under a stale
# same-node NIXL environment.
source "${ROOT_DIR}/benchmarks/pap/scripts/configure_same_node_nixl.sh"
pap_configure_same_node_nixl "${ROOT_DIR}"

mkdir -p "${RUN_ROOT}"
ENGINE_LOG="${RUN_ROOT}/engine.log"
RAW_RESULT="${RUN_ROOT}/raw_result.json"
RESULT="${RUN_ROOT}/result.json"
REPORT="${RUN_ROOT}/report.md"
CONFIG="${RUN_ROOT}/effective_config.env"
MPS_PIPE_DIR="${PAP_PREFILL_MICROBENCH_MPS_PIPE_DIR:-/tmp/pap-prefill-mps-${USER:-user}-$$}"
MPS_LOG_DIR="${RUN_ROOT}/mps/log"
mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"

MPS_STARTED=0
GPU_UUID=""
PARTITION_ID=""

mps_control() {
  local command="$1"
  timeout 10 env \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    nvidia-cuda-mps-control <<< "${command}"
}

cleanup() {
  local short_partition
  if [[ -n "${PARTITION_ID}" && -n "${GPU_UUID}" ]]; then
    short_partition="${PARTITION_ID#"${GPU_UUID}/"}"
    mps_control "sm_partition rm ${GPU_UUID} ${short_partition}" \
      >/dev/null 2>&1 || true
  fi
  if (( MPS_STARTED )); then
    mps_control quit >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

compute_pids="$(
  nvidia-smi \
    --id="${GPU_INDEX}" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${compute_pids//[[:space:]]/}" ]]; then
  die "GPU ${GPU_INDEX} has active compute processes: ${compute_pids}"
fi

echo "Starting static MPS on physical GPU ${GPU_INDEX}"
if ! env \
  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  nvidia-cuda-mps-control -d -S; then
  die "failed to start static MPS; inspect ${MPS_LOG_DIR}"
fi
MPS_STARTED=1

GPU_UUID="$(
  nvidia-smi \
    --id="${GPU_INDEX}" \
    --query-gpu=uuid \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
[[ -n "${GPU_UUID}" ]] || die "could not resolve GPU ${GPU_INDEX} UUID"
partition_response="$(
  mps_control "sm_partition add ${GPU_UUID} ${PREFILL_CHUNKS}" 2>&1
)"
if [[ "${partition_response}" == Partition\ *\ created ]]; then
  PARTITION_ID="${partition_response#Partition }"
  PARTITION_ID="${PARTITION_ID% created}"
fi
[[ -n "${PARTITION_ID}" ]] \
  || die "unexpected static MPS response: ${partition_response}"

visible_sms="$(
  env \
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    CUDA_MPS_SM_PARTITION="${PARTITION_ID}" \
    "${PYTHON_BIN}" -c \
      'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)'
)"
[[ "${visible_sms}" == "${EXPECTED_SMS}" ]] \
  || die "static MPS exposed ${visible_sms} SMs; expected ${EXPECTED_SMS}"

tracked_worktree_dirty=0
git -C "${ROOT_DIR}" diff --quiet || tracked_worktree_dirty=1
git -C "${ROOT_DIR}" diff --cached --quiet || tracked_worktree_dirty=1
{
  printf 'RUN_ID=%q\n' "${RUN_ID}"
  printf 'RUN_SCRIPT=%q\n' "${BASH_SOURCE[0]}"
  printf 'GIT_COMMIT=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  printf 'GIT_TRACKED_WORKTREE_DIRTY=%q\n' "${tracked_worktree_dirty}"
  printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
  printf 'MODEL_CONFIG_SHA256=%q\n' "$(sha256sum "${MODEL_PATH}/config.json" | cut -d' ' -f1)"
  printf 'PHYSICAL_GPU_INDEX=%q\n' "${GPU_INDEX}"
  printf 'GPU_UUID=%q\n' "${GPU_UUID}"
  printf 'GPU_NAME=%q\n' "$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader)"
  printf 'DRIVER_VERSION=%q\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  printf 'MPS_PARTITION_ID=%q\n' "${PARTITION_ID}"
  printf 'PREFILL_CHUNKS=%q\n' "${PREFILL_CHUNKS}"
  printf 'PREFILL_VISIBLE_SMS=%q\n' "${visible_sms}"
  printf 'EXPERIMENT_GROUPS=%q\n' "${EXPERIMENT_GROUPS}"
  printf 'WARMUP_REPEATS=%q\n' "${WARMUP_REPEATS}"
  printf 'MEASURE_REPEATS=%q\n' "${MEASURE_REPEATS}"
  printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN}"
  printf 'MAX_BATCHED_TOKENS=%q\n' "${MAX_BATCHED_TOKENS}"
  printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS}"
  printf 'GPU_MEMORY_UTILIZATION=%q\n' "${GPU_MEMORY_UTILIZATION}"
  printf 'EXECUTION_MODE=eager\n'
  printf 'ASYNC_SCHEDULING=1\n'
  printf 'BATCH_GATE=sleep_level_0_enqueue_wake_scheduling\n'
  printf 'PREFIX_CACHING=0\n'
  printf 'VLLM_USE_FLASHINFER_SAMPLER=0\n'
  printf 'NIXL_DATA_PATH_EXERCISED=0\n'
  printf 'PAP_NIXL_RUNTIME_MODE=%q\n' "${PAP_NIXL_RUNTIME_MODE}"
  printf 'PAP_NIXL_UCX_VERSION=%q\n' "${PAP_NIXL_UCX_VERSION}"
  printf 'PYTORCH_CUDA=%q\n' "$(env CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" CUDA_MPS_SM_PARTITION="${PARTITION_ID}" "${PYTHON_BIN}" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
} > "${CONFIG}"

echo "Running Prefill saturation matrix; output=${RUN_ROOT}"
set +e
env \
  CUDA_VISIBLE_DEVICES=0 \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  CUDA_MPS_SM_PARTITION="${PARTITION_ID}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  PAP_MODEL_HOOKS=1 \
  "${PYTHON_BIN}" "${MICROBENCH}" run \
    --model "${MODEL_PATH}" \
    --output "${RAW_RESULT}" \
    --groups "${EXPERIMENT_GROUPS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --measure-repeats "${MEASURE_REPEATS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    2>&1 | tee "${ENGINE_LOG}" | sed -u -n '/PAP_PREFILL_SAMPLE_/p'
benchmark_status="${PIPESTATUS[0]}"
set -e
(( benchmark_status == 0 )) \
  || die "Prefill benchmark failed; inspect ${ENGINE_LOG}"

"${PYTHON_BIN}" "${MICROBENCH}" analyze \
  --input "${RAW_RESULT}" \
  --engine-log "${ENGINE_LOG}" \
  --output "${RESULT}" \
  --report "${REPORT}" \
  --require-single-context-iteration

echo "Prefill saturation microbenchmark completed"
echo "Result: ${RESULT}"
echo "Report: ${REPORT}"
