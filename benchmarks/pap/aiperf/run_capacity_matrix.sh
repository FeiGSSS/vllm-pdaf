#!/usr/bin/env bash
set -euo pipefail

# Fixed four-GPU PAP/PD capacity comparison. Every point restarts all services.

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
CORPUS_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${AIPERF_ROOT}/.venv/bin/aiperf}"
PAP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
PD_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pd_multiturn_topology.sh"
DATASET_GENERATOR="${ROOT_DIR}/benchmarks/pap/aiperf/generate_multiturn_dataset.py"
RUN_SUMMARIZER="${ROOT_DIR}/benchmarks/pap/aiperf/summarize_capacity_run.py"
MATRIX_SUMMARIZER="${ROOT_DIR}/benchmarks/pap/aiperf/summarize_capacity_matrix.py"

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
MATRIX_ID="${PAP_CAPACITY_MATRIX_ID:-$(date +%Y%m%d_%H%M%S)_aiperf_capacity}"
MATRIX_ROOT="${PAP_CAPACITY_MATRIX_ROOT:-${RESULTS_ROOT}/capacity/${MATRIX_ID}}"
POINTS_CSV="${PAP_CAPACITY_POINTS:-4,8,12,16,24,32}"
ARCHITECTURES_CSV="${PAP_CAPACITY_ARCHITECTURES:-pap_3pa1p,pd_1p3d,pd_2p2d,pd_3p1d}"
REPETITIONS="${PAP_CAPACITY_REPETITIONS:-1}"
STOP_AFTER_RELAXED_FAIL="${PAP_CAPACITY_STOP_AFTER_RELAXED_FAIL:-1}"
RESUME="${PAP_CAPACITY_RESUME:-1}"
WAIT_FOR_GPUS="${PAP_CAPACITY_WAIT_FOR_GPUS:-1}"

TURNS=10
DOCUMENT_TOKENS=8192
APPEND_TOKENS=512
OUTPUT_TOKENS=256
MAX_MODEL_LEN=20000
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=32
PAP_PA_MAX_NUM_SEQS=12
PAP_GPU_MEMORY_UTILIZATION=0.76
PD_GPU_MEMORY_UTILIZATION=0.90
PAP_PREFILL_CHUNKS=18
PAP_ATTENTION_CHUNKS=5
PAP_PREFILL_SMS=72
PAP_ATTENTION_SMS=20
PAP_PREFILL_PERCENT=80
PAP_ATTENTION_PERCENT=20
REQUEST_TIMEOUT_SECONDS=600
PAP_RUN_TIMEOUT_SECONDS=3600

IFS=, read -r -a POINTS <<< "${POINTS_CSV}"
IFS=, read -r -a ARCHITECTURES <<< "${ARCHITECTURES_CSV}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

for required in "${PYTHON_BIN}" "${AIPERF_BIN}" "${MODEL_PATH}" \
  "${CORPUS_PATH}" "${PAP_RUNNER}" "${PD_RUNNER}" \
  "${DATASET_GENERATOR}" "${RUN_SUMMARIZER}" "${MATRIX_SUMMARIZER}"; do
  [[ -e "${required}" ]] || die "required path is missing: ${required}"
done
[[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]] \
  || die "PAP_CAPACITY_REPETITIONS must be positive"

MAX_SESSIONS=0
for concurrency in "${POINTS[@]}"; do
  [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] \
    || die "invalid concurrency point: ${concurrency}"
  (( concurrency <= MAX_NUM_SEQS )) \
    || die "concurrency ${concurrency} exceeds fixed max_num_seqs"
  if (( concurrency > MAX_SESSIONS )); then
    MAX_SESSIONS="${concurrency}"
  fi
done

for architecture in "${ARCHITECTURES[@]}"; do
  case "${architecture}" in
    pap_3pa1p | pd_1p3d | pd_2p2d | pd_3p1d) ;;
    *) die "unsupported architecture: ${architecture}" ;;
  esac
done

mkdir -p "${MATRIX_ROOT}/dataset" "${MATRIX_ROOT}/runs"
DATASET_FILE="${MATRIX_ROOT}/dataset/multiturn_8k_plus512_o256_t10.jsonl"

if [[ ! -f "${DATASET_FILE}" ]]; then
  "${PYTHON_BIN}" "${DATASET_GENERATOR}" \
    --model "${MODEL_PATH}" \
    --corpus "${CORPUS_PATH}" \
    --output "${DATASET_FILE}" \
    --sessions "${MAX_SESSIONS}" \
    --turns "${TURNS}" \
    --document-tokens "${DOCUMENT_TOKENS}" \
    --append-tokens "${APPEND_TOKENS}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --session-prefix "${MATRIX_ID}-session"
fi
DATASET_MANIFEST="${DATASET_FILE}.manifest.json"
if ! jq -e \
  --argjson sessions "${MAX_SESSIONS}" \
  --argjson turns "${TURNS}" \
  --argjson document_tokens "${DOCUMENT_TOKENS}" \
  --argjson append_tokens "${APPEND_TOKENS}" \
  --argjson output_tokens "${OUTPUT_TOKENS}" \
  '.sessions >= $sessions
    and .turns_per_session == $turns
    and .requested_document_tokens == $document_tokens
    and .requested_append_tokens == $append_tokens
    and .output_tokens == $output_tokens' \
  "${DATASET_MANIFEST}" >/dev/null; then
  die "dataset manifest does not match the fixed testbed"
fi

{
  printf 'SCHEMA_VERSION=1\nMATRIX_ID=%q\n' "${MATRIX_ID}"
  printf 'MODEL_PATH=%q\nCORPUS_PATH=%q\n' \
    "${MODEL_PATH}" "${CORPUS_PATH}"
  printf 'AIPERF_TIMING_MODE=concurrency\nAIPERF_REQUEST_RATE=\n'
  printf 'CONCURRENCY_POINTS=%q\nARCHITECTURES=%q\n' \
    "${POINTS_CSV}" "${ARCHITECTURES_CSV}"
  printf 'REPETITIONS=%q\nSTOP_AFTER_RELAXED_FAIL=%q\n' \
    "${REPETITIONS}" "${STOP_AFTER_RELAXED_FAIL}"
  printf 'SESSIONS_MAX=%q\nTURNS=%q\n' "${MAX_SESSIONS}" "${TURNS}"
  printf 'DOCUMENT_TOKENS=%q\nAPPEND_TOKENS=%q\nOUTPUT_TOKENS=%q\n' \
    "${DOCUMENT_TOKENS}" "${APPEND_TOKENS}" "${OUTPUT_TOKENS}"
  printf 'MAX_MODEL_LEN=%q\nMAX_NUM_BATCHED_TOKENS=%q\n' \
    "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}"
  printf 'MAX_NUM_SEQS=%q\nPAP_PA_MAX_NUM_SEQS=%q\n' \
    "${MAX_NUM_SEQS}" "${PAP_PA_MAX_NUM_SEQS}"
  printf 'PAP_GPU_MEMORY_UTILIZATION=%q\n' \
    "${PAP_GPU_MEMORY_UTILIZATION}"
  printf 'PD_GPU_MEMORY_UTILIZATION=%q\n' \
    "${PD_GPU_MEMORY_UTILIZATION}"
  printf 'PAP_STATIC_MPS_CHUNKS=%q\nPAP_STATIC_MPS_SMS=%q\n' \
    "${PAP_PREFILL_CHUNKS}/${PAP_ATTENTION_CHUNKS}" \
    "${PAP_PREFILL_SMS}/${PAP_ATTENTION_SMS}"
  printf 'SLO_STRICT=%q\nSLO_STANDARD=%q\nSLO_RELAXED=%q\n' \
    'TTFT<=5000ms,ITL<=50ms,good>=0.95' \
    'TTFT<=10000ms,ITL<=75ms,good>=0.95' \
    'TTFT<=20000ms,ITL<=100ms,good>=0.95'
} > "${MATRIX_ROOT}/matrix_config.env"

wait_for_four_gpus() {
  [[ "${WAIT_FOR_GPUS}" == "1" ]] || return 0
  local busy gpu processes
  while true; do
    busy=0
    for gpu in 0 1 2 3; do
      processes="$(
        nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null || true
      )"
      if [[ -n "${processes//[[:space:]]/}" ]]; then
        echo "GPU ${gpu} is occupied by PID(s): ${processes//$'\n'/,}"
        busy=1
      fi
    done
    (( busy == 1 )) || return 0
    echo "Waiting 60 seconds for GPUs 0-3 to become idle..."
    sleep 60
  done
}

summarize_point() {
  local architecture="$1"
  local topology="$2"
  local concurrency="$3"
  local repetition="$4"
  local run_root="$5"
  "${PYTHON_BIN}" "${RUN_SUMMARIZER}" \
    --run-root "${run_root}" \
    --architecture "${architecture}" \
    --topology "${topology}" \
    --concurrency "${concurrency}" \
    --sessions "${concurrency}" \
    --turns "${TURNS}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --repetition "${repetition}"
}

run_pap_point() {
  local concurrency="$1"
  local run_root="$2"
  env \
    PAP_ROOT="${ROOT_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATASET_PATH="${CORPUS_PATH}" \
    AIPERF_ROOT="${AIPERF_ROOT}" \
    AIPERF_BIN="${AIPERF_BIN}" \
    RUN_ID="$(basename "${run_root}")" \
    RUN_ROOT="${run_root}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    PAP_BENCH_CLIENT_MODE=aiperf_multiturn \
    PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=1 \
    PAP_BENCH_STRICT_CORRECTNESS_AUDIT=1 \
    PAP_TOPOLOGY=3pa1p \
    PAP_PREFILL_GPUS=0,1,2 \
    PAP_PROJECTION_GPUS=3 \
    PAP_ROUTING_POLICY=conversation_affinity \
    PAP_OFFLOAD_EXEC_TRANSPORT=local_fast \
    PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc \
    PAP_DIRECT_MAILBOX_OUTPUT=1 \
    PAP_VLLM_DTYPE=float16 \
    PAP_PREFILL_GPU_MEMORY_UTILIZATION="${PAP_GPU_MEMORY_UTILIZATION}" \
    PAP_PROJECTION_GPU_MEMORY_UTILIZATION="${PAP_GPU_MEMORY_UTILIZATION}" \
    PAP_PREFILL_MPS_PERCENT="${PAP_PREFILL_PERCENT}" \
    PAP_ATTENTION_MPS_PERCENT="${PAP_ATTENTION_PERCENT}" \
    PAP_STATIC_PREFILL_CHUNKS="${PAP_PREFILL_CHUNKS}" \
    PAP_STATIC_ATTENTION_CHUNKS="${PAP_ATTENTION_CHUNKS}" \
    PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_PREFILL_SMS}" \
    PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_ATTENTION_SMS}" \
    PAP_MULTITURN_LOAD_ROUNDS="${TURNS}" \
    PAP_MULTITURN_LOAD_CONVERSATIONS="${concurrency}" \
    PAP_MULTITURN_APPEND_TOKENS="${APPEND_TOKENS}" \
    INPUT_LEN="${DOCUMENT_TOKENS}" \
    OUTPUT_LEN="${OUTPUT_TOKENS}" \
    PAP_AIPERF_INPUT_FILE="${DATASET_FILE}" \
    PAP_AIPERF_OUTPUT_DIR="${run_root}/aiperf" \
    PAP_AIPERF_CONCURRENCY="${concurrency}" \
    PAP_AIPERF_TIMING_MODE=concurrency \
    PAP_AIPERF_REQUEST_RATE= \
    AIPERF_EXPORT_LEVEL=records \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    PAP_PREFILL_MAX_NUM_SEQS="${PAP_PA_MAX_NUM_SEQS}" \
    PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    PAP_PROJECTION_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=512 \
    BENCH_TIMEOUT="${PAP_RUN_TIMEOUT_SECONDS}" \
    SERVER_START_TIMEOUT=900 \
    CLUSTER_READY_WAIT_SECONDS=30 \
    "${PAP_RUNNER}"
}

run_pd_point() {
  local topology="$1"
  local concurrency="$2"
  local run_root="$3"
  env \
    PAP_ROOT="${ROOT_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATASET_PATH="${CORPUS_PATH}" \
    AIPERF_ROOT="${AIPERF_ROOT}" \
    AIPERF_BIN="${AIPERF_BIN}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    PD_LOAD_RUN_ID="$(basename "${run_root}")" \
    PD_LOAD_RUN_ROOT="${run_root}" \
    PD_LOAD_CLIENT_MODE=aiperf_multiturn \
    PD_LOAD_TOPOLOGY="${topology}" \
    PD_LOAD_ROUNDS="${TURNS}" \
    PD_LOAD_CONVERSATIONS="${concurrency}" \
    PD_LOAD_DOCUMENT_TOKENS="${DOCUMENT_TOKENS}" \
    PD_LOAD_APPEND_TOKENS="${APPEND_TOKENS}" \
    PD_LOAD_OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
    PD_LOAD_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    PD_LOAD_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    PD_LOAD_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PD_LOAD_GPU_MEMORY_UTILIZATION="${PD_GPU_MEMORY_UTILIZATION}" \
    PD_LOAD_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
    PD_AIPERF_INPUT_FILE="${DATASET_FILE}" \
    PD_AIPERF_OUTPUT_DIR="${run_root}/aiperf" \
    PD_AIPERF_CONCURRENCY="${concurrency}" \
    PD_AIPERF_TIMING_MODE=concurrency \
    PD_AIPERF_REQUEST_RATE= \
    AIPERF_EXPORT_LEVEL=records \
    "${PD_RUNNER}" oneway
}

run_point() {
  local architecture_tag="$1"
  local concurrency="$2"
  local repetition="$3"
  local architecture topology run_name run_root launcher_code
  if [[ "${architecture_tag}" == pap_* ]]; then
    architecture=pap
    topology="${architecture_tag#pap_}"
  else
    architecture=pd
    topology="${architecture_tag#pd_}"
  fi
  run_name="${architecture_tag}_c${concurrency}_r${repetition}"
  run_root="${MATRIX_ROOT}/runs/${run_name}"

  if [[ "${RESUME}" == "1" \
    && -f "${run_root}/capacity_summary.json" ]]; then
    echo "Reusing ${run_name}"
    return
  fi
  if [[ -d "${run_root}" && -n "$(find "${run_root}" -mindepth 1 -print -quit)" ]]; then
    die "run directory already has data: ${run_root}"
  fi

  wait_for_four_gpus
  mkdir -p "${run_root}"
  echo "=== ${architecture_tag} concurrency=${concurrency} rep=${repetition} ==="
  set +e
  if [[ "${architecture}" == "pap" ]]; then
    run_pap_point "${concurrency}" "${run_root}" \
      > "${run_root}/launcher.log" 2>&1
    launcher_code="$?"
  else
    run_pd_point "${topology}" "${concurrency}" "${run_root}" \
      > "${run_root}/launcher.log" 2>&1
    launcher_code="$?"
  fi
  set -e
  printf '%s\n' "${launcher_code}" > "${run_root}/launcher_exit_code.txt"
  summarize_point \
    "${architecture}" "${topology}" "${concurrency}" \
    "${repetition}" "${run_root}"
}

for architecture in "${ARCHITECTURES[@]}"; do
  relaxed_failed=0
  for concurrency in "${POINTS[@]}"; do
    if (( relaxed_failed == 1 )) && [[ "${STOP_AFTER_RELAXED_FAIL}" == "1" ]]; then
      echo "Skipping ${architecture} above its first relaxed-SLO failure"
      break
    fi
    point_passed=1
    for (( repetition=1; repetition<=REPETITIONS; repetition++ )); do
      run_point "${architecture}" "${concurrency}" "${repetition}"
      summary="${MATRIX_ROOT}/runs/${architecture}_c${concurrency}_r${repetition}/capacity_summary.json"
      if [[ "$(jq -r '.slo.relaxed.passed' "${summary}")" != "true" ]]; then
        point_passed=0
      fi
    done
    if (( point_passed == 0 )); then
      relaxed_failed=1
    fi
    "${PYTHON_BIN}" "${MATRIX_SUMMARIZER}" "${MATRIX_ROOT}"
  done
done

"${PYTHON_BIN}" "${MATRIX_SUMMARIZER}" "${MATRIX_ROOT}"
echo "PAP_CAPACITY_MATRIX_ROOT=${MATRIX_ROOT}"
