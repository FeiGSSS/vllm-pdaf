#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
RUNNER="${ROOT_DIR}/.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh"
COMPARER="${ROOT_DIR}/benchmarks/multi_turn/compare_pap_pd_multiturn_load.py"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
MODE="${1:-quick}"
LOAD_SHAPE="${2:-c4}"

case "${MODE}" in
  quick)
    DEFAULT_REPETITIONS=1
    DEFAULT_REQUIRE_CLEAN=0
    ;;
  formal)
    DEFAULT_REPETITIONS=3
    DEFAULT_REQUIRE_CLEAN=1
    ;;
  *)
    echo "usage: $0 [quick|formal] [c1|c2|c4]" >&2
    exit 2
    ;;
esac
case "${LOAD_SHAPE}" in
  c1) CONVERSATIONS=1 ;;
  c2) CONVERSATIONS=2 ;;
  c4) CONVERSATIONS=4 ;;
  *)
    echo "usage: $0 [quick|formal] [c1|c2|c4]" >&2
    exit 2
    ;;
esac

MPS_PROFILE="${PAP_LOAD_MPS_PROFILE:-baseline_static_64_28}"
MPS_MODE=dynamic
STATIC_PREFILL_CHUNKS=16
STATIC_ATTENTION_CHUNKS=7
case "${MPS_PROFILE}" in
  baseline_70_30)
    PREFILL_MPS_PERCENT=70
    ATTENTION_MPS_PERCENT=30
    ;;
  diagnostic_80_20)
    PREFILL_MPS_PERCENT=80
    ATTENTION_MPS_PERCENT=20
    ;;
  baseline_static_64_28 | diagnostic_static_64_28)
    MPS_MODE=static
    PREFILL_MPS_PERCENT=70
    ATTENTION_MPS_PERCENT=30
    ;;
  *)
    echo "unknown PAP_LOAD_MPS_PROFILE: ${MPS_PROFILE}" >&2
    exit 2
    ;;
esac

REPETITIONS="${PAP_LOAD_REPETITIONS:-${DEFAULT_REPETITIONS}}"
REQUIRE_CLEAN="${PAP_LOAD_REQUIRE_CLEAN_TRACKED_WORKTREE:-${DEFAULT_REQUIRE_CLEAN}}"
if [[ "${REPETITIONS}" != "1" && "${REPETITIONS}" != "3" ]]; then
  echo "PAP load requires one quick or three formal repetitions" >&2
  exit 2
fi
case "${REQUIRE_CLEAN}" in
  0|1) ;;
  *)
    echo "PAP_LOAD_REQUIRE_CLEAN_TRACKED_WORKTREE must be 0 or 1" >&2
    exit 2
    ;;
esac
[[ -x "${PYTHON_BIN}" ]] || {
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
}
for required in "${RUNNER}" "${COMPARER}" "${DATASET_PATH}"; do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 1
  }
done

ensure_gpu_idle() {
  local gpu pid_output
  for gpu in 1 2; do
    pid_output="$(
      nvidia-smi -i "${gpu}" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    if [[ -n "${pid_output//[[:space:]]/}" ]]; then
      echo "GPU ${gpu} is occupied by PID(s): ${pid_output}" >&2
      return 1
    fi
  done
}

hardware_signature() {
  local -a names=()
  mapfile -t names < <(
    nvidia-smi -i 1,2 --query-gpu=name --format=csv,noheader \
      | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  )
  [[ "${#names[@]}" == "2" ]] || return 1
  if [[ "${names[0]}" == "${names[1]}" ]]; then
    printf '%sx2\n' "${names[0]}"
  else
    printf '%s+%s\n' "${names[0]}" "${names[1]}"
  fi
}

cd "${ROOT_DIR}"
ensure_gpu_idle
HARDWARE_SIGNATURE="$(hardware_signature)"
GIT_SHORT="$(git rev-parse --short HEAD)"
GROUP_RUN_ID="${PAP_LOAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${GIT_SHORT}_pap_load_${LOAD_SHAPE}}"
GROUP_ROOT="${PAP_LOAD_RUN_ROOT:-${RESULTS_ROOT}/runs/${GROUP_RUN_ID}}"
mkdir -p "${GROUP_ROOT}"

RESULT_ARGS=()
for (( rep=1; rep<=REPETITIONS; rep++ )); do
  ensure_gpu_idle
  REP_ROOT="${GROUP_ROOT}/rep${rep}"
  PORT_SHIFT=$(((rep - 1) * 100))
  PAP_BENCH_CLIENT_MODE=multiturn_load \
  PAP_TOPOLOGY=1pa1p \
  PAP_PREFILL_GPUS=1 \
  PAP_PROJECTION_GPUS=2 \
  PAP_VLLM_DTYPE=float16 \
  PAP_OFFLOAD_EXEC_TRANSPORT=local_fast \
  PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc \
  PAP_DIRECT_MAILBOX_OUTPUT=1 \
  PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND=1 \
  PAP_BATCHED_ROUTE_COPY=1 \
  PAP_LOCAL_FAST_STREAM_ORDERED=1 \
  PAP_LOCAL_FAST_SLOT_COUNT=2 \
  PAP_LOCAL_FAST_BATCH_PLAN=1 \
  PAP_UNIFIED_MD_FAST_KEY=1 \
  PAP_ATTENTION_DISPATCH_MODE=legacy \
  PAP_ATTENTION_COMBINE_WAIT_US=0 \
  PAP_PREFIX_CACHE_AUDIT=0 \
  PAP_ENABLE_PROMPT_TOKENS_DETAILS=1 \
  PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=256 \
  PAP_BENCH_MPS_PROFILE="${MPS_PROFILE}" \
  PAP_MPS_MODE="${MPS_MODE}" \
  PAP_PREFILL_MPS_PERCENT="${PREFILL_MPS_PERCENT}" \
  PAP_ATTENTION_MPS_PERCENT="${ATTENTION_MPS_PERCENT}" \
  PAP_STATIC_PREFILL_CHUNKS="${STATIC_PREFILL_CHUNKS}" \
  PAP_STATIC_ATTENTION_CHUNKS="${STATIC_ATTENTION_CHUNKS}" \
  PAP_ENABLE_MPS=1 \
  PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${REQUIRE_CLEAN}" \
  PAP_BENCH_STRICT_CORRECTNESS_AUDIT=1 \
  PAP_BENCH_SESSION_DRAIN_TIMEOUT=60 \
  PAP_NORTH_STAR_HARDWARE_SIGNATURE="${HARDWARE_SIGNATURE}" \
  PAP_NORTH_STAR_CONVERSATION_ID="${GROUP_RUN_ID}-rep${rep}-conversation" \
  PAP_NORTH_STAR_CACHE_SALT="${GROUP_RUN_ID}-rep${rep}-cache-salt" \
  PAP_MULTITURN_LOAD_ROUNDS=5 \
  PAP_MULTITURN_LOAD_CONVERSATIONS="${CONVERSATIONS}" \
  PAP_MULTITURN_LOAD_REQUEST_RATE=2 \
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  INPUT_LEN=16000 \
  OUTPUT_LEN=256 \
  QPS=2 \
  MAX_MODEL_LEN=20000 \
  MAX_NUM_BATCHED_TOKENS=4096 \
  MAX_NUM_SEQS=4 \
  CLUSTER_READY_WAIT_SECONDS=5 \
  SERVER_START_TIMEOUT=900 \
  BENCH_TIMEOUT=1800 \
  RUN_ID="${GROUP_RUN_ID}_rep${rep}" \
  RUN_ROOT="${REP_ROOT}" \
  PAP_PROXY_PORT="$((19700 + PORT_SHIFT))" \
  PAP_PREFILL_PORT_BASE="$((19100 + PORT_SHIFT))" \
  PAP_PROJECTION_PORT_BASE="$((19200 + PORT_SHIFT))" \
  PAP_ATTENTION_PORT_BASE="$((19300 + PORT_SHIFT))" \
  PAP_ATTENTION_TCP_PORT_BASE="$((19400 + PORT_SHIFT))" \
  PAP_ATTENTION_ZMQ_PORT_BASE="$((19500 + PORT_SHIFT))" \
  PAP_PROJECTION_ZMQ_PORT_BASE="$((19600 + PORT_SHIFT))" \
  PAP_PREFILL_NIXL_PORT_BASE="$((5620 + PORT_SHIFT))" \
  PAP_VLLM_PREFILL_PORT_BASE="$((50200 + PORT_SHIFT))" \
  PAP_VLLM_PROJECTION_PORT_BASE="$((50220 + PORT_SHIFT))" \
  bash "${RUNNER}"
  RESULT_ARGS+=(--result "${REP_ROOT}/result.json")
done

"${PYTHON_BIN}" "${COMPARER}" aggregate \
  "${RESULT_ARGS[@]}" --output "${GROUP_ROOT}/aggregate.json"
echo "PAP_LOAD_RUN_ROOT=${GROUP_ROOT}"
