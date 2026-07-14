#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PROFILE_PATH="${PAP_PROFILE_PATH:-${ROOT_DIR}/benchmarks/pap/profiles/p17_1pa1p.toml}"
PROFILE_LOADER="${ROOT_DIR}/benchmarks/pap/profile_env.py"
RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
COMPARER="${ROOT_DIR}/benchmarks/multi_turn/compare_pap_pd_multiturn_load.py"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
MODE="${1:-quick}"
LOAD_SHAPE="${2:-c4}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
}
for required in "${PROFILE_PATH}" "${PROFILE_LOADER}" "${RUNNER}" "${COMPARER}"; do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 1
  }
done
PROFILE_ASSIGNMENTS="$("${PYTHON_BIN}" "${PROFILE_LOADER}" "${PROFILE_PATH}")"
eval "${PROFILE_ASSIGNMENTS}"
P17_ACTIVE_CONVERSATIONS="${PAP_MULTITURN_LOAD_CONVERSATIONS}"
PREFILL_GPU="${PAP_PREFILL_GPUS}"
PROJECTION_GPU="${PAP_PROJECTION_GPUS}"

MODEL_ROOT="${PAP_MODEL_ROOT:-/data/ssd1/llm-models}"
CORPUS_ROOT="${PAP_CORPUS_ROOT:-/home/fei/research/PD/refer_codes/vllm/benchmarks}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${P17_MODEL_RELATIVE_PATH}}"
DATASET_PATH="${DATASET_PATH:-${CORPUS_ROOT}/${P17_CORPUS_RELATIVE_PATH}}"

case "${MODE}" in
  quick)
    DEFAULT_REPETITIONS=1
    DEFAULT_REQUIRE_CLEAN=0
    ;;
  formal)
    DEFAULT_REPETITIONS="${P17_REPETITIONS}"
    DEFAULT_REQUIRE_CLEAN=1
    ;;
  *)
    echo "usage: $0 [quick|formal] [c1|c4]" >&2
    exit 2
    ;;
esac
case "${LOAD_SHAPE}" in
  c1) CONVERSATIONS=1 ;;
  c4)
    [[ "${P17_ACTIVE_CONVERSATIONS}" == "4" ]] || {
      echo "P17 profile must define four active conversations" >&2
      exit 2
    }
    CONVERSATIONS="${P17_ACTIVE_CONVERSATIONS}"
    ;;
  *)
    echo "usage: $0 [quick|formal] [c1|c4]" >&2
    exit 2
    ;;
esac

if [[ -v PAP_LOAD_MPS_PROFILE ]]; then
  echo "PAP_LOAD_MPS_PROFILE was removed; P17 always uses static 64/28 MPS" >&2
  exit 2
fi
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
for required in "${DATASET_PATH}"; do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 1
  }
done

ensure_gpu_idle() {
  local gpu pid_output
  for gpu in "${PREFILL_GPU}" "${PROJECTION_GPU}"; do
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
    nvidia-smi -i "${PREFILL_GPU},${PROJECTION_GPU}" \
      --query-gpu=name --format=csv,noheader \
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
  PAP_TOPOLOGY="${PAP_TOPOLOGY}" \
  PAP_PREFILL_GPUS="${PAP_PREFILL_GPUS}" \
  PAP_PROJECTION_GPUS="${PAP_PROJECTION_GPUS}" \
  PAP_VLLM_DTYPE="${PAP_VLLM_DTYPE}" \
  PAP_TP_SIZE="${PAP_TP_SIZE}" \
  PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT}" \
  PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT}" \
  PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY}" \
  PAP_DIRECT_MAILBOX_OUTPUT="${PAP_DIRECT_MAILBOX_OUTPUT}" \
  PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND="${PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND}" \
  PAP_LOCAL_FAST_STREAM_ORDERED="${PAP_LOCAL_FAST_STREAM_ORDERED}" \
  PAP_LOCAL_FAST_SLOT_COUNT="${PAP_LOCAL_FAST_SLOT_COUNT}" \
  PAP_LOCAL_FAST_BATCH_PLAN="${PAP_LOCAL_FAST_BATCH_PLAN}" \
  PAP_DECODE_SLOT_PLAN_CACHE_LIMIT="${PAP_DECODE_SLOT_PLAN_CACHE_LIMIT}" \
  PAP_PREFIX_CACHE_AUDIT="${PAP_PREFIX_CACHE_AUDIT}" \
  PAP_ENABLE_PROMPT_TOKENS_DETAILS="${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" \
  PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS}" \
  PAP_PREFILL_MPS_PERCENT="${PAP_PREFILL_MPS_PERCENT}" \
  PAP_ATTENTION_MPS_PERCENT="${PAP_ATTENTION_MPS_PERCENT}" \
  PAP_STATIC_PREFILL_CHUNKS="${PAP_STATIC_PREFILL_CHUNKS}" \
  PAP_STATIC_ATTENTION_CHUNKS="${PAP_STATIC_ATTENTION_CHUNKS}" \
  PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_STATIC_PREFILL_EXPECTED_SMS}" \
  PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_STATIC_ATTENTION_EXPECTED_SMS}" \
  PAP_ENABLE_MPS=1 \
  PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${REQUIRE_CLEAN}" \
  PAP_BENCH_STRICT_CORRECTNESS_AUDIT="${PAP_BENCH_STRICT_CORRECTNESS_AUDIT}" \
  PAP_BENCH_SESSION_DRAIN_TIMEOUT="${PAP_BENCH_SESSION_DRAIN_TIMEOUT}" \
  PAP_NORTH_STAR_HARDWARE_SIGNATURE="${HARDWARE_SIGNATURE}" \
  PAP_NORTH_STAR_CONVERSATION_ID="${GROUP_RUN_ID}-rep${rep}-conversation" \
  PAP_NORTH_STAR_CACHE_SALT="${GROUP_RUN_ID}-rep${rep}-cache-salt" \
  PAP_MULTITURN_LOAD_ROUNDS="${PAP_MULTITURN_LOAD_ROUNDS}" \
  PAP_MULTITURN_LOAD_CONVERSATIONS="${CONVERSATIONS}" \
  PAP_MULTITURN_LOAD_REQUEST_RATE="${PAP_MULTITURN_LOAD_REQUEST_RATE}" \
  PAP_MULTITURN_APPEND_TOKENS="${PAP_MULTITURN_APPEND_TOKENS}" \
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  INPUT_LEN="${INPUT_LEN}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  QPS="${PAP_MULTITURN_LOAD_REQUEST_RATE}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PAP_MULTITURN_BLOCK_SIZE="${PAP_MULTITURN_BLOCK_SIZE}" \
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
