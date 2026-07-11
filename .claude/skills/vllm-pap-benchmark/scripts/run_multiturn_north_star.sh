#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
RUNNER="${ROOT_DIR}/.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh"
COMPARER="${ROOT_DIR}/benchmarks/multi_turn/compare_pap_pd_multiturn.py"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
PROFILE_ID="qwen3_8b_chat_16k_2turn_o256_c1_v1"
REFERENCE_DIR="${PAP_NORTH_STAR_REFERENCE_DIR:-${ROOT_DIR}/test/baseline/pap/references/${PROFILE_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
MODE="${1:-quick}"

case "${MODE}" in
  quick)
    REPETITIONS=1
    REQUIRE_CLEAN=0
    ;;
  formal)
    REPETITIONS=3
    REQUIRE_CLEAN=1
    ;;
  *)
    echo "usage: $0 [quick|formal]" >&2
    exit 2
    ;;
esac

[[ -x "${PYTHON_BIN}" ]] || {
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
}
[[ -f "${RUNNER}" ]] || {
  echo "missing PAP runner: ${RUNNER}" >&2
  exit 1
}
[[ -f "${COMPARER}" ]] || {
  echo "missing comparer: ${COMPARER}" >&2
  exit 1
}

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
    nvidia-smi -i 1,2 --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  )
  [[ "${#names[@]}" == "2" ]] || {
    echo "failed to read GPU 1/2 names" >&2
    return 1
  }
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
GROUP_RUN_ID="${PAP_NORTH_STAR_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${GIT_SHORT}_pap_multiturn_${MODE}}"
GROUP_ROOT="${PAP_NORTH_STAR_RUN_ROOT:-${RESULTS_ROOT}/runs/${GROUP_RUN_ID}}"
mkdir -p "${GROUP_ROOT}"

RESULT_ARGS=()
for (( rep=1; rep<=REPETITIONS; rep++ )); do
  ensure_gpu_idle
  REP_ROOT="${GROUP_ROOT}/rep${rep}"
  PORT_SHIFT=$(((rep - 1) * 100))
  PAP_BENCH_CLIENT_MODE=multiturn_north_star \
  PAP_TOPOLOGY=1pa1p \
  PAP_PREFILL_GPUS=1 \
  PAP_PROJECTION_GPUS=2 \
  PAP_VLLM_DTYPE=float16 \
  PAP_PREFIX_CACHE_AUDIT=0 \
  PAP_ENABLE_PROMPT_TOKENS_DETAILS=1 \
  PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=256 \
  PAP_PREFILL_MPS_PERCENT=70 \
  PAP_ATTENTION_MPS_PERCENT=30 \
  PAP_ENABLE_MPS=1 \
  PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${REQUIRE_CLEAN}" \
  PAP_BENCH_STRICT_CORRECTNESS_AUDIT=1 \
  PAP_BENCH_SESSION_DRAIN_TIMEOUT=30 \
  PAP_NORTH_STAR_HARDWARE_SIGNATURE="${HARDWARE_SIGNATURE}" \
  PAP_NORTH_STAR_CONVERSATION_ID="${GROUP_RUN_ID}-rep${rep}-conversation-0" \
  PAP_NORTH_STAR_CACHE_SALT="${GROUP_RUN_ID}-rep${rep}-cache-salt" \
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  INPUT_LEN=16000 \
  OUTPUT_LEN=256 \
  MAX_MODEL_LEN=20000 \
  MAX_NUM_BATCHED_TOKENS=4096 \
  MAX_NUM_SEQS=2 \
  CLUSTER_READY_WAIT_SECONDS=5 \
  SERVER_START_TIMEOUT=900 \
  BENCH_TIMEOUT=900 \
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

AGGREGATE_PATH="${GROUP_ROOT}/aggregate.json"
"${PYTHON_BIN}" "${COMPARER}" aggregate \
  "${RESULT_ARGS[@]}" \
  --output "${AGGREGATE_PATH}"

if [[ "${MODE}" == "formal" ]]; then
  cp "${AGGREGATE_PATH}" /tmp/pap_multiturn_reference_candidate.json
fi

PD_REFERENCE="${REFERENCE_DIR}/pd_reference.json"
PAP_REFERENCE="${REFERENCE_DIR}/pap_reference.json"
if [[ -f "${PD_REFERENCE}" && -f "${PAP_REFERENCE}" ]]; then
  "${PYTHON_BIN}" "${COMPARER}" compare \
    --candidate "${AGGREGATE_PATH}" \
    --pd-reference "${PD_REFERENCE}" \
    --pap-reference "${PAP_REFERENCE}" \
    --output-json "${GROUP_ROOT}/comparison.json" \
    --output-markdown "${GROUP_ROOT}/report.md"
else
  printf 'STATUS=uninitialized\nPD_REFERENCE=%q\nPAP_REFERENCE=%q\n' \
    "${PD_REFERENCE}" "${PAP_REFERENCE}" \
    > "${GROUP_ROOT}/comparison_uninitialized.env"
  echo "References are not initialized; aggregate candidate was preserved."
fi

echo "NORTH_STAR_RUN_ROOT=${GROUP_ROOT}"
