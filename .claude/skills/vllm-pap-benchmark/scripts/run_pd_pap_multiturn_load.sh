#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PD_RUNNER="${ROOT_DIR}/.claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh"
PAP_RUNNER="${ROOT_DIR}/.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh"
COMPARER="${ROOT_DIR}/benchmarks/multi_turn/compare_pap_pd_multiturn_load.py"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
MODE="${1:-quick}"
LOAD_SHAPE="${2:-c4}"

case "${MODE}" in
  quick)
    REPETITIONS=1
    REQUIRE_CLEAN=0
    ORDER=(pd pap)
    ;;
  formal)
    REPETITIONS=3
    REQUIRE_CLEAN=1
    ORDER=(pd pap pap pd pd pap)
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

for required in "${PD_RUNNER}" "${PAP_RUNNER}" "${COMPARER}"; do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 1
  }
done
[[ -x "${PYTHON_BIN}" ]] || {
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
}

cd "${ROOT_DIR}"
if [[ "${REQUIRE_CLEAN}" == "1" ]] \
  && { ! git diff --quiet || ! git diff --cached --quiet; }; then
  echo "formal PD/PAP load requires a clean tracked worktree" >&2
  exit 1
fi

GIT_SHORT="$(git rev-parse --short HEAD)"
GROUP_RUN_ID="${PD_PAP_LOAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${GIT_SHORT}_pd_pap_load_${LOAD_SHAPE}_${MODE}}"
GROUP_ROOT="${PD_PAP_LOAD_RUN_ROOT:-${RESULTS_ROOT}/runs/${GROUP_RUN_ID}}"
mkdir -p "${GROUP_ROOT}/pd" "${GROUP_ROOT}/pap"
PD_RESULT_ARGS=()
PAP_RESULT_ARGS=()
PD_INDEX=0
PAP_INDEX=0

for architecture in "${ORDER[@]}"; do
  if [[ "${architecture}" == "pd" ]]; then
    PD_INDEX=$((PD_INDEX + 1))
    RUN_ROOT="${GROUP_ROOT}/pd/run${PD_INDEX}"
    PD_LOAD_REPETITIONS=1 \
    PD_LOAD_ROUNDS=5 \
    PD_LOAD_CONVERSATIONS="${CONVERSATIONS}" \
    PD_LOAD_REQUEST_RATE=2 \
    PD_LOAD_REQUIRE_CLEAN_TRACKED_WORKTREE="${REQUIRE_CLEAN}" \
    PD_LOAD_RUN_ID="${GROUP_RUN_ID}_pd${PD_INDEX}" \
    PD_LOAD_RUN_ROOT="${RUN_ROOT}" \
    bash "${PD_RUNNER}"
    PD_RESULT_ARGS+=(--result "${RUN_ROOT}/rep1/result.json")
  else
    PAP_INDEX=$((PAP_INDEX + 1))
    RUN_ROOT="${GROUP_ROOT}/pap/run${PAP_INDEX}"
    PAP_LOAD_REPETITIONS=1 \
    PAP_LOAD_REQUIRE_CLEAN_TRACKED_WORKTREE="${REQUIRE_CLEAN}" \
    PAP_LOAD_RUN_ID="${GROUP_RUN_ID}_pap${PAP_INDEX}" \
    PAP_LOAD_RUN_ROOT="${RUN_ROOT}" \
    bash "${PAP_RUNNER}" quick "${LOAD_SHAPE}"
    PAP_RESULT_ARGS+=(--result "${RUN_ROOT}/rep1/result.json")
  fi
done

[[ "${PD_INDEX}" == "${REPETITIONS}" \
  && "${PAP_INDEX}" == "${REPETITIONS}" ]] || {
  echo "internal repetition accounting mismatch" >&2
  exit 1
}

"${PYTHON_BIN}" "${COMPARER}" aggregate \
  "${PD_RESULT_ARGS[@]}" --output "${GROUP_ROOT}/pd_aggregate.json"
"${PYTHON_BIN}" "${COMPARER}" aggregate \
  "${PAP_RESULT_ARGS[@]}" --output "${GROUP_ROOT}/pap_aggregate.json"
"${PYTHON_BIN}" "${COMPARER}" compare \
  --pd "${GROUP_ROOT}/pd_aggregate.json" \
  --pap "${GROUP_ROOT}/pap_aggregate.json" \
  --output-json "${GROUP_ROOT}/comparison.json" \
  --output-markdown "${GROUP_ROOT}/report.md"

{
  printf 'STATUS=passed\nMODE=%q\nLOAD_SHAPE=%q\n' \
    "${MODE}" "${LOAD_SHAPE}"
  printf 'ROUNDS=5\nACTIVE_CONVERSATIONS=%q\nREQUEST_RATE=2\n' \
    "${CONVERSATIONS}"
  printf 'DOCUMENT_TOKENS=16000\nAPPEND_TOKENS=120\nOUTPUT_TOKENS=256\n'
  printf 'PD_TRANSFER_MODE=nixl-push\nPAP_TRANSFER_MODE=local-fast\n'
  printf 'REPETITIONS_PER_ARCHITECTURE=%q\n' "${REPETITIONS}"
} > "${GROUP_ROOT}/testbed.env"
echo "PD_PAP_LOAD_RUN_ROOT=${GROUP_ROOT}"
