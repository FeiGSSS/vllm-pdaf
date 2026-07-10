#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
SKILL_DIR="${ROOT_DIR}/.claude/skills/vllm-pap-benchmark"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
PROXY_SCRIPT="${SKILL_DIR}/scripts/nixl_pd_proxy.py"

MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_NAME="${DATASET_NAME:-sonnet}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PREFIX_LEN="${PREFIX_LEN:-50}"
QPS="${QPS:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
CLUSTER_READY_WAIT_SECONDS="${CLUSTER_READY_WAIT_SECONDS:-5}"

RESULTS_ROOT="${RESULTS_ROOT:-/home/fei/research/PD/test/baseline/nixl_disaggregated/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_ROOT}/service_logs}"

PROXY_PORT="${PD_PROXY_PORT:-19410}"
PREFILL_PORT="${PD_PREFILL_PORT:-18100}"
DECODE_PORT="${PD_DECODE_PORT:-19100}"
PREFILL_SIDE_CHANNEL_PORT="${PD_PREFILL_SIDE_CHANNEL_PORT:-18600}"
DECODE_SIDE_CHANNEL_PORT="${PD_DECODE_SIDE_CHANNEL_PORT:-18700}"
VLLM_PREFILL_PORT="${PD_VLLM_PREFILL_PORT:-59000}"
VLLM_DECODE_PORT="${PD_VLLM_DECODE_PORT:-59020}"
PREFILL_GPU="${PD_PREFILL_GPU:-1}"
DECODE_GPU="${PD_DECODE_GPU:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PREFILL_GPU_MEMORY_UTILIZATION="${PD_PREFILL_GPU_MEMORY_UTILIZATION:-0.80}"
DECODE_GPU_MEMORY_UTILIZATION="${PD_DECODE_GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_DTYPE="${PD_VLLM_DTYPE:-float16}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

append_no_proxy() {
  local var_name="$1"
  local current="${!var_name:-}"
  local host
  for host in 127.0.0.1 localhost; do
    case ",${current}," in
      *",${host},"*) ;;
      *) current="${current:+${current},}${host}" ;;
    esac
  done
  printf -v "${var_name}" '%s' "${current}"
  export "${var_name}"
}

append_no_proxy NO_PROXY
append_no_proxy no_proxy

PIDS=()
PGIDS=()

cleanup() {
  local code=$?
  set +e
  local pgid pid
  for pgid in "${PGIDS[@]:-}"; do
    kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
  done
  sleep 2
  for pgid in "${PGIDS[@]:-}"; do
    kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

die() {
  echo "ERROR: $*" >&2
  exit 1
}

ensure_ports_free() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import socket
import sys

busy = []
for raw_port in sys.argv[1:]:
    port = int(raw_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            busy.append(str(port))
    finally:
        sock.close()
if busy:
    raise SystemExit(f"ports already in use: {', '.join(busy)}")
PY
}

wait_for_http() {
  local url="$1"
  local name="$2"
  local started
  started="$(date +%s)"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} is ready at ${url}"
      return 0
    fi
    if (( "$(date +%s)" - started > SERVER_START_TIMEOUT )); then
      die "Timed out waiting for ${name} at ${url}"
    fi
    sleep 2
  done
}

check_children_alive() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -0 "${pid}" >/dev/null 2>&1 \
      || die "managed child exited unexpectedly: pid=${pid}"
  done
}

wait_cluster_stable() {
  local remaining="${CLUSTER_READY_WAIT_SECONDS}"
  while (( remaining > 0 )); do
    check_children_alive
    sleep 1
    remaining=$((remaining - 1))
  done
}

validate_result() {
  NUM_PROMPTS="${NUM_PROMPTS}" "${PYTHON_BIN}" - "$1" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    result = json.load(f)
expected = int(os.environ["NUM_PROMPTS"])
completed = int(result.get("completed", 0))
failed = int(result.get("failed", 0))
if completed != expected or failed != 0:
    raise SystemExit(
        f"incomplete PD result: completed={completed}, failed={failed}, "
        f"expected={expected}"
    )
PY
}

audit_logs() {
  local matches="${RUN_ROOT}/correctness_audit_matches.log"
  local pattern='Traceback|EngineDeadError|KV transfer.*failed|NIXL.*failed'
  rg -n -i "${pattern}" "${RUN_LOG_DIR}" > "${matches}" || true
  local count
  count="$(wc -l < "${matches}" | tr -d ' ')"
  if [[ "${count}" != "0" ]]; then
    {
      printf 'STATUS=failed\n'
      printf 'MATCH_COUNT=%q\n' "${count}"
    } > "${RUN_ROOT}/correctness_audit.env"
    die "PD correctness audit found ${count} error matches"
  fi
  {
    printf 'STATUS=passed\n'
    printf 'MATCH_COUNT=0\n'
  } > "${RUN_ROOT}/correctness_audit.env"
}

write_metadata() {
  local commit short_commit dirty
  commit="$(git rev-parse HEAD)"
  short_commit="$(git rev-parse --short HEAD)"
  dirty=0
  git status --short > "${RUN_ROOT}/git_status.txt"
  git diff --binary > "${RUN_ROOT}/tracked_worktree.patch"
  [[ -s "${RUN_ROOT}/tracked_worktree.patch" ]] && dirty=1
  {
    printf 'MODE=nixl_disaggregated\n'
    printf 'TOPOLOGY=1p1d\n'
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'DATASET_NAME=%q\n' "${DATASET_NAME}"
    printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
    printf 'PREFIX_LEN=%q\n' "${PREFIX_LEN}"
    printf 'NUM_PROMPTS=%q\n' "${NUM_PROMPTS}"
    printf 'INPUT_LENS_CSV=%q\n' "${INPUT_LEN}"
    printf 'OUTPUT_LENS_CSV=%q\n' "${OUTPUT_LEN}"
    printf 'QPS_CSV=%q\n' "${QPS}"
    printf 'BENCH_NUM_WARMUPS=%q\n' "${BENCH_NUM_WARMUPS}"
    printf 'RUN_ROOT=%q\n' "${RUN_ROOT}"
    printf 'PROXY_PORT=%q\n' "${PROXY_PORT}"
    printf 'PREFILL_PORT=%q\n' "${PREFILL_PORT}"
    printf 'DECODE_PORT=%q\n' "${DECODE_PORT}"
    printf 'PREFILL_GPU=%q\n' "${PREFILL_GPU}"
    printf 'DECODE_GPU=%q\n' "${DECODE_GPU}"
    printf 'PREFILL_GPU_MEMORY_UTILIZATION=%q\n' "${PREFILL_GPU_MEMORY_UTILIZATION}"
    printf 'DECODE_GPU_MEMORY_UTILIZATION=%q\n' "${DECODE_GPU_MEMORY_UTILIZATION}"
    printf 'VLLM_DTYPE=%q\n' "${VLLM_DTYPE}"
    printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN}"
    printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "${MAX_NUM_BATCHED_TOKENS}"
    printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS}"
    printf 'VLLM_BIN=%q\n' "${VLLM_BIN}"
    printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
    printf 'VLLM_USE_FLASHINFER_SAMPLER=%q\n' "${VLLM_USE_FLASHINFER_SAMPLER}"
    printf 'NO_PROXY=%q\n' "${NO_PROXY}"
    printf 'no_proxy=%q\n' "${no_proxy}"
    printf 'GIT_COMMIT=%q\n' "${commit}"
    printf 'GIT_COMMIT_SHORT=%q\n' "${short_commit}"
    printf 'GIT_TRACKED_WORKTREE_DIRTY=%q\n' "${dirty}"
  } > "${RUN_ROOT}/effective_config.env"

  RUN_ROOT="${RUN_ROOT}" RUN_ID="${RUN_ID}" MODEL_PATH="${MODEL_PATH}" \
  INPUT_LEN="${INPUT_LEN}" OUTPUT_LEN="${OUTPUT_LEN}" PREFIX_LEN="${PREFIX_LEN}" \
  QPS="${QPS}" NUM_PROMPTS="${NUM_PROMPTS}" GIT_COMMIT="${commit}" \
  GIT_TRACKED_WORKTREE_DIRTY="${dirty}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime

metadata = {
    "mode": "nixl_disaggregated",
    "topology": "1p1d",
    "run_id": os.environ["RUN_ID"],
    "result_root": os.environ["RUN_ROOT"],
    "model_path": os.environ["MODEL_PATH"],
    "input_lens": [os.environ["INPUT_LEN"]],
    "output_lens": [os.environ["OUTPUT_LEN"]],
    "prefix_len": os.environ["PREFIX_LEN"],
    "qps": [os.environ["QPS"]],
    "num_prompts": os.environ["NUM_PROMPTS"],
    "git_commit": os.environ["GIT_COMMIT"],
    "git_tracked_worktree_dirty": (
        os.environ["GIT_TRACKED_WORKTREE_DIRTY"] == "1"
    ),
    "config_dir": "self-contained skill runner",
    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}
with open(
    os.path.join(os.environ["RUN_ROOT"], "run_metadata.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")
PY
}

mkdir -p "${RUN_LOG_DIR}"
ensure_ports_free \
  "${PROXY_PORT}" "${PREFILL_PORT}" "${DECODE_PORT}" \
  "${PREFILL_SIDE_CHANNEL_PORT}" "${DECODE_SIDE_CHANNEL_PORT}" \
  "${VLLM_PREFILL_PORT}" "${VLLM_DECODE_PORT}"
write_metadata

echo "Starting PD Prefill on GPU ${PREFILL_GPU}"
setsid env \
  CUDA_VISIBLE_DEVICES="${PREFILL_GPU}" \
  VLLM_PORT="${VLLM_PREFILL_PORT}" \
  VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT}" \
  VLLM_KV_CACHE_LAYOUT=HND \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "${PREFILL_PORT}" \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype "${VLLM_DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-chunked-prefill \
    --trust-remote-code \
    --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}" \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}' \
    > "${RUN_LOG_DIR}/prefill_0.log" 2>&1 &
PIDS+=("$!")
PGIDS+=("$!")

echo "Starting PD Decode on GPU ${DECODE_GPU}"
setsid env \
  CUDA_VISIBLE_DEVICES="${DECODE_GPU}" \
  VLLM_PORT="${VLLM_DECODE_PORT}" \
  VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNEL_PORT}" \
  VLLM_KV_CACHE_LAYOUT=HND \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "${DECODE_PORT}" \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype "${VLLM_DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    --gpu-memory-utilization "${DECODE_GPU_MEMORY_UTILIZATION}" \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}' \
    > "${RUN_LOG_DIR}/decode_0.log" 2>&1 &
PIDS+=("$!")
PGIDS+=("$!")

wait_for_http "http://127.0.0.1:${PREFILL_PORT}/v1/models" "PD Prefill"
wait_for_http "http://127.0.0.1:${DECODE_PORT}/v1/models" "PD Decode"

echo "Starting bundled PD proxy on port ${PROXY_PORT}"
setsid "${PYTHON_BIN}" "${PROXY_SCRIPT}" \
  --host 127.0.0.1 \
  --port "${PROXY_PORT}" \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports "${PREFILL_PORT}" \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports "${DECODE_PORT}" \
  > "${RUN_LOG_DIR}/proxy.log" 2>&1 &
PIDS+=("$!")
PGIDS+=("$!")

wait_for_http "http://127.0.0.1:${PROXY_PORT}/healthcheck" "PD proxy"
wait_cluster_stable

TAG="1P1D_i${INPUT_LEN}_o${OUTPUT_LEN}_q${QPS}"
echo "=== Running ${TAG} on port ${PROXY_PORT} ==="
timeout "${BENCH_TIMEOUT}" "${VLLM_BIN}" bench serve \
  --backend vllm \
  --model "${MODEL_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --sonnet-input-len "${INPUT_LEN}" \
  --sonnet-output-len "${OUTPUT_LEN}" \
  --sonnet-prefix-len "${PREFIX_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --port "${PROXY_PORT}" \
  --save-result \
  --result-dir "${RUN_ROOT}" \
  --result-filename "${TAG}.json" \
  --request-rate "${QPS}" \
  --num-warmups "${BENCH_NUM_WARMUPS}" \
  2>&1 | tee "${RUN_ROOT}/${TAG}.log"

validate_result "${RUN_ROOT}/${TAG}.json"
audit_logs
echo "RUN_ROOT=${RUN_ROOT}"
