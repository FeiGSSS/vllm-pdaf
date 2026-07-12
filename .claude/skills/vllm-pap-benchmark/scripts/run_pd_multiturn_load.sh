#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
CLIENT="${ROOT_DIR}/benchmarks/multi_turn/pap_pd_multiturn_load_client.py"
AUDITOR="${ROOT_DIR}/benchmarks/multi_turn/pd_multiturn_load_reuse_metrics.py"
FINALIZER="${ROOT_DIR}/benchmarks/multi_turn/finalize_pap_pd_multiturn.py"
COMPARER="${ROOT_DIR}/benchmarks/multi_turn/compare_pap_pd_multiturn_load.py"
PROXY="${ROOT_DIR}/examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
REPETITIONS="${PD_LOAD_REPETITIONS:-1}"
ROUNDS="${PD_LOAD_ROUNDS:-5}"
CONVERSATIONS="${PD_LOAD_CONVERSATIONS:-4}"
REQUEST_RATE="${PD_LOAD_REQUEST_RATE:-2}"
REQUIRE_CLEAN="${PD_LOAD_REQUIRE_CLEAN_TRACKED_WORKTREE:-0}"
PREFILL_CUDA_VISIBLE_DEVICES="${PD_PREFILL_CUDA_VISIBLE_DEVICES:-1}"
DECODE_CUDA_VISIBLE_DEVICES="${PD_DECODE_CUDA_VISIBLE_DEVICES:-2}"

for name in REPETITIONS ROUNDS CONVERSATIONS; do
  value="${!name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer: ${value}" >&2
    exit 2
  fi
done
if [[ "${REPETITIONS}" != "1" && "${REPETITIONS}" != "3" ]]; then
  echo "PD load requires one quick or three formal repetitions" >&2
  exit 2
fi
if [[ "${ROUNDS}" != "5" ]] \
  || [[ "${CONVERSATIONS}" != "1" \
    && "${CONVERSATIONS}" != "2" \
    && "${CONVERSATIONS}" != "4" ]]; then
  echo "PD load is fixed to five rounds and C1/C2/C4" >&2
  exit 2
fi
if [[ "${REQUEST_RATE}" != "2" && "${REQUEST_RATE}" != "2.0" ]]; then
  echo "PD load is fixed to two requests/s: ${REQUEST_RATE}" >&2
  exit 2
fi

for required in "${PYTHON_BIN}" "${VLLM_BIN}"; do
  [[ -x "${required}" ]] || {
    echo "required executable is missing: ${required}" >&2
    exit 1
  }
done
for required in "${CLIENT}" "${AUDITOR}" "${FINALIZER}" "${COMPARER}" \
  "${PROXY}" "${DATASET_PATH}"; do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 1
  }
done

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=0
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"
export UCX_PROTO_EMULATION_ENABLE=n
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

PIDS=()
PGIDS=()

cleanup_current() {
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
  PIDS=()
  PGIDS=()
  set -e
}

on_exit() {
  local code=$?
  cleanup_current
  exit "${code}"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

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

wait_for_http() {
  local url="$1"
  local name="$2"
  local deadline=$((SECONDS + 900))
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for ${name}: ${url}" >&2
      return 1
    fi
    sleep 2
  done
  echo "${name} is ready at ${url}"
}

cd "${ROOT_DIR}"
if [[ "${REQUIRE_CLEAN}" == "1" ]] \
  && { ! git diff --quiet || ! git diff --cached --quiet; }; then
  echo "formal PD load requires a clean tracked worktree" >&2
  exit 1
fi
ensure_gpu_idle
HARDWARE_SIGNATURE="$(hardware_signature)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_SHORT="$(git rev-parse --short HEAD)"
GIT_TRACKED_WORKTREE_DIRTY=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  GIT_TRACKED_WORKTREE_DIRTY=1
fi
GROUP_RUN_ID="${PD_LOAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${GIT_SHORT}_pd_push_load_c${CONVERSATIONS}}"
GROUP_ROOT="${PD_LOAD_RUN_ROOT:-${RESULTS_ROOT}/runs/${GROUP_RUN_ID}}"
mkdir -p "${GROUP_ROOT}"
NIXL_VERSION="$(
  "${PYTHON_BIN}" -c \
    'import importlib.metadata as m; print(m.version("nixl"))'
)"
RESULT_ARGS=()

for (( rep=1; rep<=REPETITIONS; rep++ )); do
  ensure_gpu_idle
  REP_ROOT="${GROUP_ROOT}/rep${rep}"
  LOG_ROOT="${REP_ROOT}/service_logs"
  mkdir -p "${LOG_ROOT}"
  PORT_SHIFT=$(((rep - 1) * 100))
  PREFILL_PORT=$((21100 + PORT_SHIFT))
  DECODE_PORT=$((21200 + PORT_SHIFT))
  PROXY_PORT=$((21300 + PORT_SHIFT))
  PREFILL_SIDE_PORT=$((5660 + PORT_SHIFT))
  DECODE_SIDE_PORT=$((5661 + PORT_SHIFT))
  VLLM_PREFILL_PORT=$((50600 + PORT_SHIFT))
  VLLM_DECODE_PORT=$((50620 + PORT_SHIFT))
  PREFILL_ENGINE_ID="${GROUP_RUN_ID}-rep${rep}-prefill"
  DECODE_ENGINE_ID="${GROUP_RUN_ID}-rep${rep}-decode"

  P_CONFIG="$(printf '%s' \
    '{"kv_connector":"NixlPushConnector","engine_id":"' \
    "${PREFILL_ENGINE_ID}" '","kv_role":"kv_producer",' \
    '"kv_load_failure_policy":"fail","kv_connector_extra_config":{' \
    '"bidirectional_kv_xfer":false,"enable_cross_layers_blocks":"True"}}')"
  D_CONFIG="$(printf '%s' \
    '{"kv_connector":"NixlPushConnector","engine_id":"' \
    "${DECODE_ENGINE_ID}" '","kv_role":"kv_consumer",' \
    '"kv_load_failure_policy":"fail","kv_connector_extra_config":{' \
    '"bidirectional_kv_xfer":false,"enable_cross_layers_blocks":"True"}}')"

  git status --short > "${REP_ROOT}/git_status.txt"
  git diff --binary > "${REP_ROOT}/tracked_worktree.patch"
  git diff --cached --binary > "${REP_ROOT}/tracked_index.patch"
  {
    printf 'MODE=pd\nPD_TRANSFER_MODE=push\nTOPOLOGY=1p1d\n'
    printf 'MODEL_PATH=%q\nDATASET_PATH=%q\n' \
      "${MODEL_PATH}" "${DATASET_PATH}"
    printf 'ROUNDS=%q\nACTIVE_CONVERSATIONS=%q\nREQUEST_RATE=%q\n' \
      "${ROUNDS}" "${CONVERSATIONS}" "${REQUEST_RATE}"
    printf 'DOCUMENT_TOKENS=16000\nAPPEND_TOKENS=120\nOUTPUT_TOKENS=256\n'
    printf 'DTYPE=float16\nMAX_MODEL_LEN=20000\n'
    printf 'MAX_NUM_BATCHED_TOKENS=4096\nMAX_NUM_SEQS=4\n'
    printf 'PREFILL_GPU=1\nDECODE_GPU=2\n'
    printf 'PREFILL_CUDA_VISIBLE_DEVICES=%q\n' \
      "${PREFILL_CUDA_VISIBLE_DEVICES}"
    printf 'DECODE_CUDA_VISIBLE_DEVICES=%q\n' \
      "${DECODE_CUDA_VISIBLE_DEVICES}"
    printf 'VLLM_USE_V2_MODEL_RUNNER=%q\n' \
      "${VLLM_USE_V2_MODEL_RUNNER}"
    printf 'ENABLE_CROSS_LAYERS_BLOCKS=True\n'
    printf 'UCX_TLS=%q\nUCX_NET_DEVICES=%q\n' \
      "${UCX_TLS}" "${UCX_NET_DEVICES}"
    printf 'UCX_RCACHE_MAX_UNRELEASED=%q\n' \
      "${UCX_RCACHE_MAX_UNRELEASED}"
    printf 'UCX_PROTO_EMULATION_ENABLE=%q\n' \
      "${UCX_PROTO_EMULATION_ENABLE}"
    printf 'UCX_PROTO_INFO=%q\nUCX_LOG_LEVEL=%q\n' \
      "${UCX_PROTO_INFO:-}" "${UCX_LOG_LEVEL:-}"
    printf 'NIXL_VERSION=%q\n' "${NIXL_VERSION}"
    printf 'PREFILL_KV_TRANSFER_CONFIG=%q\n' "${P_CONFIG}"
    printf 'DECODE_KV_TRANSFER_CONFIG=%q\n' "${D_CONFIG}"
    printf 'GIT_COMMIT=%q\nGIT_TRACKED_WORKTREE_DIRTY=%q\n' \
      "${GIT_COMMIT}" "${GIT_TRACKED_WORKTREE_DIRTY}"
    printf 'HARDWARE_SIGNATURE=%q\n' "${HARDWARE_SIGNATURE}"
  } > "${REP_ROOT}/effective_config.env"

  echo "Starting PD push Prefill rep ${rep} on GPU 1"
  setsid env \
    CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES}" \
    VLLM_PORT="${VLLM_PREFILL_PORT}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_PORT}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${PREFILL_PORT}" --enforce-eager \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --max-model-len 20000 --max-num-batched-tokens 4096 \
      --max-num-seqs 4 --enable-chunked-prefill --enable-prefix-caching \
      --block-size 16 --gpu-memory-utilization 0.80 \
      --kv-transfer-config "${P_CONFIG}" \
      > "${LOG_ROOT}/prefill.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")

  echo "Starting PD push Decode rep ${rep} on GPU 2"
  setsid env \
    CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES}" \
    VLLM_PORT="${VLLM_DECODE_PORT}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_PORT}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${DECODE_PORT}" --enforce-eager \
      --generation-config vllm --dtype float16 --tensor-parallel-size 1 \
      --max-model-len 20000 --max-num-batched-tokens 4096 \
      --max-num-seqs 4 --enable-chunked-prefill --enable-prefix-caching \
      --block-size 16 --gpu-memory-utilization 0.80 \
      --kv-transfer-config "${D_CONFIG}" \
      > "${LOG_ROOT}/decode.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")

  wait_for_http "http://127.0.0.1:${PREFILL_PORT}/health" "PD Prefill"
  wait_for_http "http://127.0.0.1:${DECODE_PORT}/health" "PD Decode"

  echo "Starting official NIXL push proxy rep ${rep}"
  setsid "${PYTHON_BIN}" "${PROXY}" \
    --model "${MODEL_PATH}" \
    --prefill "127.0.0.1:${PREFILL_PORT}" \
    --decode "127.0.0.1:${DECODE_PORT}" \
    --prefill-engine-id "${PREFILL_ENGINE_ID}" \
    --prefill-kv-host 127.0.0.1 \
    --prefill-side-channel-port "${PREFILL_SIDE_PORT}" \
    --prefill-tp-size 1 --port "${PROXY_PORT}" \
    > "${LOG_ROOT}/proxy.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
  wait_for_http "http://127.0.0.1:${PROXY_PORT}/status" "PD push proxy"
  sleep 5

  "${PYTHON_BIN}" "${CLIENT}" \
    --base-url "http://127.0.0.1:${PROXY_PORT}" \
    --model "${MODEL_PATH}" --corpus "${DATASET_PATH}" \
    --result "${REP_ROOT}/result.json" --architecture pd \
    --topology 1p1d \
    --conversation-id-prefix "${GROUP_RUN_ID}-rep${rep}-conversation" \
    --cache-salt-prefix "${GROUP_RUN_ID}-rep${rep}-cache-salt" \
    --hardware-signature "${HARDWARE_SIGNATURE}" \
    --git-commit "${GIT_COMMIT}" \
    --git-tracked-worktree-dirty "${GIT_TRACKED_WORKTREE_DIRTY}" \
    --offload-exec-transport nixl-push \
    --direct-mailbox-output 0 --unified-md-fast-key 0 \
    --document-tokens 16000 --append-tokens 120 --output-tokens 256 \
    --rounds "${ROUNDS}" --active-conversations "${CONVERSATIONS}" \
    --request-rate "${REQUEST_RATE}" --block-size 16 --dtype float16 \
    --tensor-parallel-size 1 --max-model-len 20000 \
    --max-num-batched-tokens 4096 --max-num-seqs 4 \
    2>&1 | tee "${REP_ROOT}/client.log"

  sleep 2
  curl -fsS "http://127.0.0.1:${PREFILL_PORT}/metrics" \
    -o "${REP_ROOT}/prefill_metrics.prom"
  curl -fsS "http://127.0.0.1:${DECODE_PORT}/metrics" \
    -o "${REP_ROOT}/decode_metrics.prom"
  "${PYTHON_BIN}" "${AUDITOR}" \
    --result "${REP_ROOT}/result.json" \
    --prefill-metrics "${REP_ROOT}/prefill_metrics.prom" \
    --decode-metrics "${REP_ROOT}/decode_metrics.prom" \
    --effective-config "${REP_ROOT}/effective_config.env"

  if rg -n -i \
    'CUDA out of memory|EngineDeadError|Traceback|NIXL.*failed|NIXL_ERR' \
    "${LOG_ROOT}" > "${REP_ROOT}/correctness_audit_matches.log"; then
    printf 'STATUS=failed\n' > "${REP_ROOT}/correctness_audit.env"
    echo "PD load correctness audit failed in rep ${rep}" >&2
    exit 1
  fi
  : > "${REP_ROOT}/correctness_audit_matches.log"
  printf 'STATUS=passed\nMATCH_COUNT=0\n' \
    > "${REP_ROOT}/correctness_audit.env"

  "${PYTHON_BIN}" "${FINALIZER}" \
    --result "${REP_ROOT}/result.json" --architecture pd \
    --passed-gate pd_reuse_metrics --passed-gate correctness_logs \
    --artifact "proxy_log=${LOG_ROOT}/proxy.log" \
    --artifact "prefill_metrics=${REP_ROOT}/prefill_metrics.prom" \
    --artifact "decode_metrics=${REP_ROOT}/decode_metrics.prom" \
    --artifact "effective_config=${REP_ROOT}/effective_config.env" \
    --artifact "correctness_logs=${REP_ROOT}/correctness_audit.env" \
    --artifact "tracked_worktree_patch=${REP_ROOT}/tracked_worktree.patch" \
    --artifact "tracked_index_patch=${REP_ROOT}/tracked_index.patch"
  RESULT_ARGS+=(--result "${REP_ROOT}/result.json")
  cleanup_current
done

"${PYTHON_BIN}" "${COMPARER}" aggregate \
  "${RESULT_ARGS[@]}" --output "${GROUP_ROOT}/aggregate.json"
echo "PD_LOAD_RUN_ROOT=${GROUP_ROOT}"
