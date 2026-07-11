#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
CLIENT="${ROOT_DIR}/benchmarks/multi_turn/pap_pd_multiturn_client.py"
COMPARER="${ROOT_DIR}/benchmarks/multi_turn/compare_pap_pd_multiturn.py"
OFFICIAL_PROXY="${ROOT_DIR}/examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/test/baseline/pap/results}"
REPETITIONS=3

for required in "${PYTHON_BIN}" "${VLLM_BIN}"; do
  [[ -x "${required}" ]] || {
    echo "required executable is missing: ${required}" >&2
    exit 1
  }
done
for required in "${CLIENT}" "${COMPARER}" "${OFFICIAL_PROXY}"; do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 1
  }
done

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=1
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
    nvidia-smi -i 1,2 --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
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

audit_and_close_pd_result() {
  local result_path="$1"
  local proxy_log="$2"
  local prefill_metrics="$3"
  local decode_metrics="$4"
  "${PYTHON_BIN}" - \
    "${result_path}" \
    "${proxy_log}" \
    "${prefill_metrics}" \
    "${decode_metrics}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
proxy_log = Path(sys.argv[2]).read_text(encoding="utf-8")
prefill_metrics_text = Path(sys.argv[3]).read_text(encoding="utf-8")
decode_metrics_text = Path(sys.argv[4]).read_text(encoding="utf-8")

metric_pattern = re.compile(
    r'vllm:prompt_tokens_by_source_total\{[^\n]*?source="([^"]+)"'
    r'[^\n]*?\}\s+([\d.eE+\-]+)'
)


def token_sources(metrics_text: str) -> dict[str, float]:
    sources = {
        "local_compute": 0.0,
        "local_cache_hit": 0.0,
        "external_kv_transfer": 0.0,
    }
    for source, raw_value in metric_pattern.findall(metrics_text):
        if source in sources:
            sources[source] += float(raw_value)
    return sources


proxy_misses = len(re.findall(r"cache MISS", proxy_log))
proxy_hits = len(re.findall(r"cache HIT", proxy_log))
if proxy_misses != 2 or proxy_hits != 0:
    raise SystemExit(
        "official streaming PD cache semantics changed: "
        f"misses={proxy_misses}, hits={proxy_hits}"
    )

prefill_sources = token_sources(prefill_metrics_text)
decode_sources = token_sources(decode_metrics_text)
if prefill_sources["local_cache_hit"] < 16:
    raise SystemExit(
        "PD Prefill did not reuse a complete local cache block: "
        f"{prefill_sources}"
    )
if prefill_sources["external_kv_transfer"] != 0:
    raise SystemExit(
        "PD Prefill unexpectedly received external KV in the official "
        f"streaming path: {prefill_sources}"
    )
if decode_sources["local_cache_hit"] < 16:
    raise SystemExit(
        "PD Decode did not reuse a complete local cache block: "
        f"{decode_sources}"
    )
if decode_sources["external_kv_transfer"] < 16:
    raise SystemExit(
        "PD Decode did not receive Prefill KV via NIXL: "
        f"{decode_sources}"
    )

result = json.loads(path.read_text(encoding="utf-8"))
cache = result.get("cache_validation") or {}
if int(cache.get("decode_derived_hit_tokens", 0)) < 16:
    raise SystemExit(f"PD result has no Decode-derived LCP: {cache}")
status = "official_streaming_metrics_passed"
cache["status"] = status
result["cache_validation"] = cache
result["pd_reuse_validation"] = {
    "status": status,
    "mode": "official_streaming_local_cache_plus_p_to_d",
    "proxy_cache_misses": proxy_misses,
    "proxy_cache_hits": proxy_hits,
    "prefill_prompt_tokens_by_source": prefill_sources,
    "decode_prompt_tokens_by_source": decode_sources,
}
result["validity"] = {
    "status": "passed",
    "cache_gate": status,
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

cd "${ROOT_DIR}"
git diff --quiet || {
  echo "PD reference bootstrap requires a clean tracked worktree" >&2
  exit 1
}
ensure_gpu_idle
HARDWARE_SIGNATURE="$(hardware_signature)"
GIT_SHORT="$(git rev-parse --short HEAD)"
GROUP_RUN_ID="${PD_NORTH_STAR_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${GIT_SHORT}_pd_multiturn_formal}"
GROUP_ROOT="${PD_NORTH_STAR_RUN_ROOT:-${RESULTS_ROOT}/runs/${GROUP_RUN_ID}}"
mkdir -p "${GROUP_ROOT}"

P_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"bidirectional_kv_xfer":true,"kv_recompute_threshold":64,"decoder_kv_blocks_ttl":480}}'
D_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"bidirectional_kv_xfer":true,"kv_recompute_threshold":64,"decoder_kv_blocks_ttl":480}}'
RESULT_ARGS=()

for (( rep=1; rep<=REPETITIONS; rep++ )); do
  ensure_gpu_idle
  REP_ROOT="${GROUP_ROOT}/rep${rep}"
  LOG_ROOT="${REP_ROOT}/service_logs"
  mkdir -p "${LOG_ROOT}"
  PORT_SHIFT=$(((rep - 1) * 100))
  PREFILL_PORT=$((18100 + PORT_SHIFT))
  DECODE_PORT=$((18200 + PORT_SHIFT))
  PROXY_PORT=$((18300 + PORT_SHIFT))
  PREFILL_SIDE_PORT=$((5610 + PORT_SHIFT))
  DECODE_SIDE_PORT=$((5611 + PORT_SHIFT))
  VLLM_PREFILL_PORT=$((50100 + PORT_SHIFT))
  VLLM_DECODE_PORT=$((50120 + PORT_SHIFT))

  git status --short > "${REP_ROOT}/git_status.txt"
  git diff --binary > "${REP_ROOT}/tracked_worktree.patch"
  {
    printf 'MODE=pd\n'
    printf 'TOPOLOGY=1p1d\n'
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
    printf 'PREFILL_GPU=1\nDECODE_GPU=2\n'
    printf 'DTYPE=float16\nMAX_MODEL_LEN=20000\n'
    printf 'MAX_NUM_BATCHED_TOKENS=4096\nMAX_NUM_SEQS=2\n'
    printf 'GIT_COMMIT=%q\n' "$(git rev-parse HEAD)"
    printf 'HARDWARE_SIGNATURE=%q\n' "${HARDWARE_SIGNATURE}"
  } > "${REP_ROOT}/effective_config.env"

  echo "Starting official PD Prefill rep ${rep} on GPU 1"
  setsid env \
    CUDA_VISIBLE_DEVICES=1 \
    UCX_NET_DEVICES=all \
    VLLM_PORT="${VLLM_PREFILL_PORT}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_PORT}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PREFILL_PORT}" \
      --enforce-eager \
      --generation-config vllm \
      --dtype float16 \
      --tensor-parallel-size 1 \
      --max-model-len 20000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 2 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --block-size 16 \
      --gpu-memory-utilization 0.80 \
      --kv-transfer-config "${P_CONFIG}" \
      > "${LOG_ROOT}/prefill.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")

  echo "Starting official PD Decode rep ${rep} on GPU 2"
  setsid env \
    CUDA_VISIBLE_DEVICES=2 \
    UCX_NET_DEVICES=all \
    VLLM_PORT="${VLLM_DECODE_PORT}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_PORT}" \
    VLLM_KV_CACHE_LAYOUT=HND \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${DECODE_PORT}" \
      --enforce-eager \
      --generation-config vllm \
      --dtype float16 \
      --tensor-parallel-size 1 \
      --max-model-len 20000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 2 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --block-size 16 \
      --gpu-memory-utilization 0.80 \
      --kv-transfer-config "${D_CONFIG}" \
      > "${LOG_ROOT}/decode.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")

  wait_for_http "http://127.0.0.1:${PREFILL_PORT}/health" "PD Prefill"
  wait_for_http "http://127.0.0.1:${DECODE_PORT}/health" "PD Decode"

  echo "Starting unchanged official PD multi-turn proxy rep ${rep}"
  setsid "${PYTHON_BIN}" "${OFFICIAL_PROXY}" \
    --host 127.0.0.1 \
    --port "${PROXY_PORT}" \
    --prefiller-host 127.0.0.1 \
    --prefiller-port "${PREFILL_PORT}" \
    --decoder-host 127.0.0.1 \
    --decoder-port "${DECODE_PORT}" \
    > "${LOG_ROOT}/proxy.log" 2>&1 &
  PIDS+=("$!")
  PGIDS+=("$!")
  wait_for_http "http://127.0.0.1:${PROXY_PORT}/health" "PD proxy"
  sleep 5

  "${PYTHON_BIN}" "${CLIENT}" \
    --base-url "http://127.0.0.1:${PROXY_PORT}" \
    --model "${MODEL_PATH}" \
    --corpus "${DATASET_PATH}" \
    --result "${REP_ROOT}/result.json" \
    --architecture "pd" \
    --topology "1p1d" \
    --conversation-id "${GROUP_RUN_ID}-rep${rep}-conversation-0" \
    --cache-salt "${GROUP_RUN_ID}-rep${rep}-cache-salt" \
    --hardware-signature "${HARDWARE_SIGNATURE}" \
    --document-tokens 16000 \
    --append-tokens 120 \
    --output-tokens 256 \
    --block-size 16 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 20000 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 2 \
    2>&1 | tee "${REP_ROOT}/client.log"

  curl -fsS "http://127.0.0.1:${PREFILL_PORT}/metrics" \
    -o "${REP_ROOT}/prefill_metrics.prom"
  curl -fsS "http://127.0.0.1:${DECODE_PORT}/metrics" \
    -o "${REP_ROOT}/decode_metrics.prom"
  audit_and_close_pd_result \
    "${REP_ROOT}/result.json" \
    "${LOG_ROOT}/proxy.log" \
    "${REP_ROOT}/prefill_metrics.prom" \
    "${REP_ROOT}/decode_metrics.prom"
  if rg -n -i 'CUDA out of memory|EngineDeadError|Traceback|NIXL.*failed' \
    "${LOG_ROOT}" > "${REP_ROOT}/correctness_audit_matches.log"; then
    printf 'STATUS=failed\n' > "${REP_ROOT}/correctness_audit.env"
    echo "PD correctness audit failed in rep ${rep}" >&2
    exit 1
  fi
  : > "${REP_ROOT}/correctness_audit_matches.log"
  printf 'STATUS=passed\nMATCH_COUNT=0\n' \
    > "${REP_ROOT}/correctness_audit.env"
  RESULT_ARGS+=(--result "${REP_ROOT}/result.json")
  cleanup_current
done

AGGREGATE_PATH="${GROUP_ROOT}/aggregate.json"
"${PYTHON_BIN}" "${COMPARER}" aggregate \
  "${RESULT_ARGS[@]}" \
  --output "${AGGREGATE_PATH}"
cp "${AGGREGATE_PATH}" /tmp/pap_pd_multiturn_reference_candidate.json

echo "PD_NORTH_STAR_RUN_ROOT=${GROUP_ROOT}"
echo "PD_REFERENCE_CANDIDATE=/tmp/pap_pd_multiturn_reference_candidate.json"
