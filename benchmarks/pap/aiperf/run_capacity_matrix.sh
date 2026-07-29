#!/usr/bin/env bash
set -euo pipefail

# PAP/PD/DP randomized-length comparison using AIPerf workloads.

ROOT_DIR="${PAP_ROOT:-/home/fei/research/PD/vllm-pap}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
CORPUS_PATH="${DATASET_PATH:-/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt}"
AIPERF_ROOT="${AIPERF_ROOT:-/home/fei/research/PD/refer_codes/aiperf}"
AIPERF_BIN="${AIPERF_BIN:-${AIPERF_ROOT}/.venv/bin/aiperf}"
AIPERF_PYTHON="${AIPERF_PYTHON:-${AIPERF_ROOT}/.venv/bin/python}"
PAP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
PD_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_pd_multiturn_topology.sh"
DP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_dp_multiturn.sh"
DATASET_GENERATOR="${ROOT_DIR}/benchmarks/pap/aiperf/generate_multiturn_dataset.py"

EXPERIMENTS_ROOT="${PAP_EXPERIMENTS_ROOT:-${ROOT_DIR}/benchmarks/pap/experiments}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXPERIMENTS_ROOT}/_staging}"
OUTPUT_TOKENS="${PAP_CAPACITY_OUTPUT_TOKENS_MEAN:-${PAP_CAPACITY_OUTPUT_TOKENS:-16}}"
OUTPUT_TOKENS_MEDIAN="${PAP_CAPACITY_OUTPUT_TOKENS_MEDIAN:-$((OUTPUT_TOKENS * 15 / 16))}"
OUTPUT_TOKENS_MIN="${PAP_CAPACITY_OUTPUT_TOKENS_MIN:-$((OUTPUT_TOKENS / 2))}"
OUTPUT_TOKENS_MAX="${PAP_CAPACITY_OUTPUT_TOKENS_MAX:-$((OUTPUT_TOKENS * 2))}"
RANDOM_SEED="${PAP_CAPACITY_RANDOM_SEED:-42}"
MATRIX_ID="${PAP_CAPACITY_MATRIX_ID:-$(date +%Y%m%d_%H%M%S)_aiperf_capacity_random_o${OUTPUT_TOKENS}}"
MATRIX_ROOT="${PAP_CAPACITY_MATRIX_ROOT:-${RESULTS_ROOT}/capacity/${MATRIX_ID}}"
ARCHITECTURES_CSV="${PAP_CAPACITY_ARCHITECTURES:-dp_8,pd_6p2d,pap_6pa2p}"
TOTAL_SESSIONS="${PAP_CAPACITY_SESSIONS:-128}"
GPU_COUNT="${PAP_CAPACITY_GPU_COUNT:-8}"
DEFAULT_POINTS_CSV="${PAP_CAPACITY_POINTS:-16,24,32,48}"
REPETITIONS="${PAP_CAPACITY_REPETITIONS:-1}"
RESUME="${PAP_CAPACITY_RESUME:-1}"
WAIT_FOR_GPUS="${PAP_CAPACITY_WAIT_FOR_GPUS:-1}"
RESTART_BETWEEN_POINTS="${PAP_CAPACITY_RESTART_BETWEEN_POINTS:-1}"
GPU_IDLE_STABILITY_SECONDS="${PAP_CAPACITY_GPU_IDLE_STABILITY_SECONDS:-15}"
AIPERF_SWEEP_COOLDOWN_SECONDS="${PAP_CAPACITY_SWEEP_COOLDOWN_SECONDS:-30}"
AIPERF_PROFILE_RUN_COOLDOWN_SECONDS="${PAP_CAPACITY_PROFILE_RUN_COOLDOWN_SECONDS:-30}"
DEFAULT_AIPERF_GOODPUT_SLO="time_to_first_token:10000 inter_token_latency:75"
AIPERF_GOODPUT_SLO="${PAP_CAPACITY_GOODPUT_SLO:-${DEFAULT_AIPERF_GOODPUT_SLO}}"
EXECUTION_MODE="${PAP_CAPACITY_EXECUTION_MODE:-eager}"
PAP_ROUTING_POLICY="${PAP_CAPACITY_PAP_ROUTING_POLICY:-conversation_affinity}"
PAP_MIGRATION_MIN_PEAK_GAIN_RATIO="${PAP_CAPACITY_PAP_MIGRATION_MIN_PEAK_GAIN_RATIO:-0.30}"
SLO_STRICT="${PAP_CAPACITY_SLO_STRICT:-TTFT<=5000ms,ITL<=50ms,good>=0.95}"
SLO_STANDARD="${PAP_CAPACITY_SLO_STANDARD:-TTFT<=10000ms,ITL<=75ms,good>=0.95}"
SLO_RELAXED="${PAP_CAPACITY_SLO_RELAXED:-TTFT<=20000ms,ITL<=100ms,good>=0.95}"

TURNS="${PAP_CAPACITY_TURNS:-5}"
DATASET_SESSION_PREFIX="pap-pd-dp-s${TOTAL_SESSIONS}-t${TURNS}-seed${RANDOM_SEED}"
DOCUMENT_TOKENS="${PAP_CAPACITY_DOCUMENT_TOKENS_MEAN:-4096}"
DOCUMENT_TOKENS_MEDIAN="${PAP_CAPACITY_DOCUMENT_TOKENS_MEDIAN:-4000}"
DOCUMENT_TOKENS_MIN="${PAP_CAPACITY_DOCUMENT_TOKENS_MIN:-2048}"
DOCUMENT_TOKENS_MAX="${PAP_CAPACITY_DOCUMENT_TOKENS_MAX:-5632}"
# The upper bound truncates the log-normal tail. With seed 42 and 128x5
# requests, these parameters sample an append mean near 0.7K (about 44:1).
APPEND_TOKENS="${PAP_CAPACITY_APPEND_TOKENS_MEAN:-1100}"
APPEND_TOKENS_MEDIAN="${PAP_CAPACITY_APPEND_TOKENS_MEDIAN:-400}"
APPEND_TOKENS_MIN="${PAP_CAPACITY_APPEND_TOKENS_MIN:-4}"
APPEND_TOKENS_MAX="${PAP_CAPACITY_APPEND_TOKENS_MAX:-2125}"
SAMPLED_MEAN_TOLERANCE="${PAP_CAPACITY_SAMPLED_MEAN_TOLERANCE:-0.40}"
THINK_TIME_MS="${PAP_CAPACITY_THINK_TIME_MS:-1000}"
TOOL_TIME_MS="${PAP_CAPACITY_TOOL_TIME_MS:-300}"
TOOL_EVERY="${PAP_CAPACITY_TOOL_EVERY:-3}"
MAX_MODEL_LEN="${PAP_CAPACITY_MAX_MODEL_LEN:-32768}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PAP_CAPACITY_PREFILL_MAX_NUM_BATCHED_TOKENS:-32768}"
DECODE_MAX_NUM_BATCHED_TOKENS="${PAP_CAPACITY_DECODE_MAX_NUM_BATCHED_TOKENS:-256}"
MAX_NUM_SEQS="${PAP_CAPACITY_MAX_NUM_SEQS:-256}"
PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90
PD_GPU_MEMORY_UTILIZATION=0.90
PAP_PREFILL_CHUNKS="${PAP_CAPACITY_PAP_PREFILL_CHUNKS:-18}"
PAP_ATTENTION_CHUNKS="${PAP_CAPACITY_PAP_ATTENTION_CHUNKS:-5}"
PAP_PREFILL_SMS="${PAP_CAPACITY_PAP_PREFILL_EXPECTED_SMS:-$((PAP_PREFILL_CHUNKS * 4))}"
PAP_ATTENTION_SMS="${PAP_CAPACITY_PAP_ATTENTION_EXPECTED_SMS:-$((PAP_ATTENTION_CHUNKS * 4))}"
PAP_PREFILL_PERCENT="${PAP_CAPACITY_PAP_PREFILL_PERCENT:-80}"
PAP_ATTENTION_PERCENT="${PAP_CAPACITY_PAP_ATTENTION_PERCENT:-20}"
REQUEST_TIMEOUT_SECONDS=600
PAP_RUN_TIMEOUT_SECONDS=3600
SUMMARY_RUNNER="${ROOT_DIR}/benchmarks/pap/aiperf/summarize_capacity_run.py"

IFS=, read -r -a ARCHITECTURES <<< "${ARCHITECTURES_CSV}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

points_for_architecture() {
  local architecture="$1"
  local suffix="${architecture^^}"
  local variable="PAP_CAPACITY_${suffix}_POINTS"
  printf '%s' "${!variable:-${DEFAULT_POINTS_CSV}}"
}

for required in "${PYTHON_BIN}" "${AIPERF_BIN}" "${AIPERF_PYTHON}" "${MODEL_PATH}" \
  "${CORPUS_PATH}" "${PAP_RUNNER}" "${PD_RUNNER}" "${DP_RUNNER}" \
  "${DATASET_GENERATOR}"; do
  [[ -e "${required}" ]] || die "required path is missing: ${required}"
done
[[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]] \
  || die "PAP_CAPACITY_REPETITIONS must be positive"
(( REPETITIONS <= 10 )) \
  || die "PAP_CAPACITY_REPETITIONS exceeds AIPerf's limit of 10"
[[ "${TOTAL_SESSIONS}" =~ ^[1-9][0-9]*$ ]] \
  || die "PAP_CAPACITY_SESSIONS must be positive"
[[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] \
  || die "PAP_CAPACITY_GPU_COUNT must be positive"
for value in "${PAP_PREFILL_CHUNKS}" "${PAP_ATTENTION_CHUNKS}" \
  "${PAP_PREFILL_SMS}" "${PAP_ATTENTION_SMS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || die "PAP static-MPS chunk and SM counts must be positive"
done
(( PAP_PREFILL_CHUNKS + PAP_ATTENTION_CHUNKS == 23 )) \
  || die "PAP capacity runs require all 23 L20 MPS chunks"
(( PAP_PREFILL_SMS == PAP_PREFILL_CHUNKS * 4 \
  && PAP_ATTENTION_SMS == PAP_ATTENTION_CHUNKS * 4 )) \
  || die "PAP capacity MPS audit expects four visible SMs per L20 chunk"
[[ "${RESTART_BETWEEN_POINTS}" =~ ^[01]$ ]] \
  || die "PAP_CAPACITY_RESTART_BETWEEN_POINTS must be 0 or 1"
[[ "${OUTPUT_TOKENS}" =~ ^[1-9][0-9]*$ ]] \
  || die "PAP_CAPACITY_OUTPUT_TOKENS must be positive"
for value in "${OUTPUT_TOKENS_MEDIAN}" "${OUTPUT_TOKENS_MIN}" \
  "${OUTPUT_TOKENS_MAX}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || die "output length distribution values must be positive"
done
(( OUTPUT_TOKENS_MEDIAN <= OUTPUT_TOKENS )) \
  || die "output token median must not exceed its mean"
(( OUTPUT_TOKENS_MIN <= OUTPUT_TOKENS_MAX )) \
  || die "output token minimum exceeds its maximum"
[[ "${GPU_IDLE_STABILITY_SECONDS}" =~ ^[0-9]+$ ]] \
  || die "PAP_CAPACITY_GPU_IDLE_STABILITY_SECONDS must be non-negative"
for value in "${AIPERF_SWEEP_COOLDOWN_SECONDS}" \
  "${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}"; do
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || die "AIPerf cooldown values must be non-negative numbers"
done
case "${EXECUTION_MODE}" in
  eager | piecewise) ;;
  *) die "PAP_CAPACITY_EXECUTION_MODE must be eager or piecewise" ;;
esac
case "${PAP_ROUTING_POLICY}" in
  conversation_affinity | attention_load) ;;
  *) die "unsupported PAP capacity routing policy: ${PAP_ROUTING_POLICY}" ;;
esac

IFS=, read -r -a ALL_POINTS <<< "${DEFAULT_POINTS_CSV}"
for concurrency in "${ALL_POINTS[@]}"; do
  [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] \
    || die "invalid concurrency point: ${concurrency}"
  (( concurrency <= TOTAL_SESSIONS )) \
    || die "concurrency ${concurrency} exceeds total sessions"
done

for architecture in "${ARCHITECTURES[@]}"; do
  if [[ "${architecture}" =~ ^pap_([1-9][0-9]*)pa([1-9][0-9]*)p$ ]]; then
    (( BASH_REMATCH[1] + BASH_REMATCH[2] == GPU_COUNT )) \
      || die "${architecture} does not use ${GPU_COUNT} GPUs"
  elif [[ "${architecture}" =~ ^pd_([1-9][0-9]*)p([1-9][0-9]*)d$ ]]; then
    (( BASH_REMATCH[1] + BASH_REMATCH[2] == GPU_COUNT )) \
      || die "${architecture} does not use ${GPU_COUNT} GPUs"
  elif [[ "${architecture}" =~ ^dp_([1-9][0-9]*)$ ]]; then
    (( BASH_REMATCH[1] == GPU_COUNT )) \
      || die "${architecture} does not use ${GPU_COUNT} GPUs"
  else
    die "unsupported architecture: ${architecture}"
  fi
  IFS=, read -r -a architecture_points \
    <<< "$(points_for_architecture "${architecture}")"
  if [[ "${architecture}" == pap_* \
    && "${PAP_ROUTING_POLICY}" == "attention_load" \
    && ${#architecture_points[@]} -gt 1 ]]; then
    die "attention_load requires a single-point run until gateway reset exists"
  fi
  for concurrency in "${architecture_points[@]}"; do
    [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] \
      || die "invalid ${architecture} concurrency point: ${concurrency}"
    (( concurrency <= TOTAL_SESSIONS )) \
      || die "${architecture} concurrency ${concurrency} exceeds sessions"
  done
done

mkdir -p "${MATRIX_ROOT}/dataset" "${MATRIX_ROOT}/runs"
DATASET_FILE="${MATRIX_ROOT}/dataset/multiturn_s${TOTAL_SESSIONS}_longtail_random_o${OUTPUT_TOKENS}_t${TURNS}_seed${RANDOM_SEED}.jsonl"

if [[ ! -f "${DATASET_FILE}" ]]; then
  "${PYTHON_BIN}" "${DATASET_GENERATOR}" \
    --model "${MODEL_PATH}" \
    --corpus "${CORPUS_PATH}" \
    --output "${DATASET_FILE}" \
    --sessions "${TOTAL_SESSIONS}" \
    --turns "${TURNS}" \
    --document-tokens "${DOCUMENT_TOKENS}" \
    --document-tokens-median "${DOCUMENT_TOKENS_MEDIAN}" \
    --document-tokens-min "${DOCUMENT_TOKENS_MIN}" \
    --document-tokens-max "${DOCUMENT_TOKENS_MAX}" \
    --append-tokens "${APPEND_TOKENS}" \
    --append-tokens-median "${APPEND_TOKENS_MEDIAN}" \
    --append-tokens-min "${APPEND_TOKENS_MIN}" \
    --append-tokens-max "${APPEND_TOKENS_MAX}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --output-tokens-median "${OUTPUT_TOKENS_MEDIAN}" \
    --output-tokens-min "${OUTPUT_TOKENS_MIN}" \
    --output-tokens-max "${OUTPUT_TOKENS_MAX}" \
    --random-seed "${RANDOM_SEED}" \
    --sampled-mean-tolerance "${SAMPLED_MEAN_TOLERANCE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --think-time-ms "${THINK_TIME_MS}" \
    --tool-time-ms "${TOOL_TIME_MS}" \
    --tool-every "${TOOL_EVERY}" \
    --session-prefix "${DATASET_SESSION_PREFIX}"
fi
DATASET_MANIFEST="${DATASET_FILE}.manifest.json"
if ! jq -e \
  --arg session_prefix "${DATASET_SESSION_PREFIX}" \
  --argjson sessions "${TOTAL_SESSIONS}" \
  --argjson turns "${TURNS}" \
  --argjson document_tokens "${DOCUMENT_TOKENS}" \
  --argjson document_median "${DOCUMENT_TOKENS_MEDIAN}" \
  --argjson document_min "${DOCUMENT_TOKENS_MIN}" \
  --argjson document_max "${DOCUMENT_TOKENS_MAX}" \
  --argjson append_tokens "${APPEND_TOKENS}" \
  --argjson append_median "${APPEND_TOKENS_MEDIAN}" \
  --argjson append_min "${APPEND_TOKENS_MIN}" \
  --argjson append_max "${APPEND_TOKENS_MAX}" \
  --argjson output_tokens "${OUTPUT_TOKENS}" \
  --argjson output_median "${OUTPUT_TOKENS_MEDIAN}" \
  --argjson output_min "${OUTPUT_TOKENS_MIN}" \
  --argjson output_max "${OUTPUT_TOKENS_MAX}" \
  --argjson random_seed "${RANDOM_SEED}" \
  --argjson think_time_ms "${THINK_TIME_MS}" \
  --argjson tool_time_ms "${TOOL_TIME_MS}" \
  --argjson tool_every "${TOOL_EVERY}" \
  '.schema_version == 2
    and .session_prefix == $session_prefix
    and .sessions == $sessions
    and .turns_per_session == $turns
    and .requested_document_tokens == $document_tokens
    and .requested_append_tokens == $append_tokens
    and .output_tokens == $output_tokens
    and .distribution_semantics == "aiperf_lognormal_mean_median"
    and .random_seed == $random_seed
    and .length_distributions.document_content_tokens.configured.median == $document_median
    and .length_distributions.document_content_tokens.configured.min == $document_min
    and .length_distributions.document_content_tokens.configured.max == $document_max
    and .length_distributions.append_content_tokens.configured.median == $append_median
    and .length_distributions.append_content_tokens.configured.min == $append_min
    and .length_distributions.append_content_tokens.configured.max == $append_max
    and .length_distributions.output_tokens.configured.median == $output_median
    and .length_distributions.output_tokens.configured.min == $output_min
    and .length_distributions.output_tokens.configured.max == $output_max
    and .length_distributions.document_content_tokens.sampled.min
      < .length_distributions.document_content_tokens.sampled.max
    and .length_distributions.append_content_tokens.sampled.min
      < .length_distributions.append_content_tokens.sampled.max
    and .length_distributions.output_tokens.sampled.min
      < .length_distributions.output_tokens.sampled.max
    and .validation.status == "passed"
    and .delay_profile.think_time_ms == $think_time_ms
    and .delay_profile.tool_time_ms == $tool_time_ms
    and .delay_profile.tool_every == $tool_every' \
  "${DATASET_MANIFEST}" >/dev/null; then
  die "dataset manifest does not match the randomized testbed"
fi

"${AIPERF_PYTHON}" - "${DATASET_FILE}" "${TOTAL_SESSIONS}" "${TURNS}" <<'PY'
import json
import sys
from pathlib import Path

from aiperf.dataset.loader.models import MultiTurn

path = Path(sys.argv[1])
expected_sessions = int(sys.argv[2])
expected_turns = int(sys.argv[3])
records = [
    MultiTurn.model_validate(json.loads(line))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(records) != expected_sessions:
    raise SystemExit("AIPerf dataset session count mismatch")
for record in records:
    if len(record.turns) != expected_turns:
        raise SystemExit(f"AIPerf dataset turn count mismatch: {record.session_id}")
    for turn in record.turns:
        if turn.extra.get("ignore_eos") is not True:
            raise SystemExit("AIPerf dataset does not force exact output")
        if turn.extra.get("min_tokens") != turn.output_length:
            raise SystemExit("AIPerf min_tokens/output_length mismatch")
print(f"AIPerf validated {len(records)} randomized multi-turn sessions")
PY

{
  printf 'SCHEMA_VERSION=2\nMATRIX_ID=%q\n' "${MATRIX_ID}"
  printf 'MODEL_PATH=%q\nCORPUS_PATH=%q\n' \
    "${MODEL_PATH}" "${CORPUS_PATH}"
  printf 'AIPERF_TIMING_MODE=concurrency\nAIPERF_REQUEST_RATE=\n'
  printf 'ARCHITECTURES=%q\nTOTAL_SESSIONS=%q\n' \
    "${ARCHITECTURES_CSV}" "${TOTAL_SESSIONS}"
  printf 'GPU_COUNT=%q\nDEFAULT_POINTS=%q\n' \
    "${GPU_COUNT}" "${DEFAULT_POINTS_CSV}"
  for architecture in "${ARCHITECTURES[@]}"; do
    printf 'POINTS_%s=%q\n' "${architecture^^}" \
      "$(points_for_architecture "${architecture}")"
  done
  printf 'REPETITIONS=%q\n' "${REPETITIONS}"
  printf 'SWEEP_OWNER=aiperf\nSWEEP_MODE=repeated\n'
  printf 'SERVICE_RESTART_BETWEEN_POINTS=%q\n' \
    "${RESTART_BETWEEN_POINTS}"
  printf 'AIPERF_SWEEP_SAME_SEED=1\n'
  printf 'AIPERF_SWEEP_COOLDOWN_SECONDS=%q\n' \
    "${AIPERF_SWEEP_COOLDOWN_SECONDS}"
  printf 'AIPERF_PROFILE_RUN_COOLDOWN_SECONDS=%q\n' \
    "${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}"
  printf 'AIPERF_GOODPUT_SLO=%q\n' "${AIPERF_GOODPUT_SLO}"
  printf 'GPU_IDLE_STABILITY_SECONDS=%q\n' \
    "${GPU_IDLE_STABILITY_SECONDS}"
  printf 'TURNS=%q\n' "${TURNS}"
  printf 'DATASET_SESSION_PREFIX=%q\n' "${DATASET_SESSION_PREFIX}"
  printf 'LENGTH_DISTRIBUTION=aiperf_lognormal_mean_median\nRANDOM_SEED=%q\n' \
    "${RANDOM_SEED}"
  printf 'DOCUMENT_TOKENS_MEAN=%q\nDOCUMENT_TOKENS_MEDIAN=%q\n' \
    "${DOCUMENT_TOKENS}" "${DOCUMENT_TOKENS_MEDIAN}"
  printf 'DOCUMENT_TOKENS_MIN=%q\nDOCUMENT_TOKENS_MAX=%q\n' \
    "${DOCUMENT_TOKENS_MIN}" "${DOCUMENT_TOKENS_MAX}"
  printf 'APPEND_TOKENS_MEAN=%q\nAPPEND_TOKENS_MEDIAN=%q\n' \
    "${APPEND_TOKENS}" "${APPEND_TOKENS_MEDIAN}"
  printf 'APPEND_TOKENS_MIN=%q\nAPPEND_TOKENS_MAX=%q\n' \
    "${APPEND_TOKENS_MIN}" "${APPEND_TOKENS_MAX}"
  printf 'SAMPLED_MEAN_TOLERANCE=%q\n' "${SAMPLED_MEAN_TOLERANCE}"
  printf 'APPEND_SAMPLED_MEAN=%q\nAPPEND_SAMPLED_MEDIAN=%q\n' \
    "$(jq -r '.length_distributions.append_content_tokens.sampled.mean' \
      "${DATASET_MANIFEST}")" \
    "$(jq -r '.length_distributions.append_content_tokens.sampled.median' \
      "${DATASET_MANIFEST}")"
  printf 'OUTPUT_TOKENS_MEAN=%q\nOUTPUT_TOKENS_MEDIAN=%q\n' \
    "${OUTPUT_TOKENS}" "${OUTPUT_TOKENS_MEDIAN}"
  printf 'OUTPUT_TOKENS_MIN=%q\nOUTPUT_TOKENS_MAX=%q\n' \
    "${OUTPUT_TOKENS_MIN}" "${OUTPUT_TOKENS_MAX}"
  printf 'THINK_TIME_MS=%q\nTOOL_TIME_MS=%q\nTOOL_EVERY=%q\n' \
    "${THINK_TIME_MS}" "${TOOL_TIME_MS}" "${TOOL_EVERY}"
  printf 'MAX_MODEL_LEN=%q\nMAX_NUM_SEQS=%q\n' \
    "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}"
  printf 'PREFILL_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${PREFILL_MAX_NUM_BATCHED_TOKENS}"
  printf 'DECODE_MAX_NUM_BATCHED_TOKENS=%q\n' \
    "${DECODE_MAX_NUM_BATCHED_TOKENS}"
  printf 'MAX_NUM_PARTIAL_PREFILLS=default_1\n'
  printf 'LONG_PREFILL_TOKEN_THRESHOLD=default_0\n'
  printf 'PAP_DECODE_CAPACITY=request_max_tokens_fallback_64\n'
  printf 'EXECUTION_MODE=%q\n' "${EXECUTION_MODE}"
  printf 'PAP_PREFILL_GPU_MEMORY_UTILIZATION=%q\n' \
    "${PAP_PREFILL_GPU_MEMORY_UTILIZATION}"
  printf 'PAP_ROUTING_POLICY=%q\n' "${PAP_ROUTING_POLICY}"
  printf 'PAP_MIGRATION_MIN_PEAK_GAIN_RATIO=%q\n' \
    "${PAP_MIGRATION_MIN_PEAK_GAIN_RATIO}"
  printf 'PAP_PROJECTION_MEMORY_POLICY=%q\n' 'model_weights_x1.20'
  printf 'PD_GPU_MEMORY_UTILIZATION=%q\n' \
    "${PD_GPU_MEMORY_UTILIZATION}"
  printf 'PAP_STATIC_MPS_CHUNKS=%q\nPAP_STATIC_MPS_SMS=%q\n' \
    "${PAP_PREFILL_CHUNKS}/${PAP_ATTENTION_CHUNKS}" \
    "${PAP_PREFILL_SMS}/${PAP_ATTENTION_SMS}"
  printf 'SLO_STRICT=%q\nSLO_STANDARD=%q\nSLO_RELAXED=%q\n' \
    "${SLO_STRICT}" \
    "${SLO_STANDARD}" \
    "${SLO_RELAXED}"
} > "${MATRIX_ROOT}/matrix_config.env"

wait_for_gpus() {
  [[ "${WAIT_FOR_GPUS}" == "1" ]] || return 0
  local busy gpu processes stable=0
  while true; do
    busy=0
    for (( gpu=0; gpu<GPU_COUNT; gpu++ )); do
      if ! processes="$(
        nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits
      )"; then
        die "nvidia-smi failed when checking GPU ${gpu}; set WAIT_FOR_GPUS=0 or fix NVIDIA runtime"
      fi
      if [[ -n "${processes//[[:space:]]/}" ]]; then
        echo "GPU ${gpu} is occupied by PID(s): ${processes//$'\n'/,}"
        busy=1
      fi
    done
    if (( busy == 1 )); then
      stable=0
      echo "Waiting 60 seconds for GPUs 0-$((GPU_COUNT - 1)) to become idle..."
      sleep 60
    elif (( stable == 0 && GPU_IDLE_STABILITY_SECONDS > 0 )); then
      stable=1
      echo "GPUs 0-$((GPU_COUNT - 1)) are idle; verifying for ${GPU_IDLE_STABILITY_SECONDS} seconds..."
      sleep "${GPU_IDLE_STABILITY_SECONDS}"
    else
      return 0
    fi
  done
}

gpu_csv() {
  local start="$1"
  local count="$2"
  seq -s, "${start}" "$((start + count - 1))"
}

read_env_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="$key" '$1==key {print $2}' "${file}"
}

summarize_aiperf_variation() {
  local run_root="$1"
  local aiperf_root="$2"
  local architecture="$3"
  local topology="$4"
  local concurrency="$5"
  local repetition="${6:-1}"
  local summary_root="$7"
  local exit_code="${8:-0}"
  local runtime_repetitions="${9:-1}"

  ${PYTHON_BIN} "${SUMMARY_RUNNER}" \
    --run-root "${run_root}" \
    --aiperf-root "${aiperf_root}" \
    --architecture "${architecture}" \
    --topology "${topology}" \
    --concurrency "${concurrency}" \
    --sessions "${TOTAL_SESSIONS}" \
    --turns "${TURNS}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --dataset-file "${DATASET_FILE}" \
    --repetition "${repetition}" \
    --runtime-repetitions "${runtime_repetitions}" \
    --launcher-exit-code "${exit_code}" \
    --output "${summary_root}"
}

summarize_sweep_run() {
  local run_root="$1"
  local architecture="$2"
  local topology="$3"
  local concurrency_points="$4"
  local exit_code="${5:-0}"
  local point output
  local -a points
  local point_root repetition found_point has_point_dirs point_count=0 trial_root
  local runtime_repetitions
  local -a profile_run_roots

  if [[ -z "${concurrency_points}" ]]; then
    return
  fi
  IFS=, read -r -a points <<< "${concurrency_points//\\,/,}"

  for point in "${points[@]}"; do
    [[ -n "${point}" ]] && (( ++point_count ))
  done
  if (( point_count == 0 )); then
    return
  fi

  if [[ -d "${run_root}/points" ]]; then
    for point in "${points[@]}"; do
      [[ -n "${point}" ]] || continue
      point_root="${run_root}/points/concurrency_${point}"
      if [[ ! -d "${point_root}/aiperf" ]]; then
        echo "WARNING: missing isolated AIPerf point C=${point}" >&2
        continue
      fi
      exit_code=0
      if [[ -f "${point_root}/launcher_exit_code.txt" ]]; then
        exit_code="$(
          cat "${point_root}/launcher_exit_code.txt" 2>/dev/null || echo 1
        )"
      fi
      shopt -s nullglob
      profile_run_roots=(
        "${point_root}/aiperf/profile_runs/run_"[0-9]*
      )
      shopt -u nullglob
      if (( ${#profile_run_roots[@]} > 0 )); then
        runtime_repetitions="${#profile_run_roots[@]}"
        for trial_root in "${profile_run_roots[@]}"; do
          repetition="${trial_root##*/run_}"
          repetition="$((10#${repetition}))"
          output="${run_root}/capacity_summary_c${point}_r${repetition}.json"
          summarize_aiperf_variation \
            "${point_root}" \
            "${trial_root}" \
            "${architecture}" \
            "${topology}" \
            "${point}" \
            "${repetition}" \
            "${output}" \
            "${exit_code}" \
            "${runtime_repetitions}"
        done
        continue
      fi
      output="${run_root}/capacity_summary_c${point}_r1.json"
      summarize_aiperf_variation \
        "${point_root}" \
        "${point_root}/aiperf" \
        "${architecture}" \
        "${topology}" \
        "${point}" \
        1 \
        "${output}" \
        "${exit_code}"
    done
    return
  fi

  if [[ -d "${run_root}/aiperf/profile_runs" ]]; then
    shopt -s nullglob
    for trial_root in "${run_root}/aiperf/profile_runs/trial_"*; do
      [[ -d "${trial_root}" ]] || continue
      repetition="${trial_root##*/}"
      repetition="${repetition#trial_}"
      [[ "${repetition}" =~ ^[0-9]+$ ]] || repetition="1"
      found_point=0
      for point in "${points[@]}"; do
        [[ -z "${point}" ]] && continue
        point_root="${trial_root}/concurrency_${point}"
        if [[ -d "${point_root}" ]]; then
          output="${run_root}/capacity_summary_c${point}_r${repetition}.json"
          summarize_aiperf_variation \
            "${run_root}" \
            "${point_root}" \
            "${architecture}" \
            "${topology}" \
            "${point}" \
            "${repetition}" \
            "${output}" \
            "${exit_code}"
          found_point=1
          continue
        fi
        if [[ -d "${point_root}/aggregate/sweep_aggregate" ]]; then
          output="${run_root}/capacity_summary_c${point}_r${repetition}.json"
          summarize_aiperf_variation \
            "${run_root}" \
            "${point_root}/aggregate/sweep_aggregate" \
            "${architecture}" \
            "${topology}" \
            "${point}" \
            "${repetition}" \
            "${output}" \
            "${exit_code}"
          found_point=1
        fi
      done
      if (( found_point == 0 )); then
        point="${points[0]}"
        output="${run_root}/capacity_summary_c${point}_r${repetition}.json"
        summarize_aiperf_variation \
          "${run_root}" \
          "${trial_root}" \
          "${architecture}" \
          "${topology}" \
          "${point}" \
          "${repetition}" \
          "${output}" \
          "${exit_code}"
      fi
    done
    shopt -u nullglob
    return
  fi

  if [[ -d "${run_root}/aiperf" ]]; then
    has_point_dirs=0
    for point in "${points[@]}"; do
      [[ -z "${point}" ]] && continue
      if [[ -d "${run_root}/aiperf/concurrency_${point}" ]] \
        || [[ -d "${run_root}/aiperf/concurrency_${point}/aggregate/sweep_aggregate" ]]; then
        has_point_dirs=1
        break
      fi
    done

    if (( has_point_dirs == 1 )); then
      for point in "${points[@]}"; do
        [[ -z "${point}" ]] && continue
        point_root="${run_root}/aiperf/concurrency_${point}"
        if [[ -d "${point_root}/aggregate/sweep_aggregate" ]]; then
          point_root="${point_root}/aggregate/sweep_aggregate"
        fi
        if [[ -d "${point_root}" ]] || [[ -f "${point_root}/profile.json" ]]; then
          output="${run_root}/capacity_summary_c${point}_r1.json"
          summarize_aiperf_variation \
            "${run_root}" \
            "${point_root}" \
            "${architecture}" \
            "${topology}" \
            "${point}" \
            1 \
            "${output}" \
            "${exit_code}"
        else
          echo "WARNING: missing aiperf point data for C=${point} in ${run_root}" >&2
        fi
      done
      return
    fi

    if (( point_count == 1 )); then
      point="${points[0]}"
      point_root="${run_root}/aiperf"
      output="${run_root}/capacity_summary_c${point}_r1.json"
      summarize_aiperf_variation \
        "${run_root}" \
        "${point_root}" \
        "${architecture}" \
        "${topology}" \
        "${point}" \
        1 \
        "${output}" \
        "${exit_code}"
    fi
    return
  fi
}

run_pap_architecture() {
  local topology="$1"
  local concurrency_points="$2"
  local run_root="$3"
  [[ "${topology}" =~ ^([1-9][0-9]*)pa([1-9][0-9]*)p$ ]] \
    || die "invalid PAP topology: ${topology}"
  local pa_count="${BASH_REMATCH[1]}"
  local projection_count="${BASH_REMATCH[2]}"
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
    PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE="${PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE:-1}" \
    PAP_BENCH_STRICT_CORRECTNESS_AUDIT=1 \
    PAP_TOPOLOGY="${topology}" \
    PAP_PREFILL_GPUS="$(gpu_csv 0 "${pa_count}")" \
    PAP_PROJECTION_GPUS="$(gpu_csv "${pa_count}" "${projection_count}")" \
    PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY}" \
    PAP_ATTENTION_LOAD_MIGRATION_MIN_PEAK_GAIN_RATIO=\
"${PAP_MIGRATION_MIN_PEAK_GAIN_RATIO}" \
    PAP_OFFLOAD_EXEC_TRANSPORT=local_fast \
    PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc \
    PAP_DIRECT_MAILBOX_OUTPUT=1 \
    PAP_VLLM_DTYPE=float16 \
    PAP_PREFILL_GPU_MEMORY_UTILIZATION="${PAP_PREFILL_GPU_MEMORY_UTILIZATION}" \
    PAP_PREFILL_MPS_PERCENT="${PAP_PREFILL_PERCENT}" \
    PAP_ATTENTION_MPS_PERCENT="${PAP_ATTENTION_PERCENT}" \
    PAP_STATIC_PREFILL_CHUNKS="${PAP_PREFILL_CHUNKS}" \
    PAP_STATIC_ATTENTION_CHUNKS="${PAP_ATTENTION_CHUNKS}" \
    PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_PREFILL_SMS}" \
    PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_ATTENTION_SMS}" \
    PAP_AIPERF_TURNS="${TURNS}" \
    PAP_AIPERF_SESSIONS="${TOTAL_SESSIONS}" \
    PAP_AIPERF_APPEND_TOKENS="${APPEND_TOKENS}" \
    INPUT_LEN="${DOCUMENT_TOKENS}" \
    OUTPUT_LEN="${OUTPUT_TOKENS}" \
    PAP_AIPERF_INPUT_FILE="${DATASET_FILE}" \
    PAP_AIPERF_OUTPUT_DIR="${run_root}/aiperf" \
    PAP_AIPERF_CONCURRENCY="${concurrency_points}" \
    PAP_AIPERF_TIMING_MODE=concurrency \
    PAP_AIPERF_REQUEST_RATE= \
    AIPERF_RANDOM_SEED="${RANDOM_SEED}" \
    AIPERF_NUM_PROFILE_RUNS="${REPETITIONS}" \
    AIPERF_PROFILE_RUN_COOLDOWN_SECONDS=\
"${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}" \
    AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS=\
"${AIPERF_SWEEP_COOLDOWN_SECONDS}" \
    AIPERF_PARAMETER_SWEEP_MODE=repeated \
    AIPERF_GOODPUT_SLO="${AIPERF_GOODPUT_SLO}" \
    AIPERF_EXPORT_LEVEL=records \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    PAP_PREFILL_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS}" \
    PAP_PROJECTION_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${OUTPUT_TOKENS_MAX}" \
    PAP_EXECUTION_MODE="${EXECUTION_MODE}" \
    BENCH_TIMEOUT="${PAP_RUN_TIMEOUT_SECONDS}" \
    SERVER_START_TIMEOUT=900 \
    CLUSTER_READY_WAIT_SECONDS=30 \
    "${PAP_RUNNER}"
}

run_pd_architecture() {
  local topology="$1"
  local concurrency_points="$2"
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
    PD_LOAD_TOPOLOGY="${topology}" \
    PD_LOAD_ROUNDS="${TURNS}" \
    PD_LOAD_CONVERSATIONS="${TOTAL_SESSIONS}" \
    PD_LOAD_DOCUMENT_TOKENS="${DOCUMENT_TOKENS}" \
    PD_LOAD_APPEND_TOKENS="${APPEND_TOKENS}" \
    PD_LOAD_OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
    PD_LOAD_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    PD_LOAD_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    PD_LOAD_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PD_LOAD_PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    PD_LOAD_PREFILL_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PD_LOAD_DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS}" \
    PD_LOAD_DECODE_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PD_LOAD_GPU_MEMORY_UTILIZATION="${PD_GPU_MEMORY_UTILIZATION}" \
    PD_LOAD_EXECUTION_MODE="${EXECUTION_MODE}" \
    PD_LOAD_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
    PD_AIPERF_INPUT_FILE="${DATASET_FILE}" \
    PD_AIPERF_OUTPUT_DIR="${run_root}/aiperf" \
    PD_AIPERF_CONCURRENCY="${concurrency_points}" \
    PD_AIPERF_TIMING_MODE=concurrency \
    PD_AIPERF_REQUEST_RATE= \
    AIPERF_RANDOM_SEED="${RANDOM_SEED}" \
    AIPERF_NUM_PROFILE_RUNS="${REPETITIONS}" \
    AIPERF_PROFILE_RUN_COOLDOWN_SECONDS=\
"${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}" \
    AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS=\
"${AIPERF_SWEEP_COOLDOWN_SECONDS}" \
    AIPERF_PARAMETER_SWEEP_MODE=repeated \
    AIPERF_GOODPUT_SLO="${AIPERF_GOODPUT_SLO}" \
    AIPERF_EXPORT_LEVEL=records \
    "${PD_RUNNER}" oneway
}

run_dp_architecture() {
  local topology="$1"
  local concurrency_points="$2"
  local run_root="$3"
  [[ "${topology}" =~ ^([1-9][0-9]*)dp$ ]] \
    || die "invalid DP topology: ${topology}"
  local dp_size="${BASH_REMATCH[1]}"
  env \
    PAP_ROOT="${ROOT_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_PATH="${MODEL_PATH}" \
    AIPERF_ROOT="${AIPERF_ROOT}" \
    AIPERF_BIN="${AIPERF_BIN}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    DP_LOAD_RUN_ID="$(basename "${run_root}")" \
    DP_LOAD_RUN_ROOT="${run_root}" \
    DP_LOAD_SIZE="${dp_size}" \
    DP_LOAD_GPUS="$(gpu_csv 0 "${dp_size}")" \
    DP_LOAD_ROUNDS="${TURNS}" \
    DP_LOAD_CONVERSATIONS="${TOTAL_SESSIONS}" \
    DP_LOAD_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    DP_LOAD_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
    DP_LOAD_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    DP_LOAD_GPU_MEMORY_UTILIZATION="${PD_GPU_MEMORY_UTILIZATION}" \
    DP_LOAD_EXECUTION_MODE="${EXECUTION_MODE}" \
    DP_LOAD_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
    DP_AIPERF_INPUT_FILE="${DATASET_FILE}" \
    DP_AIPERF_OUTPUT_DIR="${run_root}/aiperf" \
    DP_AIPERF_CONCURRENCY="${concurrency_points}" \
    AIPERF_RANDOM_SEED="${RANDOM_SEED}" \
    AIPERF_NUM_PROFILE_RUNS="${REPETITIONS}" \
    AIPERF_PROFILE_RUN_COOLDOWN_SECONDS=\
"${AIPERF_PROFILE_RUN_COOLDOWN_SECONDS}" \
    AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS=\
"${AIPERF_SWEEP_COOLDOWN_SECONDS}" \
    AIPERF_PARAMETER_SWEEP_MODE=repeated \
    AIPERF_GOODPUT_SLO="${AIPERF_GOODPUT_SLO}" \
    AIPERF_EXPORT_LEVEL=records \
    "${DP_RUNNER}"
}

run_architecture_launcher() {
  local architecture="$1"
  local topology="$2"
  local concurrency_points="$3"
  local run_root="$4"

  if [[ "${architecture}" == "pap" ]]; then
    run_pap_architecture "${topology}" "${concurrency_points}" "${run_root}"
  elif [[ "${architecture}" == "pd" ]]; then
    run_pd_architecture "${topology}" "${concurrency_points}" "${run_root}"
  else
    run_dp_architecture "${topology}" "${concurrency_points}" "${run_root}"
  fi
}

run_isolated_architecture_points() {
  local architecture="$1"
  local topology="$2"
  local concurrency_points="$3"
  local run_root="$4"
  local point point_root launcher_code failed_points=0
  local completion_file status
  local -a points

  IFS=, read -r -a points <<< "${concurrency_points//\,/,}"
  mkdir -p "${run_root}/points"
  for point in "${points[@]}"; do
    [[ -n "${point}" ]] || continue
    point_root="${run_root}/points/concurrency_${point}"
    if [[ "${RESUME}" == "1" \
      && -f "${point_root}/aiperf_sweep_complete.env" ]]; then
      echo "Reusing isolated ${architecture} ${topology} C=${point}"
      launcher_code="$(
        cat "${point_root}/launcher_exit_code.txt" 2>/dev/null || echo 0
      )"
      summarize_sweep_run \
        "${run_root}" \
        "${architecture}" \
        "${topology}" \
        "${point}" \
        "${launcher_code}"
      continue
    fi
    if [[ -d "${point_root}" ]]; then
      if [[ "${RESUME}" == "1" ]]; then
        echo "Cleaning incomplete isolated point ${point_root}" >&2
        rm -rf "${point_root}"
      elif [[ -n "$(find "${point_root}" -mindepth 1 -print -quit)" ]]; then
        die "isolated point directory already has data: ${point_root}"
      fi
    fi

    wait_for_gpus
    mkdir -p "${point_root}"
    echo "=== ${architecture} ${topology} isolated C=${point} ==="
    set +e
    run_architecture_launcher \
      "${architecture}" \
      "${topology}" \
      "${point}" \
      "${point_root}" \
      > "${point_root}/launcher.log" 2>&1
    launcher_code="$?"
    set -e
    printf '%s\n' "${launcher_code}" > "${point_root}/launcher_exit_code.txt"
    if [[ -z "$(find "${point_root}/aiperf" -type f \
      -name 'profile*.json' -size +0c -print -quit 2>/dev/null)" ]]; then
      echo "WARNING: ${architecture} ${topology} C=${point} produced no profile" >&2
      (( ++failed_points ))
      continue
    fi
    if (( launcher_code != 0 )); then
      echo "WARNING: ${architecture} ${topology} C=${point} exited ${launcher_code}" >&2
      (( ++failed_points ))
    else
      {
        printf 'STATUS=passed\n'
        printf 'ARCHITECTURE=%q\nTOPOLOGY=%q\n' \
          "${architecture}" "${topology}"
        printf 'CONCURRENCY_POINTS=%q\n' "${point}"
        printf 'PROFILE_RUNS=%q\n' "${REPETITIONS}"
        printf 'DATASET_FILE=%q\n' "${DATASET_FILE}"
      } > "${point_root}/aiperf_sweep_complete.env"
    fi
    summarize_sweep_run \
      "${run_root}" \
      "${architecture}" \
      "${topology}" \
      "${point}" \
      "${launcher_code}"
  done

  completion_file="${run_root}/aiperf_sweep_complete.env"
  status=passed
  if (( failed_points > 0 )); then
    completion_file="${run_root}/aiperf_sweep_incomplete.env"
    status=incomplete
  fi
  {
    printf 'STATUS=%q\n' "${status}"
    printf 'ARCHITECTURE=%q\nTOPOLOGY=%q\n' \
      "${architecture}" "${topology}"
    printf 'CONCURRENCY_POINTS=%q\n' "${concurrency_points}"
    printf 'PROFILE_RUNS=%q\n' "${REPETITIONS}"
    printf 'DATASET_FILE=%q\n' "${DATASET_FILE}"
    printf 'FAILED_POINTS=%q\n' "${failed_points}"
  } > "${completion_file}"
  printf '%s\n' "$((failed_points > 0))" > "${run_root}/launcher_exit_code.txt"
}

run_architecture() {
  local architecture_tag="$1"
  local concurrency_points="$2"
  local architecture topology run_root launcher_code
  local resume_code=0
  if [[ "${architecture_tag}" == pap_* ]]; then
    architecture=pap
    topology="${architecture_tag#pap_}"
  elif [[ "${architecture_tag}" == pd_* ]]; then
    architecture=pd
    topology="${architecture_tag#pd_}"
  else
    architecture=dp
    topology="${architecture_tag#dp_}dp"
  fi
  run_root="${MATRIX_ROOT}/runs/${architecture_tag}"

  if [[ "${RESUME}" == "1" \
    && -f "${run_root}/aiperf_sweep_complete.env" ]]; then
    if [[ -f "${run_root}/launcher_exit_code.txt" ]]; then
      resume_code="$(cat "${run_root}/launcher_exit_code.txt" 2>/dev/null || echo 0)"
      if [[ -z "${resume_code}" ]]; then
        resume_code=0
      fi
    fi
    echo "Reusing completed AIPerf sweep for ${architecture_tag}"
    summarize_sweep_run \
      "${run_root}" \
      "${architecture}" \
      "${topology}" \
      "${concurrency_points}" \
      "${resume_code}"
    return
  fi
  if [[ "${RESTART_BETWEEN_POINTS}" == "1" ]]; then
    mkdir -p "${run_root}"
    run_isolated_architecture_points \
      "${architecture}" \
      "${topology}" \
      "${concurrency_points}" \
      "${run_root}"
    return
  fi
  if [[ -d "${run_root}" ]]; then
    if [[ "${RESUME}" == "1" ]]; then
      if [[ -n "$(find "${run_root}" -mindepth 1 -type f -name 'launcher_exit_code.txt' -print -quit)" ]]; then
        existing_code=""
        existing_code="$(cat "${run_root}/launcher_exit_code.txt" 2>/dev/null || echo '')"
        if [[ "${existing_code}" == "0" ]]; then
          if [[ -n "$(find "${run_root}/aiperf" -type f -name 'profile*.json' -size +0c -print -quit)" ]]; then
            echo "Found completed sweep artifacts for ${architecture_tag}."
            summarize_sweep_run \
              "${run_root}" \
              "${architecture}" \
              "${topology}" \
              "${concurrency_points}" \
              "${existing_code}"
            return
          fi
          echo "Launcher exit=0 but no profile output for ${architecture_tag}; will rerun." >&2
        fi
      fi
      echo "Resume enabled: cleaning stale/incomplete run directory ${run_root}" >&2
      rm -rf "${run_root}"
    elif [[ -n "$(find "${run_root}" -mindepth 1 -print -quit)" ]]; then
      die "run directory already has data: ${run_root}"
    fi
  fi

  wait_for_gpus
  mkdir -p "${run_root}"
  echo "=== ${architecture_tag} AIPerf concurrency=${concurrency_points} ==="
  set +e
  run_architecture_launcher \
    "${architecture}" \
    "${topology}" \
    "${concurrency_points}" \
    "${run_root}" \
    > "${run_root}/launcher.log" 2>&1
  launcher_code="$?"
  set -e
  printf '%s\n' "${launcher_code}" > "${run_root}/launcher_exit_code.txt"
  (( launcher_code == 0 )) \
    || die "${architecture_tag} AIPerf sweep failed; see ${run_root}"
  if [[ -z "$(find "${run_root}/aiperf" -type f \
    -name 'profile*.json' -size +0c -print -quit)" ]]; then
    die "${architecture_tag} AIPerf sweep produced no profile JSON"
  fi
  {
    printf 'STATUS=passed\n'
    printf 'ARCHITECTURE=%q\nTOPOLOGY=%q\n' \
      "${architecture}" "${topology}"
    printf 'CONCURRENCY_POINTS=%q\n' "${concurrency_points}"
    printf 'PROFILE_RUNS=%q\n' "${REPETITIONS}"
    printf 'DATASET_FILE=%q\n' "${DATASET_FILE}"
  } > "${run_root}/aiperf_sweep_complete.env"

  summarize_sweep_run \
    "${run_root}" \
    "${architecture}" \
    "${topology}" \
    "${concurrency_points}" \
    "${launcher_code}"
}

for architecture in "${ARCHITECTURES[@]}"; do
  points_csv="$(points_for_architecture "${architecture}")"
  run_architecture "${architecture}" "${points_csv}"
done

echo "PAP_CAPACITY_MATRIX_ROOT=${MATRIX_ROOT}"
