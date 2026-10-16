#!/usr/bin/env bash
# Run one service start of one variant.
#
# Usage: bash run.sh <variant> <repetition>
#
# variants/<variant>.env selects RUNNER=pap|dynamo and is sourced after the
# runner's base file (pap.env or dynamo.env). PAP runs use the current checkout;
# Dynamo runs are followed by the TTFT breakdown analysis. A variant that sets
# EXPECTED_PAT_SO_SHA256 refuses to start unless that PAT build is installed.
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"

variant="${1:?variant is required}"
repetition="${2:?repetition is required}"
variant_env="${EXPERIMENT_DIR}/variants/${variant}.env"
[[ -f "${variant_env}" ]] || { echo "ERROR: missing ${variant_env}" >&2; exit 2; }
runner="$(sed -n 's/^RUNNER=//p' "${variant_env}")"
case "${runner}" in
  pap) base_env="${EXPERIMENT_DIR}/pap.env" ;;
  dynamo) base_env="${EXPERIMENT_DIR}/dynamo.env" ;;
  *) echo "ERROR: ${variant_env} must set RUNNER=pap|dynamo" >&2; exit 2 ;;
esac

# The launchers' CUDA Graph log audit requires ripgrep.
PATH="${PAP_EXTRA_PATH:-/home/fei/.cache/opencode/bin}:${PATH}"
command -v rg >/dev/null \
  || { echo "ERROR: rg is required; set PAP_EXTRA_PATH to its directory" >&2; exit 2; }

pat_sha="$(find "${ROOT_DIR}/.venv" -name '_prefix_attn*.so' -exec sha256sum {} \; \
  | cut -d' ' -f1)"
expected_pat="$(sed -n 's/^EXPECTED_PAT_SO_SHA256=//p' "${variant_env}")"
if [[ -n "${expected_pat}" && "${pat_sha}" != "${expected_pat}" ]]; then
  echo "ERROR: installed PAT ${pat_sha} does not match ${variant} (${expected_pat})" >&2
  exit 2
fi

# The previous service can hold GPU memory for a few seconds after it exits.
for _ in $(seq 120); do
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
  sleep 1
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${EXPERIMENT_DIR}/runs/${stamp}_${variant}_r${repetition}"
mkdir -p "${run_root}"
cat "${base_env}" "${variant_env}" > "${run_root}/requested.env"

launch_env=(
  "HOME=${HOME}" "USER=${USER:-fei}" "PATH=${PATH}" "LANG=C.UTF-8"
  "PYTHON_BIN=${ROOT_DIR}/.venv/bin/python"
  "VLLM_BIN=${ROOT_DIR}/.venv/bin/vllm"
  "AIPERF_BIN=${ROOT_DIR}/.venv-aiperf/bin/aiperf"
  "AIPERF_PYTHON=${ROOT_DIR}/.venv-aiperf/bin/python"
)
if [[ "${runner}" == pap ]]; then
  launch_env+=(
    "RUN_ROOT=${run_root}"
    "RUN_LOG_DIR=${run_root}/service_logs"
    "PAP_AIPERF_OUTPUT_DIR=${run_root}/aiperf"
    "PAP_DYNAMO_SITE_PACKAGES=${ROOT_DIR}/.local/pap-dynamo-router"
    "PAP_NVSHMEM_NVCC=${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
    "PAP_NVSHMEM_PREFIX=${ROOT_DIR}/${NVSHMEM_PREFIX}"
  )
  workload="${ROOT_DIR}/benchmarks/pap/scripts/run_pap_workload.sh"
else
  run_id="e2d_${stamp}_${variant}_r${repetition}"
  launch_env+=(
    "DYNAMO_RUN_ID=${run_id}"
    "DYNAMO_RUN_ROOT=${run_root}"
    "DYNAMO_AIPERF_OUTPUT_DIR=${run_root}/aiperf"
    "DYNAMO_NAMESPACE=pap-dynamo-${run_id}"
    "DYNAMO_FILE_KV=${run_root}/discovery"
  )
  workload="${ROOT_DIR}/benchmarks/pap/scripts/run_dynamo_workload.sh"
fi

{
  printf 'variant=%s\nrunner=%s\nrepetition=%s\n' "${variant}" "${runner}" "${repetition}"
  printf 'source_head=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  printf 'source_dirty_files=%s\n' \
    "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=no | wc -l)"
  printf 'pat_so_sha256=%s\n' "${pat_sha}"
  printf '%s\n' "${launch_env[@]}"
} > "${run_root}/launch.env"
git -C "${ROOT_DIR}" diff HEAD > "${run_root}/source.diff"

cd "${ROOT_DIR}"
set +e
# shellcheck disable=SC2016  # expanded by the inner bash, not here
env -i "${launch_env[@]}" bash -c \
  'set -a; source "$1"; set +a; exec bash "$2"' paper-run \
  "${run_root}/requested.env" "${workload}" \
  > "${run_root}/runner.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${run_root}/exit_code"

if [[ "${runner}" == dynamo && "${status}" == 0 ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/benchmarks/pap/tooling/analyze_dynamo_ttft.py" \
    "${run_root}" --block-size 16 --output "${run_root}/ttft_breakdown.json" \
    > "${run_root}/ttft_breakdown.log" 2>&1 \
    || echo "WARNING: TTFT breakdown failed; see ${run_root}/ttft_breakdown.log" >&2
fi
echo "${variant} r${repetition} exit=${status} run_root=${run_root}"
exit "${status}"
