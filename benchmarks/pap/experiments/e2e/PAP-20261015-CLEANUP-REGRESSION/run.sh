#!/usr/bin/env bash
# Run one PAP 7PA1P C30 point for the cleanup A/B regression.
#
# Usage: bash run.sh <current|base> <repetition>
#
# "current" runs this checkout (the cleaned working tree). "base" runs the
# pre-cleanup commit from BASE_WORKTREE with its own NVSHMEM bridge build.
# Both use requested.env, which is byte-identical to the PAP settings of
# PAP-20260907-ALIGNED-C30.
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"

variant="${1:?variant must be current or base}"
repetition="${2:?repetition is required}"
case "${variant}" in
  current) source_root="${ROOT_DIR}" ;;
  base) source_root="${ROOT_DIR}/${BASE_WORKTREE}" ;;
  *) echo "ERROR: variant must be current or base" >&2; exit 2 ;;
esac
[[ -f "${source_root}/benchmarks/pap/scripts/run_pap_workload.sh" ]] \
  || { echo "ERROR: missing source tree ${source_root}" >&2; exit 2; }
# The launcher's CUDA Graph log audit requires ripgrep.
PATH="${PAP_EXTRA_PATH:+${PAP_EXTRA_PATH}:}${PATH}"
command -v rg >/dev/null \
  || { echo "ERROR: rg is required; set PAP_EXTRA_PATH to its directory" >&2; exit 2; }

run_root="${EXPERIMENT_DIR}/runs/$(date -u +%Y%m%dT%H%M%SZ)_${variant}_r${repetition}"
mkdir -p "${run_root}"
cp "${EXPERIMENT_DIR}/requested.env" "${run_root}/requested.env"

launch_env=(
  "HOME=${HOME}" "USER=${USER:-fei}" "PATH=${PATH}" "LANG=C.UTF-8"
  "RUN_ROOT=${run_root}"
  "RUN_LOG_DIR=${run_root}/service_logs"
  "PAP_AIPERF_OUTPUT_DIR=${run_root}/aiperf"
  "PYTHON_BIN=${ROOT_DIR}/.venv/bin/python"
  "VLLM_BIN=${ROOT_DIR}/.venv/bin/vllm"
  "AIPERF_BIN=${ROOT_DIR}/.venv-aiperf/bin/aiperf"
  "AIPERF_PYTHON=${ROOT_DIR}/.venv-aiperf/bin/python"
  "PAP_DYNAMO_SITE_PACKAGES=${ROOT_DIR}/.local/pap-dynamo-router"
  "PAP_NVSHMEM_NVCC=${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
)
if [[ "${variant}" == base ]]; then
  launch_env+=(
    "PYTHONPATH=${source_root}"
    "PAP_NVSHMEM_PREFIX=${ROOT_DIR}/${BASE_NVSHMEM_PREFIX}"
  )
else
  launch_env+=("PAP_NVSHMEM_PREFIX=${ROOT_DIR}/${CURRENT_NVSHMEM_PREFIX}")
fi

{
  printf 'variant=%s\nrepetition=%s\nsource_root=%s\n' \
    "${variant}" "${repetition}" "${source_root}"
  printf 'source_head=%s\n' "$(git -C "${source_root}" rev-parse HEAD)"
  printf 'source_dirty_files=%s\n' \
    "$(git -C "${source_root}" status --porcelain --untracked-files=no | wc -l)"
  printf '%s\n' "${launch_env[@]}"
} > "${run_root}/launch.env"
git -C "${source_root}" diff HEAD > "${run_root}/source.diff"

cd "${source_root}"
set +e
# shellcheck disable=SC2016  # expanded by the inner bash, not here
env -i "${launch_env[@]}" bash -c \
  'set -a; source "$1"; set +a; exec bash "$2"' cleanup-regression \
  "${run_root}/requested.env" \
  "${source_root}/benchmarks/pap/scripts/run_pap_workload.sh" \
  > "${run_root}/runner.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${run_root}/exit_code"
echo "${variant} r${repetition} exit=${status} run_root=${run_root}"
exit "${status}"
