#!/usr/bin/env bash
# NIXL READ microbenchmark of one PD P->D KV handoff per context length.
#
# Usage: bash run.sh
#
# Order: same-NUMA 1->1, cross-NUMA 1->1, then 3->1 fan-in last. The kernel
# log is checked for PCIe/AER/GHES errors after every step; the first one stops
# the run (PAP-20260915-P2P-PCIE-INGRESS documents the host fatal reset).
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)"
source "${EXPERIMENT_DIR}/experiment.env"

run_root="${EXPERIMENT_DIR}/runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${run_root}"
started_at="$(date '+%Y-%m-%d %H:%M:%S')"
nvidia-smi topo -m > "${run_root}/topo.txt"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv \
  > "${run_root}/gpus.csv"
git -C "${ROOT_DIR}" rev-parse HEAD > "${run_root}/source_head"

cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source benchmarks/pap/scripts/configure_same_node_nixl.sh
pap_configure_same_node_nixl "${ROOT_DIR}"
env | grep -E '^(UCX|NIXL)_' | sort > "${run_root}/transport.env"

check_kernel_log() {
  if journalctl -k --since "${started_at}" --no-pager 2>/dev/null \
      | grep -Ei '\bA[E]R\b|GHES|Hardware Error|PCIe Bus Error' > "${run_root}/kernel_errors.log"; then
    echo "STOP: kernel PCIe error after $1; see ${run_root}/kernel_errors.log" >&2
    exit 3
  fi
}

probe() {
  "${ROOT_DIR}/.venv/bin/python" benchmarks/pap/tooling/nixl_read_write_probe.py \
    --initiator "$1" --peer "$2" --operation READ --regions "${REGIONS}" \
    --total-mib "$3" --segment-kib "${SEGMENT_KIB}" --repetitions "${REPETITIONS}" \
    2>>"${run_root}/probe_stderr.log" | grep '^operation='
}

results="${run_root}/results.txt"
for pair in "${SAME_NUMA_PAIR}" "${CROSS_NUMA_PAIR}"; do
  for size in ${CONTEXT_SIZES}; do
    tokens="${size%%:*}"; mib="${size##*:}"
    line="$(probe "${DECODE_GPU}" "${pair##*:}" "${mib}")"
    echo "case=${pair%%:*} context_tokens=${tokens} total_mib=${mib} ${line}" | tee -a "${results}"
    check_kernel_log "${pair%%:*} ${tokens}"
  done
done

for size in ${CONTEXT_SIZES}; do
  tokens="${size%%:*}"; mib="${size##*:}"
  pids=()
  for peer in ${FAN_IN_PEERS}; do
    probe "${DECODE_GPU}" "${peer}" "${mib}" > "${run_root}/fanin_${tokens}_${peer}.txt" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  for peer in ${FAN_IN_PEERS}; do
    echo "case=fanin3 context_tokens=${tokens} total_mib=${mib} $(cat "${run_root}/fanin_${tokens}_${peer}.txt")" \
      | tee -a "${results}"
  done
  check_kernel_log "fan-in ${tokens}"
done
echo "done run_root=${run_root}"
