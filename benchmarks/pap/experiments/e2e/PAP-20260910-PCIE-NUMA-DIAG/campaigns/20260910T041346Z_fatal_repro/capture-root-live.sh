#!/usr/bin/env bash
# Bounded, read-only PCIe register capture into the current run's mirrored logs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $EUID == 0 ]] || { echo 'Run with sudo.' >&2; exit 2; }
umask 022
deadline=$((SECONDS + 7200))
while (( SECONDS < deadline )); do
  run="$(cat "${HERE}/CURRENT_RUN")"
  [[ "${run}" == "${HERE}/runs/"* && -d "${run}" ]] || exit 2
  if [[ -f "${run}/status.json" ]]; then sleep 2; continue; fi
  {
    date --iso-8601=ns
    for bdf in 0000:30:01.1 0000:31:00.0 0000:80:01.1 0000:81:00.0 0000:90:01.1 0000:91:00.0 0000:e0:01.1 0000:e1:00.0 0000:f0:01.1 0000:f1:00.0; do
      echo "BDF ${bdf}"
      timeout 5 lspci -D -vv -s "${bdf}" || echo "READ_FAILED ${bdf}"
    done
  } >> "${run}/root-pcie-live.log" 2>&1
  chmod a+r "${run}/root-pcie-live.log"
  sync -f "${run}/root-pcie-live.log"
  sleep 2
done
