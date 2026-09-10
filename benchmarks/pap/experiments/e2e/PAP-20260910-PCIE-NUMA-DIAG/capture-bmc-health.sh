#!/usr/bin/env bash
# Read-only BMC collection through the local IPMI interface.
# No LAN credentials, sensor changes, SEL clearing, reset, or power commands.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPMI="${HERE}/tools/ipmitool/usr/bin/ipmitool"
LIBDIR="${HERE}/tools/ipmitool/usr/lib/x86_64-linux-gnu"
[[ $EUID == 0 ]] || { echo 'Run with sudo to read /dev/ipmi0.' >&2; exit 2; }
[[ -x "${IPMI}" && -c /dev/ipmi0 ]] || { echo 'Local IPMI tool/device unavailable.' >&2; exit 2; }
umask 022
OUT="${HERE}/hardware-snapshots/bmc-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "${OUT}"
failures=0
read_bmc() {
  local name="$1" code=0
  shift
  echo "Reading ${name} ..."
  timeout 60 env LD_LIBRARY_PATH="${LIBDIR}" "${IPMI}" -I open "$@" > "${OUT}/${name}.txt" 2>&1 || code=$?
  # Some ipmitool commands return zero despite protocol errors in their output.
  if (( code == 0 )) && grep -Eq 'Get Device ID command failed|Received a response with unexpected ID|Unable to establish|Could not open device|Error sending message' "${OUT}/${name}.txt"; then
    code=3
  fi
  printf '%s\t%s\n' "${name}" "${code}" >> "${OUT}/command-status.tsv"
  if (( code != 0 )); then failures=$((failures + 1)); fi
  sync -f "${OUT}"
}
{
  date --iso-8601=ns
  hostname
  cat /proc/sys/kernel/random/boot_id
  env LD_LIBRARY_PATH="${LIBDIR}" "${IPMI}" -V
} > "${OUT}/host-identity.txt"
read_bmc bmc-info mc info
read_bmc bmc-selftest mc selftest
read_bmc chassis-status chassis status
read_bmc current-sdr sdr elist all
read_bmc sensor-thresholds sensor list
read_bmc event-log sel elist
read_bmc event-log-info sel info
read_bmc bmc-clock sel time get
chmod a+r "${OUT}"/*
sync -f "${OUT}"
echo "Snapshot saved: ${OUT}"
if (( failures )); then
  echo "${failures} command(s) failed or timed out; details are preserved in command-status.tsv and the corresponding files."
  exit 2
fi
