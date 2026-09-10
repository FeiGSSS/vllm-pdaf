#!/usr/bin/env bash
# Read-only hardware snapshot. Does not clear status, change PCI config, or reboot.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $EUID == 0 ]] || { echo 'Run this script with sudo.' >&2; exit 2; }
OUT="${HERE}/hardware-snapshots/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT}"
umask 022
{
  date --iso-8601=ns
  cat /proc/sys/kernel/random/boot_id
  cat /proc/cmdline
} > "${OUT}/identity.txt"
for bdf in 0000:30:01.1 0000:31:00.0 0000:80:01.1 0000:81:00.0 0000:90:01.1 0000:91:00.0 0000:e0:01.1 0000:e1:00.0 0000:f0:01.1 0000:f1:00.0; do
  timeout 10 lspci -D -vv -s "${bdf}" > "${OUT}/${bdf}.txt" 2>&1 || {
    echo "Failed or timed out: ${bdf}" >> "${OUT}/errors.txt"
  }
done
dmidecode -t 9 > "${OUT}/slots.txt" 2>&1
journalctl -k -b --no-pager -o short-iso-precise > "${OUT}/kernel-current.log"
journalctl -k -b -1 --no-pager -o short-iso-precise > "${OUT}/kernel-previous.log"
journalctl --no-pager --all -o json SYSLOG_IDENTIFIER=systemd-pstore > "${OUT}/persistent-panic-journal.jsonl"
if command -v ipmitool >/dev/null 2>&1; then
  timeout 30 ipmitool sel elist > "${OUT}/bmc-sel.txt" 2>&1 || true
  timeout 30 ipmitool sdr elist > "${OUT}/bmc-sensors.txt" 2>&1 || true
fi
chmod a+r "${OUT}"/*
sync -f "${OUT}"
echo "Snapshot saved: ${OUT}"
