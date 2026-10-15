#!/usr/bin/env bash
# Rebuild data/summary.json and data/requests.csv from the raw AIPerf archives
# committed on branch feature/pap-v026 of FeiGSSS/vllm-pdaf.
# Usage: make_data.sh <vllm-pdaf checkout> [git ref]
# Uses the experiment's own analyze.py (copied here as analyze_upstream.py),
# so the numbers are identical to PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/report.md.
set -euo pipefail
REPO=${1:?repo path}; REF=${2:-origin/feature/pap-v026}
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$HERE/../data
E=benchmarks/pap/experiments/e2e; RAW=$(mktemp -d)
ex() { mkdir -p "$RAW/$2"; git -C "$REPO" show "$REF:$E/$1" | tar xz -C "$RAW/$2"; }
A=PAP-20260907-ALIGNED-C30/results/archives
for r in dp8_r1 dp8_r2 6p2d_r1 6p2d_r2 4p4d_r1 4p4d_r2 2p6d_r1 2p6d_r2; do ex $A/$r.tar.gz $r; done
for r in run_1 run_2; do ex PAP-20260907-ALIGNED-C30-7P1D/results/archives/$r.tar.gz 7p1d_$r; done
for r in 7 8; do ex PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/results/archives/pap_native32k_r$r.tar.gz pap_r$r; done
run() { find "$RAW/$1" -name profile.json -path '*aiperf*' | head -1 | xargs dirname | xargs dirname; }
python3 -I "$HERE/analyze_upstream.py" --output-dir "$OUT" \
  --run dp8:r1="$(run dp8_r1)" --run dp8:r2="$(run dp8_r2)" \
  --run pd6p2d:r1="$(run 6p2d_r1)" --run pd6p2d:r2="$(run 6p2d_r2)" \
  --run pd4p4d:r1="$(run 4p4d_r1)" --run pd4p4d:r2="$(run 4p4d_r2)" \
  --run pd2p6d:r1="$(run 2p6d_r1)" --run pd2p6d:r2="$(run 2p6d_r2)" \
  --run pd7p1d:r1="$(run 7p1d_run_1)" --run pd7p1d:r2="$(run 7p1d_run_2)" \
  --run pap7pa1p_kv_migration:r7="$(run pap_r7)" --run pap7pa1p_kv_migration:r8="$(run pap_r8)"
rm -rf "$RAW"; echo "wrote $OUT/summary.json $OUT/requests.csv"
