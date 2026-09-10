#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../.." && pwd)"
config="${1:?usage: run.sh 6p2d|4p4d|2p6d}"
[[ "$config" =~ ^(6p2d|4p4d|2p6d)$ ]] || exit 2
set -a
# shellcheck source=/dev/null
source "${HERE}/${config}.env"
set +a
export DYNAMO_RUN_ID="${DYNAMO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_cross_host_${config}}"
export DYNAMO_RUN_ROOT="${HERE}/runs/${DYNAMO_RUN_ID}"
export DYNAMO_REMOTE_RUN_ROOT="${ROOT}/benchmarks/pap/experiments/_staging/${DYNAMO_RUN_ID}_remote"
[[ ! -e "${DYNAMO_RUN_ROOT}" ]] || { echo "Run already exists" >&2; exit 2; }
mkdir -p "${DYNAMO_RUN_ROOT}"
cp "${HERE}/${config}.env" "${DYNAMO_RUN_ROOT}/replay.env"
cp "${ROOT}/benchmarks/pap/scripts/run_cross_host_dynamo_7p1d_aiperf.sh" "${DYNAMO_RUN_ROOT}/runner_snapshot.sh"
exec bash "${ROOT}/benchmarks/pap/scripts/run_cross_host_dynamo_7p1d_aiperf.sh"
